import logging
import os
import re
import warnings

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.args import build_test_parser, build_train_parser, finalize_test_args, format_args_for_logging
from core.runtime import build_test_run_context, resolve_device
from data import BaseDataSets
from models.factory import create_model
from strategies import create_strategy
from utils.common import build_run_output_dir, get_fold_seqs_from_path, load_checkpoint

warnings.filterwarnings('ignore', message='.*timm.models.layers.*')

TEST_NUM_WORKERS = 4
LAYER_NAMES = ['e1', 'e2', 'e3', 'e4', 'e5']
HEATMAP_SIZE = 112
BAR_WIDTH = 160
BAR_HEIGHT = 112


def build_gamma_beta_parser():
    parser = build_test_parser()
    parser.add_argument("--vis_max_cases", type=int, default=5, help="max cases to visualize")
    parser.add_argument("--vis_output_dir", type=str, default=None, help="output dir override")
    return parser


def create_inference_strategy(args, model, device):
    strategy_args = build_train_parser().parse_args(["--task", str(args.task)])
    for key, value in vars(args).items():
        setattr(strategy_args, key, value)
    optimizer = torch.optim.Adam(model.parameters(), lr=strategy_args.lr, betas=(0.9, 0.99), weight_decay=0.0001)
    return create_strategy(strategy_args.way, strategy_args, model, optimizer, device)


def parse_requested_folds(raw_folds, num_folds):
    if num_folds is None or num_folds <= 1:
        return [None]
    if raw_folds in (None, [], ()):
        return list(range(num_folds))
    if isinstance(raw_folds, (int, str)):
        raw_folds = [raw_folds]
    tokens = []
    for value in raw_folds:
        text = str(value).strip()
        if not text:
            continue
        tokens.extend(part.strip() for part in text.split(',') if part.strip())
    if not tokens or '-1' in tokens:
        return list(range(num_folds))
    folds = []
    for token in tokens:
        try:
            fold = int(token)
        except ValueError as exc:
            raise ValueError(f'Invalid fold value: {token!r}') from exc
        if fold < 0 or fold >= num_folds:
            raise ValueError(f'Fold {fold} is out of range for num_folds={num_folds}')
        folds.append(fold)
    return sorted(set(folds))


def _collate_test_batch(batch):
    images, labels, cases, original_shapes, original_images = [], [], [], [], []
    depth3_list, depth1_list = [], []
    has_depth3 = False
    has_depth1 = False
    for item in batch:
        images.append(item['image'])
        labels.append(item['label'])
        cases.append(item['case'])
        original_shapes.append(item.get('original_shape'))
        original_images.append(item.get('original_image'))
        depth3 = item.get('depth3')
        depth1 = item.get('depth1')
        depth3_list.append(depth3)
        depth1_list.append(depth1)
        has_depth3 = has_depth3 or depth3 is not None
        has_depth1 = has_depth1 or depth1 is not None
    collated = {
        'image': images, 'label': labels, 'case': cases,
        'original_shape': original_shapes, 'original_image': original_images,
    }
    if has_depth3:
        collated['depth3'] = depth3_list
    if has_depth1:
        collated['depth1'] = depth1_list
    return collated


def _build_test_loader(args, fold, fold_map):
    depth_channels = args.use_depth if args.use_depth else None
    depth_uint = int(args.depth_uint)
    is_depth = bool(depth_channels)
    test_dataset = BaseDataSets(
        base_dir=args.root_path, split='test', fold=fold,
        resize_size=tuple(args.resize_size), load_mode='path',
        num_classes=args.num_classes,
        depth_channels=depth_channels if is_depth else None,
        depth_uint=depth_uint, normalize_method=args.normalize,
        fold_map=fold_map, use_val=False, for_inference=True,
        is_depth=is_depth, task=args.task,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=TEST_NUM_WORKERS, pin_memory=True, collate_fn=_collate_test_batch,
    )
    return test_dataset, test_loader, is_depth, depth_channels


def _predict_logits(args, model, strategy, sample_batch, device, use_grad=False):
    strategy_model = strategy.model if strategy is not None else None
    if strategy is not None and strategy_model is None:
        outputs = strategy.validation_step(sample_batch)
    else:
        target_model = strategy_model if strategy_model is not None else model
        volume = sample_batch["image"].to(device)
        use_depth = int(args.use_depth or 0)
        if strategy is not None:
            depth_tensor = strategy._get_depth_tensor(sample_batch)
        elif use_depth:
            depth_key = f"depth{use_depth}"
            raw_depth = sample_batch.get(depth_key)
            depth_tensor = raw_depth.to(device) if raw_depth is not None else None
        else:
            depth_tensor = None
        if depth_tensor is not None:
            if args.way == "only_depth_input":
                volume = depth_tensor.repeat(1, 3, 1, 1) if depth_tensor.shape[1] == 1 else depth_tensor
            else:
                volume = torch.cat([volume, depth_tensor], dim=1)
        if use_grad:
            outputs = target_model(volume)
        else:
            with torch.no_grad():
                outputs = target_model(volume)
    if not isinstance(outputs, tuple):
        return outputs
    if args.way == "urpc":
        return outputs[0]
    if args.way == "w2s":
        return outputs[0][0]
    return outputs[0]


def _find_depth_guiders(model):
    for name, module in model.named_modules():
        if hasattr(module, 'depth_guiders'):
            return module.depth_guiders
    return None


def _patch_depth_guiders(guiders):
    records = []
    original_forwards = []

    for guider in guiders:
        original_forwards.append(guider.forward)

    for idx, guider in enumerate(guiders):
        orig_fwd = original_forwards[idx]
        layer_idx = idx
        has_fc = hasattr(guider, 'fc_gamma') and hasattr(guider, 'fc_beta')

        if has_fc:
            fc_gamma = guider.fc_gamma
            fc_beta = guider.fc_beta
            depth_enc = guider.depth_encoder

            def make_patched_fc(orig, li, enc, g_mod, b_mod):
                def patched(rgb_feat, depth):
                    B, C, H, W = rgb_feat.shape
                    if depth.shape[2:] != (H, W):
                        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
                    depth_zero = torch.zeros_like(depth)
                    depth_feat = enc(depth).view(B, -1)
                    depth_feat_zero = enc(depth_zero).view(B, -1)
                    gamma = g_mod(depth_feat).view(B, C, 1, 1)
                    beta = b_mod(depth_feat).view(B, C, 1, 1)
                    gamma_zero = g_mod(depth_feat_zero).view(B, C, 1, 1)
                    beta_zero = b_mod(depth_feat_zero).view(B, C, 1, 1)
                    modulated_feat = gamma * rgb_feat + beta
                    modulated_feat_zero = gamma_zero * rgb_feat + beta_zero
                    output = modulated_feat + rgb_feat
                    output_zero = modulated_feat_zero + rgb_feat
                    records.append({
                        'layer_idx': li,
                        'feat_in': rgb_feat.detach().cpu(),
                        'feat_mod': modulated_feat.detach().cpu(),
                        'feat_out': output.detach().cpu(),
                        'feat_out_zero': output_zero.detach().cpu(),
                        'feat_delta_depth': (output - output_zero).detach().cpu(),
                        'gamma': gamma.detach().cpu(),
                        'beta': beta.detach().cpu(),
                        'gamma_zero': gamma_zero.detach().cpu(),
                        'beta_zero': beta_zero.detach().cpu(),
                        'gamma_delta': (gamma - gamma_zero).detach().cpu(),
                        'beta_delta': (beta - beta_zero).detach().cpu(),
                    })
                    return output
                return patched

            guider.forward = make_patched_fc(orig_fwd, layer_idx, depth_enc, fc_gamma, fc_beta)
        else:
            def make_patched_simple(orig, li):
                def patched(rgb_feat, depth):
                    output = orig(rgb_feat, depth)
                    output_zero = orig(rgb_feat, torch.zeros_like(depth))
                    records.append({
                        'layer_idx': li,
                        'feat_in': rgb_feat.detach().cpu(),
                        'feat_mod': output.detach().cpu(),
                        'feat_out': output.detach().cpu(),
                        'feat_out_zero': output_zero.detach().cpu(),
                        'feat_delta_depth': (output - output_zero).detach().cpu(),
                        'gamma': None,
                        'beta': None,
                        'gamma_zero': None,
                        'beta_zero': None,
                        'gamma_delta': None,
                        'beta_delta': None,
                    })
                    return output
                return patched

            guider.forward = make_patched_simple(orig_fwd, layer_idx)

    return records, original_forwards


def _restore_depth_guiders(guiders, original_forwards):
    for guider, original_forward in zip(guiders, original_forwards):
        guider.forward = original_forward


def _heatmap_to_rgb(heatmap):
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


def _diff_to_rgb(diff_2d, size=HEATMAP_SIZE):
    diff_resized = cv2.resize(diff_2d.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    vmax = max(abs(diff_resized.min()), abs(diff_resized.max()))
    if vmax < 1e-8:
        return np.ones((size, size, 3), dtype=np.uint8) * 255
    norm = diff_resized / vmax
    norm = np.clip(norm, -1, 1)
    r = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(-norm, 0, 1) * 255).astype(np.uint8)
    g = (255 - np.abs(norm) * 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def _effect_to_rgb(effect_2d, size=HEATMAP_SIZE):
    effect_resized = cv2.resize(effect_2d.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    vmax = float(effect_resized.max())
    if vmax < 1e-8:
        return np.ones((size, size, 3), dtype=np.uint8) * 255
    return _heatmap_to_rgb(effect_resized / vmax)


def _resize_for_vis(arr, size=HEATMAP_SIZE):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    denom = arr.max()
    if denom > 1e-8:
        arr /= denom
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)


def _feat_to_heatmap(feat_tensor):
    mean_map = feat_tensor.mean(dim=0).numpy()
    return _resize_for_vis(mean_map)


def _feat_to_heatmap_paired(feat_a, feat_b, size=HEATMAP_SIZE):
    map_a = feat_a.mean(dim=0).numpy().astype(np.float32)
    map_b = feat_b.mean(dim=0).numpy().astype(np.float32)
    vmin = min(map_a.min(), map_b.min())
    vmax = max(map_a.max(), map_b.max())
    denom = vmax - vmin
    if denom < 1e-8:
        return np.zeros((size, size), dtype=np.float32), np.zeros((size, size), dtype=np.float32)
    norm_a = (map_a - vmin) / denom
    norm_b = (map_b - vmin) / denom
    return cv2.resize(norm_a, (size, size), interpolation=cv2.INTER_LINEAR), cv2.resize(norm_b, (size, size), interpolation=cv2.INTER_LINEAR)


def _feat_to_heatmap_shared(feat_tensors, size=HEATMAP_SIZE):
    maps = [feat.mean(dim=0).numpy().astype(np.float32) for feat in feat_tensors]
    vmin = min(m.min() for m in maps)
    vmax = max(m.max() for m in maps)
    denom = vmax - vmin
    if denom < 1e-8:
        return [np.zeros((size, size), dtype=np.float32) for _ in maps]
    return [cv2.resize((m - vmin) / denom, (size, size), interpolation=cv2.INTER_LINEAR) for m in maps]


def _make_bar_chart(values, title, compare_values=None, width=BAR_WIDTH, height=BAR_HEIGHT):
    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
    n = len(values)
    x = np.arange(n)
    if compare_values is None:
        colors = ['#2196F3' if v >= 0 else '#F44336' for v in values]
        ax.bar(x, values, color=colors, width=0.8)
    else:
        compare_values = np.asarray(compare_values)
        bar_w = 0.38
        ax.bar(x - bar_w / 2, values, color='#1E88E5', width=bar_w, label='real')
        ax.bar(x + bar_w / 2, compare_values, color='#FB8C00', width=bar_w, alpha=0.85, label='zero')
        ax.legend(loc='lower right', fontsize=4, frameon=False, borderaxespad=0.1, handlelength=1.2)
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(labelsize=5)
    ax.set_xlim(-0.5, n - 0.5)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if compare_values is None:
        ax.axhline(y=mean_val, color='green', linestyle='--', linewidth=0.5)
        text = f'mean={mean_val:.3f}\nstd={std_val:.3f}'
    else:
        delta = values - compare_values
        text = f'r={mean_val:.3f}\nz={np.mean(compare_values):.3f}\n|d|={np.mean(np.abs(delta)):.3f}'
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=5, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def _make_header(total_width):
    header_h = 40
    header = np.ones((header_h, total_width, 3), dtype=np.uint8) * 255
    labels = ['RGB', 'Depth', 'Feat_in', 'Feat_out', 'Feat_zero', 'Depth_eff_abs', 'Gamma(r/z)', 'Beta(r/z)']
    col_widths = [HEATMAP_SIZE, HEATMAP_SIZE, HEATMAP_SIZE, HEATMAP_SIZE, HEATMAP_SIZE, HEATMAP_SIZE, BAR_WIDTH, BAR_WIDTH]
    x_offset = 0
    for label, w in zip(labels, col_widths):
        cx = x_offset + w // 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(header, label, (cx - text_size[0] // 2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        x_offset += w
    return header


def _make_layer_row(record, rgb_img, depth_img, layer_idx):
    feat_in_t = record['feat_in'][0]
    feat_out_t = record['feat_out'][0]
    feat_out_zero_t = record['feat_out_zero'][0]
    feat_in_map, feat_out_map, feat_zero_map = _feat_to_heatmap_shared([feat_in_t, feat_out_t, feat_out_zero_t])
    feat_in_bgr = _heatmap_to_rgb(feat_in_map)
    feat_out_bgr = _heatmap_to_rgb(feat_out_map)
    feat_zero_bgr = _heatmap_to_rgb(feat_zero_map)
    effect = record['feat_delta_depth'][0].abs().mean(dim=0).numpy().astype(np.float32)
    effect_bgr = _effect_to_rgb(effect, HEATMAP_SIZE)
    eff_mean = float(record['feat_delta_depth'][0].abs().mean().item())
    eff_rel = eff_mean / (float(feat_in_t.abs().mean().item()) + 1e-8)
    cv2.putText(effect_bgr, f'm={eff_mean:.4f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(effect_bgr, f'r={eff_rel:.3f}', (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    rgb_resized = cv2.resize(rgb_img, (HEATMAP_SIZE, HEATMAP_SIZE))
    if len(rgb_resized.shape) == 2:
        rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_GRAY2BGR)
    elif rgb_resized.shape[2] == 4:
        rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGRA2BGR)

    depth_resized = cv2.resize(depth_img, (HEATMAP_SIZE, HEATMAP_SIZE))
    if len(depth_resized.shape) == 2:
        depth_vis = _heatmap_to_rgb(_resize_for_vis(depth_resized.astype(np.float32)))
    else:
        depth_vis = depth_resized

    gamma = record['gamma']
    gamma_zero = record['gamma_zero']
    beta = record['beta']
    beta_zero = record['beta_zero']
    if gamma is not None and gamma.ndim >= 2:
        gamma_vals = gamma[0].squeeze().numpy()
        gamma_zero_vals = gamma_zero[0].squeeze().numpy()
    else:
        gamma_vals = np.zeros(1)
        gamma_zero_vals = np.zeros(1)
    if beta is not None and beta.ndim >= 2:
        beta_vals = beta[0].squeeze().numpy()
        beta_zero_vals = beta_zero[0].squeeze().numpy()
    else:
        beta_vals = np.zeros(1)
        beta_zero_vals = np.zeros(1)

    C = len(gamma_vals)
    gamma_img = _make_bar_chart(gamma_vals, f'L{layer_idx} gamma (C={C})', compare_values=gamma_zero_vals)
    beta_img = _make_bar_chart(beta_vals, f'L{layer_idx} beta (C={C})', compare_values=beta_zero_vals)

    row = np.hstack([rgb_resized, depth_vis, feat_in_bgr, feat_out_bgr, feat_zero_bgr, effect_bgr, gamma_img, beta_img])
    return row


def visualize_gamma_beta(rgb_img, depth_img, records, save_path):
    if not records:
        return
    records.sort(key=lambda r: r['layer_idx'])

    total_width = 6 * HEATMAP_SIZE + 2 * BAR_WIDTH
    header = _make_header(total_width)
    rows = [header]
    for rec in records:
        row = _make_layer_row(rec, rgb_img, depth_img, rec['layer_idx'])
        rows.append(row)

    min_w = min(r.shape[1] for r in rows)
    rows = [r[:, :min_w] for r in rows]
    canvas = np.vstack(rows)
    cv2.imwrite(save_path, canvas)


def _save_per_channel_vis(record, output_dir, case_name):
    layer_idx = record['layer_idx']
    feat_in = record['feat_in'][0]
    feat_out = record['feat_out'][0]
    feat_out_zero = record['feat_out_zero'][0]
    C = feat_in.shape[0]
    n_show = C

    feat_in_np = feat_in.numpy().astype(np.float32)
    feat_out_np = feat_out.numpy().astype(np.float32)
    feat_out_zero_np = feat_out_zero.numpy().astype(np.float32)
    diff_np = feat_out_np - feat_out_zero_np

    gamma_vals = record['gamma'][0].squeeze().numpy() if record['gamma'] is not None else None
    beta_vals = record['beta'][0].squeeze().numpy() if record['beta'] is not None else None
    gamma_zero_vals = record['gamma_zero'][0].squeeze().numpy() if record['gamma_zero'] is not None else None
    beta_zero_vals = record['beta_zero'][0].squeeze().numpy() if record['beta_zero'] is not None else None
    gamma_delta_vals = record['gamma_delta'][0].squeeze().numpy() if record['gamma_delta'] is not None else None
    beta_delta_vals = record['beta_delta'][0].squeeze().numpy() if record['beta_delta'] is not None else None

    cell_h = cell_w = 64
    text_w = 300
    n_cols = 4
    header_h = 30
    canvas = np.ones((header_h + n_show * cell_h, n_cols * cell_w + text_w, 3), dtype=np.uint8) * 255

    def _put_center(text, col, y_pos):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(canvas, text, (col * cell_w + (cell_w - tw) // 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    _put_center('feat_in', 0, 22)
    _put_center('feat_out', 1, 22)
    _put_center('feat_zero', 2, 22)
    _put_center('depth_eff', 3, 22)
    cv2.putText(canvas, 'g/b real zero delta', (n_cols * cell_w + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for ch in range(n_show):
        cell_maps = [feat_in_np[ch], feat_out_np[ch], feat_out_zero_np[ch]]
        vmin = min(m.min() for m in cell_maps)
        vmax = max(m.max() for m in cell_maps)
        denom = vmax - vmin
        if denom < 1e-8:
            norm_maps = [np.zeros((cell_w, cell_h), dtype=np.float32) for _ in cell_maps]
        else:
            norm_maps = [cv2.resize((m - vmin) / denom, (cell_w, cell_h)) for m in cell_maps]
        in_bgr = _heatmap_to_rgb(norm_maps[0])
        out_bgr = _heatmap_to_rgb(norm_maps[1])
        zero_bgr = _heatmap_to_rgb(norm_maps[2])
        diff_bgr = _diff_to_rgb(diff_np[ch], cell_w)

        y = header_h + ch * cell_h
        canvas[y:y + cell_h, 0:cell_w] = in_bgr
        canvas[y:y + cell_h, cell_w:2 * cell_w] = out_bgr
        canvas[y:y + cell_h, 2 * cell_w:3 * cell_w] = zero_bgr
        canvas[y:y + cell_h, 3 * cell_w:4 * cell_w] = diff_bgr

        g = gamma_vals[ch] if gamma_vals is not None else float('nan')
        gz = gamma_zero_vals[ch] if gamma_zero_vals is not None else float('nan')
        dg = gamma_delta_vals[ch] if gamma_delta_vals is not None else float('nan')
        b = beta_vals[ch] if beta_vals is not None else float('nan')
        bz = beta_zero_vals[ch] if beta_zero_vals is not None else float('nan')
        db = beta_delta_vals[ch] if beta_delta_vals is not None else float('nan')
        txt_x = n_cols * cell_w + 5
        cv2.putText(canvas, f'ch{ch}', (txt_x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(canvas, f'g={g:.4f} gz={gz:.4f} dg={dg:.4f}', (txt_x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
        cv2.putText(canvas, f'b={b:.4f} bz={bz:.4f} db={db:.4f}', (txt_x, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 0), 1)
        cv2.putText(canvas, f'|eff|={(np.abs(diff_np[ch]).mean()):.4f}', (txt_x, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (64, 64, 64), 1)

    save_path = os.path.join(output_dir, f'{case_name}_L{layer_idx}_channels.png')
    cv2.imwrite(save_path, canvas)


def run_one_sample(args, model, strategy, device, sample_batch, case_name, output_dir):
    logger = logging.getLogger(__name__)
    target_model = strategy.model if strategy is not None else model
    guiders = _find_depth_guiders(target_model)
    if guiders is None:
        logger.error("No DepthGuiders found in model")
        return
    os.makedirs(output_dir, exist_ok=True)

    patched_records, original_forwards = _patch_depth_guiders(list(guiders))

    strategy.eval() if strategy is not None else model.eval()
    try:
        with torch.no_grad():
            _predict_logits(args, model, strategy, sample_batch, device, use_grad=False)
    finally:
        _restore_depth_guiders(list(guiders), original_forwards)

    if not patched_records:
        logger.warning("No DepthGuider records captured for %s", case_name)
        return

    rgb_tensor = sample_batch['image'][0]
    if rgb_tensor.shape[0] >= 3:
        rgb_np = rgb_tensor[:3].permute(1, 2, 0).numpy()
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_np = (rgb_tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)

    depth_tensor = sample_batch.get('depth1')
    if depth_tensor is None:
        depth_tensor = sample_batch.get('depth3')
    if depth_tensor is not None:
        if depth_tensor.dim() == 4:
            depth_np = depth_tensor[0, 0].numpy()
        else:
            depth_np = depth_tensor[0].numpy()
    else:
        depth_np = np.zeros(rgb_np.shape[:2], dtype=np.float32)

    save_path = os.path.join(output_dir, f'{case_name}_gamma_beta.png')
    visualize_gamma_beta(rgb_np, depth_np, patched_records, save_path)
    logger.info('Saved: %s', save_path)

    for rec in patched_records:
        _save_per_channel_vis(rec, output_dir, case_name)


def main():
    args = finalize_test_args(build_gamma_beta_parser().parse_args())
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    logger.info('Args: %s', format_args_for_logging(args))
    logger.info('Strategy: %s', args.way)

    requested_folds = parse_requested_folds(args.fold, args.num_folds)
    args.fold = None
    context = build_test_run_context(args)
    fold_map = get_fold_seqs_from_path(args.root_path, args.task) if args.num_folds and args.num_folds > 1 else {}

    device = resolve_device(args)
    max_cases = max(0, int(args.vis_max_cases))

    for fold in requested_folds:
        fold_label = f"f{0 if fold is None else fold}"
        snapshot_path = args.snapshot_path or build_run_output_dir(args, mode='train', fold=fold)
        logger.info('Fold %s | snapshot: %s', fold_label, snapshot_path)

        model = create_model(args).to(device)
        checkpoint_name = 'model_best.pth' if context.checkpoint_type == 'best' else 'model_final.pth'
        checkpoint_path = os.path.join(snapshot_path, checkpoint_name)
        load_checkpoint(model, checkpoint_path)

        strategy = create_inference_strategy(args, model, device)
        test_dataset, test_loader, _, _ = _build_test_loader(args, fold, fold_map)

        output_dir = args.vis_output_dir or os.path.join(
            build_run_output_dir(args, mode='test', fold=fold), 'gamma_beta_vis'
        )
        os.makedirs(output_dir, exist_ok=True)

        saved = 0
        for batch in tqdm(test_loader, desc=f'Fold {fold_label}'):
            if saved >= max_cases:
                break
            for index, case in enumerate(batch['case']):
                if saved >= max_cases:
                    break
                sample_batch = {"image": batch["image"][index].unsqueeze(0)}
                for depth_key in ("depth3", "depth1"):
                    depth_values = batch.get(depth_key)
                    if depth_values is None:
                        continue
                    depth_value = depth_values[index]
                    if depth_value is not None:
                        sample_batch[depth_key] = depth_value.unsqueeze(0)
                sample_batch["label"] = batch["label"][index]
                run_one_sample(args, model, strategy, device, sample_batch, case, output_dir)
                saved += 1

        logger.info('Fold %s: saved %d visualizations to %s', fold_label, saved, output_dir)


if __name__ == '__main__':
    main()
