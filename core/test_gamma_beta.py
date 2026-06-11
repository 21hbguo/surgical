import logging
import os
import warnings

import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
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
HEATMAP_SIZE = 112


def build_gamma_beta_parser():
    parser = build_test_parser()
    parser.add_argument("--vis_max_cases", type=int, default=5, help="max cases to visualize")
    parser.add_argument("--vis_output_dir", type=str, default=None, help="output dir override")
    parser.add_argument("--vis_topk_channels", type=int, default=0, help="deprecated; per-layer channel visualization always exports all channels")
    return parser


def create_inference_strategy(args, model, device):
    strategy_args = build_train_parser().parse_args(["--task", str(args.task)])
    for key, value in vars(args).items():
        setattr(strategy_args, key, value)
    return create_strategy(strategy_args.way, strategy_args, model, torch.optim.Adam(model.parameters(), lr=strategy_args.lr, betas=(0.9, 0.99), weight_decay=0.0001), device)


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
    is_depth = bool(depth_channels)
    test_dataset = BaseDataSets(
        base_dir=args.root_path, split='test', fold=fold,
        resize_size=tuple(args.resize_size), load_mode='path',
        num_classes=args.num_classes,
        depth_channels=depth_channels if is_depth else None,
        depth_uint=int(args.depth_uint), normalize_method=args.normalize,
        fold_map=fold_map, use_val=False, for_inference=True,
        is_depth=is_depth, task=args.task,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed) + (0 if fold is None else int(fold)))
    return test_dataset, DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=TEST_NUM_WORKERS, pin_memory=True, collate_fn=_collate_test_batch,
        generator=loader_generator,
    ), is_depth, depth_channels


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
            raw_depth = sample_batch.get(f"depth{use_depth}")
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
    for module in model.modules():
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
        has_delta = hasattr(guider, 'compute_delta')
        has_affine = hasattr(guider, 'compute_gamma_beta')

        if has_delta:
            def make_patched_delta(orig, li, guider_mod):
                def patched(rgb_feat, depth):
                    depth_zero = torch.zeros_like(depth)
                    delta = guider_mod.compute_delta(rgb_feat, depth)
                    delta_zero = guider_mod.compute_delta(rgb_feat, depth_zero)
                    output = rgb_feat + delta
                    output_zero = rgb_feat + delta_zero
                    records.append({
                        'layer_idx': li,
                        'vis_mode': 'delta',
                        'feat_in': rgb_feat.detach().cpu(),
                        'feat_mod': delta.detach().cpu(),
                        'feat_out': output.detach().cpu(),
                        'feat_out_zero': output_zero.detach().cpu(),
                        'feat_delta_depth': (output - output_zero).detach().cpu(),
                        'delta': delta.detach().cpu(),
                        'delta_zero': delta_zero.detach().cpu(),
                        'delta_diff': (delta - delta_zero).detach().cpu(),
                        'gamma': None,
                        'beta': None,
                        'gamma_zero': None,
                        'beta_zero': None,
                        'gamma_delta': None,
                        'beta_delta': None,
                    })
                    return output
                return patched

            guider.forward = make_patched_delta(orig_fwd, idx, guider)
        elif has_affine:
            def make_patched_fc(orig, li, guider_mod):
                def patched(rgb_feat, depth):
                    depth_zero = torch.zeros_like(depth)
                    gamma, beta = guider_mod.compute_gamma_beta(rgb_feat, depth)
                    gamma_zero, beta_zero = guider_mod.compute_gamma_beta(rgb_feat, depth_zero)
                    modulated_feat = gamma * rgb_feat + beta
                    modulated_feat_zero = gamma_zero * rgb_feat + beta_zero
                    output = rgb_feat + modulated_feat
                    output_zero = rgb_feat + modulated_feat_zero
                    records.append({
                        'layer_idx': li,
                        'vis_mode': 'affine',
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

            guider.forward = make_patched_fc(orig_fwd, idx, guider)
        else:
            def make_patched_simple(orig, li):
                def patched(rgb_feat, depth):
                    output = orig(rgb_feat, depth)
                    output_zero = orig(rgb_feat, torch.zeros_like(depth))
                    records.append({
                        'layer_idx': li,
                        'vis_mode': 'simple',
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

            guider.forward = make_patched_simple(orig_fwd, idx)

    return records, original_forwards


def _restore_depth_guiders(guiders, original_forwards):
    for guider, original_forward in zip(guiders, original_forwards):
        guider.forward = original_forward


def _heatmap_to_rgb(heatmap):
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


def _diff_to_rgb(diff_2d, size=HEATMAP_SIZE, vmax_override=None):
    diff_resized = cv2.resize(diff_2d.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    vmax = float(vmax_override) if vmax_override is not None else max(abs(diff_resized.min()), abs(diff_resized.max()))
    if vmax < 1e-8:
        return np.ones((size, size, 3), dtype=np.uint8) * 255
    norm = diff_resized / vmax
    norm = np.clip(norm, -1, 1)
    return np.stack([
        (np.clip(-norm, 0, 1) * 255).astype(np.uint8),
        (255 - np.abs(norm) * 255).astype(np.uint8),
        (np.clip(norm, 0, 1) * 255).astype(np.uint8),
    ], axis=-1)


def _resize_for_vis(arr, size=HEATMAP_SIZE):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    denom = arr.max()
    if denom > 1e-8:
        arr /= denom
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)


def _feat_to_heatmap_shared(feat_tensors, size=HEATMAP_SIZE):
    maps = [feat.mean(dim=0).numpy().astype(np.float32) for feat in feat_tensors]
    vmin = min(m.min() for m in maps)
    vmax = max(m.max() for m in maps)
    denom = vmax - vmin
    if denom < 1e-8:
        return [np.zeros((size, size), dtype=np.float32) for _ in maps]
    return [cv2.resize((m - vmin) / denom, (size, size), interpolation=cv2.INTER_LINEAR) for m in maps]


def _make_header(total_width):
    header_h = 40
    header = np.ones((header_h, total_width, 3), dtype=np.uint8) * 255
    labels = ['RGB', 'Depth', 'Feat_in', 'Feat_out', 'Delta']
    col_widths = [HEATMAP_SIZE] * len(labels)
    x_offset = 0
    for label, w in zip(labels, col_widths):
        cx = x_offset + w // 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(header, label, (cx - text_size[0] // 2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        x_offset += w
    return header


def _make_layer_row(record, rgb_img, depth_img):
    feat_in_t = record['feat_in'][0]
    feat_out_t = record['feat_out'][0]
    feat_in_map, feat_out_map = _feat_to_heatmap_shared([feat_in_t, feat_out_t])
    feat_in_bgr = _heatmap_to_rgb(feat_in_map)
    feat_out_bgr = _heatmap_to_rgb(feat_out_map)

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

    vis_mode = record.get('vis_mode', 'affine')
    if vis_mode == 'delta':
        delta_map = record['delta'][0].mean(dim=0).numpy().astype(np.float32)
        delta_view = _diff_to_rgb(delta_map, HEATMAP_SIZE)
        eff_tensor = record['delta'][0]
        eff_mean = float(eff_tensor.abs().mean().item())
        eff_std = float(eff_tensor.std().item())
        eff_maxabs = float(eff_tensor.abs().max().item())
        eff_rel = eff_mean / (float(feat_in_t.abs().mean().item()) + 1e-8)
        cv2.putText(delta_view, f'm={float(delta_map.mean()):+.4f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'std={eff_std:.4f}', (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'max={eff_maxabs:.4f}', (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'|delta|={eff_mean:.4f}', (4, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'r={eff_rel:.3f}', (4, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    else:
        delta_map = record['feat_delta_depth'][0].mean(dim=0).numpy().astype(np.float32)
        delta_view = _diff_to_rgb(delta_map, HEATMAP_SIZE)
        eff_tensor = record['feat_delta_depth'][0]
        eff_mean = float(eff_tensor.abs().mean().item())
        eff_std = float(eff_tensor.std().item())
        eff_maxabs = float(eff_tensor.abs().max().item())
        eff_rel = eff_mean / (float(feat_in_t.abs().mean().item()) + 1e-8)
        cv2.putText(delta_view, f'm={float(delta_map.mean()):+.4f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'std={eff_std:.4f}', (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'max={eff_maxabs:.4f}', (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'|delta|={eff_mean:.4f}', (4, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(delta_view, f'r={eff_rel:.3f}', (4, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    return np.hstack([rgb_resized, depth_vis, feat_in_bgr, feat_out_bgr, delta_view])


def visualize_gamma_beta(rgb_img, depth_img, records, save_path):
    if not records:
        return
    records.sort(key=lambda r: r['layer_idx'])

    total_width = 5 * HEATMAP_SIZE
    header = _make_header(total_width)
    rows = [header]
    for rec in records:
        rows.append(_make_layer_row(rec, rgb_img, depth_img))

    min_w = min(r.shape[1] for r in rows)
    rows = [r[:, :min_w] for r in rows]
    canvas = np.vstack(rows)
    cv2.imwrite(save_path, canvas)


def _save_per_channel_vis(record, output_dir, case_name):
    layer_idx = record['layer_idx']
    feat_in = record['feat_in'][0]
    feat_out = record['feat_out'][0]
    C = feat_in.shape[0]
    delta_np = record['delta'][0].numpy().astype(np.float32) if record.get('delta') is not None else record['feat_delta_depth'][0].numpy().astype(np.float32)
    score = np.abs(delta_np).mean(axis=(1, 2))
    order = np.argsort(-score)
    indices = order[:C]
    n_show = len(indices)

    feat_in_np = feat_in.numpy().astype(np.float32)
    feat_out_np = feat_out.numpy().astype(np.float32)
    shared_delta_vmax = float(np.max(np.abs(delta_np[indices]))) if n_show > 0 else 0.0

    cell_h = cell_w = 64
    text_w = 220
    n_cols = 3
    header_h = 30
    canvas = np.ones((header_h + n_show * cell_h, n_cols * cell_w + text_w, 3), dtype=np.uint8) * 255

    def _put_center(text, col, y_pos):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(canvas, text, (col * cell_w + (cell_w - tw) // 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    _put_center('feat_in', 0, 22)
    _put_center('feat_out', 1, 22)
    _put_center('delta', 2, 22)
    detail_title = 'all channels sorted by |delta|'
    cv2.putText(canvas, detail_title, (n_cols * cell_w + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f'shared_vmax={shared_delta_vmax:.4f}', (n_cols * cell_w + 10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (64, 64, 64), 1)

    for row_idx, ch in enumerate(indices):
        cell_maps = [feat_in_np[ch], feat_out_np[ch]]
        vmin = min(m.min() for m in cell_maps)
        vmax = max(m.max() for m in cell_maps)
        denom = vmax - vmin
        if denom < 1e-8:
            norm_maps = [np.zeros((cell_w, cell_h), dtype=np.float32) for _ in cell_maps]
        else:
            norm_maps = [cv2.resize((m - vmin) / denom, (cell_w, cell_h)) for m in cell_maps]
        in_bgr = _heatmap_to_rgb(norm_maps[0])
        out_bgr = _heatmap_to_rgb(norm_maps[1])
        delta_bgr = _diff_to_rgb(delta_np[ch], cell_w, vmax_override=shared_delta_vmax)

        y = header_h + row_idx * cell_h
        canvas[y:y + cell_h, 0:cell_w] = in_bgr
        canvas[y:y + cell_h, cell_w:2 * cell_w] = out_bgr
        canvas[y:y + cell_h, 2 * cell_w:3 * cell_w] = delta_bgr

        txt_x = n_cols * cell_w + 5
        cv2.putText(canvas, f'ch{ch}', (txt_x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        dd = float(delta_np[ch].mean())
        ds = float(delta_np[ch].std())
        dm = float(np.abs(delta_np[ch]).max())
        da = float(np.abs(delta_np[ch]).mean())
        cv2.putText(canvas, f'mean={dd:.4f}', (txt_x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
        cv2.putText(canvas, f'std={ds:.4f}', (txt_x, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 0), 1)
        cv2.putText(canvas, f'max={dm:.4f}', (txt_x, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (64, 64, 64), 1)
        cv2.putText(canvas, f'|mean|={da:.4f}', (txt_x, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 64, 0), 1)

    save_path = os.path.join(output_dir, f'{case_name}_L{layer_idx}_all_channels.png')
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

    save_path = os.path.join(output_dir, f'{case_name}_depth_guider.png')
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

    if args.num_folds is None or args.num_folds <= 1:
        requested_folds = [None]
    elif args.fold in (None, []):
        requested_folds = list(range(args.num_folds))
    elif -1 in args.fold:
        requested_folds = list(range(args.num_folds))
    else:
        requested_folds = sorted(set(args.fold))
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
        load_checkpoint(model, os.path.join(snapshot_path, 'model_best.pth' if context.checkpoint_type == 'best' else 'model_final.pth'))

        strategy = create_inference_strategy(args, model, device)
        _, test_loader, _, _ = _build_test_loader(args, fold, fold_map)

        output_dir = args.vis_output_dir or os.path.join(
            build_run_output_dir(args, mode='test', fold=fold), 'depth_guider_vis'
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
