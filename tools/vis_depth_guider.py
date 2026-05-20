"""
可视化 DepthGuider 的 delta 或 gamma/beta 值。
对 test 的第一张图，每层选 top-K 通道的特征图，标注 depth guider 输出并拼接。
用法与 test.py 一致，--pth best/final 自动解析 checkpoint 路径。
"""

import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from core.args import build_test_parser, finalize_test_args
from core.runtime import build_test_run_context, resolve_device
from data import BaseDataSets
from models.factory import create_model
from utils.common import load_checkpoint


# ── Hook 注册 ──────────────────────────────────────────────
class DepthGuiderHook:
    """注册到每个 DepthGuider，捕获调制量和输入特征。"""

    def __init__(self):
        self.mode = None
        self.gamma = None
        self.beta = None
        self.delta = None
        self.rgb_feat = None
        self.modulated_feat = None

    def __call__(self, module, input, output):
        rgb_feat = input[0]
        depth = input[1]
        if hasattr(module, "compute_delta"):
            self.mode = "delta"
            self.delta = module.compute_delta(rgb_feat, depth).detach().cpu()
            self.gamma = None
            self.beta = None
        else:
            self.mode = "affine"
            gamma, beta = module.compute_gamma_beta(rgb_feat, depth)
            self.gamma = gamma.detach().cpu()
            self.beta = beta.detach().cpu()
            self.delta = None
        self.rgb_feat = rgb_feat.detach().cpu()
        self.modulated_feat = output.detach().cpu()


def register_hooks(model):
    hooks = {}
    for i, guider in enumerate(model.depth_guiders):
        h = DepthGuiderHook()
        guider.register_forward_hook(h)
        hooks[f'layer{i+1}'] = h
    return hooks


# ── 可视化 ──────────────────────────────────────────────────
def normalize_feat(f):
    f = f.float()
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    return f


def select_topk_channels(feat_2d, k):
    """feat_2d: (C, H, W)，选方差最大的 k 个通道。"""
    var = feat_2d.var(dim=(1, 2))
    indices = var.argsort(descending=True)[:k]
    return indices.numpy()


def reduce_channel_values(feat):
    values = feat[0]
    if values.ndim == 1:
        return values.numpy()
    if values.ndim == 3:
        return values.mean(dim=(1, 2)).numpy()
    return values.reshape(values.shape[0], -1).mean(dim=1).numpy()


def draw_layer_viz(hook, layer_name, out_dir, topk=8, img_size=128):
    """画一层 top-K 通道的 depth guider 标注特征图，拼成一张大图。"""
    if hook.mode == "delta":
        delta = reduce_channel_values(hook.delta)
    else:
        gamma = reduce_channel_values(hook.gamma)
        beta = reduce_channel_values(hook.beta)
    feat = hook.rgb_feat[0]
    C = feat.shape[0]

    k = min(topk, C)
    indices = select_topk_channels(feat, k)

    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6))
    if k == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        ch_feat = normalize_feat(feat[idx])
        ch_resized = F.interpolate(ch_feat.unsqueeze(0).unsqueeze(0),
                                   size=(img_size, img_size),
                                   mode='bilinear', align_corners=False
                                   ).squeeze().numpy()

        primary_val = float(delta[idx]) if hook.mode == "delta" else float(gamma[idx])
        if primary_val > 0:
            color = (0.85, 0.15, 0.15)
        elif primary_val < 0:
            color = (0.15, 0.15, 0.85)
        else:
            color = (0.3, 0.3, 0.3)

        ax = axes[0, col]
        ax.imshow(ch_resized, cmap='viridis')
        ax.set_title(f'Ch {idx}', fontsize=9, fontweight='bold', color=color)
        ax.axis('off')

        ax.text(0.5, -0.08,
                f'δ={float(delta[idx]):+.3f}' if hook.mode == "delta" else f'γ={float(gamma[idx]):+.3f}\nβ={float(beta[idx]):+.3f}',
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color))

        mod_ch = normalize_feat(hook.modulated_feat[0][idx])
        mod_resized = F.interpolate(mod_ch.unsqueeze(0).unsqueeze(0),
                                    size=(img_size, img_size),
                                    mode='bilinear', align_corners=False
                                    ).squeeze().numpy()
        axes[1, col].imshow(mod_resized, cmap='viridis')
        axes[1, col].set_title('After Mod', fontsize=8, color='green')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Before', fontsize=11, fontweight='bold', rotation=0, labelpad=40)
    axes[1, 0].set_ylabel('After', fontsize=11, fontweight='bold', rotation=0, labelpad=40)

    if hook.mode == "delta":
        fig.suptitle(f'{layer_name}  |  δ range: [{delta.min():+.4f}, {delta.max():+.4f}]',
                     fontsize=11, fontweight='bold')
        save_path = os.path.join(out_dir, f'{layer_name}_delta.png')
    else:
        fig.suptitle(f'{layer_name}  |  γ range: [{gamma.min():+.4f}, {gamma.max():+.4f}]  |  '
                     f'β range: [{beta.min():+.4f}, {beta.max():+.4f}]',
                     fontsize=11, fontweight='bold')
        save_path = os.path.join(out_dir, f'{layer_name}_gamma_beta.png')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def draw_summary(hooks, out_dir):
    """画所有层的 depth guider 总览图。"""
    layer_names = sorted(hooks.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 2, figsize=(8, 3 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, 2)

    for row, name in enumerate(layer_names):
        hook = hooks[name]
        ax = axes[row, 0]
        ax.set_ylabel(name, fontsize=9, fontweight='bold')
        if hook.mode == "delta":
            delta = reduce_channel_values(hook.delta)
            ax.bar(range(len(delta)), delta,
                   color=['#d32f2f' if d > 0 else '#1565c0' if d < 0 else '#757575' for d in delta])
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            if row == 0:
                ax.set_title('δ (delta)', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax = axes[row, 1]
            ax.bar(range(len(delta)), abs(delta), color='#43a047')
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            if row == 0:
                ax.set_title('|δ|', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)
        else:
            gamma = reduce_channel_values(hook.gamma)
            beta = reduce_channel_values(hook.beta)
            ax.bar(range(len(gamma)), gamma,
                   color=['#d32f2f' if g > 0 else '#1565c0' if g < 0 else '#757575' for g in gamma])
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            if row == 0:
                ax.set_title('γ (gamma)', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax = axes[row, 1]
            ax.bar(range(len(beta)), beta,
                   color=['#e57373' if b > 0 else '#64b5f6' if b < 0 else '#bdbdbd' for b in beta])
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            if row == 0:
                ax.set_title('β (beta)', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)

    summary_title = 'DepthGuider δ Summary (per channel)' if hooks[layer_names[0]].mode == "delta" else 'DepthGuider γ / β Summary (per channel)'
    fig.suptitle(summary_title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'summary_delta.png' if hooks[layer_names[0]].mode == "delta" else 'summary_gamma_beta.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ── 主流程 ──────────────────────────────────────────────────
def main():
    parser = build_test_parser()
    parser.add_argument("--ckpt", type=str, default=None, help="手动指定 checkpoint 路径（可选，不指定则自动解析）")
    parser.add_argument("--topk", type=int, default=8, help="每层显示的通道数")
    parser.add_argument("--img_size", type=int, default=128, help="特征图缩放尺寸")
    args = parser.parse_args()
    finalize_test_args(args)

    # --ckpt 优先，否则和 test.py 一样自动解析
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        from core.test import parse_requested_folds
        from utils.common import build_run_output_dir
        context = build_test_run_context(args)
        checkpoint_name = 'model_best.pth' if context.checkpoint_type == 'best' else 'model_final.pth'
        fold = parse_requested_folds(args.fold, args.num_folds)
        fold = fold[0] if fold else None
        snapshot_path = build_run_output_dir(args, mode="train", fold=fold)
        ckpt_path = os.path.join(snapshot_path, checkpoint_name)
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint not found: {ckpt_path}')
            print(f'Args: --task {args.task} --way {args.way} --exp {args.exp} '
                  f'--labeled_num {args.labeled_num} --pretrain {args.pretrain} '
                  f'--lr {args.lr} --pth {args.pth} --fold {fold}')
            print(f'Use --ckpt /path/to/model_best.pth to specify manually.')
            raise SystemExit(1)

    device = resolve_device(args)
    model = create_model(args)
    info = load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')

    # 注册 hooks
    hooks = register_hooks(model)

    # 构建数据集，取第一张
    strategy = None
    if args.way:
        from core.test import create_inference_strategy
        strategy = create_inference_strategy(args, model, device)

    depth_channels = args.use_depth if args.use_depth else None
    ds = BaseDataSets(
        base_dir=args.root_path,
        split='test',
        fold=None,
        resize_size=tuple(args.resize_size),
        load_mode='path',
        num_classes=args.num_classes,
        depth_channels=depth_channels if depth_channels else None,
        depth_uint=int(args.depth_uint),
        normalize_method=args.normalize,
        task=args.task,
    )
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v.unsqueeze(0)

    # 构造输入
    volume = sample["image"].to(device)
    use_depth = int(args.use_depth or 0)
    if strategy is not None:
        depth_tensor = strategy._get_depth_tensor(sample)
    elif use_depth:
        depth_key = f"depth{use_depth}"
        raw_depth = sample.get(depth_key)
        depth_tensor = raw_depth.to(device) if raw_depth is not None else None
    else:
        depth_tensor = None

    if depth_tensor is not None:
        if args.way == "only_depth_input":
            volume = depth_tensor.repeat(1, 3, 1, 1) if depth_tensor.shape[1] == 1 else depth_tensor
        else:
            volume = torch.cat([volume, depth_tensor], dim=1)

    # 前向
    with torch.no_grad():
        _ = model(volume)

    # 输出目录
    out_dir = os.path.join(os.path.dirname(ckpt_path), 'depth_guider_vis')
    os.makedirs(out_dir, exist_ok=True)

    # 画每层
    for layer_name, hook in sorted(hooks.items()):
        save_path = draw_layer_viz(hook, layer_name, out_dir, topk=args.topk, img_size=args.img_size)
        print(f'  saved: {save_path}')

    # 总览图
    save_path = draw_summary(hooks, out_dir)
    print(f'  saved: {save_path}')
    print(f'\nDone! Output: {out_dir}')


if __name__ == '__main__':
    main()
