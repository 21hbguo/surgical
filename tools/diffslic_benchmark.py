#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import mark_boundaries

try:
    from tools.diffSLIC import DiffSLIC, spixel_upsampling
except ImportError:  # pragma: no cover - direct script execution fallback
    from diffSLIC import DiffSLIC, spixel_upsampling


def load_image_as_bchw(image_path: Path, resize_height: int | None = None, resize_width: int | None = None) -> tuple[torch.Tensor, np.ndarray]:
    pil_img = Image.open(image_path)
    if resize_height and resize_width:
        pil_img = pil_img.resize((resize_width, resize_height), Image.BILINEAR)

    if pil_img.mode in ("I", "I;16", "F"):
        img_np = np.array(pil_img, dtype=np.float32)
        img_np = img_np - img_np.min()
        if img_np.max() > 0:
            img_np = img_np / img_np.max()
        vis_np = np.repeat(img_np[..., None], 3, axis=2)
        tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
        return tensor.contiguous(), vis_np.astype(np.float32)

    if pil_img.mode == "L":
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        vis_np = np.repeat(img_np[..., None], 3, axis=2)
        tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
        return tensor.contiguous(), vis_np.astype(np.float32)

    pil_img = pil_img.convert("RGB")
    img_np = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.contiguous(), img_np.astype(np.float32)


def build_position_features(height: int, width: int, n_freq: int, pos_weight: float) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0)

    freqs = 2 ** torch.arange(n_freq, dtype=torch.float32)
    shape = coords.shape[:-1] + (-1,)
    scaled = (coords[..., None, :] * freqs[..., None]).reshape(shape)
    scaled = torch.stack([scaled, scaled + 0.5 * torch.pi], dim=-2).reshape(shape)
    embedded = torch.sin(scaled).permute(0, 3, 1, 2).contiguous()
    return embedded * pos_weight


def compute_superpixel_labels(
    image_bchw: torch.Tensor,
    n_spix: int,
    n_iter: int,
    tau: float,
    candidate_radius: int,
    n_freq: int,
    pos_weight: float,
) -> np.ndarray:
    _, _, height, width = image_bchw.shape
    pos_feats = build_position_features(height, width, n_freq=n_freq, pos_weight=pos_weight)
    inputs = torch.cat([image_bchw, pos_feats], dim=1)

    model = DiffSLIC(
        n_spixels=n_spix,
        n_iter=n_iter,
        tau=tau,
        candidate_radius=candidate_radius,
        normalize=True,
        stable=True,
    )
    model.eval()

    with torch.no_grad():
        clst_feats, p2s_assign, _ = model(inputs)
        height_s, width_s = clst_feats.shape[-2:]
        hard_assign = F.one_hot(
            p2s_assign.argmax(dim=1), num_classes=(2 * candidate_radius + 1) ** 2
        ).permute(0, 3, 1, 2).contiguous().float()
        label_seed = torch.arange(height_s * width_s, dtype=torch.float32).reshape(1, 1, height_s, width_s)
        labels = spixel_upsampling(label_seed, hard_assign, candidate_radius=candidate_radius)
    return labels[0, 0].long().cpu().numpy()


def overlay_superpixels(vis_np: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return mark_boundaries(vis_np, labels, color=(1.0, 0.0, 0.0), mode="thick").astype(np.float32)


def average_value_superpixels(vis_np: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vis_np, dtype=np.float32)
    for label_id in np.unique(labels):
        mask = labels == label_id
        out[mask] = vis_np[mask].mean(axis=0)
    return out


def run_visualize(args):
    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    image_bchw, vis_np = load_image_as_bchw(args.image)
    labels = compute_superpixel_labels(
        image_bchw=image_bchw,
        n_spix=args.n_spix,
        n_iter=args.n_iter,
        tau=args.tau,
        candidate_radius=args.candidate_radius,
        n_freq=args.n_freq,
        pos_weight=args.pos_weight,
    )
    overlay = overlay_superpixels(vis_np, labels)
    merged = np.concatenate([vis_np, overlay], axis=1)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.save_path, merged)
    print(f"Saved: {args.save_path}")
    if not args.no_show:
        plt.figure(figsize=(14, 6))
        plt.imshow(merged, cmap="gray" if image_bchw.shape[1] == 1 else None)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def run_param_grid(args):
    for image_path in args.image:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

    baseline = {
        "n_iter": 5,
        "tau": 0.01,
        "candidate_radius": 1,
        "n_freq": 2,
        "pos_weight": 1.0,
    }
    param_grid = {
        "n_iter": [1, 3, 5, 8],
        "tau": [0.005, 0.01, 0.05, 0.1],
        "candidate_radius": [1, 2, 3, 4],
        "n_freq": [1, 2, 4, 6],
        "pos_weight": [0.1, 0.5, 1.0, 2.0],
    }

    row_names = ["n_iter", "tau", "candidate_radius", "n_freq", "pos_weight"]
    n_rows = len(row_names)
    n_cols = len(next(iter(param_grid.values())))
    n_images = len(args.image)
    fig, axes = plt.subplots(n_rows * n_images, n_cols, figsize=(4 * n_cols, 3.6 * n_rows * n_images), squeeze=False)
    fig_sp, axes_sp = plt.subplots(n_rows * n_images, n_cols, figsize=(4 * n_cols, 3.6 * n_rows * n_images), squeeze=False)

    for image_idx, image_path in enumerate(args.image):
        image_bchw, vis_np = load_image_as_bchw(image_path, args.resize_height, args.resize_width)
        row_offset = image_idx * n_rows
        for row_idx, param_name in enumerate(row_names):
            for col_idx, value in enumerate(param_grid[param_name]):
                cur = dict(baseline)
                cur[param_name] = value
                labels = compute_superpixel_labels(
                    image_bchw=image_bchw,
                    n_spix=args.n_spix,
                    n_iter=cur["n_iter"],
                    tau=cur["tau"],
                    candidate_radius=cur["candidate_radius"],
                    n_freq=cur["n_freq"],
                    pos_weight=cur["pos_weight"],
                )
                ax = axes[row_offset + row_idx][col_idx]
                ax.imshow(overlay_superpixels(vis_np, labels))
                ax.axis("off")
                ax.set_title(f"{param_name}={value}", fontsize=10)

                ax_sp = axes_sp[row_offset + row_idx][col_idx]
                ax_sp.imshow(average_value_superpixels(vis_np, labels))
                ax_sp.axis("off")
                ax_sp.set_title(f"{param_name}={value}", fontsize=10)

    fig.suptitle("DiffSLIC Parameter Grid", fontsize=14)
    fig_sp.suptitle("DiffSLIC Superpixel Maps", fontsize=14)
    plt.tight_layout()
    fig_sp.tight_layout()

    if not args.no_save:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=180, bbox_inches="tight")
        args.save_spixel_path.parent.mkdir(parents=True, exist_ok=True)
        fig_sp.savefig(args.save_spixel_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {args.save_path}")
        print(f"Saved: {args.save_spixel_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_sp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiffSLIC benchmark/visualization toolkit.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_viz = subparsers.add_parser("visualize", help="Single-image overlay visualization")
    parser_viz.add_argument("--image", type=Path, default=Path("vis_depth_triplet.png"))
    parser_viz.add_argument("--n-spix", type=int, default=200)
    parser_viz.add_argument("--n-iter", type=int, default=1)
    parser_viz.add_argument("--tau", type=float, default=0.01)
    parser_viz.add_argument("--candidate-radius", type=int, default=1)
    parser_viz.add_argument("--n-freq", type=int, default=2)
    parser_viz.add_argument("--pos-weight", type=float, default=1.0)
    parser_viz.add_argument("--save-path", type=Path, default=Path("outputs/diffslic_visualize.png"))
    parser_viz.add_argument("--no-show", action="store_true")

    parser_grid = subparsers.add_parser("param-grid", help="Parameter matrix visualization")
    parser_grid.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=[
            Path("/home/guo/project/ssl4mis/data/endovis2017/data/depth1c_slices/seq_1_frame003.png"),
            Path("/home/guo/project/ssl4mis/data/endovis2017/data/depth3c_slices/seq_1_frame003.png"),
            Path("/home/guo/project/ssl4mis/data/endovis2017/data/images/seq_1_frame003.png"),
        ],
    )
    parser_grid.add_argument("--n-spix", type=int, default=200)
    parser_grid.add_argument("--resize-height", type=int, default=224)
    parser_grid.add_argument("--resize-width", type=int, default=224)
    parser_grid.add_argument("--save-path", type=Path, default=Path("outputs/diffslic_param_grid.png"))
    parser_grid.add_argument("--save-spixel-path", type=Path, default=Path("outputs/diffslic_param_grid_spixels.png"))
    parser_grid.add_argument("--no-show", action="store_true")
    parser_grid.add_argument("--no-save", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "visualize":
        run_visualize(args)
    elif args.mode == "param-grid":
        run_param_grid(args)


if __name__ == "__main__":
    main()
