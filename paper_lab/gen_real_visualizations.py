"""Generate publication-quality risk map visualizations from real model inference.

Produces an 8-panel figure per sample:
  (a) Input RGB  (b) Depth map  (c) Teacher uncertainty  (d) Depth discontinuity
  (e) Geometry-semantic conflict  (f) Risk map + M_r overlay  (g) Pseudo-label error
  (h) Prediction overlay on GT boundary

Output: paper/figures/risk_map_real_{case}.pdf
"""

import os
import sys
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.transforms import _normalize_array, _resize_numpy_array
from models.networks.unet import UNet_DepthGuiderV4_GeoRiskSPC
from strategies.semi_georisk_spc import (
    _local_relative_normalization,
    _sobel_gradient_magnitude_single,
    _sobel_gradient_magnitude,
    _normalize_map,
)

# ============ Configuration ============
DATA_ROOT = "/home/guo/project/ssl4mis/data/endovis2017"
CHECKPOINT_PATH = (
    "/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
    "/GeoRiskSPC_DGv4/20_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C/f0/model_best.pth"
)
OUTPUT_DIR = "/home/guo/project/ssl4mis/code_all_vibe_v2/paper/figures"
NUM_SAMPLES = 4
RESIZE_SIZE = (224, 224)
NUM_CLASSES = 2
FILTER_NUM = 16

RISK_WINDOW_SIZE = 16
RISK_TAU_R = 0.5
RISK_TAU_C = 0.9
RISK_LAMBDA_C = 0.5
RISK_DROPOUT_RATE = 0.3
RISK_NOISE_STD = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matplotlib publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

RISK_CMAP = LinearSegmentedColormap.from_list("risk", [
    (0.0, "#2166ac"), (0.3, "#67a9cf"), (0.5, "#f7f7f7"),
    (0.7, "#ef8a62"), (1.0, "#b2182b"),
])


# ============ Data loading ============
def load_single_sample(data_root, case):
    img_path = os.path.join(data_root, "data", "images", f"{case}.png")
    lab_path = os.path.join(data_root, "data", "labels_task1_binary", f"{case}.png")
    depth_path = os.path.join(data_root, "data", "depth1c_slices_uint16", f"{case}.png")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_numpy_array(img, RESIZE_SIZE)
    img_255 = img.copy()
    img_norm = _normalize_array(img.copy(), method="255")

    lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    lab = _resize_numpy_array(lab, RESIZE_SIZE)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth.max() > 1.0:
        depth = depth / 65535.0 if depth.max() > 255 else depth / 255.0
    depth = _resize_numpy_array(depth, RESIZE_SIZE)

    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    lab_t = torch.from_numpy(lab).long()
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()

    return {
        "image": img_t, "label": lab_t, "depth1": depth_t,
        "case": case, "img_rgb": img_255, "depth_np": depth,
    }


def load_test_samples(data_root, num_samples):
    list_path = os.path.join(data_root, "test_slices.list")
    with open(list_path) as f:
        all_cases = [line.strip() for line in f if line.strip()]
    random.seed(42)
    selected = random.sample(all_cases, min(num_samples, len(all_cases)))
    return [load_single_sample(data_root, c) for c in selected]


# ============ Model loading ============
def load_model(checkpoint_path):
    model = UNet_DepthGuiderV4_GeoRiskSPC(
        in_chns=4, class_num=NUM_CLASSES, filter_num=FILTER_NUM,
        dropout_rate=RISK_DROPOUT_RATE, noise_std=RISK_NOISE_STD,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(DEVICE).eval()
    print(f"  Loaded {len(cleaned)} params, best_perf={ckpt.get('best_performance', 'N/A')}")
    return model


# ============ Risk map computation ============
def compute_risk_components(depth, teacher_pred):
    d_rel = _local_relative_normalization(depth, RISK_WINDOW_SIZE)
    G_d = _sobel_gradient_magnitude_single(d_rel)
    U_t = -torch.sum(teacher_pred * torch.log(teacher_pred + 1e-8), dim=1, keepdim=True)
    B_p = _sobel_gradient_magnitude(teacher_pred).mean(dim=1, keepdim=True)

    G_d_n = _normalize_map(G_d)
    B_p_n = _normalize_map(B_p)
    C_conf = torch.abs(G_d_n - B_p_n)
    U_t_n = _normalize_map(U_t)
    R = U_t_n * G_d_n + RISK_LAMBDA_C * C_conf

    conf = teacher_pred.max(dim=1, keepdim=True)[0]
    M_r = (R > RISK_TAU_R).float()
    M_l = (conf > RISK_TAU_C).float() * (1.0 - M_r)

    return {
        "d_rel": d_rel, "G_d": G_d, "G_d_n": G_d_n,
        "U_t": U_t, "U_t_n": U_t_n,
        "B_p": B_p, "B_p_n": B_p_n,
        "C_conf": C_conf, "R": R, "M_r": M_r, "M_l": M_l,
    }


# ============ Visualization ============
def _add_panel_label(ax, label, loc="tl"):
    """Add a white bold label with dark outline."""
    x, y, ha = (0.03, 0.97, "left") if loc == "tl" else (0.03, 0.03, "left")
    txt = ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
                  fontweight="bold", ha=ha, va="top" if loc == "tl" else "bottom",
                  color="white")
    txt.set_path_effects([
        patheffects.withStroke(linewidth=2.5, foreground="black"),
    ])


def generate_figure(sample, teacher_pred, risk, save_path):
    """Generate 2x4 publication figure."""
    img_rgb = sample["img_rgb"].astype(np.float32) / 255.0
    depth_np = sample["depth_np"]
    label_np = sample["label"].numpy()
    pred_np = teacher_pred[0].argmax(dim=0).cpu().numpy()
    prob_fg = teacher_pred[0, 1].cpu().numpy()

    # Pseudo-label error map (teacher prediction vs GT)
    pseudo_error = (pred_np != label_np).astype(np.float32)

    # GT boundary
    gt_boundary = _compute_boundary_numpy(label_np)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))

    # (a) Input RGB
    axes[0, 0].imshow(img_rgb)
    _add_panel_label(axes[0, 0], "(a)")
    axes[0, 0].set_title("Input Image", fontsize=9)
    axes[0, 0].axis("off")

    # (b) Depth map
    im_d = axes[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=1)
    _add_panel_label(axes[0, 1], "(b)")
    axes[0, 1].set_title("Depth Map", fontsize=9)
    axes[0, 1].axis("off")

    # (c) Teacher uncertainty
    U_np = risk["U_t_n"][0, 0].cpu().numpy()
    im_u = axes[0, 2].imshow(U_np, cmap="hot", vmin=0, vmax=1)
    _add_panel_label(axes[0, 2], "(c)")
    axes[0, 2].set_title("Teacher Uncertainty $U_t$", fontsize=9)
    axes[0, 2].axis("off")

    # (d) Depth discontinuity
    G_np = risk["G_d_n"][0, 0].cpu().numpy()
    im_g = axes[0, 3].imshow(G_np, cmap="inferno", vmin=0, vmax=1)
    _add_panel_label(axes[0, 3], "(d)")
    axes[0, 3].set_title("Depth Discontinuity $G_d$", fontsize=9)
    axes[0, 3].axis("off")

    # (e) Geometry-semantic conflict
    C_np = risk["C_conf"][0, 0].cpu().numpy()
    im_c = axes[1, 0].imshow(C_np, cmap="YlOrRd", vmin=0, vmax=1)
    _add_panel_label(axes[1, 0], "(e)")
    axes[1, 0].set_title("Conflict $|G_d - B_p|$", fontsize=9)
    axes[1, 0].axis("off")

    # (f) Risk map + M_r overlay
    R_np = risk["R"][0, 0].cpu().numpy()
    Mr_np = risk["M_r"][0, 0].cpu().numpy()
    axes[1, 1].imshow(img_rgb, alpha=0.5)
    im_r = axes[1, 1].imshow(R_np, cmap=RISK_CMAP, vmin=0, vmax=1, alpha=0.7)
    # M_r boundary contour
    if Mr_np.sum() > 0:
        Mr_boundary = _compute_boundary_numpy(Mr_np)
        axes[1, 1].contour(Mr_boundary, levels=[0.5], colors=["yellow"], linewidths=0.8)
    _add_panel_label(axes[1, 1], "(f)")
    axes[1, 1].set_title("Risk Map $R$ + $M_r$", fontsize=9)
    axes[1, 1].axis("off")
    plt.colorbar(im_r, ax=axes[1, 1], fraction=0.046, pad=0.02)

    # (g) Pseudo-label error map
    axes[1, 2].imshow(img_rgb, alpha=0.4)
    error_vis = np.ma.masked_where(pseudo_error < 0.5, pseudo_error)
    axes[1, 2].imshow(error_vis, cmap="Reds", vmin=0, vmax=1, alpha=0.8)
    _add_panel_label(axes[1, 2], "(g)")
    n_err = int(pseudo_error.sum())
    axes[1, 2].set_title(f"Pseudo-label Error ({n_err}px)", fontsize=9)
    axes[1, 2].axis("off")

    # (h) Prediction overlay + GT boundary
    overlay = img_rgb.copy()
    # Prediction in green channel
    overlay[:, :, 1] = np.where(pred_np > 0,
                                np.clip(overlay[:, :, 1] * 0.3 + 0.5, 0, 1),
                                overlay[:, :, 1])
    # GT boundary in red
    overlay[:, :, 0] = np.where(gt_boundary > 0, 1.0, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(gt_boundary > 0, 0.0, overlay[:, :, 1])
    overlay[:, :, 2] = np.where(gt_boundary > 0, 0.0, overlay[:, :, 2])
    axes[1, 3].imshow(overlay)
    _add_panel_label(axes[1, 3], "(h)")
    dice_val = _dice_numpy(pred_np > 0, label_np > 0)
    axes[1, 3].set_title(f"Prediction (Dice={dice_val:.3f})", fontsize=9)
    axes[1, 3].axis("off")

    plt.tight_layout(pad=0.3)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _compute_boundary_numpy(mask, band=2):
    from scipy.ndimage import binary_dilation, binary_erosion
    mask_bin = mask > 0 if mask.max() <= 1 else mask.astype(bool)
    dilated = binary_dilation(mask_bin, iterations=band)
    eroded = binary_erosion(mask_bin, iterations=band)
    return (dilated & ~eroded).astype(np.float32)


def _dice_numpy(pred, gt, smooth=1e-6):
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


# ============ Main ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)

    print(f"Loading {NUM_SAMPLES} test samples...")
    samples = load_test_samples(DATA_ROOT, NUM_SAMPLES)

    for i, sample in enumerate(samples):
        case = sample["case"]
        print(f"\n[{i+1}/{len(samples)}] {case}")

        image = sample["image"].to(DEVICE)
        depth = sample["depth1"].to(DEVICE)
        volume = torch.cat([image, depth], dim=1)

        with torch.no_grad():
            output = model(volume)
            if isinstance(output, tuple):
                output = output[0]
            teacher_pred = torch.softmax(output, dim=1)

        with torch.no_grad():
            risk = compute_risk_components(depth, teacher_pred)

        Mr_ratio = risk["M_r"].mean().item()
        dice = _dice_numpy(
            teacher_pred[0].argmax(dim=0).cpu().numpy() > 0,
            sample["label"].numpy() > 0,
        )
        print(f"  Dice={dice:.4f}, M_r ratio={Mr_ratio:.4f}")

        save_path = os.path.join(OUTPUT_DIR, f"risk_map_real_{case}.pdf")
        generate_figure(sample, teacher_pred, risk, save_path)

    print(f"\nDone! Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
