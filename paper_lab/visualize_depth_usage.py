"""可视化 DepthGuiderV4 各层深度信息使用情况（论文级）

输出：depth_usage_per_layer.png (300 dpi, 论文规范)
"""

import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.networks.unet import UNet_DepthGuiderV4_GeoRiskSPC
from data.transforms import _normalize_array, _resize_numpy_array

DATA_ROOT = "/home/guo/project/ssl4mis/data/endovis2017"
CHECKPOINT_PATH = "/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1/GeoRiskSPC_DGv4/40_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C/f0/model_best.pth"
OUTPUT_DIR = "/home/guo/project/ssl4mis/code_all_vibe_v2/paper_lab/picture"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CASES = ["seq_4_frame228", "seq_8_frame263", "seq_7_frame232"]


def load_model(checkpoint_path):
    model = UNet_DepthGuiderV4_GeoRiskSPC(
        in_chns=4, class_num=2, filter_num=16,
        dropout_rate=0.3, noise_std=0.1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"]["model"]
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def load_sample(case):
    img_path = os.path.join(DATA_ROOT, "data", "images", f"{case}.png")
    depth_path = os.path.join(DATA_ROOT, "data", "depth1c_slices_uint16", f"{case}.png")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_numpy_array(img, (224, 224))
    img = _normalize_array(img, method="255")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth.max() > 1.0:
        depth = depth / 65535.0 if depth.max() > 255 else depth / 255.0
    depth = _resize_numpy_array(depth, (224, 224))

    rgb_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    return rgb_t, depth_t, img, depth


def extract_depth_features(model, rgb, depth):
    encoder = model.encoder
    depth_feats, geom_feats = [], []

    def make_hook(storage):
        def hook_fn(module, input, output):
            storage.append(output.detach())
        return hook_fn

    hooks = []
    for dg in encoder.depth_guiders:
        hooks.append(dg.depth_encoder.register_forward_hook(make_hook(depth_feats)))
        hooks.append(dg.geometry_encoder.register_forward_hook(make_hook(geom_feats)))

    with torch.no_grad():
        x0 = encoder.depth_guiders[0](encoder.in_conv(rgb), depth)
        x1 = encoder.depth_guiders[1](encoder.down1(x0), depth)
        x2 = encoder.depth_guiders[2](encoder.down2(x1), depth)
        x3 = encoder.depth_guiders[3](encoder.down3(x2), depth)
        x4 = encoder.depth_guiders[4](encoder.down4(x3), depth)

    for h in hooks:
        h.remove()
    return depth_feats, geom_feats


def to_numpy_heatmap(tensor_2d, global_max=1.0):
    arr = tensor_2d.detach().cpu().numpy()
    arr = arr / (global_max + 1e-8)
    arr = np.clip(arr, 0, 1)
    return arr


def visualize_depth_usage(model, cases, save_path):
    n_cases = len(cases)
    n_layers = 5
    target_h, target_w = 224, 224

    all_data = []
    for case in cases:
        rgb, depth, img_np, depth_np = load_sample(case)
        rgb, depth = rgb.to(DEVICE), depth.to(DEVICE)
        df, gf = extract_depth_features(model, rgb, depth)
        all_data.append((case, img_np, depth_np, df, gf))

    # 全局最大值
    global_max_d = max(
        df[0].mean(dim=0).max().item()
        for _, _, _, df_list, _ in all_data
        for df in df_list
    )
    global_max_g = max(
        gf[0].mean(dim=0).max().item()
        for _, _, _, _, gf_list in all_data
        for gf in gf_list
    )

    # === 紧凑布局: 每个样本单独一行 ===
    fig, axes = plt.subplots(
        n_cases, 1 + n_layers,
        figsize=(10, 2.8 * n_cases),
        dpi=300,
        gridspec_kw={"wspace": 0.05, "hspace": 0.25}
    )

    for case_idx in range(n_cases):
        case_name, img_np, depth_np, df_list, gf_list = all_data[case_idx]

        # 输入列 (RGB + Depth 叠加显示)
        ax_input = axes[case_idx, 0]
        ax_input.imshow(img_np)
        # 叠加半透明 depth
        depth_vis = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        ax_input.imshow(depth_vis, cmap="jet", alpha=0.3)
        ax_input.set_title(case_name.replace("_", " "), fontsize=8, pad=2)
        ax_input.axis("off")

        if case_idx == 0:
            ax_input.set_ylabel("Input\n+ Depth", fontsize=8, fontweight='bold',
                              rotation=0, labelpad=55, ha="right", va="center")

        # L0-L4 depth_feat
        for layer_idx in range(n_layers):
            ax = axes[case_idx, 1 + layer_idx]
            df_mean = df_list[layer_idx][0].mean(dim=0)
            df_resized = F.interpolate(
                df_mean.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w), mode="bilinear", align_corners=False
            )[0, 0]
            arr = to_numpy_heatmap(df_resized, global_max_d)

            if df_mean.max() < 1e-6:
                ax.imshow(np.ones_like(arr)*0.6, cmap="gray", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "OFF", ha="center", va="center",
                       fontsize=9, fontweight='bold', color='gray',
                       transform=ax.transAxes, alpha=0.7)
            else:
                ax.imshow(arr, cmap="jet", vmin=0, vmax=1)
            ax.axis("off")

            # 列标签 (只在第一行显示)
            if case_idx == 0:
                ax.set_title(f"L{layer_idx}", fontsize=9, fontweight='bold', pad=2)

        # 行标签
        if case_idx == 0:
            axes[case_idx, 1].set_ylabel("Depth\nEnc.", fontsize=8, fontweight='bold',
                                        rotation=0, labelpad=45, ha="right", va="center")

    # 添加 colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Activation", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)
    print("Visualizing depth usage per layer...")
    save_path = os.path.join(OUTPUT_DIR, "depth_usage_per_layer.png")
    visualize_depth_usage(model, CASES, save_path)
    print("Done!")


if __name__ == "__main__":
    main()
