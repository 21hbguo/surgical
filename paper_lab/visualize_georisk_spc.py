"""GeoRisk-SPC 全模块可视化脚本

可视化内容：
1. Risk Map 组件 (depth, G_d, U_t, B_p, C_conf, R, M_r, M_l)
2. DepthGuiderV4 中间特征
3. 特征扰动前后对比
4. Encoder 各层特征 (PCA 降维)
5. 边界对比 (B_p vs G_d + 冲突区域)

输出保存到 paper_lab/picture/
"""

import os
import sys
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import BaseDataSets
from models.networks.unet import UNet_DepthGuiderV4_GeoRiskSPC
from strategies.semi_georisk_spc import (
    _local_relative_normalization,
    _sobel_gradient_magnitude_single,
    _sobel_gradient_magnitude,
    _normalize_map,
)

# ============ 配置 ============
DATA_ROOT = "/home/guo/project/ssl4mis/data/endovis2017"
CHECKPOINT_PATH = "/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1/GeoRiskSPC_DGv4/40_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C/f0/model_best.pth"
OUTPUT_DIR = "/home/guo/project/ssl4mis/code_all_vibe_v2/paper_lab/picture"
NUM_SAMPLES = 5
RESIZE_SIZE = (224, 224)
NUM_CLASSES = 2
FILTER_NUM = 16

# Risk map 参数
RISK_WINDOW_SIZE = 16
RISK_TAU_R = 0.5
RISK_TAU_C = 0.9
RISK_LAMBDA_C = 0.5
RISK_DROPOUT_RATE = 0.3
RISK_NOISE_STD = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_single_sample(data_root, case):
    """直接加载单个样本，跳过 BaseDataSets 的全量预加载"""
    import cv2
    from data.transforms import _normalize_array, _resize_numpy_array

    img_path = os.path.join(data_root, "data", "images", f"{case}.png")
    lab_path = os.path.join(data_root, "data", "labels_task1_binary", f"{case}.png")
    depth_path = os.path.join(data_root, "data", "depth1c_slices_uint16", f"{case}.png")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_numpy_array(img, RESIZE_SIZE)
    img = _normalize_array(img, method="255")

    lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    lab = _resize_numpy_array(lab, RESIZE_SIZE)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth.max() > 1.0:
        depth = depth / 65535.0 if depth.max() > 255 else depth / 255.0
    depth = _resize_numpy_array(depth, RESIZE_SIZE)

    # 转 tensor
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
    lab_t = torch.from_numpy(lab).long()
    depth_t = torch.from_numpy(depth).unsqueeze(0).float()

    return {"image": img_t, "label": lab_t, "depth1": depth_t, "case": case}


def load_test_samples(data_root, num_samples=5):
    """从 test_slices.list 随机加载样本"""
    list_path = os.path.join(data_root, "test_slices.list")
    with open(list_path, "r") as f:
        all_cases = [line.strip() for line in f if line.strip()]

    random.seed(42)
    selected = random.sample(all_cases, min(num_samples, len(all_cases)))

    samples = []
    for case in selected:
        sample = load_single_sample(data_root, case)
        samples.append(sample)

    return samples


def load_model(checkpoint_path):
    """加载训练好的模型"""
    model = UNet_DepthGuiderV4_GeoRiskSPC(
        in_chns=4,
        class_num=NUM_CLASSES,
        filter_num=FILTER_NUM,
        dropout_rate=RISK_DROPOUT_RATE,
        noise_std=RISK_NOISE_STD,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # 解析 state_dict
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 处理各种 key 前缀
    cleaned = {}
    for k, v in state_dict.items():
        # 移除 _orig_mod. 前缀 (torch.compile)
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        # 移除 module. 前缀 (DataParallel)
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Checkpoint best_performance: {checkpoint.get('best_performance', 'N/A')}")
    print(f"  Loaded {len(cleaned)} parameters")
    return model


def to_uint8_heatmap(tensor_2d, colormap=cv2.COLORMAP_JET):
    """[H, W] tensor -> BGR uint8 heatmap"""
    arr = tensor_2d.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    return cv2.applyColorMap(arr, colormap)


def to_uint8_binary(mask_2d):
    """[H, W] binary tensor -> white/black uint8"""
    arr = (mask_2d.detach().cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def put_label(img, text, pos=(4, 16)):
    """在图像上添加标签"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


# ============ 可视化 1: Risk Map 组件 ============
def visualize_risk_components(depth, teacher_pred, save_path):
    """Risk Map 各组件可视化"""
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

    # 预测图：所有非背景类别的预测 (argmax，排除类别0)
    pred_argmax = teacher_pred[0].argmax(dim=0).float()  # [H, W]
    pred_nonbg = (pred_argmax > 0).float()  # 所有非背景区域

    # 布局：3行 × 3列
    panels = [
        ("Depth", depth[0, 0], False),
        ("Prediction", pred_nonbg, True),
        ("M_r (high-risk)", M_r[0, 0], True),
        ("G_d (depth edge)", G_d[0, 0], False),
        ("B_p (pred edge)", B_p[0, 0], False),
        ("C_conf (conflict)", C_conf[0, 0], False),
        ("U_t (entropy)", U_t[0, 0], False),
        ("R (risk map)", R[0, 0], False),
        ("M_l (low-risk)", M_l[0, 0], True),
    ]

    imgs = []
    for title, tensor, is_binary in panels:
        img = to_uint8_binary(tensor) if is_binary else to_uint8_heatmap(tensor)
        img = put_label(img, title)
        imgs.append(img)

    row1 = np.hstack(imgs[:3])
    row2 = np.hstack(imgs[3:6])
    row3 = np.hstack(imgs[6:9])
    panel = np.vstack([row1, row2, row3])
    cv2.imwrite(str(save_path), panel)
    print(f"  Saved: {save_path}")


# ============ 可视化 2: DepthGuiderV4 中间特征 ============
def visualize_depth_guider_features(model, volume, save_path):
    """DepthGuiderV4 各层中间特征"""
    rgb = volume[:, :3, :, :]
    depth = volume[:, 3:4, :, :]

    encoder = model.encoder
    features = []

    # Hook 来捕获中间特征
    depth_feats = []
    geom_feats = []
    scale_weights = []

    def make_hook(name, storage):
        def hook_fn(module, input, output):
            storage.append(output.detach())
        return hook_fn

    # 注册 hooks
    hooks = []
    for i, dg in enumerate(encoder.depth_guiders):
        hooks.append(dg.depth_encoder.register_forward_hook(make_hook(f"depth_{i}", depth_feats)))
        hooks.append(dg.geometry_encoder.register_forward_hook(make_hook(f"geom_{i}", geom_feats)))

    with torch.no_grad():
        # 手动前向以获取各层特征
        x0 = encoder.depth_guiders[0](encoder.in_conv(rgb), depth)
        x1 = encoder.depth_guiders[1](encoder.down1(x0), depth)
        x2 = encoder.depth_guiders[2](encoder.down2(x1), depth)
        x3 = encoder.depth_guiders[3](encoder.down3(x2), depth)
        x4 = encoder.depth_guiders[4](encoder.down4(x3), depth)

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 可视化 depth_feat 和 geom_feat (每层取 mean)
    target_h, target_w = volume.shape[2:]
    panels = []

    for i, (df, gf) in enumerate(zip(depth_feats, geom_feats)):
        df_mean = df[0].mean(dim=0)
        gf_mean = gf[0].mean(dim=0)
        df_resized = F.interpolate(df_mean.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False)[0, 0]
        gf_resized = F.interpolate(gf_mean.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False)[0, 0]

        panels.append((f"L{i} depth_feat", df_resized))
        panels.append((f"L{i} geom_feat", gf_resized))

    imgs = []
    for title, tensor in panels[:8]:  # 只显示前 4 层
        img = to_uint8_heatmap(tensor)
        img = put_label(img, title)
        imgs.append(img)

    row1 = np.hstack(imgs[:4])
    row2 = np.hstack(imgs[4:8])
    panel = np.vstack([row1, row2])
    cv2.imwrite(str(save_path), panel)
    print(f"  Saved: {save_path}")


# ============ 可视化 3: 特征扰动对比 ============
def visualize_perturbation(model, volume, risk_mask, save_path):
    """特征扰动前后对比"""
    target_h, target_w = volume.shape[2:]

    with torch.no_grad():
        rgb = volume[:, :3, :, :]
        depth = volume[:, 3:4, :, :]
        feature = model.encoder(rgb, depth)
        feat_clean = feature[4]
        p_clean = model.decoder(feature)

        feat_pert = model.perturbation(feat_clean.clone(), risk_mask)
        feature_pert = list(feature)
        feature_pert[4] = feat_pert
        p_pert = model.decoder_pert(feature_pert)

    feat_diff = (feat_clean - feat_pert).abs().mean(dim=1, keepdim=True)
    pred_diff = (p_clean - p_pert).abs().mean(dim=1, keepdim=True)
    risk_resized = F.interpolate(risk_mask, size=feat_clean.shape[2:], mode="nearest")

    panels = [
        ("Risk mask", risk_resized[0, 0], True, True),
        ("Clean feat (mean)", feat_clean[0].mean(dim=0), False, True),
        ("Pert feat (mean)", feat_pert[0].mean(dim=0), False, True),
        ("Feat |diff|", feat_diff[0, 0], False, True),
        ("Clean pred", p_clean[0].argmax(dim=0).float(), False, False),
        ("Pert pred", p_pert[0].argmax(dim=0).float(), False, False),
        ("Pred |diff|", pred_diff[0, 0], False, False),
    ]

    imgs = []
    for title, tensor, is_binary, need_resize in panels:
        if need_resize:
            tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode="bilinear", align_corners=False)[0, 0]
        img = to_uint8_binary(tensor) if is_binary else to_uint8_heatmap(tensor)
        img = put_label(img, title)
        imgs.append(img)

    row1 = np.hstack(imgs[:4])
    row2 = np.hstack(imgs[4:])
    max_w = max(row1.shape[1], row2.shape[1])
    if row2.shape[1] < max_w:
        row2 = np.hstack([row2, np.zeros((row2.shape[0], max_w - row2.shape[1], 3), dtype=np.uint8)])
    panel = np.vstack([row1, row2])
    cv2.imwrite(str(save_path), panel)
    print(f"  Saved: {save_path}")


# ============ 可视化 4: Encoder 各层特征 (PCA) ============
def visualize_encoder_features_pca(model, volume, save_path):
    """Encoder 各层特征 PCA 降维可视化"""
    target_h, target_w = volume.shape[2:]

    with torch.no_grad():
        rgb = volume[:, :3, :, :]
        depth = volume[:, 3:4, :, :]
        features = model.encoder(rgb, depth)

    imgs = []
    for i, feat in enumerate(features):
        # PCA 降维到 3 通道
        B, C, H, W = feat.shape
        feat_flat = feat[0].reshape(C, -1).cpu().numpy()
        mean = feat_flat.mean(axis=1, keepdims=True)
        feat_centered = feat_flat - mean
        cov = np.cov(feat_centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 取前 3 个主成分
        pca = eigvecs[:, -3:].T @ feat_centered
        pca = pca.reshape(3, H, W)
        # 归一化到 [0, 255]
        for c in range(3):
            pca[c] = (pca[c] - pca[c].min()) / (pca[c].max() - pca[c].min() + 1e-8) * 255
        pca = pca.astype(np.uint8).transpose(1, 2, 0)  # [H, W, 3]
        pca_bgr = cv2.cvtColor(pca, cv2.COLOR_RGB2BGR)
        pca_bgr = cv2.resize(pca_bgr, (target_w, target_h))
        pca_bgr = put_label(pca_bgr, f"x{i} ({C}ch)")
        imgs.append(pca_bgr)

    row1 = np.hstack(imgs[:3])
    row2 = np.hstack(imgs[3:5] + [np.zeros_like(imgs[0])])  # 补齐到 3 个
    panel = np.vstack([row1, row2])
    cv2.imwrite(str(save_path), panel)
    print(f"  Saved: {save_path}")


# ============ 可视化 5: 边界对比 ============
def visualize_boundary_comparison(depth, teacher_pred, save_path):
    """B_p vs G_d + 冲突区域高亮"""
    # 几何边界
    d_rel = _local_relative_normalization(depth, RISK_WINDOW_SIZE)
    G_d = _sobel_gradient_magnitude_single(d_rel)
    G_d_n = _normalize_map(G_d)

    # 语义边界
    B_p = _sobel_gradient_magnitude(teacher_pred).mean(dim=1, keepdim=True)
    B_p_n = _normalize_map(B_p)

    # 冲突区域
    C_conf = torch.abs(G_d_n - B_p_n)

    # 创建 RGB 边界叠加图
    target_h, target_w = depth.shape[2:]
    G_d_np = G_d_n[0, 0].cpu().numpy()
    B_p_np = B_p_n[0, 0].cpu().numpy()
    C_conf_np = C_conf[0, 0].cpu().numpy()

    # 红色=G_d, 绿色=B_p, 黄色=两者重叠
    boundary_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    boundary_rgb[:, :, 2] = (G_d_np * 255).astype(np.uint8)  # R
    boundary_rgb[:, :, 1] = (B_p_np * 255).astype(np.uint8)  # G
    boundary_rgb = cv2.cvtColor(boundary_rgb, cv2.COLOR_RGB2BGR)

    panels = [
        ("G_d (depth edge)", to_uint8_heatmap(G_d[0, 0])),
        ("B_p (pred edge)", to_uint8_heatmap(B_p[0, 0])),
        ("C_conf (conflict)", to_uint8_heatmap(C_conf[0, 0])),
        ("Overlay (R=G_d, G=B_p)", boundary_rgb),
    ]

    imgs = [put_label(img, title) for title, img in panels]
    panel = np.hstack(imgs)
    cv2.imwrite(str(save_path), panel)
    print(f"  Saved: {save_path}")


# ============ 固定随机数 ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============ Dice 计算 ============
def compute_dice(pred, gt, smooth=1e-6):
    """计算 dice"""
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return dice


# ============ 主函数 ============
def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)

    # 打印模型参数量确认加载正确
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    print("Loading test samples...")
    samples = load_test_samples(DATA_ROOT, NUM_SAMPLES)

    for i, sample in enumerate(samples):
        case = sample["case"]
        print(f"\n[{i+1}/{len(samples)}] Processing {case}...")

        image = sample["image"].unsqueeze(0).to(DEVICE)
        depth = sample["depth1"].unsqueeze(0).to(DEVICE)
        label = sample["label"]

        volume = torch.cat([image, depth], dim=1)

        # Teacher forward (模拟 teacher 预测)
        with torch.no_grad():
            output = model(volume)
            if isinstance(output, tuple):
                output = output[0]
            teacher_pred = torch.softmax(output, dim=1)

        # 计算预测和 GT 的 dice
        pred_argmax = teacher_pred[0].argmax(dim=0).cpu().numpy()
        gt = label.cpu().numpy()
        dice = compute_dice(pred_argmax > 0, gt > 0)
        print(f"  Prediction vs GT Dice: {dice:.4f}")

        # 计算 risk mask
        with torch.no_grad():
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
            print(f"  Risk mask M_r ratio: {M_r.mean().item():.4f}")
            print(f"  Low-risk mask M_l ratio: {M_l.mean().item():.4f}")

        prefix = os.path.join(OUTPUT_DIR, f"{case}")

        # 1. Risk Map
        visualize_risk_components(depth, teacher_pred, f"{prefix}_1_risk_components.png")

        # 2. DepthGuider 特征
        visualize_depth_guider_features(model, volume, f"{prefix}_2_depth_guider.png")

        # 3. 特征扰动
        visualize_perturbation(model, volume, M_r, f"{prefix}_3_perturbation.png")

        # 4. Encoder 特征 PCA
        visualize_encoder_features_pca(model, volume, f"{prefix}_4_encoder_pca.png")

        # 5. 边界对比
        visualize_boundary_comparison(depth, teacher_pred, f"{prefix}_5_boundary.png")

    print(f"\nDone! All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
