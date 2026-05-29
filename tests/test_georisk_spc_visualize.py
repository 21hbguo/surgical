"""Visualize GeoRisk-SPC: risk mask selection + perturbation comparison.

Outputs (saved to tests/outputs/):
  risk_components.png  — depth, G_d, U_t, B_p, C_conf, R, M_r, M_l
  perturbation_compare.png — clean vs perturbed feature & prediction diff
"""

import os
import sys
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strategies.semi_georisk_spc import (
    GeoRiskSPCStrategy,
    _local_relative_normalization,
    _sobel_gradient_magnitude_single,
    _sobel_gradient_magnitude,
    _normalize_map,
)
from models.networks.unet import UNet_GeoRiskSPC

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def _to_uint8_heatmap(tensor_2d, colormap=cv2.COLORMAP_JET):
    """[H, W] tensor -> BGR uint8 heatmap."""
    arr = tensor_2d.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    return cv2.applyColorMap(arr, colormap)


def _to_uint8_binary(mask_2d):
    """[H, W] binary tensor -> white/black uint8."""
    arr = (mask_2d.detach().cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def _make_synthetic_data(H=128, W=128, num_classes=2):
    """Create synthetic data with clear depth edges for visible risk regions."""
    image = torch.rand(1, 3, H, W)
    depth = torch.zeros(1, 1, H, W)
    # left half close (high depth), right half far (low depth) -> strong edge at midline
    depth[:, :, :, : W // 2] = 0.8
    depth[:, :, :, W // 2 :] = 0.2
    # add a circular hole in the left side -> another depth edge
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cx, cy, r = W // 4, H // 2, H // 6
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2).float().sqrt() < r
    depth[0, 0][circle] = 0.1
    label = torch.zeros(1, H, W, dtype=torch.long)
    label[:, :, W // 2 :] = 1
    return image, depth, label


def _make_teacher_pred(label, num_classes=2, noise_std=0.3):
    """Convert label to one-hot teacher prediction with noise."""
    B, H, W = label.shape
    one_hot = F.one_hot(label, num_classes).permute(0, 3, 1, 2).float()
    noise = torch.randn_like(one_hot) * noise_std
    pred = torch.softmax(one_hot + noise, dim=1)
    return pred


def _visualize_risk_components(depth, teacher_pred, args, save_path):
    """Compute and save all risk map intermediate components."""
    H, W = depth.shape[2:]
    d_rel = _local_relative_normalization(depth, args.risk_window_size)
    G_d = _sobel_gradient_magnitude_single(d_rel)
    U_t = -torch.sum(teacher_pred * torch.log(teacher_pred + 1e-8), dim=1, keepdim=True)
    B_p = _sobel_gradient_magnitude(teacher_pred).mean(dim=1, keepdim=True)
    G_d_n = _normalize_map(G_d)
    B_p_n = _normalize_map(B_p)
    C_conf = torch.abs(G_d_n - B_p_n)
    U_t_n = _normalize_map(U_t)
    R = U_t_n * G_d_n + args.risk_lambda_c * C_conf
    conf = teacher_pred.max(dim=1, keepdim=True)[0]
    M_r = (R > args.risk_tau_r).float()
    M_l = (conf > args.risk_tau_c).float() * (1.0 - M_r)

    panels = [
        ("Depth", depth[0, 0]),
        ("d_rel", d_rel[0, 0]),
        ("G_d (edge)", G_d[0, 0]),
        ("U_t (entropy)", U_t[0, 0]),
        ("B_p (pred edge)", B_p[0, 0]),
        ("C_conf (conflict)", C_conf[0, 0]),
        ("R (risk map)", R[0, 0]),
        ("M_r (high-risk)", M_r[0, 0]),
        ("M_l (low-risk)", M_l[0, 0]),
    ]
    rows = []
    for i in range(0, len(panels), 3):
        row_imgs = []
        for title, tensor in panels[i : i + 3]:
            if "M_" in title:
                img = _to_uint8_binary(tensor)
            else:
                img = _to_uint8_heatmap(tensor)
            cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            row_imgs.append(img)
        rows.append(np.hstack(row_imgs))
    panel = np.vstack(rows)
    cv2.imwrite(str(save_path), panel)
    return R, M_r, M_l


def _resize_to(tensor_2d, target_h, target_w):
    """Resize a [H, W] tensor to (target_h, target_w) for display."""
    return F.interpolate(
        tensor_2d.unsqueeze(0).unsqueeze(0).float(),
        size=(target_h, target_w), mode="bilinear", align_corners=False,
    )[0, 0]


def _visualize_perturbation(model, volume, risk_mask, save_path):
    """Compare clean vs perturbed features and predictions."""
    model.eval()
    with torch.no_grad():
        feature = model.encoder(volume)
        feat_clean = feature[4]
        p_clean = model.decoder(feature)
        feat_pert = model.perturbation(feat_clean.clone(), risk_mask)
        feature_pert = list(feature)
        feature_pert[4] = feat_pert
        p_pert = model.decoder_pert(feature_pert)

    target_h, target_w = volume.shape[2:]  # display size = input size

    # feature difference (mean across channels), resize to input resolution
    feat_diff = (feat_clean - feat_pert).abs().mean(dim=1, keepdim=True)
    pred_diff = (p_clean - p_pert).abs().mean(dim=1, keepdim=True)
    risk_resized = F.interpolate(risk_mask, size=feat_clean.shape[2:], mode="nearest")

    panels = [
        ("Risk mask (feat res)", risk_resized[0, 0], True, True),
        ("Clean feature (mean)", feat_clean[0].mean(dim=0), False, True),
        ("Perturbed feature (mean)", feat_pert[0].mean(dim=0), False, True),
        ("Feature |diff|", feat_diff[0, 0], False, True),
        ("Clean pred (argmax)", p_clean[0].argmax(dim=0).float(), False, False),
        ("Pert pred (argmax)", p_pert[0].argmax(dim=0).float(), False, False),
        ("Pred |diff|", pred_diff[0, 0], False, False),
    ]

    imgs = []
    for title, tensor, is_binary, need_resize in panels:
        if need_resize:
            tensor = _resize_to(tensor, target_h, target_w)
        if is_binary:
            img = _to_uint8_binary(tensor)
        else:
            img = _to_uint8_heatmap(tensor)
        cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        imgs.append(img)

    row1 = np.hstack(imgs[:3])
    row2 = np.hstack(imgs[3:7])
    max_w = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] < max_w:
        row1 = np.hstack([row1, np.zeros((row1.shape[0], max_w - row1.shape[1], 3), dtype=np.uint8)])
    if row2.shape[1] < max_w:
        row2 = np.hstack([row2, np.zeros((row2.shape[0], max_w - row2.shape[1], 3), dtype=np.uint8)])
    panel = np.vstack([row1, row2])
    cv2.imwrite(str(save_path), panel)

    feat_diff_val = feat_diff.mean().item()
    pred_diff_val = pred_diff.mean().item()
    risk_ratio = risk_mask.float().mean().item()
    return feat_diff_val, pred_diff_val, risk_ratio


class TestGeoRiskSPCVisualize(unittest.TestCase):

    def setUp(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.args = type("Args", (), {
            "risk_window_size": 16,
            "risk_tau_r": 0.5,
            "risk_tau_c": 0.9,
            "risk_lambda_c": 0.5,
            "risk_dropout_rate": 0.3,
            "risk_noise_std": 0.1,
        })()

    def test_risk_components_visualization(self):
        image, depth, label = _make_synthetic_data()
        teacher_pred = _make_teacher_pred(label, num_classes=2)
        save_path = os.path.join(OUTPUT_DIR, "risk_components.png")
        R, M_r, M_l = _visualize_risk_components(depth, teacher_pred, self.args, save_path)
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(M_r.sum().item(), 0, "M_r should have non-zero high-risk pixels")
        self.assertGreater(M_l.sum().item(), 0, "M_l should have non-zero low-risk pixels")
        print(f"\n  M_r ratio: {M_r.mean().item():.3f}, M_l ratio: {M_l.mean().item():.3f}")
        print(f"  Saved: {save_path}")

    def test_perturbation_comparison(self):
        image, depth, label = _make_synthetic_data(H=128, W=128)
        teacher_pred = _make_teacher_pred(label, num_classes=2)
        R, M_r, _ = _visualize_risk_components(
            depth, teacher_pred, self.args,
            os.path.join(OUTPUT_DIR, "risk_components.png"),
        )
        model = UNet_GeoRiskSPC(in_chns=4, class_num=2, filter_num=16,
                                dropout_rate=0.3, noise_std=0.1)
        model.eval()
        volume = torch.cat([image, depth], dim=1)
        save_path = os.path.join(OUTPUT_DIR, "perturbation_compare.png")
        feat_diff, pred_diff, risk_ratio = _visualize_perturbation(
            model, volume, M_r, save_path
        )
        self.assertTrue(os.path.exists(save_path))
        print(f"\n  Risk mask ratio: {risk_ratio:.3f}")
        print(f"  Feature |diff|: {feat_diff:.4f}")
        print(f"  Pred    |diff|: {pred_diff:.4f}")
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    unittest.main()
