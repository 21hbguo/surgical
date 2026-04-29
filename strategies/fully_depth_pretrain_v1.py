"""
fully_depth_pretrain:
面向深度重建的全监督预训练策略（MAE风格简化版）。

方案描述：
1. 输入由完整 RGB 与随机掩码后的 3 通道深度图拼接而成。
2. 随机掩码在策略内部生成，不依赖数据增强模块外部实现。
3. 模型输出目标为 3 通道深度图；若模型输出为 1 通道则在策略内扩展为 3 通道。
4. 训练目标为深度重建损失（L1 + MSE）。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base_strategy import BaseTrainingStrategy


class FullyDepthPretrainStrategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--depth_pretrain_mask_ratio", type=float, default=0.75)
        parser.add_argument("--depth_l1_weight", type=float, default=1.0)
        parser.add_argument("--depth_loss_weight", type=float, default=1.0)

    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self.mask_ratio = float(getattr(args, "depth_pretrain_mask_ratio", 0.75))
        self.l1_weight = float(getattr(args, "depth_l1_weight", 1.0))
        self.mse_weight = float(getattr(args, "depth_loss_weight", 1.0))
        self.vis_root = Path(__file__).resolve().parents[1] / "outputs"

    def _build_random_mask(self, depth):
        keep_prob = max(0.0, min(1.0, 1.0 - self.mask_ratio))
        mask = (torch.rand(
            depth.shape[0], 1, depth.shape[2], depth.shape[3],
            device=depth.device, dtype=depth.dtype,
        ) < keep_prob).to(depth.dtype)
        return mask

    def _to_depth3(self, pred, target_size):
        if pred.shape[2:] != target_size:
            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
        if pred.shape[1] == 3:
            return pred
        if pred.shape[1] == 1:
            return pred.repeat(1, 3, 1, 1)
        if pred.shape[1] > 3:
            return pred[:, :3]
        pad_c = 3 - pred.shape[1]
        return F.pad(pred, (0, 0, 0, 0, 0, pad_c), mode="constant", value=0.0)

    def _tensor_to_vis_uint8(self, tensor):
        img = tensor.detach().cpu()
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)
        return (img * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    def _save_simple_visualization(self, depth3, masked_depth3, pred_depth3):
        self.vis_root.mkdir(parents=True, exist_ok=True)
        gt_img = self._tensor_to_vis_uint8(depth3[0])
        masked_img = self._tensor_to_vis_uint8(masked_depth3[0])
        pred_img = self._tensor_to_vis_uint8(pred_depth3[0])
        merged = np.concatenate([gt_img, masked_img, pred_img], axis=1)
        Image.fromarray(merged).save(self.vis_root / "vis_depth_triplet.png")

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        rgb = batch_data["image"].to(self.device)
        depth3 = batch_data.get("depth3")
        if depth3 is None:
            raise KeyError("fully_depth_pretrain requires batch_data['depth3'] (3-channel depth)")
        depth3 = depth3.to(self.device)
        if depth3.shape[1] == 1:
            depth3 = depth3.repeat(1, 3, 1, 1)

        mask = self._build_random_mask(depth3)
        masked_depth3 = depth3 * mask
        model_input = torch.cat([rgb, masked_depth3], dim=1)

        model_output = self.model(model_input)
        if isinstance(model_output, (tuple, list)):
            depth_pred = model_output[1] if len(model_output) > 1 else model_output[0]
        else:
            depth_pred = model_output
        depth_pred3 = self._to_depth3(depth_pred, target_size=depth3.shape[2:])

        loss_l1 = F.l1_loss(depth_pred3, depth3)
        loss_mse = F.mse_loss(depth_pred3, depth3)
        total_loss = self.l1_weight * loss_l1 + self.mse_weight * loss_mse

        return {
            "total": total_loss,
            "depth_l1": loss_l1,
            "depth_mse": loss_mse,
            "mask_ratio": torch.tensor(self.mask_ratio, device=total_loss.device),
        }

    def validation_step(self, batch_data):
        with torch.no_grad():
            rgb = batch_data["image"].to(self.device)
            depth3 = batch_data.get("depth3")
            if depth3 is None:
                raise KeyError("fully_depth_pretrain requires batch_data['depth3'] (3-channel depth)")
            depth3 = depth3.to(self.device)
            if depth3.shape[1] == 1:
                depth3 = depth3.repeat(1, 3, 1, 1)
            mask = self._build_random_mask(depth3)
            masked_depth3 = depth3 * mask
            model_input = torch.cat([rgb, masked_depth3], dim=1)
            output = self.model(model_input)
            if isinstance(output, (tuple, list)):
                output = output[1] if len(output) > 1 else output[0]

            pred_depth3 = self._to_depth3(output, target_size=depth3.shape[2:])
            # self._save_simple_visualization(depth3, masked_depth3, pred_depth3)
            return pred_depth3
