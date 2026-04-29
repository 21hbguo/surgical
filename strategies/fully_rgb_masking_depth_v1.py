"""
fully_rgb_masking_depth_v1:
面向分割任务的全监督 RGB-Depth 互补策略。

方案描述：
1. 输入使用 RGB 与 3 通道 depth 的互补结果，不额外拼接 depth 通道。
2. 在策略内部对 RGB 随机生成 mask，默认遮挡比例为 75%。
3. RGB 被 masking 的区域直接用对应位置的 3 通道 depth 替换，形成新的 3 通道输入。
4. 模型使用 base 分割模型，输出仍为分割 logits，不做深度重建分支。
5. 训练目标沿用全监督分割损失（CE + Dice）。
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .base_strategy import BaseTrainingStrategy


class FullyRGBMaskingDepthV1Strategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--rgb_masking_ratio", type=float, default=0.75)

    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        repo_root = Path(__file__).resolve().parents[1]
        self.mask_ratio = float(getattr(args, "rgb_masking_ratio", 0.75))
        self.vis_path = repo_root / "outputs" / "fully_rgb_masking_depth_v1_complementary.png"

    def _build_random_mask(self, rgb):
        keep_prob = max(0.0, min(1.0, 1.0 - self.mask_ratio))
        return (
            torch.rand(
                rgb.shape[0],
                1,
                rgb.shape[2],
                rgb.shape[3],
                device=rgb.device,
                dtype=rgb.dtype,
            )
            < keep_prob
        ).to(rgb.dtype)

    def _build_complementary_input(self, rgb, depth3):
        mask = self._build_random_mask(rgb)
        return rgb * mask + depth3 * (1.0 - mask)

    def _tensor_to_vis_uint8(self, tensor):
        image = tensor.detach().cpu()
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min + 1e-8)
        return (image * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    def _save_complementary_visualization(self, volume):
        self.vis_path.parent.mkdir(parents=True, exist_ok=True)
        vis = self._tensor_to_vis_uint8(volume[0])
        Image.fromarray(vis).save(self.vis_path)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        rgb = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)
        depth3 = batch_data.get("depth3")
        if depth3 is None:
            raise KeyError("fully_rgb_masking_depth_v1 requires batch_data['depth3'] (3-channel depth)")
        depth3 = depth3.to(self.device)
        if depth3.shape[1] == 1:
            depth3 = depth3.repeat(1, 3, 1, 1)

        volume = self._build_complementary_input(rgb, depth3)
        # self._save_complementary_visualization(volume)
        output = self.model(volume)
        if isinstance(output, tuple):
            output = output[0]

        loss_ce = self.ce_loss(output, label.long())
        loss_dice = self.dice_loss(F.softmax(output, dim=1), label.unsqueeze(1))
        loss = 0.5 * (loss_dice + loss_ce)
        return {"total": loss, "ce": loss_ce, "dice": loss_dice}

    def validation_step(self, batch_data):
        with torch.no_grad():
            rgb = batch_data["image"].to(self.device)
            depth3 = batch_data.get("depth3")
            if depth3 is None:
                raise KeyError("fully_rgb_masking_depth_v1 requires batch_data['depth3'] (3-channel depth)")
            depth3 = depth3.to(self.device)
            if depth3.shape[1] == 1:
                depth3 = depth3.repeat(1, 3, 1, 1)
            volume = self._build_complementary_input(rgb, depth3)
            # self._save_complementary_visualization(volume)
            return self.model(volume)
