# -*- coding: utf-8 -*-
"""
fully_contrast_v1:
基于 fully 监督策略扩展出的有标签对比学习版本。

主要改进：
1. 去掉原来的 prototype loss，不再在 logit 空间做类原型收缩。
2. 新增 encoder 侧 projector，输出独立的 contrast feature map。
3. 使用真实标注在 contrast feature 上构造像素类别归属，并进一步拆分边界区 / 非边界区。
4. 采用 labeled-only 的 boundary-guided contrastive loss，为后续正负样本划分提供直接依据。

方案概述：
- 分割分支仍保持 fully / resnet 的原始 encoder-decoder 路径；
- contrast 分支只从 encoder 末端特征提取，不参与 decoder 输入；
- 对每个类别使用同类像素均值作为 anchor，拉近同类边界和内部像素，推远其他类别像素。
"""

import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy


class FullyContrastV1Strategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--contrast_feature_dim", type=int, default=256)
        parser.add_argument("--contrast_loss_weight", type=float, default=0.05)
        parser.add_argument("--contrast_temperature", type=float, default=0.1)
        parser.add_argument("--contrast_boundary_width", type=int, default=1)
        parser.add_argument("--contrast_min_pixels", type=int, default=8)
        parser.add_argument("--contrast_max_samples", type=int, default=64)

    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self.num_classes = int(args.num_classes)
        self.contrast_weight = float(args.contrast_loss_weight)
        self.temperature = float(args.contrast_temperature)
        self.boundary_width = max(int(args.contrast_boundary_width), 0)
        self.min_pixels = int(args.contrast_min_pixels)
        self.max_samples = int(args.contrast_max_samples)
        self.eps = 1e-6

    def _build_class_assignment(self, labels, target_size):
        return F.interpolate(labels.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1).long()

    def _build_boundary_region_masks(self, label_assign):
        class_ids = torch.arange(self.num_classes, device=label_assign.device).view(1, -1, 1, 1)
        class_masks = label_assign.unsqueeze(1) == class_ids
        if self.boundary_width == 0:
            interior_masks = class_masks.clone()
        else:
            kernel_size = 2 * self.boundary_width + 1
            pooled = F.avg_pool2d(class_masks.float(), kernel_size=kernel_size, stride=1, padding=self.boundary_width)
            interior_masks = pooled >= (1.0 - self.eps)
        boundary_masks = class_masks & (~interior_masks)
        return boundary_masks, interior_masks

    def _limit_samples(self, feats):
        if feats.shape[0] <= self.max_samples:
            return feats
        idx = torch.linspace(0, feats.shape[0] - 1, steps=self.max_samples, device=feats.device).long()
        return feats[idx]

    def _contrastive_branch_loss(self, anchor, positives, negatives):
        if positives.numel() == 0 or negatives.numel() == 0:
            return None
        positives = F.normalize(self._limit_samples(positives), p=2, dim=1)
        negatives = F.normalize(self._limit_samples(negatives), p=2, dim=1)
        pos_logits = anchor @ positives.t() / self.temperature
        neg_logits = anchor @ negatives.t() / self.temperature
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        return -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(all_logits, dim=1)).mean()

    def _compute_contrastive_loss(self, contrast_feat, labels):
        label_assign = self._build_class_assignment(labels, target_size=contrast_feat.shape[2:])
        boundary_masks, interior_masks = self._build_boundary_region_masks(label_assign)

        feat = F.normalize(contrast_feat, p=2, dim=1)
        flat_feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        flat_labels = label_assign.reshape(-1)
        flat_boundary = boundary_masks.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        flat_interior = interior_masks.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        loss_terms = []
        for cls_id in range(self.num_classes):
            class_mask = flat_labels == cls_id
            if int(class_mask.sum().item()) < self.min_pixels:
                continue

            class_feats = flat_feat[class_mask]
            negatives = flat_feat[flat_labels != cls_id]
            if negatives.numel() == 0:
                continue

            anchor = F.normalize(class_feats.mean(dim=0, keepdim=True), p=2, dim=1)
            boundary_pos = flat_feat[flat_boundary[:, cls_id]]
            interior_pos = flat_feat[flat_interior[:, cls_id]]

            branch_losses = []
            boundary_loss = self._contrastive_branch_loss(anchor, boundary_pos, negatives)
            if boundary_loss is not None:
                branch_losses.append(boundary_loss)

            interior_loss = self._contrastive_branch_loss(anchor, interior_pos, negatives)
            if interior_loss is not None:
                branch_losses.append(interior_loss)

            if not branch_losses:
                fallback_loss = self._contrastive_branch_loss(anchor, class_feats, negatives)
                if fallback_loss is not None:
                    branch_losses.append(fallback_loss)

            loss_terms.extend(branch_losses)

        if not loss_terms:
            return torch.tensor(0.0, device=contrast_feat.device)
        return torch.stack(loss_terms).mean()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        outputs = self.model(volume)
        if isinstance(outputs, tuple):
            output, contrast_feat = outputs[0], outputs[1]

        loss_ce = self.ce_loss(output, label.long())
        loss_dice = self.dice_loss(F.softmax(output, dim=1), label.unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)
        contrastive_loss = self._compute_contrastive_loss(contrast_feat, label) if contrast_feat is not None else torch.tensor(0.0, device=output.device)
        total_loss = supervised_loss + self.contrast_weight * contrastive_loss

        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "contrastive": contrastive_loss,
            "contrast_weight": torch.tensor(self.contrast_weight, device=total_loss.device),
        }
