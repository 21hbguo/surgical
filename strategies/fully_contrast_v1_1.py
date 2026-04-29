# -*- coding: utf-8 -*-
"""
fully_contrast_v1_1:
基于 fully_contrast_v1 的简化版本，只保留类别级对比损失。

主要改进：
1. 复用现有 contrast_v1 模型族，不新增 projector 或 decoder 结构。
2. 去掉 v1 中的边界区 / 内部区拆分，不再做 boundary-guided 对比。
3. 仅基于类别归属执行 class-only contrastive loss：同类像素为正样本，异类像素为负样本。
4. 保持 fully 监督分割主损失不变，便于和 v1 做直接消融比较。

方案概述：
- 分割分支仍使用原始 fully 模式的监督训练；
- projector 输出的 contrast feature 先与下采样 GT 标签对齐得到类别归属；
- 每个类别以类内均值特征为 anchor，聚合同类像素并排斥其他类别像素。
"""

import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy


class FullyContrastV11Strategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self.num_classes = int(args.num_classes)
        self.contrast_weight = float(getattr(args, "contrast_loss_weight", 0.05))
        self.temperature = float(getattr(args, "contrast_temperature", 0.1))
        self.min_pixels = int(getattr(args, "contrast_min_pixels", 8))
        self.max_samples = int(getattr(args, "contrast_max_samples", 64))
        self.eps = 1e-6

    def _build_class_assignment(self, labels, target_size):
        return F.interpolate(labels.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1).long()

    def _limit_samples(self, feats):
        if feats.shape[0] <= self.max_samples:
            return feats
        idx = torch.linspace(0, feats.shape[0] - 1, steps=self.max_samples, device=feats.device).long()
        return feats[idx]

    def _class_contrastive_loss(self, anchor, positives, negatives):
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
        feat = F.normalize(contrast_feat, p=2, dim=1)
        flat_feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        flat_labels = label_assign.reshape(-1)

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
            cls_loss = self._class_contrastive_loss(anchor, class_feats, negatives)
            if cls_loss is not None:
                loss_terms.append(cls_loss)

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
        else:
            output, contrast_feat = outputs, None

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
