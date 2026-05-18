"""方案简介（Semi Mean Teacher + Contrast V1）：
在 Mean Teacher 半监督分割框架上增加对比学习分支。
学生端输出分割 logits 与对比特征，教师端仅提供无标注一致性伪监督。
监督项为 CE + Dice；无监督项为学生/教师分割一致性；额外对比项在类别内外样本上约束特征判别性。
"""

import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy


class MeanTeacherContrastV1Strategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--contrast_feature_dim", type=int, default=256)
        parser.add_argument("--contrast_loss_weight", type=float, default=0.05)
        parser.add_argument("--contrast_temperature", type=float, default=0.1)
        parser.add_argument("--contrast_boundary_width", type=int, default=1)
        parser.add_argument("--contrast_min_pixels", type=int, default=8)
        parser.add_argument("--contrast_max_samples", type=int, default=64)

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self.num_classes = int(args.num_classes)
        self.contrast_weight = float(args.contrast_loss_weight)
        self.temperature = float(args.contrast_temperature)
        self.boundary_width = max(int(args.contrast_boundary_width), 0)
        self.min_pixels = int(args.contrast_min_pixels)
        self.max_samples = int(args.contrast_max_samples)
        self.eps = 1e-6
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)

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

        stud_volume = self._add_noise(volume, strong_flag="s", unlabeled_only=True)

        outputs = self.model(stud_volume)
        if isinstance(outputs, tuple):
            output, contrast_feat = outputs[0], outputs[1]
        else:
            output, contrast_feat = outputs, None
        output_soft = torch.softmax(output, dim=1)

        labeled_output = output[: self.labeled_bs]
        labeled_output_soft = output_soft[: self.labeled_bs]
        labeled_label = label[: self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs :]

        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag="t")
            ema_outputs = self.ema_model(ema_inputs)
            ema_output = ema_outputs[0] if isinstance(ema_outputs, tuple) else ema_outputs
            ema_output_soft = torch.softmax(ema_output, dim=1)

        batch_data["teacher_pred"] = ema_output_soft

        loss_ce = self.ce_loss(labeled_output, labeled_label.long())
        loss_dice = self.dice_loss(labeled_output_soft, labeled_label.unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        if contrast_feat is not None and unlabeled_volume.shape[0] > 0:
            pseudo_label = torch.argmax(ema_output_soft, dim=1)
            contrastive_loss = self._compute_contrastive_loss(contrast_feat[self.labeled_bs :], pseudo_label)
        else:
            contrastive_loss = torch.tensor(0.0, device=output.device)

        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            consistency_loss = torch.mean((output_soft[self.labeled_bs :] - ema_output_soft) ** 2)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)

        total_loss = supervised_loss + self.contrast_weight * contrastive_loss + consistency_weight * consistency_loss

        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "contrastive": contrastive_loss,
            "contrast_weight": torch.tensor(self.contrast_weight, device=total_loss.device),
            "consistency": consistency_loss,
            "consistency_weight": consistency_weight,
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict["total"], optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict
