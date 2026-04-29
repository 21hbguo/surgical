"""
semi_proto_v1:
work！
基于 Mean Teacher 的半监督原型学习策略（Proto）。

方案描述：
1. 主分支使用学生模型进行分割监督训练（CE + Dice）。
2. 无标签分支使用 EMA 教师生成伪标签，并计算一致性损失（MSE）。
3. Proto 分支从模型输出的 proto feature 中按类别聚合 anchor，
   与可学习原型库（Learnable_Prototypes）做 NT-Xent 对比约束。
4. 总损失：supervised + consistency_weight * consistency + proto_weight * proto。
5. 参数更新：主模型优化器与原型库优化器同时反向更新。
6. Checkpoint 仅保存主模型与可学习原型库参数，不保存 EMA 教师参数。
"""

import torch
import torch.nn.functional as F

from models.networks.prototype import Learnable_Prototypes
from utils import losses

from .base_strategy import BaseTrainingStrategy


class ProtoV1Strategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self._enable_ema_support()

        self.labeled_bs = int(args.labeled_bs)
        self.num_classes = int(args.num_classes)

        self.proto_weight = float(args.proto_pixel_weight)
        self.q_low = int(args.proto_entropy_q_low)
        self.q_high = int(args.proto_entropy_q_high)
        self.proto_feature_dim = int(args.proto_feature_dim)
        self.consistency_start_iters = int(args.consistency_start_iters)

        self.learnable_prototypes_model = Learnable_Prototypes(
            num_classes=self.num_classes,
            feat_dim=self.proto_feature_dim,
        ).to(self.device)
        for _, param in self.learnable_prototypes_model.named_parameters():
            param.requires_grad = True
        self.learnable_prototypes_model.train()

        self.proto_optimizer = torch.optim.Adam(
            self.learnable_prototypes_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0001,
        )
        self.contrastive_loss_model = losses.NTXentLoss(temperature=0.07).to(self.device)

        print("[Proto] Learnable prototype bank enabled")
        print(f"[Proto] Proto weight: {self.proto_weight}, q_low={self.q_low}%, q_high={self.q_high}%")

    def _resize_label_to_feat(self, label_map, target_size):
        return F.interpolate(label_map.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1).long()

    def _build_entropy_mask(self, prob_map, target_size):
        if prob_map.numel() == 0:
            return prob_map.new_empty((0, *target_size), dtype=torch.bool)

        entropy = -(prob_map * torch.log(prob_map.clamp(min=1e-8))).sum(dim=1)
        flat_entropy = entropy.reshape(-1)
        q_low = max(0, min(100, self.q_low)) / 100.0
        q_high = max(0, min(100, self.q_high)) / 100.0

        low_thr = torch.quantile(flat_entropy, q_low)
        high_thr = torch.quantile(flat_entropy, q_high)
        valid = (entropy >= low_thr) & (entropy <= high_thr)

        valid = F.interpolate(valid.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1) > 0.5
        return valid

    def _build_proto_pairs(self, feat_map, class_assign, valid_mask, prototypes):
        feat_map = F.normalize(feat_map, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        flat_feat = feat_map.permute(0, 2, 3, 1).reshape(-1, feat_map.shape[1])
        flat_cls = class_assign.reshape(-1)
        flat_valid = valid_mask.reshape(-1)

        anchors = []
        proto_targets = []
        for cls_id in range(self.num_classes):
            cls_mask = (flat_cls == cls_id) & flat_valid
            if int(cls_mask.sum().item()) == 0:
                continue
            cls_feat = flat_feat[cls_mask]
            cls_anchor = F.normalize(cls_feat.mean(dim=0, keepdim=True), p=2, dim=1)
            anchors.append(cls_anchor)
            proto_targets.append(prototypes[cls_id : cls_id + 1])

        if not anchors:
            return None, None
        return torch.cat(anchors, dim=0), torch.cat(proto_targets, dim=0)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)

        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        labeled_label = label[: self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs :]

        stud_volume = self._add_noise(volume, strong_flag="s", unlabeled_only=True)
        model_output = self.model(stud_volume)
        if isinstance(model_output, (tuple, list)):
            student_output, student_proto_feat = model_output[0], model_output[1]
        else:
            student_output, student_proto_feat = model_output, None
        student_output_soft = torch.softmax(student_output, dim=1)

        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag="t")
            ema_output = self.ema_model(ema_inputs)
            ema_output = ema_output[0] if isinstance(ema_output, (tuple, list)) else ema_output
            ema_output_soft = torch.softmax(ema_output, dim=1)

        loss_ce = self.ce_loss(student_output[: self.labeled_bs], labeled_label.long())
        loss_dice = self.dice_loss(student_output_soft[: self.labeled_bs], labeled_label.unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            consistency_loss = torch.mean((student_output_soft[self.labeled_bs :] - ema_output_soft) ** 2)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)

        proto_loss = torch.tensor(0.0, device=self.device)
        prototypes = self.learnable_prototypes_model()

        feat_map = student_proto_feat
        if feat_map is not None and iter_num >= self.consistency_start_iters:
            target_size = feat_map.shape[2:]
            labeled_assign = self._resize_label_to_feat(labeled_label, target_size)
            labeled_valid = torch.ones_like(labeled_assign, dtype=torch.bool)

            if unlabeled_volume.shape[0] > 0:
                unlabeled_assign = self._resize_label_to_feat(torch.argmax(ema_output_soft, dim=1), target_size)
                unlabeled_valid = self._build_entropy_mask(ema_output_soft, target_size)
                class_assign = torch.cat([labeled_assign, unlabeled_assign], dim=0)
                valid_mask = torch.cat([labeled_valid, unlabeled_valid], dim=0)

            z_i, z_j = self._build_proto_pairs(feat_map, class_assign, valid_mask, prototypes)
            if z_i is not None and z_i.shape[0] > 1:
                proto_loss = self.contrastive_loss_model(z_i, z_j)

        total_loss = supervised_loss + consistency_weight * consistency_loss + self.proto_weight * proto_loss

        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "proto": proto_loss,
            "proto_weight": torch.tensor(self.proto_weight, device=total_loss.device),
            "consistency": consistency_loss,
            "consistency_weight": consistency_weight,
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        self.proto_optimizer.zero_grad(set_to_none=True)

        loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss_dict["total"].backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.learnable_prototypes_model.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()
        self.proto_optimizer.step()
        self._update_ema(iter_num)
        return loss_dict

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        self.learnable_prototypes_model.train(mode=training)

    def get_state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "learnable_prototypes_model": self.learnable_prototypes_model.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "learnable_prototypes_model" in state_dict:
                self.learnable_prototypes_model.load_state_dict(state_dict["learnable_prototypes_model"])
            return
        self.model.load_state_dict(state_dict)
