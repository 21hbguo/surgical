"""
semi_rdnet:
基于 CVPR 2025 RDNet 论文的完整双分支半监督策略。

核心创新点：
1. 双流架构：RGB流 + Depth流，各自独立模型 + EMA教师
2. DPA (Depth-guided Patch Augmentation)：基于深度方差的自适应困难样本挖掘
3. 伪标签跨模态更新：用深度流高置信预测更新RGB流伪标签
4. 对比学习：Dice系数作为相似度的InfoNCE风格损失
5. 特征一致性：MSE + L2损失约束双流特征对齐
"""

import inspect
import random

import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy
from utils.losses import (
    rdnet_contrastive_loss,
    update_pseudo_labels,
    feature_l2_loss,
    mse_consistency_loss,
    dice_coefficient,
)


class RDNetStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()

        self.num_classes = int(args.num_classes)
        self.consistency_start_iters = int(args.consistency_start_iters)

        # RDNet超参数（与原始论文一致）
        self.rdnet_thresh = getattr(args, 'rdnet_thresh', 0.85)
        self.rdnet_thresh_depth = getattr(args, 'rdnet_thresh_depth', 0.85)
        self.rdnet_thresh_dpa = getattr(args, 'rdnet_thresh_dpa', 0.95)
        self.rdnet_beta = getattr(args, 'rdnet_beta', 0.3)

        # 损失权重
        self.rdnet_unsup_rgb_weight = 0.5
        self.rdnet_unsup_depth_weight = 1.2
        self.rdnet_cross_weight = 0.5
        self.rdnet_dpa_weight = 1.0
        self.rdnet_contrast_weight = 1.0
        self.rdnet_mse_weight = 0.5
        self.rdnet_feature_l2_weight = 0.5

        # 创建depth分支模型
        self.depth_model = self._create_branch_model(self.model)
        self.depth_optimizer = torch.optim.Adam(
            self.depth_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0001,
        )

    def _create_branch_model(self, model):
        orig_model = getattr(model, "_orig_mod", model)
        sig = inspect.signature(type(orig_model).__init__)
        init_kwargs = {}
        model_params = getattr(orig_model, "params", None)

        if model_params is not None:
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                value = self._resolve_model_param_value(model_params, param_name, param)
                if value != inspect.Parameter.empty:
                    init_kwargs[param_name] = value

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name not in init_kwargs and hasattr(orig_model, param_name):
                init_kwargs[param_name] = getattr(orig_model, param_name)

        branch_model = type(orig_model)(**init_kwargs)
        branch_model.load_state_dict(orig_model.state_dict(), strict=False)
        if not hasattr(branch_model, "params") and hasattr(orig_model, "params"):
            branch_model.params = orig_model.params
        return branch_model.to(self.device)

    def _extract_logits(self, output):
        return output[0] if isinstance(output, (tuple, list)) else output

    def _match_expected_channels(self, x, expected_channels):
        if x.shape[1] == expected_channels:
            return x
        if x.shape[1] > expected_channels:
            return x[:, :expected_channels]
        if x.shape[1] == 1:
            return x.repeat(1, expected_channels, 1, 1)
        repeat_factor = (expected_channels + x.shape[1] - 1) // x.shape[1]
        x_rep = x.repeat(1, repeat_factor, 1, 1)
        return x_rep[:, :expected_channels]

    def _compute_masked_consistency(self, student_prob, target_prob, confidence):
        if student_prob.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        diff = (student_prob - target_prob) ** 2
        diff = diff.mean(dim=1, keepdim=True)
        masked = diff * confidence
        denom = confidence.sum() + 1e-6
        return masked.sum() / denom

    def _dpa_augment(self, depth_map, images, pred_u, epoch, total_epochs):
        """DPA: 基于深度方差的Patch增强"""
        B, C, H, W = images.shape
        if B <= 1:
            return images, pred_u

        # 计算patch大小
        h_patch = random.choice([10, 20, 40])
        w_patch = random.choice([10, 20, 40])

        # 计算每个patch的深度方差作为hardness score
        depth_patches = depth_map.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
        patch_variance = depth_patches.var(dim=(-2, -1))
        hardness_scores = patch_variance.flatten(1)

        num_patches = hardness_scores.shape[1]

        # 计算要保留的hard patch数量（渐进式）
        if epoch < total_epochs:
            k = int(self.rdnet_beta * (epoch / total_epochs) * num_patches)
        else:
            k = int(self.rdnet_beta * num_patches)
        k = max(1, min(k, num_patches - 1))

        _, indices = torch.topk(hardness_scores, k, dim=1, largest=True)

        augmented_imgs = images.clone()
        augmented_labels = pred_u.clone()

        grid_h = H // h_patch
        grid_w = W // w_patch

        for i in range(B):
            mask = torch.zeros(num_patches, dtype=torch.float32, device=images.device)
            mask[indices[i]] = 1.0
            available = [j for j in range(num_patches) if mask[j] == 0]

            if available:
                chosen_patch = random.choice(available)
                ph = chosen_patch // grid_w
                pw = chosen_patch % grid_w
                y_start, y_end = ph * h_patch, (ph + 1) * h_patch
                x_start, x_end = pw * w_patch, (pw + 1) * w_patch

                next_idx = (i + 1) % B
                augmented_imgs[i, :, y_start:y_end, x_start:x_end] = images[next_idx, :, y_start:y_end, x_start:x_end]
                augmented_labels[i, :, y_start:y_end, x_start:x_end] = pred_u[next_idx, :, y_start:y_end, x_start:x_end]

        return augmented_imgs, augmented_labels

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)

        # RDNet需要depth3（3通道，depth分支输入）和depth1（1通道，DPA用）
        depth3 = batch_data.get("depth3")
        if depth3 is not None:
            depth3 = depth3.to(self.device)
        else:
            depth3 = self._get_depth_tensor(batch_data)  # fallback

        depth1 = batch_data.get("depth1")
        if depth1 is not None:
            depth1 = depth1.to(self.device)

        labeled_bs = self.labeled_bs
        labeled_label = label[:labeled_bs]
        unlabeled_image = image[labeled_bs:]
        unlabeled_depth3 = depth3[labeled_bs:] if depth3 is not None else None
        unlabeled_depth1 = depth1[labeled_bs:] if depth1 is not None else None

        expected_channels = int(getattr(self.model, "params", {}).get("in_chns", image.shape[1]))

        # 准备输入
        rgb_input = self._match_expected_channels(image, expected_channels)
        depth_input = self._match_expected_channels(depth3, expected_channels) if depth3 is not None else rgb_input

        rgb_input = self._add_noise(rgb_input, strong_flag="s", unlabeled_only=True)
        depth_input = self._add_noise(depth_input, strong_flag="s", unlabeled_only=True)

        # 前向传播
        rgb_logits = self._extract_logits(self.model(rgb_input))
        depth_logits = self._extract_logits(self.depth_model(depth_input))

        rgb_prob = torch.softmax(rgb_logits, dim=1)
        depth_prob = torch.softmax(depth_logits, dim=1)

        # EMA教师预测
        with torch.no_grad():
            unlabeled_teacher_input = self._match_expected_channels(
                self._add_noise(unlabeled_image, strong_flag="t"), expected_channels
            )
            ema_logits = self._extract_logits(self.ema_model(unlabeled_teacher_input))
            ema_prob = torch.softmax(ema_logits, dim=1)
            ema_conf = torch.max(ema_prob, dim=1, keepdim=True)[0]
            conf_mask = (ema_conf >= self.rdnet_thresh).float()

        # ====== 1. 监督损失 ======
        loss_ce_rgb = self.ce_loss(rgb_logits[:labeled_bs], labeled_label.long())
        loss_dice_rgb = self.dice_loss(rgb_prob[:labeled_bs], labeled_label.unsqueeze(1))
        sup_rgb = 0.5 * (loss_ce_rgb + loss_dice_rgb)

        loss_ce_depth = self.ce_loss(depth_logits[:labeled_bs], labeled_label.long())
        loss_dice_depth = self.dice_loss(depth_prob[:labeled_bs], labeled_label.unsqueeze(1))
        sup_depth = 0.5 * (loss_ce_depth + loss_dice_depth)

        consistency_weight = self._get_consistency_weight(iter_num)

        if iter_num >= self.consistency_start_iters and unlabeled_image.shape[0] > 0:
            rgb_u = rgb_prob[labeled_bs:]
            depth_u = depth_prob[labeled_bs:]
            rgb_l = rgb_prob[:labeled_bs]
            depth_l = depth_prob[:labeled_bs]

            # ====== 2. 无监督一致性损失 ======
            unsup_rgb = self._compute_masked_consistency(rgb_u, ema_prob, conf_mask)
            unsup_depth = self._compute_masked_consistency(depth_u, ema_prob, conf_mask)

            # ====== 3. 跨模态一致性 ======
            # 伪标签跨模态更新
            updated_pseudo = update_pseudo_labels(ema_prob, depth_u, gamma=self.rdnet_thresh_depth)
            depth_conf = torch.max(depth_u, dim=1, keepdim=True)[0]
            depth_conf_mask = (depth_conf >= self.rdnet_thresh_depth).float()
            cross_cons = self._compute_masked_consistency(rgb_u, updated_pseudo.detach(), depth_conf_mask)

            # ====== 4. DPA增强一致性 ======
            if unlabeled_depth1 is not None:
                dpa_input, dpa_teacher = self._dpa_augment(
                    unlabeled_depth1, unlabeled_image, ema_prob, epoch, max(1, self.args.max_epoch)
                )
                dpa_input = self._match_expected_channels(dpa_input, expected_channels)
                dpa_input = self._add_noise(dpa_input, strong_flag="s", unlabeled_only=False)
                dpa_logits = self._extract_logits(self.model(dpa_input))
                dpa_prob = torch.softmax(dpa_logits, dim=1)
                dpa_conf = torch.max(dpa_teacher, dim=1, keepdim=True)[0]
                dpa_mask = (dpa_conf >= self.rdnet_thresh_dpa).float()
                dpa_cons = self._compute_masked_consistency(dpa_prob, dpa_teacher, dpa_mask)
            else:
                dpa_cons = torch.tensor(0.0, device=self.device)

            # ====== 5. 对比学习损失 ======
            loss_contrast = rdnet_contrastive_loss(rgb_l, rgb_u, depth_l, depth_u)

            # ====== 6. MSE一致性损失 ======
            loss_mse = mse_consistency_loss(rgb_u, depth_u.detach())

            # ====== 7. 特征L2损失（使用输出logits近似） ======
            loss_feat_l2 = feature_l2_loss(
                rgb_logits[labeled_bs:], depth_logits[labeled_bs:].detach()
            )
        else:
            unsup_rgb = torch.tensor(0.0, device=self.device)
            unsup_depth = torch.tensor(0.0, device=self.device)
            cross_cons = torch.tensor(0.0, device=self.device)
            dpa_cons = torch.tensor(0.0, device=self.device)
            loss_contrast = torch.tensor(0.0, device=self.device)
            loss_mse = torch.tensor(0.0, device=self.device)
            loss_feat_l2 = torch.tensor(0.0, device=self.device)

        # ====== 总损失 ======
        # RGB分支损失
        loss_rgb = (sup_rgb + self.rdnet_unsup_rgb_weight * unsup_rgb + dpa_cons + self.rdnet_thresh_depth * cross_cons) / 3.0
        # Depth分支损失
        loss_depth = (sup_depth + self.rdnet_unsup_depth_weight * unsup_depth) / 2.0
        # 跨模态损失
        loss_con = (self.rdnet_contrast_weight * loss_contrast +
                    self.rdnet_mse_weight * loss_mse +
                    self.rdnet_feature_l2_weight * loss_feat_l2) / 2.0

        total_loss = (loss_rgb + loss_depth + loss_con) / 3.0

        return {
            "total": total_loss,
            "ce_rgb": loss_ce_rgb,
            "dice_rgb": loss_dice_rgb,
            "ce_depth": loss_ce_depth,
            "dice_depth": loss_dice_depth,
            "sup_rgb": sup_rgb,
            "sup_depth": sup_depth,
            "unsup_rgb": unsup_rgb,
            "unsup_depth": unsup_depth,
            "cross_cons": cross_cons,
            "dpa_cons": dpa_cons,
            "contrast": loss_contrast,
            "mse": loss_mse,
            "feat_l2": loss_feat_l2,
            "consistency_weight": consistency_weight,
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        self.depth_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss = loss_dict["total"]

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                self.scaler.unscale_(self.depth_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.depth_model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.step(self.depth_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.depth_model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.depth_optimizer.step()
        self._update_ema(iter_num)
        return loss_dict

    def validation_step(self, batch_data):
        image = batch_data["image"].to(self.device)
        expected_channels = int(getattr(self.model, "params", {}).get("in_chns", image.shape[1]))
        rgb_input = self._match_expected_channels(image, expected_channels)
        output = self.model(rgb_input)
        return self._extract_logits(output)

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        self.depth_model.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "depth_model": self.depth_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "depth_model" in state_dict:
                self.depth_model.load_state_dict(state_dict["depth_model"])
            return
        self.model.load_state_dict(state_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--rdnet_thresh', type=float, default=0.85, help='RGB伪标签置信阈值')
        parser.add_argument('--rdnet_thresh_depth', type=float, default=0.85, help='Depth伪标签置信阈值')
        parser.add_argument('--rdnet_thresh_dpa', type=float, default=0.95, help='DPA增强置信阈值')
        parser.add_argument('--rdnet_beta', type=float, default=0.3, help='DPA patch保留比例')
