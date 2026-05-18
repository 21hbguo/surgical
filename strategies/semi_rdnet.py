"""
semi_rdnet:
基于 RDNet 训练思路的双分支半监督策略。

实现要点：
1. RGB 分支：学生模型 + EMA 教师。
2. Depth 分支：与学生同构的独立模型（可训练）。
3. 监督损失：RGB 与 Depth 分支都使用 fully 策略一致的 CE+Dice。
4. 无监督损失：
   - RGB 分支对 EMA 伪标签的一致性（高置信区域）。
   - Depth 分支对 EMA 伪标签的一致性（高置信区域）。
   - RGB / Depth 两分支之间一致性（高置信区域）。
   - DPA：基于深度引导的混合一致性。
"""

import inspect

import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy


class RDNetStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()

        self.num_classes = int(args.num_classes)
        self.consistency_start_iters = int(args.consistency_start_iters)

        self.rdnet_thresh = 0.85
        self.rdnet_thresh_dpa = 0.95
        self.rdnet_unsup_rgb_weight = 0.5
        self.rdnet_unsup_depth_weight = 0.5
        self.rdnet_cross_weight = 0.5
        self.rdnet_dpa_weight = 1.0

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
        # student_prob/target_prob: [B,C,H,W], confidence: [B,1,H,W] 学生/目标概率: [B,C,H,W], 置信度: [B,1,H,W]
        if student_prob.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        diff = (student_prob - target_prob) ** 2
        diff = diff.mean(dim=1, keepdim=True)
        masked = diff * confidence
        denom = confidence.sum() + 1e-6
        return masked.sum() / denom

    def _build_depth_guided_mix_mask(self, depth_guidance, target_size):
        if depth_guidance.shape[1] > 1:
            depth_map = depth_guidance.mean(dim=1, keepdim=True)
        else:
            depth_map = depth_guidance
        if depth_map.shape[2:] != target_size:
            depth_map = F.interpolate(depth_map, size=target_size, mode="bilinear", align_corners=False)

        flat = depth_map.flatten(2)
        median = flat.median(dim=2).values.view(depth_map.shape[0], 1, 1, 1)
        mix_mask = (depth_map > median).float()

        # 防止全0/全1导致退化
        mix_ratio = mix_mask.mean(dim=(1, 2, 3), keepdim=True)
        degenerate = (mix_ratio < 0.05) | (mix_ratio > 0.95)
        if degenerate.any():
            rand_mask = (torch.rand_like(mix_mask) > 0.5).float()
            mix_mask = torch.where(degenerate, rand_mask, mix_mask)
        return mix_mask

    def _dpa_mix(self, unlabeled_img, ema_prob, depth_guidance):
        bsz = unlabeled_img.shape[0]
        if bsz <= 1:
            return unlabeled_img, ema_prob

        perm = torch.randperm(bsz, device=unlabeled_img.device)
        mix_mask = self._build_depth_guided_mix_mask(depth_guidance, unlabeled_img.shape[2:])

        mixed_img = unlabeled_img * (1.0 - mix_mask) + unlabeled_img[perm] * mix_mask
        mixed_teacher = ema_prob * (1.0 - mix_mask) + ema_prob[perm] * mix_mask
        return mixed_img, mixed_teacher

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)
        depth3 = self._get_depth_tensor(batch_data)
        depth1 = batch_data.get("depth1")

        labeled_label = label[: self.labeled_bs]
        unlabeled_image = image[self.labeled_bs :]
        unlabeled_depth3 = depth3[self.labeled_bs :]
        unlabeled_depth1 = depth1[self.labeled_bs :]

        expected_channels = int(getattr(self.model, "params", {}).get("in_chns", image.shape[1]))
        rgb_input = self._match_expected_channels(image, expected_channels)
        depth_input = self._match_expected_channels(depth3, expected_channels)

        rgb_input = self._add_noise(rgb_input, strong_flag="s", unlabeled_only=True)
        depth_input = self._add_noise(depth_input, strong_flag="s", unlabeled_only=True)

        rgb_logits = self._extract_logits(self.model(rgb_input))
        depth_logits = self._extract_logits(self.depth_model(depth_input))

        rgb_prob = torch.softmax(rgb_logits, dim=1)
        depth_prob = torch.softmax(depth_logits, dim=1)

        with torch.no_grad():
            unlabeled_teacher_input = self._match_expected_channels(
                self._add_noise(unlabeled_image, strong_flag="t"), expected_channels
            )
            ema_logits = self._extract_logits(self.ema_model(unlabeled_teacher_input))
            ema_prob = torch.softmax(ema_logits, dim=1)
            ema_conf = torch.max(ema_prob, dim=1, keepdim=True)[0]
            conf_mask = (ema_conf >= self.rdnet_thresh).float()

        # 监督损失（与fully相同）: 0.5 * (CE + Dice)
        loss_ce_rgb = self.ce_loss(rgb_logits[: self.labeled_bs], labeled_label.long())
        loss_dice_rgb = self.dice_loss(rgb_prob[: self.labeled_bs], labeled_label.unsqueeze(1))
        sup_rgb = 0.5 * (loss_ce_rgb + loss_dice_rgb)

        loss_ce_depth = self.ce_loss(depth_logits[: self.labeled_bs], labeled_label.long())
        loss_dice_depth = self.dice_loss(depth_prob[: self.labeled_bs], labeled_label.unsqueeze(1))
        sup_depth = 0.5 * (loss_ce_depth + loss_dice_depth)

        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_image.shape[0] > 0:
            rgb_u = rgb_prob[self.labeled_bs :]
            depth_u = depth_prob[self.labeled_bs :]

            unsup_rgb = self._compute_masked_consistency(rgb_u, ema_prob, conf_mask)
            unsup_depth = self._compute_masked_consistency(depth_u, ema_prob, conf_mask)
            cross_cons = self._compute_masked_consistency(rgb_u, depth_u.detach(), conf_mask)

            # DPA: 基于depth3引导的混合一致性
            dpa_input, dpa_teacher = self._dpa_mix(unlabeled_image, ema_prob, unlabeled_depth1)
            dpa_input = self._match_expected_channels(dpa_input, expected_channels)
            dpa_input = self._add_noise(dpa_input, strong_flag="s", unlabeled_only=False)
            dpa_logits = self._extract_logits(self.model(dpa_input))
            dpa_prob = torch.softmax(dpa_logits, dim=1)
            dpa_conf = torch.max(dpa_teacher, dim=1, keepdim=True)[0]
            dpa_mask = (dpa_conf >= self.rdnet_thresh_dpa).float()
            dpa_cons = self._compute_masked_consistency(dpa_prob, dpa_teacher, dpa_mask)
        else:
            unsup_rgb = torch.tensor(0.0, device=self.device)
            unsup_depth = torch.tensor(0.0, device=self.device)
            cross_cons = torch.tensor(0.0, device=self.device)
            dpa_cons = torch.tensor(0.0, device=self.device)

        total_loss = (
            sup_rgb
            + sup_depth
            + consistency_weight
            * (
                self.rdnet_unsup_rgb_weight * unsup_rgb
                + self.rdnet_unsup_depth_weight * unsup_depth
                + self.rdnet_cross_weight * cross_cons
                + self.rdnet_dpa_weight * dpa_cons
            )
        )

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
        with torch.no_grad():
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
