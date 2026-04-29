"""
semi_mt_depth_guider_proto_teacher_v1:
结合 depth1c 引导 + mt_depth_guider_proto_v1 的原型对比，并额外引入来自 mt_depth_teacher_v1 的可训练 depth teacher 分支。
"""

import inspect

import torch
import torch.nn.functional as F

from models.networks.prototype import Learnable_Prototypes
from utils import losses

from .base_strategy import BaseTrainingStrategy


class MTDepthGuiderProtoTeacherV1Strategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--proto_feature_dim", type=int, default=256)
        parser.add_argument("--proto_pixel_weight", type=float, default=0.05)
        parser.add_argument("--proto_momentum", type=float, default=0.999)
        parser.add_argument("--proto_entropy_q_low", type=int, default=20)
        parser.add_argument("--proto_entropy_q_high", type=int, default=95)
        parser.add_argument("--proto_entropy_temp", type=float, default=0.1)
        parser.add_argument("--proto_entropy_num_samples", type=int, default=1024)

    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        if int(args.use_depth or 0) not in (1, 13):
            raise ValueError(
                "mt_depth_guider_proto_teacher_v1 requires --use_depth 1 or 13 (depth1c guider input)."
            )
        self._enable_ema_support()

        self.labeled_bs = int(args.labeled_bs)
        self.num_classes = int(args.num_classes)
        self.proto_weight = float(args.proto_pixel_weight)
        self.q_low = int(args.proto_entropy_q_low)
        self.q_high = int(args.proto_entropy_q_high)
        self.proto_feature_dim = int(args.proto_feature_dim)
        self.consistency_start_iters = int(args.consistency_start_iters)

        self.depth_teacher_model = self._create_branch_model(self.model)
        self.depth_teacher_optimizer = torch.optim.Adam(
            self.depth_teacher_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0001,
        )

        self.learnable_prototypes_model = Learnable_Prototypes(
            num_classes=self.num_classes, feat_dim=self.proto_feature_dim
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

    def _create_branch_model(self, model):
        sig = inspect.signature(type(model).__init__)
        init_kwargs = {}
        model_params = getattr(model, "params", None)

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
            if param_name not in init_kwargs and hasattr(model, param_name):
                init_kwargs[param_name] = getattr(model, param_name)

        branch_model = type(model)(**init_kwargs)
        branch_model.load_state_dict(model.state_dict(), strict=False)
        if not hasattr(branch_model, "params") and hasattr(model, "params"):
            branch_model.params = model.params
        return branch_model.to(self.device)

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
        return F.interpolate(valid.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1) > 0.5

    def _build_proto_pairs(self, feat_map, class_assign, valid_mask, prototypes):
        feat_map = F.normalize(feat_map, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        flat_feat = feat_map.permute(0, 2, 3, 1).reshape(-1, feat_map.shape[1])
        flat_cls = class_assign.reshape(-1)
        flat_valid = valid_mask.reshape(-1)
        anchors, proto_targets = [], []
        for cls_id in range(self.num_classes):
            cls_mask = (flat_cls == cls_id) & flat_valid
            if int(cls_mask.sum().item()) == 0:
                continue
            cls_anchor = F.normalize(flat_feat[cls_mask].mean(dim=0, keepdim=True), p=2, dim=1)
            anchors.append(cls_anchor)
            proto_targets.append(prototypes[cls_id : cls_id + 1])
        if not anchors:
            return None, None
        return torch.cat(anchors, dim=0), torch.cat(proto_targets, dim=0)

    def _extract_logits(self, output):
        return output[0] if isinstance(output, (tuple, list)) else output

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        depth1 = batch_data.get("depth1")
        if depth1 is None:
            raise KeyError("mt_depth_guider_proto_teacher_v1 requires batch_data['depth1'] (1-channel depth guider).")
        depth1 = depth1.to(self.device)
        volume = torch.cat([image, depth1], dim=1)
        volume_depth_teacher = volume
        label = batch_data["label"].to(self.device)

        labeled_label = label[: self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs :]
        
        stud_volume = self._add_noise(volume, strong_flag="s", unlabeled_only=True)
        model_output = self.model(stud_volume)
        if isinstance(model_output, (tuple, list)):
            student_logits, student_proto_feat = model_output[0], model_output[1]
        else:
            student_logits, student_proto_feat = model_output, None
        student_probs = torch.softmax(student_logits, dim=1)

        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag="t")
            ema_logits = self._extract_logits(self.ema_model(ema_inputs))
            ema_probs = torch.softmax(ema_logits, dim=1)

        depth_teacher_inputs = self._add_noise(volume_depth_teacher, strong_flag="t")
        depth_teacher_logits = self._extract_logits(self.depth_teacher_model(depth_teacher_inputs))
        depth_teacher_probs = torch.softmax(depth_teacher_logits, dim=1)

        batch_data["teacher_pred"] = ema_probs
        batch_data["depth_teacher_pred"] = depth_teacher_probs[self.labeled_bs :]

        loss_ce = self.ce_loss(student_logits[: self.labeled_bs], labeled_label.long())
        loss_dice = self.dice_loss(student_probs[: self.labeled_bs], labeled_label.unsqueeze(1))
        loss_ce_depth_teacher = self.ce_loss(depth_teacher_logits[: self.labeled_bs], labeled_label.long())
        loss_dice_depth_teacher = self.dice_loss(depth_teacher_probs[: self.labeled_bs], labeled_label.unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce + loss_ce_depth_teacher + loss_dice_depth_teacher)

        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            ema_consistency_loss = torch.mean((student_probs[self.labeled_bs :] - ema_probs) ** 2)
            depth_teacher_consistency_loss = torch.mean((student_probs[self.labeled_bs :] - depth_teacher_probs[self.labeled_bs :]) ** 2)
        else:
            ema_consistency_loss = torch.tensor(0.0, device=self.device)
            depth_teacher_consistency_loss = torch.tensor(0.0, device=self.device)
        consistency_loss = ema_consistency_loss + depth_teacher_consistency_loss

        proto_loss = torch.tensor(0.0, device=self.device)
        prototypes = self.learnable_prototypes_model()
        if student_proto_feat is not None and iter_num >= self.consistency_start_iters:
            target_size = student_proto_feat.shape[2:]
            labeled_assign = self._resize_label_to_feat(labeled_label, target_size)
            labeled_valid = torch.ones_like(labeled_assign, dtype=torch.bool)
            if unlabeled_volume.shape[0] > 0:
                unlabeled_assign = self._resize_label_to_feat(torch.argmax(ema_probs, dim=1), target_size)
                unlabeled_valid = self._build_entropy_mask(ema_probs, target_size)
                class_assign = torch.cat([labeled_assign, unlabeled_assign], dim=0)
                valid_mask = torch.cat([labeled_valid, unlabeled_valid], dim=0)
            else:
                class_assign = labeled_assign
                valid_mask = labeled_valid
            z_i, z_j = self._build_proto_pairs(student_proto_feat, class_assign, valid_mask, prototypes)
            if z_i is not None and z_i.shape[0] > 1:
                proto_loss = self.contrastive_loss_model(z_i, z_j)

        total_loss = supervised_loss + consistency_weight * consistency_loss + self.proto_weight * proto_loss
        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "consistency": consistency_loss,
            "ema_consistency": ema_consistency_loss,
            "depth_teacher_consistency": depth_teacher_consistency_loss,
            "consistency_weight": consistency_weight,
            "proto": proto_loss,
            "proto_weight": torch.tensor(self.proto_weight, device=total_loss.device),
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        self.depth_teacher_optimizer.zero_grad(set_to_none=True)
        self.proto_optimizer.zero_grad(set_to_none=True)

        loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss_dict["total"].backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.depth_teacher_model.parameters(), max_norm=self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.learnable_prototypes_model.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()
        self.depth_teacher_optimizer.step()
        self.proto_optimizer.step()
        self._update_ema(iter_num)
        return loss_dict

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        self.depth_teacher_model.train(mode=training)
        self.learnable_prototypes_model.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "depth_teacher_model": self.depth_teacher_model.state_dict(),
            "learnable_prototypes_model": self.learnable_prototypes_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "depth_teacher_model" in state_dict:
                self.depth_teacher_model.load_state_dict(state_dict["depth_teacher_model"])
            if "learnable_prototypes_model" in state_dict:
                self.learnable_prototypes_model.load_state_dict(state_dict["learnable_prototypes_model"])
            return
        self.model.load_state_dict(state_dict)
