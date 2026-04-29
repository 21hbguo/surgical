"""
semi_mt_depth_teacher_v1:

基于 Mean Teacher 的三模型半监督策略。

实现要点：
1. Student 分支：主训练模型，负责监督分割学习。
2. EMA Teacher 分支：与原 Mean Teacher 一致，参数由 student 通过 EMA 更新。
3. Depth Teacher 分支：与 student 同构的独立模型，不使用 EMA，可通过反向传播直接更新。
4. 监督损失：student 在有标注样本上使用 CE + Dice。
5. 无监督损失：在无标注样本上，student 同时对齐 EMA Teacher 与 Depth Teacher 的 softmax 输出，使用 MSE 一致性约束。
"""

import inspect
import torch
from .base_strategy import BaseTrainingStrategy


class MTDepthTeacherV1Strategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)

        self.depth_teacher_model = self._create_branch_model(self.model)
        self.depth_teacher_optimizer = torch.optim.Adam(
            self.depth_teacher_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0001,
        )

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

    def _extract_logits(self, output):
        return output[0] if isinstance(output, (tuple, list)) else output

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        volume_depth = self._get_depth_tensor(batch_data)
        label = batch_data["label"].to(self.device)

        stud_volume = self._add_noise(volume, strong_flag="s", unlabeled_only=True)
        student_logits = self._extract_logits(self.model(stud_volume))
        student_probs = torch.softmax(student_logits, dim=1)

        unlabeled_volume = volume[self.labeled_bs :]
        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag="t")
            ema_logits = self._extract_logits(self.ema_model(ema_inputs))
            ema_probs = torch.softmax(ema_logits, dim=1)

        depth_teacher_inputs = self._add_noise(volume_depth, strong_flag="t")
        depth_teacher_logits = self._extract_logits(self.depth_teacher_model(depth_teacher_inputs))
        depth_teacher_probs = torch.softmax(depth_teacher_logits, dim=1)

        batch_data["teacher_pred"] = ema_probs
        batch_data["depth_teacher_pred"] = depth_teacher_probs[self.labeled_bs :]

        loss_ce = self.ce_loss(student_logits[: self.labeled_bs], label[: self.labeled_bs].long())
        loss_dice = self.dice_loss(student_probs[: self.labeled_bs], label[: self.labeled_bs].unsqueeze(1))
        loss_ce_depth = self.ce_loss(depth_teacher_logits[: self.labeled_bs], label[: self.labeled_bs].long())
        loss_dice_depth = self.dice_loss(depth_teacher_probs[: self.labeled_bs], label[: self.labeled_bs].unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce + loss_ce_depth + loss_dice_depth)

        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            ema_consistency_loss = torch.mean((student_probs[self.labeled_bs :] - ema_probs) ** 2)
            depth_teacher_consistency_loss = torch.mean((student_probs[self.labeled_bs :] - depth_teacher_probs[self.labeled_bs :]) ** 2)
        else:
            ema_consistency_loss = torch.tensor(0.0, device=self.device)
            depth_teacher_consistency_loss = torch.tensor(0.0, device=self.device)

        consistency_loss = ema_consistency_loss + depth_teacher_consistency_loss
        total_loss = supervised_loss + consistency_weight * consistency_loss

        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "consistency": consistency_loss,
            "ema_consistency": ema_consistency_loss,
            "depth_teacher_consistency": depth_teacher_consistency_loss,
            "consistency_weight": consistency_weight,
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        self.depth_teacher_optimizer.zero_grad(set_to_none=True)

        loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss_dict["total"].backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.depth_teacher_model.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()
        self.depth_teacher_optimizer.step()
        self._update_ema(iter_num)
        return loss_dict

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        self.depth_teacher_model.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "depth_teacher_model": self.depth_teacher_model.state_dict(),
        }

    def validation_step(self, batch_data):
        with torch.no_grad():
            volume = batch_data["image"].to(self.device)
            return self.model(volume)

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "depth_teacher_model" in state_dict:
                self.depth_teacher_model.load_state_dict(state_dict["depth_teacher_model"])
            return
        self.model.load_state_dict(state_dict)
