import inspect

import torch
import torch.nn as nn
from utils.losses import DiceLoss
from utils.losses import CoordLoss
from utils.common import sigmoid_rampup


class BaseTrainingStrategy:
    def __init__(self, args, model, optimizer, device, scaler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.amp_enabled = scaler is not None
        self.consistency = args.consistency
        self.grad_clip = args.grad_clip
        self.strong = args.strong
        self.labeled_bs = args.labeled_bs
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(args.num_classes)
        self.coord_loss = CoordLoss()
        self.use_depth = args.use_depth
        self.ema_model = None
        self.ema_decay = None

    def _get_depth_tensor(self, batch_data):
        if not self.use_depth:
            return None
        depth_key = "depth3" if int(self.use_depth) == 3 else "depth1"
        depth_tensor = batch_data.get(depth_key)
        if depth_tensor is None:
            raise KeyError(f"use_depth={self.use_depth} requires batch_data['{depth_key}']")
        return depth_tensor.to(self.device)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        raise NotImplementedError

    def _add_noise(self, data, strong_flag, unlabeled_only=False):
        if self.strong != strong_flag:
            return data

        data_noisy = data.clone()
        noise_target = data_noisy[self.labeled_bs :] if unlabeled_only else data_noisy
        noise = torch.clamp(torch.randn_like(noise_target) * 0.1, -0.2, 0.2)
        noise_target += noise
        return data_noisy

    def _get_consistency_weight(self, iter_num, rampup=None, div=None):
        if rampup is None:
            rampup = self.args.consistency_rampup
        if div is None:
            div = self.args.consistency_rampup_div
        return self.consistency * sigmoid_rampup(iter_num // div, rampup)

    def _resolve_model_param_value(self, model_params, param_name, param):
        if param_name in model_params:
            return model_params[param_name]

        aliases = {
            "in_chns": "in_channels",
            "in_channels": "in_chns",
            "num_classes": "class_num",
            "class_num": "num_classes",
        }
        alias_name = aliases.get(param_name)
        if alias_name in model_params:
            return model_params[alias_name]

        if param.default != inspect.Parameter.empty:
            return param.default
        return inspect.Parameter.empty

    def _create_ema_model(self, model):
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

        ema_model = type(orig_model)(**init_kwargs)
        ema_model.load_state_dict(orig_model.state_dict(), strict=False)
        for param in ema_model.parameters():
            param.detach_()
        if not hasattr(ema_model, "params") and hasattr(model, "params"):
            ema_model.params = model.params
        return ema_model.to(self.device)

    def _enable_ema_support(self):
        if self.ema_model is not None:
            return
        self.ema_model = self._create_ema_model(self.model)
        self.ema_decay = self.args.ema_decay
        self.ema_model.train()

    def _update_ema(self, global_step):
        if self.ema_model is None:
            return
        alpha = min(1 - 1 / (global_step + 1), self.ema_decay)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def training_step(self, batch_data, iter_num=0, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss = loss_dict["total"]
        self._backward_and_step(loss, optimizer=self.optimizer)
        return loss_dict

    def _backward_and_step(
        self, loss, optimizer=None, clip_params=None, clip_max_norm=None
    ):
        opt = optimizer if optimizer is not None else self.optimizer
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(opt)
                params = clip_params if clip_params is not None else self.model.parameters()
                max_norm = self.grad_clip if clip_max_norm is None else clip_max_norm
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                params = clip_params if clip_params is not None else self.model.parameters()
                max_norm = self.grad_clip if clip_max_norm is None else clip_max_norm
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            opt.step()

    def validation_step(self, batch_data):
        with torch.no_grad():
            volume = batch_data["image"].to(self.device)
            depth_tensor = self._get_depth_tensor(batch_data)
            if depth_tensor is not None:
                volume = torch.cat([volume, depth_tensor], dim=1)
            return self.model(volume)

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        if self.ema_model is not None:
            self.ema_model.eval()

    def eval(self):
        self._set_model_mode(False)

    def train(self):
        self._set_model_mode(True)

    def get_state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            return
        self.model.load_state_dict(state_dict)
