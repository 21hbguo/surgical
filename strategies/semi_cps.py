import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class CPSStrategy(BaseTrainingStrategy):
    """CPS: Cross Pseudo Supervision for Semi-Supervised Semantic Segmentation (CVPR 2021).

    Two independently initialized networks generate pseudo-labels for each other
    through cross-supervision. This faithfully reproduces the official SSL4MIS
    implementation which uses two independent networks (not student+EMA).
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        # Create second INDEPENDENT network with fresh random weights (not EMA)
        # This matches official CPS which uses two independently initialized networks
        self.model2 = self._create_independent_model()
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001
        )

    def _create_independent_model(self):
        """Create a fresh model with random weights (not copying from model1)."""
        import inspect
        orig_model = getattr(self.model, "_orig_mod", self.model)
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

        # Create fresh model with random weights (NO weight copying)
        new_model = type(orig_model)(**init_kwargs)
        # Do NOT load state_dict - keep random initialization
        if not hasattr(new_model, "params") and hasattr(self.model, "params"):
            new_model.params = self.model.params
        return new_model.to(self.device)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data['label'].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        # Forward pass through both independent networks
        output1 = self.model(volume)
        if isinstance(output1, tuple):
            output1 = output1[0]
        output1_soft = torch.softmax(output1, dim=1)

        output2 = self.model2(volume)
        if isinstance(output2, tuple):
            output2 = output2[0]
        output2_soft = torch.softmax(output2, dim=1)

        # Supervised loss on labeled data for both networks
        loss_ce_1 = self.ce_loss(output1[:self.labeled_bs], labeled_label.long())
        loss_dice_1 = self.dice_loss(output1_soft[:self.labeled_bs], labeled_label.unsqueeze(1))
        sup_loss_1 = 0.5 * (loss_dice_1 + loss_ce_1)

        loss_ce_2 = self.ce_loss(output2[:self.labeled_bs], labeled_label.long())
        loss_dice_2 = self.dice_loss(output2_soft[:self.labeled_bs], labeled_label.unsqueeze(1))
        sup_loss_2 = 0.5 * (loss_dice_2 + loss_ce_2)

        # Cross pseudo supervision on ALL samples (labeled + unlabeled)
        # This matches official SSL4MIS CPS implementation
        consistency_weight = self._get_consistency_weight(iter_num)
        pseudo_1 = torch.argmax(output1_soft.detach(), dim=1)
        pseudo_2 = torch.argmax(output2_soft.detach(), dim=1)

        # Network 1 learns from pseudo-labels of network 2
        cps_loss_1 = F.cross_entropy(output1, pseudo_2.long())
        # Network 2 learns from pseudo-labels of network 1
        cps_loss_2 = F.cross_entropy(output2, pseudo_1.long())

        # Total loss: each model gets its own supervised + CPS loss
        model1_loss = sup_loss_1 + consistency_weight * cps_loss_1
        model2_loss = sup_loss_2 + consistency_weight * cps_loss_2
        total_loss = model1_loss + model2_loss

        return {
            'total': total_loss,
            'ce': loss_ce_1,
            'dice': loss_dice_1,
            'consistency': 0.5 * (cps_loss_1 + cps_loss_2),
            'consistency_weight': consistency_weight
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer2.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        # Single backward pass through combined loss, then step both optimizers
        loss = loss_dict['total']
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # Unscale and step optimizer1
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            # Unscale and step optimizer2
            self.scaler.unscale_(self.optimizer2)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer2)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer2.step()
        return loss_dict

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        self.model2.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "model2": self.model2.state_dict()
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "model2" in state_dict:
                self.model2.load_state_dict(state_dict["model2"])
            return
        self.model.load_state_dict(state_dict)
