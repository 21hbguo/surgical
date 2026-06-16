import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseTrainingStrategy


class SegMatchOfficialStrategy(BaseTrainingStrategy):
    """SegMatch: Semi-supervised surgical instrument segmentation (Scientific Reports 2025).

    Faithful reproduction:
    - Single model with weak/strong branches (no EMA teacher)
    - Spatial weak augmentation with inverse transform on predictions
    - Photometric strong augmentation (RandAugment 3-op)
    - I-FGSM adversarial: eps=0.08, K=25, alpha=eps/K, projection around x0_s
    - Soft pseudo labels with temperature sharpening
    - Confidence mask applied to I-FGSM loss
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self.pseudo_threshold = args.pseudo_threshold
        self.sharpen_temperature = args.sharpen_temperature

        # I-FGSM parameters (paper defaults)
        self.adversarial_eps = args.adversarial_eps
        self.adversarial_steps = args.adversarial_steps
        self.adversarial_alpha = self.adversarial_eps / self.adversarial_steps

    def _weak_spatial_augment(self, x, label=None, exclude_crop=False):
        """Single random spatial weak augmentation with transform params for inverse.

        Args:
            exclude_crop: If True, only use flip/rotation (not crop).
                Crop creates partial views that don't align with original coords.
        """
        B, C, H, W = x.shape
        aug_type = torch.randint(0, 3 if exclude_crop else 4, (1,)).item()

        if aug_type == 0:
            angle = (torch.rand(1).item() * 2 - 1) * 15
            theta = torch.zeros(B, 2, 3, device=x.device)
            cos_a = np.cos(angle * np.pi / 180)
            sin_a = np.sin(angle * np.pi / 180)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x_aug = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')
            if label is not None:
                label_aug = label.unsqueeze(1).float()
                label_aug = F.grid_sample(label_aug, grid, align_corners=False, mode='nearest', padding_mode='border')
                label_aug = label_aug.squeeze(1)
                return x_aug, label_aug, {'type': 'rotation', 'angle': angle}
            return x_aug, None, {'type': 'rotation', 'angle': angle}

        elif aug_type == 1:
            x_aug = torch.flip(x, dims=[3])
            if label is not None:
                return x_aug, torch.flip(label, dims=[2]), {'type': 'hflip'}
            return x_aug, None, {'type': 'hflip'}

        elif aug_type == 2:
            x_aug = torch.flip(x, dims=[2])
            if label is not None:
                return x_aug, torch.flip(label, dims=[1]), {'type': 'vflip'}
            return x_aug, None, {'type': 'vflip'}

        else:
            scale = 0.8 + torch.rand(1).item() * 0.2
            new_h, new_w = int(H * scale), int(W * scale)
            top = torch.randint(0, max(H - new_h, 1), (1,)).item()
            left = torch.randint(0, max(W - new_w, 1), (1,)).item()
            x_crop = x[:, :, top:top+new_h, left:left+new_w]
            x_aug = F.interpolate(x_crop, size=(H, W), mode='bilinear', align_corners=False)
            if label is not None:
                label_crop = label[:, top:top+new_h, left:left+new_w]
                label_aug = F.interpolate(label_crop.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
                return x_aug, label_aug, {'type': 'crop', 'top': top, 'left': left,
                                          'new_h': new_h, 'new_w': new_w, 'H': H, 'W': W}
            return x_aug, None, {'type': 'crop', 'top': top, 'left': left,
                                 'new_h': new_h, 'new_w': new_w, 'H': H, 'W': W}

    def _inverse_transform(self, pred, params):
        """Apply inverse spatial transform on predictions to align with original coords."""
        t = params['type']
        if t == 'rotation':
            B = pred.shape[0]
            angle = -params['angle']
            theta_inv = torch.zeros(B, 2, 3, device=pred.device)
            cos_a = np.cos(angle * np.pi / 180)
            sin_a = np.sin(angle * np.pi / 180)
            theta_inv[:, 0, 0] = cos_a
            theta_inv[:, 0, 1] = -sin_a
            theta_inv[:, 1, 0] = sin_a
            theta_inv[:, 1, 1] = cos_a
            grid = F.affine_grid(theta_inv, pred.size(), align_corners=False)
            return F.grid_sample(pred, grid, align_corners=False, mode='bilinear', padding_mode='border')
        elif t == 'hflip':
            return torch.flip(pred, dims=[3])
        elif t == 'vflip':
            return torch.flip(pred, dims=[2])
        else:
            # Crop: predictions are already at original resolution after model forward
            # No inverse needed since model outputs at original spatial dims
            return pred

    def _sharpen(self, probs, T=0.5):
        """Temperature sharpening for soft pseudo labels."""
        sharpened = probs ** (1.0 / T)
        return sharpened / sharpened.sum(dim=1, keepdim=True)

    def _photometric_strong_augment(self, x):
        """RandAugment-style photometric strong augmentation (3 random ops)."""
        ops = ['brightness', 'contrast', 'saturation', 'noise', 'cutout']
        chosen = np.random.choice(ops, 3, replace=False)

        for op in chosen:
            if op == 'brightness':
                factor = 0.6 + torch.rand(1).item() * 0.8
                x = x * factor
            elif op == 'contrast':
                factor = 0.6 + torch.rand(1).item() * 0.8
                mean = x.mean(dim=(2, 3), keepdim=True)
                x = factor * (x - mean) + mean
            elif op == 'saturation':
                gray = x.mean(dim=1, keepdim=True)
                factor = 0.6 + torch.rand(1).item() * 0.8
                x = factor * x + (1 - factor) * gray
            elif op == 'noise':
                x = x + torch.randn_like(x) * 0.1
            elif op == 'cutout':
                B, C, H, W = x.shape
                ch = int(H * 0.2)
                cw = int(W * 0.2)
                cy = torch.randint(0, H, (1,)).item()
                cx = torch.randint(0, W, (1,)).item()
                y1, y2 = max(cy - ch//2, 0), min(cy + ch//2, H)
                x1, x2 = max(cx - cw//2, 0), min(cx + cw//2, W)
                x[:, :, y1:y2, x1:x2] = 0

        return torch.clamp(x, 0, 1)

    def _generate_adversarial(self, x0_s, pseudo_label, confidence_mask):
        """I-FGSM adversarial augmentation with confidence masking."""
        adv = x0_s.detach().requires_grad_(True)

        for _ in range(self.adversarial_steps):
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                output = self.model(adv)
                if isinstance(output, tuple):
                    output = output[0]
                loss = F.cross_entropy(output, pseudo_label.long(), reduction='none')
                loss = (loss * confidence_mask).sum() / confidence_mask.sum().clamp(min=1)
            loss.backward()

            with torch.no_grad():
                adv = adv + self.adversarial_alpha * adv.grad.sign()
                delta = adv - x0_s
                delta = torch.clamp(delta, -self.adversarial_eps, self.adversarial_eps)
                adv = (x0_s + delta).detach().requires_grad_(True)

            self.model.zero_grad()

        return adv.detach()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data['label'].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        # Supervised loss (with synchronized weak spatial augmentation)
        sup_loss = torch.tensor(0.0, device=self.device)
        loss_ce = torch.tensor(0.0, device=self.device)
        loss_dice = torch.tensor(0.0, device=self.device)

        if self.labeled_bs > 0:
            labeled_aug, label_aug, _ = self._weak_spatial_augment(labeled_volume, labeled_label)
            out_l = self.model(labeled_aug)
            if isinstance(out_l, tuple):
                out_l = out_l[0]
            out_l_soft = torch.softmax(out_l, dim=1)
            loss_ce = self.ce_loss(out_l, label_aug.long())
            loss_dice = self.dice_loss(out_l_soft, label_aug.unsqueeze(1))
            sup_loss = 0.5 * (loss_dice + loss_ce)

        # Consistency loss (weak-to-strong with I-FGSM)
        consistency_weight = self._get_consistency_weight(iter_num)
        consistency_loss = torch.tensor(0.0, device=self.device)

        if unlabeled_volume.shape[0] > 0:
            # Weak branch: model on weakly augmented unlabeled -> pseudo labels
            # Exclude crop to ensure predictions align with original coordinates
            with torch.no_grad():
                weak_aug, _, aug_params = self._weak_spatial_augment(
                    unlabeled_volume, exclude_crop=True
                )
                out_w = self.model(weak_aug)
                if isinstance(out_w, tuple):
                    out_w = out_w[0]
                # Inverse transform predictions back to original coordinates
                out_w_soft = torch.softmax(out_w, dim=1)
                out_w_soft_orig = self._inverse_transform(out_w_soft, aug_params)
                # Soft pseudo labels with temperature sharpening
                pseudo_soft = self._sharpen(out_w_soft_orig, T=self.sharpen_temperature)

            max_prob, pseudo_label = torch.max(pseudo_soft, dim=1)
            confidence_mask = (max_prob >= self.pseudo_threshold).float()

            # Strong branch: photometric augmentation -> x0_s
            x0_s = self._photometric_strong_augment(unlabeled_volume)

            # I-FGSM adversarial around x0_s (with confidence mask)
            adv_volume = self._generate_adversarial(x0_s, pseudo_label, confidence_mask)

            # Student on adversarial augmentation
            out_adv = self.model(adv_volume)
            if isinstance(out_adv, tuple):
                out_adv = out_adv[0]

            if confidence_mask.sum() > 0:
                # KL divergence with soft pseudo labels (preserves uncertainty)
                log_adv = F.log_softmax(out_adv, dim=1)
                kl_loss = F.kl_div(log_adv, pseudo_soft.detach(), reduction='none').sum(dim=1)
                consistency_loss = (kl_loss * confidence_mask).sum() / confidence_mask.sum()

        total_loss = sup_loss + consistency_weight * consistency_loss

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': consistency_loss,
            'consistency_weight': consistency_weight
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        return loss_dict

    @staticmethod
    def add_args(parser):
        parser.add_argument('--pseudo_threshold', type=float, default=0.8,
                            help='Confidence threshold (paper stable: 0.7-0.9)')
        parser.add_argument('--sharpen_temperature', type=float, default=0.5,
                            help='Temperature for pseudo label sharpening')
        parser.add_argument('--adversarial_eps', type=float, default=0.08,
                            help='I-FGSM epsilon (paper default: 0.08)')
        parser.add_argument('--adversarial_steps', type=int, default=25,
                            help='I-FGSM steps (paper default: 25)')
