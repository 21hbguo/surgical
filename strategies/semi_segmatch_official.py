import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseTrainingStrategy


class SegMatchOfficialStrategy(BaseTrainingStrategy):
    """SegMatch: Semi-supervised surgical instrument segmentation (Scientific Reports 2025).

    Faithful reproduction of official paper:
    - Spatial weak augmentation with inverse transform
    - Photometric strong augmentation (RandAugment-style)
    - I-FGSM adversarial: eps=0.08, K=25, alpha=eps/K
    - Proper projection to [-eps, eps]
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)
        self.pseudo_threshold = getattr(args, 'pseudo_threshold', 0.95)

        # I-FGSM parameters (paper defaults)
        self.adversarial_eps = getattr(args, 'adversarial_eps', 0.08)
        self.adversarial_steps = getattr(args, 'adversarial_steps', 25)
        self.adversarial_alpha = self.adversarial_eps / self.adversarial_steps

        # Spatial augmentation parameters
        self.spatial_scale = getattr(args, 'spatial_scale', 0.1)
        self.spatial_angle = getattr(args, 'spatial_angle', 10)

    def _spatial_weak_augment(self, x):
        """Spatial weak augmentation (rotation + scale + translation).

        Returns augmented image and transformation params for inverse transform.
        """
        B, C, H, W = x.shape

        # Random rotation (-10, 10 degrees)
        angle = (torch.rand(B, device=x.device) * 2 - 1) * self.spatial_angle
        # Random scale (0.9, 1.1)
        scale = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * self.spatial_scale
        # Random translation (-10%, 10%)
        tx = (torch.rand(B, device=x.device) * 2 - 1) * 0.1
        ty = (torch.rand(B, device=x.device) * 2 - 1) * 0.1

        # Build affine matrix
        theta = torch.zeros(B, 2, 3, device=x.device)
        cos_a = torch.cos(angle * np.pi / 180)
        sin_a = torch.sin(angle * np.pi / 180)

        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = -sin_a * scale
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a * scale
        theta[:, 1, 1] = cos_a * scale
        theta[:, 1, 2] = ty

        # Apply affine transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_aug = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')

        return x_aug, {'angle': angle, 'scale': scale, 'tx': tx, 'ty': ty}

    def _inverse_spatial_transform(self, x, transform_params):
        """Apply inverse spatial transform to map predictions back to original coordinates."""
        B, C, H, W = x.shape
        angle = transform_params['angle']
        scale = transform_params['scale']
        tx = transform_params['tx']
        ty = transform_params['ty']

        # Inverse affine matrix
        theta_inv = torch.zeros(B, 2, 3, device=x.device)
        cos_a = torch.cos(-angle * np.pi / 180)
        sin_a = torch.sin(-angle * np.pi / 180)
        inv_scale = 1.0 / scale

        theta_inv[:, 0, 0] = cos_a * inv_scale
        theta_inv[:, 0, 1] = -sin_a * inv_scale
        theta_inv[:, 0, 2] = -tx * inv_scale
        theta_inv[:, 1, 0] = sin_a * inv_scale
        theta_inv[:, 1, 1] = cos_a * inv_scale
        theta_inv[:, 1, 2] = -ty * inv_scale

        grid = F.affine_grid(theta_inv, x.size(), align_corners=False)
        x_inv = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')

        return x_inv

    def _photometric_strong_augment(self, x):
        """Photometric strong augmentation (RandAugment-style).

        Applies random brightness, contrast, saturation, hue jitter.
        """
        # Brightness jitter
        if torch.rand(1).item() < 0.5:
            brightness = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            x = x * brightness

        # Contrast jitter
        if torch.rand(1).item() < 0.5:
            contrast = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = contrast * (x - mean) + mean

        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * 0.1
            x = x + noise

        # Cutout
        if torch.rand(1).item() < 0.5:
            B, C, H, W = x.shape
            cut_h = int(H * 0.2)
            cut_w = int(W * 0.2)
            cx = torch.randint(0, W, (1,)).item()
            cy = torch.randint(0, H, (1,)).item()
            x1 = max(cx - cut_w // 2, 0)
            y1 = max(cy - cut_h // 2, 0)
            x2 = min(cx + cut_w // 2, W)
            y2 = min(cy + cut_h // 2, H)
            x[:, :, y1:y2, x1:x2] = 0

        return torch.clamp(x, 0, 1)

    def _generate_adversarial(self, unlabeled_volume, teacher_soft, transform_params):
        """Generate adversarial augmentation using I-FGSM.

        Paper: eps=0.08, K=25, alpha=eps/K
        Projection: [-eps, eps]
        """
        # Start from strong augmented image
        adv_volume = self._photometric_strong_augment(unlabeled_volume.clone())
        adv_volume = adv_volume.detach().requires_grad_(True)
        pseudo_label = torch.argmax(teacher_soft, dim=1)

        for _ in range(self.adversarial_steps):
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                adv_output = self.model(adv_volume)
                if isinstance(adv_output, tuple):
                    adv_output = adv_output[0]
                adv_loss = self.ce_loss(adv_output, pseudo_label.long())

            adv_loss.backward()

            with torch.no_grad():
                # I-FGSM step with alpha
                adv_volume = adv_volume + self.adversarial_alpha * adv_volume.grad.sign()
                # Project to [-eps, eps] (correct projection, not [-2eps, 2eps])
                delta = adv_volume - unlabeled_volume
                delta = torch.clamp(delta, -self.adversarial_eps, self.adversarial_eps)
                adv_volume = (unlabeled_volume + delta).detach().requires_grad_(True)

            # Zero model gradients
            self.model.zero_grad()

        return adv_volume.detach()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data['label'].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        # Supervised loss with spatial weak augmentation
        if self.labeled_bs > 0:
            labeled_aug, transform_params = self._spatial_weak_augment(labeled_volume)
            output_labeled = self.model(labeled_aug)
            if isinstance(output_labeled, tuple):
                output_labeled = output_labeled[0]
            output_labeled_soft = torch.softmax(output_labeled, dim=1)

            loss_ce = self.ce_loss(output_labeled, labeled_label.long())
            loss_dice = self.dice_loss(output_labeled_soft, labeled_label.unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
        else:
            supervised_loss = torch.tensor(0.0, device=self.device)
            loss_ce = torch.tensor(0.0, device=self.device)
            loss_dice = torch.tensor(0.0, device=self.device)

        consistency_weight = self._get_consistency_weight(iter_num)
        consistency_loss = torch.tensor(0.0, device=self.device)

        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            # Teacher with spatial weak augmentation generates pseudo-labels
            with torch.no_grad():
                weak_aug, transform_params = self._spatial_weak_augment(unlabeled_volume)
                teacher_output = self.ema_model(weak_aug)
                if isinstance(teacher_output, tuple):
                    teacher_output = teacher_output[0]
                teacher_soft = torch.softmax(teacher_output, dim=1)

                # Inverse transform pseudo-labels back to original coordinates
                teacher_soft_orig = self._inverse_spatial_transform(
                    teacher_soft.unsqueeze(1), transform_params
                ).squeeze(1)

            # Generate adversarial strong augmentation
            adv_volume = self._generate_adversarial(unlabeled_volume, teacher_soft_orig, transform_params)

            # Student with adversarial augmentation
            student_output = self.model(adv_volume)
            if isinstance(student_output, tuple):
                student_output = student_output[0]

            # Confidence mask for pseudo-labels
            max_prob, pseudo_label = torch.max(teacher_soft_orig, dim=1)
            mask = (max_prob >= self.pseudo_threshold).float()

            # Cross-entropy with pseudo-labels (only on confident pixels)
            if mask.sum() > 0:
                ce_loss = F.cross_entropy(student_output, pseudo_label.long(), reduction='none')
                consistency_loss = (ce_loss * mask).sum() / mask.sum()

        total_loss = supervised_loss + consistency_weight * consistency_loss

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
        self._update_ema(iter_num)
        return loss_dict

    @staticmethod
    def add_args(parser):
        parser.add_argument('--pseudo_threshold', type=float, default=0.95,
                            help='Confidence threshold for pseudo-labels')
        parser.add_argument('--adversarial_eps', type=float, default=0.08,
                            help='Epsilon for I-FGSM (paper default: 0.08)')
        parser.add_argument('--adversarial_steps', type=int, default=25,
                            help='Number of I-FGSM steps (paper default: 25)')
        parser.add_argument('--spatial_scale', type=float, default=0.1,
                            help='Scale range for spatial augmentation')
        parser.add_argument('--spatial_angle', type=float, default=10,
                            help='Rotation angle range for spatial augmentation')
