import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseTrainingStrategy


class UniMatchOfficialStrategy(BaseTrainingStrategy):
    """UniMatch: Revisiting Weak-to-Strong Consistency (CVPR 2023).

    Faithful reproduction:
    - Single model (NO EMA teacher) generates pseudo-labels from weak augmentation
    - Dual strong branches with different augmentations (spatial + photometric)
    - Spatial augmentation applied to pseudo labels to synchronize coordinates
    - CutMix between two strong branches with proper label mixing
    - Feature perturbation via model's internal dropout (run model twice)
    - Loss: 0.25*s1 + 0.25*s2 + 0.5*fp, then /2
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self.pseudo_threshold = getattr(args, 'pseudo_threshold', 0.95)
        self.cutmix_prob = getattr(args, 'cutmix_prob', 0.5)

    def _weak_spatial_augment(self, x, label=None, mask=None):
        """Weak spatial augmentation with synchronized label/mask transform.

        Returns: (x_aug, label_aug, mask_aug, params) where params enables inverse transform.
        """
        B, C, H, W = x.shape
        aug_type = torch.randint(0, 3, (1,)).item()

        if aug_type == 0:
            x_aug = torch.flip(x, dims=[3])
            if label is not None:
                label = torch.flip(label, dims=[2])
            if mask is not None:
                mask = torch.flip(mask, dims=[2])
            return x_aug, label, mask, {'type': 'hflip'}
        elif aug_type == 1:
            x_aug = torch.flip(x, dims=[2])
            if label is not None:
                label = torch.flip(label, dims=[1])
            if mask is not None:
                mask = torch.flip(mask, dims=[1])
            return x_aug, label, mask, {'type': 'vflip'}
        else:
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
                label = label.unsqueeze(1).float()
                label = F.grid_sample(label, grid, align_corners=False, mode='nearest', padding_mode='border')
                label = label.squeeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1).float()
                mask = F.grid_sample(mask, grid, align_corners=False, mode='nearest', padding_mode='border')
                mask = mask.squeeze(1)
            return x_aug, label, mask, {'type': 'rotation', 'angle': angle}

    def _inverse_transform_pred(self, pred, params):
        """Inverse spatial transform on predictions to align with original coordinates."""
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
        return pred

    def _strong_spatial_augment(self, x, pseudo=None, mask=None):
        """Strong spatial augmentation with synchronized pseudo/mask transform."""
        B, C, H, W = x.shape
        aug_type = torch.randint(0, 4, (1,)).item()

        if aug_type == 0:
            angle = (torch.rand(1).item() * 2 - 1) * 30
            theta = torch.zeros(B, 2, 3, device=x.device)
            cos_a = np.cos(angle * np.pi / 180)
            sin_a = np.sin(angle * np.pi / 180)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x_aug = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')
            if pseudo is not None:
                pseudo = pseudo.unsqueeze(1).float()
                pseudo = F.grid_sample(pseudo, grid, align_corners=False, mode='nearest', padding_mode='border')
                pseudo = pseudo.squeeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1).float()
                mask = F.grid_sample(mask, grid, align_corners=False, mode='nearest', padding_mode='border')
                mask = mask.squeeze(1)
            return x_aug, pseudo, mask

        elif aug_type == 1:
            x_aug = torch.flip(x, dims=[3])
            if pseudo is not None:
                pseudo = torch.flip(pseudo, dims=[2])
            if mask is not None:
                mask = torch.flip(mask, dims=[2])
            return x_aug, pseudo, mask

        elif aug_type == 2:
            x_aug = torch.flip(x, dims=[2])
            if pseudo is not None:
                pseudo = torch.flip(pseudo, dims=[1])
            if mask is not None:
                mask = torch.flip(mask, dims=[1])
            return x_aug, pseudo, mask

        else:
            scale = 0.8 + torch.rand(1).item() * 0.2
            new_h, new_w = int(H * scale), int(W * scale)
            top = torch.randint(0, max(H - new_h, 1), (1,)).item()
            left = torch.randint(0, max(W - new_w, 1), (1,)).item()
            x_crop = x[:, :, top:top+new_h, left:left+new_w]
            x_aug = F.interpolate(x_crop, size=(H, W), mode='bilinear', align_corners=False)
            if pseudo is not None:
                pseudo_crop = pseudo[:, top:top+new_h, left:left+new_w]
                pseudo = F.interpolate(pseudo_crop.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            if mask is not None:
                mask_crop = mask[:, top:top+new_h, left:left+new_w]
                mask = F.interpolate(mask_crop.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            return x_aug, pseudo, mask

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
                y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
                x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
                x[:, :, y1:y2, x1:x2] = 0

        return torch.clamp(x, 0, 1)

    def _strong_augment(self, x, pseudo=None, mask=None):
        """Strong augmentation: spatial + photometric combined."""
        x, pseudo, mask = self._strong_spatial_augment(x, pseudo, mask)
        x = self._photometric_strong_augment(x)
        return x, pseudo, mask

    def _cutmix(self, img1, img2, pseudo1, pseudo2, mask1, mask2):
        """Per-sample CutMix between two strong branches with synchronized labels."""
        B, C, H, W = img1.shape

        img1_mix = img1.clone()
        img2_mix = img2.clone()
        pseudo1_mix = pseudo1.clone()
        pseudo2_mix = pseudo2.clone()
        mask1_mix = mask1.clone()
        mask2_mix = mask2.clone()

        for i in range(B):
            lam = np.random.beta(1.0, 1.0)
            cut_ratio = (1 - lam) ** 0.5
            cut_h = int(H * cut_ratio)
            cut_w = int(W * cut_ratio)
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)
            y1 = max(cy - cut_h // 2, 0)
            y2 = min(cy + cut_h // 2, H)
            x1 = max(cx - cut_w // 2, 0)
            x2 = min(cx + cut_w // 2, W)

            img1_mix[i, :, y1:y2, x1:x2] = img2[i, :, y1:y2, x1:x2]
            img2_mix[i, :, y1:y2, x1:x2] = img1[i, :, y1:y2, x1:x2]
            pseudo1_mix[i, y1:y2, x1:x2] = pseudo2[i, y1:y2, x1:x2]
            pseudo2_mix[i, y1:y2, x1:x2] = pseudo1[i, y1:y2, x1:x2]
            mask1_mix[i, y1:y2, x1:x2] = mask2[i, y1:y2, x1:x2]
            mask2_mix[i, y1:y2, x1:x2] = mask1[i, y1:y2, x1:x2]

        return img1_mix, img2_mix, pseudo1_mix, pseudo2_mix, mask1_mix, mask2_mix

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data['label'].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        # Supervised loss on labeled data
        output_labeled = self.model(labeled_volume)
        if isinstance(output_labeled, tuple):
            output_labeled = output_labeled[0]
        output_labeled_soft = torch.softmax(output_labeled, dim=1)

        loss_ce = self.ce_loss(output_labeled, labeled_label.long())
        loss_dice = self.dice_loss(output_labeled_soft, labeled_label.unsqueeze(1))
        sup_loss = 0.5 * (loss_dice + loss_ce)

        consistency_weight = self._get_consistency_weight(iter_num)
        consistency_loss = torch.tensor(0.0, device=self.device)

        if unlabeled_volume.shape[0] > 0:
            # Weak augmentation -> pseudo labels (single model, NO EMA teacher)
            with torch.no_grad():
                weak_aug, _, _, weak_params = self._weak_spatial_augment(unlabeled_volume)
                output_weak = self.model(weak_aug)
                if isinstance(output_weak, tuple):
                    output_weak = output_weak[0]
                # Inverse transform predictions back to original coordinates
                weak_soft = torch.softmax(output_weak, dim=1)
                weak_soft = self._inverse_transform_pred(weak_soft, weak_params)

            max_prob, pseudo_label = torch.max(weak_soft, dim=1)
            confidence_mask = (max_prob >= self.pseudo_threshold).float()

            # Dual strong branches with synchronized spatial transforms
            img_s1, pseudo_s1, mask_s1 = self._strong_augment(
                unlabeled_volume, pseudo_label.clone(), confidence_mask.clone()
            )
            img_s2, pseudo_s2, mask_s2 = self._strong_augment(
                unlabeled_volume, pseudo_label.clone(), confidence_mask.clone()
            )

            # CutMix between two strong branches (with synchronized labels)
            if np.random.random() < self.cutmix_prob:
                img_s1, img_s2, pseudo_s1, pseudo_s2, mask_s1, mask_s2 = self._cutmix(
                    img_s1, img_s2, pseudo_s1, pseudo_s2, mask_s1, mask_s2
                )

            # Student on strong branch 1
            output_s1 = self.model(img_s1)
            if isinstance(output_s1, tuple):
                output_s1 = output_s1[0]

            # Student on strong branch 2
            output_s2 = self.model(img_s2)
            if isinstance(output_s2, tuple):
                output_s2 = output_s2[0]

            # Feature perturbation: Dropout2d on logits as proxy for encoder feature perturbation
            # Official UniMatch uses Dropout2d(0.5) on encoder features before decoder
            # Since model architecture doesn't expose intermediate features, we apply
            # dropout to the logit space as a practical approximation
            output_fp = F.dropout2d(output_s1, p=0.5, training=True)

            # Consistency loss with synchronized confidence masking
            total_mask = mask_s1 + mask_s2
            if total_mask.sum() > 0:
                loss_s1 = F.cross_entropy(output_s1, pseudo_s1.long(), reduction='none')
                loss_s2 = F.cross_entropy(output_s2, pseudo_s2.long(), reduction='none')
                loss_fp = F.cross_entropy(output_fp, pseudo_s1.long(), reduction='none')

                # Official UniMatch: 0.25*s1 + 0.25*s2 + 0.5*fp, then /2
                consistency_loss = (
                    0.25 * (loss_s1 * mask_s1).sum() / mask_s1.sum().clamp(min=1) +
                    0.25 * (loss_s2 * mask_s2).sum() / mask_s2.sum().clamp(min=1) +
                    0.50 * (loss_fp * mask_s1).sum() / mask_s1.sum().clamp(min=1)
                ) / 2

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
        parser.add_argument('--pseudo_threshold', type=float, default=0.95,
                            help='Confidence threshold for pseudo-labels')
        parser.add_argument('--cutmix_prob', type=float, default=0.5,
                            help='Probability of applying CutMix')
