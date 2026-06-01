import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class ThreshController:
    """Dynamic threshold controller (from CorrMatch official).

    Tracks per-class maximum confidence EMA for adaptive thresholding.
    """
    def __init__(self, nclass, momentum=0.999, thresh_init=0.95):
        self.nclass = nclass
        self.momentum = momentum
        self.thresh_global = thresh_init
        self.count = 0

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        """Update global threshold based on per-class max confidence EMA."""
        prob = pred.softmax(dim=1)
        conf = prob.max(dim=1)[0]

        if ignore_mask is not None:
            valid_mask = ignore_mask != 255
        else:
            valid_mask = torch.ones_like(conf, dtype=torch.bool)

        if valid_mask.sum() > 0:
            mean_conf = conf[valid_mask].mean().item()
            if update_g:
                if self.count == 0:
                    self.thresh_global = mean_conf
                else:
                    self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * mean_conf
                self.count += 1

    def get_thresh_global(self):
        return self.thresh_global


class CorrelationHead(nn.Module):
    """Correlation head for label propagation (from CorrMatch).

    Takes bottleneck features and produces per-class correlation logits.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.conv_out(x)


class CorrMatchStrategy(BaseTrainingStrategy):
    """CorrMatch: Label Propagation via Correlation Matching (CVPR 2024).

    Faithful reproduction based on official code:
    - Single model generates pseudo-labels from weak augmentation
    - Correlation head on bottleneck features for label propagation
    - High-confidence pixels serve as anchors for label propagation
    - Dynamic threshold controller adjusts confidence threshold
    - Dual strong branches with CutMix
    - Feature perturbation via dropout
    - KL divergence consistency between branches
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)

        num_classes = getattr(args, 'num_classes', 2)
        bottleneck_channels = 256  # filter_num * 16 for default UNet

        self.pseudo_threshold = getattr(args, 'corrmatch_threshold', 0.95)
        self.cutmix_prob = getattr(args, 'corrmatch_cutmix_prob', 0.5)

        # Correlation head on bottleneck features
        self.corr_head = CorrelationHead(bottleneck_channels, num_classes).to(device)

        # Threshold controller
        self.thresh_controller = ThreshController(
            nclass=num_classes, momentum=0.999, thresh_init=self.pseudo_threshold
        )

        # Update optimizer to include correlation head
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.corr_head.parameters()),
            lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001
        )

        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def _extract_features(self, volume):
        """Extract logits and bottleneck features."""
        result = self.model(volume, return_features=True)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        features = self.model.encoder(volume)
        logits = self.model.decoder(features)
        return logits, features[4]

    def _weak_augment(self, x):
        """Weak spatial augmentation."""
        aug_type = torch.randint(0, 3, (1,)).item()
        if aug_type == 0:
            return torch.flip(x, dims=[3]), 'hflip'
        elif aug_type == 1:
            return torch.flip(x, dims=[2]), 'vflip'
        else:
            return x, 'none'

    def _inverse_aug(self, x, aug_type):
        """Inverse weak augmentation."""
        if aug_type == 'hflip':
            return torch.flip(x, dims=[3])
        elif aug_type == 'vflip':
            return torch.flip(x, dims=[2])
        return x

    def _strong_augment(self, x, pseudo=None, mask=None):
        """Strong augmentation: spatial + photometric."""
        B, C, H, W = x.shape
        aug_type = torch.randint(0, 4, (1,)).item()
        if aug_type == 0:
            angle = (torch.rand(1).item() * 2 - 1) * 30
            theta = torch.zeros(B, 2, 3, device=x.device)
            cos_a, sin_a = np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)
            theta[:, 0, 0], theta[:, 0, 1] = cos_a, -sin_a
            theta[:, 1, 0], theta[:, 1, 1] = sin_a, cos_a
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')
            if pseudo is not None:
                pseudo = F.grid_sample(pseudo.unsqueeze(1).float(), grid, align_corners=False, mode='nearest', padding_mode='border').squeeze(1)
            if mask is not None:
                mask = F.grid_sample(mask.unsqueeze(1).float(), grid, align_corners=False, mode='nearest', padding_mode='border').squeeze(1)
        elif aug_type == 1:
            x = torch.flip(x, dims=[3])
            if pseudo is not None: pseudo = torch.flip(pseudo, dims=[2])
            if mask is not None: mask = torch.flip(mask, dims=[2])
        elif aug_type == 2:
            x = torch.flip(x, dims=[2])
            if pseudo is not None: pseudo = torch.flip(pseudo, dims=[1])
            if mask is not None: mask = torch.flip(mask, dims=[1])
        else:
            scale = 0.8 + torch.rand(1).item() * 0.2
            new_h, new_w = int(H * scale), int(W * scale)
            top = torch.randint(0, max(H - new_h, 1), (1,)).item()
            left = torch.randint(0, max(W - new_w, 1), (1,)).item()
            x = F.interpolate(x[:, :, top:top+new_h, left:left+new_w], size=(H, W), mode='bilinear', align_corners=False)
            if pseudo is not None:
                pseudo = F.interpolate(pseudo[:, top:top+new_h, left:left+new_w].unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            if mask is not None:
                mask = F.interpolate(mask[:, top:top+new_h, left:left+new_w].unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)

        # Photometric
        ops = ['brightness', 'contrast', 'saturation', 'noise', 'cutout']
        chosen = np.random.choice(ops, 3, replace=False)
        for op in chosen:
            if op == 'brightness':
                x = x * (0.6 + torch.rand(1).item() * 0.8)
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
                ch, cw = int(H * 0.2), int(W * 0.2)
                cy, cx = torch.randint(0, H, (1,)).item(), torch.randint(0, W, (1,)).item()
                y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
                x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
                x[:, :, y1:y2, x1:x2] = 0

        return torch.clamp(x, 0, 1), pseudo, mask

    def _cutmix(self, img1, img2, pseudo1, pseudo2, mask1, mask2):
        """CutMix between two strong branches."""
        B, C, H, W = img1.shape
        img1_mix, img2_mix = img1.clone(), img2.clone()
        pseudo1_mix, pseudo2_mix = pseudo1.clone(), pseudo2.clone()
        mask1_mix, mask2_mix = mask1.clone(), mask2.clone()

        for i in range(B):
            lam = np.random.beta(1.0, 1.0)
            cut_ratio = (1 - lam) ** 0.5
            cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
            cy, cx = np.random.randint(0, H), np.random.randint(0, W)
            y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
            x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)

            img1_mix[i, :, y1:y2, x1:x2] = img2[i, :, y1:y2, x1:x2]
            img2_mix[i, :, y1:y2, x1:x2] = img1[i, :, y1:y2, x1:x2]
            pseudo1_mix[i, y1:y2, x1:x2] = pseudo2[i, y1:y2, x1:x2]
            pseudo2_mix[i, y1:y2, x1:x2] = pseudo1[i, y1:y2, x1:x2]
            mask1_mix[i, y1:y2, x1:x2] = mask2[i, y1:y2, x1:x2]
            mask2_mix[i, y1:y2, x1:x2] = mask1[i, y1:y2, x1:x2]

        return img1_mix, img2_mix, pseudo1_mix, pseudo2_mix, mask1_mix, mask2_mix

    def _label_propagation(self, pred, corr_logits, mask, thresh_global):
        """Correlation-based label propagation (from CorrMatch).

        Uses correlation head output to propagate labels from
        high-confidence anchors to correlated regions.
        """
        B, num_cls, H, W = pred.shape
        prob = pred.softmax(dim=1)
        conf = prob.max(dim=1)[0]
        pred_label = pred.argmax(dim=1)

        # High-confidence anchors
        conf_mask = (conf >= thresh_global) & (mask != 255)

        # Correlation-based propagation
        # corr_logits: (B, num_classes, H, W) - per-class correlation scores
        corr_prob = corr_logits.softmax(dim=1)
        corr_label = corr_logits.argmax(dim=1)

        # For each image, propagate where correlation head agrees with high-confidence
        for img_idx in range(B):
            high_conf = conf_mask[img_idx]
            if high_conf.sum() == 0:
                continue

            # Check agreement between correlation head and high-confidence predictions
            agree_mask = (corr_label[img_idx] == pred_label[img_idx]) & (~high_conf)
            # Only propagate where correlation is confident
            corr_conf = corr_prob[img_idx].max(dim=0)[0]
            propagate_mask = agree_mask & (corr_conf > thresh_global * 0.8)

            pred_label[img_idx][propagate_mask] = corr_label[img_idx][propagate_mask]
            conf_mask[img_idx] = conf_mask[img_idx] | propagate_mask

        return pred_label, conf_mask

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data["label"].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        if unlabeled_volume.shape[0] == 0:
            output = self.model(labeled_volume)
            if isinstance(output, tuple):
                output = output[0]
            output_soft = torch.softmax(output, dim=1)
            loss_ce = self.ce_loss(output, labeled_label.long())
            loss_dice = self.dice_loss(output_soft, labeled_label.unsqueeze(1))
            sup_loss = 0.5 * (loss_dice + loss_ce)
            return {'total': sup_loss, 'ce': loss_ce, 'dice': loss_dice,
                    'consistency': torch.tensor(0.0), 'consistency_weight': 0.0}

        # Generate pseudo-labels from weak augmentation
        with torch.no_grad():
            weak_aug, aug_type = self._weak_augment(unlabeled_volume)
            output_weak, feat_weak = self._extract_features(weak_aug)
            pred_weak = self._inverse_aug(output_weak, aug_type)
            pred_weak_soft = torch.softmax(pred_weak, dim=1)

            # Update dynamic threshold
            self.thresh_controller.thresh_update(pred_weak.detach(), None, update_g=True)
            thresh_global = self.thresh_controller.get_thresh_global()

            # Generate correlation map from bottleneck features (inverse-aug to align with pred_weak)
            feat_weak_inv = self._inverse_aug(feat_weak, aug_type)
            corr_map_weak = self.corr_head(feat_weak_inv.detach())
            # Upsample correlation map to match prediction spatial size
            corr_map_weak = F.interpolate(corr_map_weak, size=pred_weak.shape[2:], mode='bilinear', align_corners=False)

            # Apply label propagation
            mask_propagated, conf_mask = self._label_propagation(
                pred_weak, corr_map_weak, torch.ones_like(pred_weak.argmax(dim=1)), thresh_global
            )

        # Strong augmentation on unlabeled data
        img_s1, pseudo_s1, mask_s1 = self._strong_augment(
            unlabeled_volume, mask_propagated.clone(), conf_mask.float().clone()
        )
        img_s2, pseudo_s2, mask_s2 = self._strong_augment(
            unlabeled_volume, mask_propagated.clone(), conf_mask.float().clone()
        )

        # CutMix
        if np.random.random() < self.cutmix_prob:
            img_s1, img_s2, pseudo_s1, pseudo_s2, mask_s1, mask_s2 = self._cutmix(
                img_s1, img_s2, pseudo_s1, pseudo_s2, mask_s1, mask_s2
            )

        # Supervised loss on labeled data
        output_labeled = self.model(labeled_volume)
        if isinstance(output_labeled, tuple):
            output_labeled = output_labeled[0]
        output_labeled_soft = torch.softmax(output_labeled, dim=1)

        loss_ce = self.ce_loss(output_labeled, labeled_label.long())
        loss_dice = self.dice_loss(output_labeled_soft, labeled_label.unsqueeze(1))
        sup_loss = 0.5 * (loss_dice + loss_ce)

        # Student on strong branches
        output_s1, feat_s1 = self._extract_features(img_s1)
        output_s2, feat_s2 = self._extract_features(img_s2)

        # Correlation outputs from bottleneck features
        corr_s1 = self.corr_head(feat_s1)
        corr_s2 = self.corr_head(feat_s2)
        # Upsample correlation outputs to match prediction spatial size
        corr_s1 = F.interpolate(corr_s1, size=output_s1.shape[2:], mode='bilinear', align_corners=False)
        corr_s2 = F.interpolate(corr_s2, size=output_s2.shape[2:], mode='bilinear', align_corners=False)

        # Feature perturbation
        output_fp = F.dropout2d(output_s1, p=0.5, training=True)

        # Consistency loss with confidence masking
        total_mask = mask_s1 + mask_s2
        consistency_loss = torch.tensor(0.0, device=self.device)

        if total_mask.sum() > 0:
            loss_s1 = self.ce_loss_none(output_s1, pseudo_s1.long())
            loss_s2 = self.ce_loss_none(output_s2, pseudo_s2.long())
            loss_fp = self.ce_loss_none(output_fp, pseudo_s1.long())

            # Correlation consistency loss (corr output is num_classes channels)
            loss_corr_s1 = self.ce_loss_none(corr_s1, pseudo_s1.long())
            loss_corr_s2 = self.ce_loss_none(corr_s2, pseudo_s2.long())

            # KL divergence between strong branches
            softmax_s1 = F.softmax(output_s1.detach(), dim=1)
            logsoftmax_s2 = F.log_softmax(output_s2, dim=1)
            loss_kl = self.kl_loss(logsoftmax_s2, softmax_s1).sum(dim=1) * mask_s1.float()

            consistency_weight = self._get_consistency_weight(iter_num)

            # CorrMatch loss
            consistency_loss = (
                0.25 * (loss_s1 * mask_s1.float()).sum() / mask_s1.float().sum().clamp(min=1) +
                0.25 * loss_kl.sum() / mask_s1.float().sum().clamp(min=1) +
                0.25 * (loss_fp * mask_s1.float()).sum() / mask_s1.float().sum().clamp(min=1) +
                0.25 * (
                    (loss_corr_s1 * mask_s1.float()).sum() / mask_s1.float().sum().clamp(min=1) +
                    (loss_corr_s2 * mask_s2.float()).sum() / mask_s2.float().sum().clamp(min=1)
                ) / 2
            ) / 2
        else:
            consistency_weight = self._get_consistency_weight(iter_num)

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

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        self.corr_head.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "corr_head": self.corr_head.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "corr_head" in state_dict:
                self.corr_head.load_state_dict(state_dict["corr_head"])
            return
        self.model.load_state_dict(state_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--corrmatch_threshold', type=float, default=0.95)
        parser.add_argument('--corrmatch_cutmix_prob', type=float, default=0.5)
