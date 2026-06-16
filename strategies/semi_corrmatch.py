import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class ThreshController:
    """Per-class max confidence EMA threshold controller (CorrMatch official)."""

    def __init__(self, nclass, momentum=0.999, thresh_init=0.95):
        self.nclass = nclass
        self.momentum = momentum
        self.thresh_per_class = [thresh_init] * nclass
        self.thresh_global = thresh_init
        self.count = 0

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        prob = pred.softmax(dim=1)
        conf = prob.max(dim=1)[0]
        pred_label = pred.argmax(dim=1)

        for c in range(self.nclass):
            class_mask = pred_label == c
            if ignore_mask is not None:
                class_mask = class_mask & (ignore_mask != 255)
            if class_mask.sum() > 0:
                max_conf = conf[class_mask].max().item()
                if self.count == 0:
                    self.thresh_per_class[c] = max_conf
                else:
                    self.thresh_per_class[c] = (
                        self.momentum * self.thresh_per_class[c]
                        + (1 - self.momentum) * max_conf
                    )

        if update_g:
            self.thresh_global = sum(self.thresh_per_class) / self.nclass
            self.count += 1

    def get_thresh_global(self):
        return self.thresh_global


class CorrelationHead(nn.Module):
    """Projects bottleneck features for cosine similarity correlation."""

    def __init__(self, in_channels, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class CorrMatchStrategy(BaseTrainingStrategy):
    """CorrMatch: Label Propagation via Correlation Matching (CVPR 2024).

    Faithful reproduction based on official code:
    - Single model generates pseudo-labels from weak augmentation
    - Feature correlation (cosine similarity) for label propagation
    - Per-class max confidence EMA threshold controller
    - Dual strong branches with CutMix between weak and strong
    - Feature perturbation via dropout on bottleneck features
    - CE + KL + feature perturbation consistency loss
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)

        num_classes = args.num_classes
        bottleneck_channels = 256  # filter_num * 16 for default UNet

        self.pseudo_threshold = args.corrmatch_threshold
        self.cutmix_prob = args.corrmatch_cutmix_prob

        self.corr_head = CorrelationHead(bottleneck_channels).to(device)

        self.thresh_controller = ThreshController(
            nclass=num_classes, momentum=0.999, thresh_init=self.pseudo_threshold
        )

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.corr_head.parameters()),
            lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001,
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
        aug_type = torch.randint(0, 3, (1,)).item()
        if aug_type == 0:
            return torch.flip(x, dims=[3]), 'hflip'
        elif aug_type == 1:
            return torch.flip(x, dims=[2]), 'vflip'
        return x, 'none'

    def _inverse_aug(self, x, aug_type):
        if aug_type == 'hflip':
            return torch.flip(x, dims=[3])
        elif aug_type == 'vflip':
            return torch.flip(x, dims=[2])
        return x

    def _strong_augment(self, x, pseudo=None, mask=None):
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

    def _cutmix(self, img_weak, img_strong, pseudo_weak, pseudo_strong, mask_weak, mask_strong):
        """CutMix between weak and strong augmented images."""
        B, C, H, W = img_weak.shape
        img1_mix, img2_mix = img_weak.clone(), img_strong.clone()
        pseudo1_mix, pseudo2_mix = pseudo_weak.clone(), pseudo_strong.clone()
        mask1_mix, mask2_mix = mask_weak.clone(), mask_strong.clone()

        for i in range(B):
            lam = np.random.beta(1.0, 1.0)
            cut_ratio = (1 - lam) ** 0.5
            cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
            cy, cx = np.random.randint(0, H), np.random.randint(0, W)
            y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
            x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)

            img1_mix[i, :, y1:y2, x1:x2] = img_strong[i, :, y1:y2, x1:x2]
            img2_mix[i, :, y1:y2, x1:x2] = img_weak[i, :, y1:y2, x1:x2]
            pseudo1_mix[i, y1:y2, x1:x2] = pseudo_strong[i, y1:y2, x1:x2]
            pseudo2_mix[i, y1:y2, x1:x2] = pseudo_weak[i, y1:y2, x1:x2]
            mask1_mix[i, y1:y2, x1:x2] = mask_strong[i, y1:y2, x1:x2]
            mask2_mix[i, y1:y2, x1:x2] = mask_weak[i, y1:y2, x1:x2]

        return img1_mix, img2_mix, pseudo1_mix, pseudo2_mix, mask1_mix, mask2_mix

    def _label_propagation(self, pred, feat_proj, thresh_global):
        """Correlation-based label propagation using cosine similarity."""
        B, num_cls, H, W = pred.shape
        C, h, w = feat_proj.shape[1], feat_proj.shape[2], feat_proj.shape[3]

        # Downsample pred to match feat_proj spatial size
        pred_down = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
        prob = pred_down.softmax(dim=1)
        conf = prob.max(dim=1)[0]
        pred_label = pred_down.argmax(dim=1)

        # Flatten and normalize features for cosine similarity
        feat_flat = feat_proj.view(B, C, -1)  # (B, C, hw)
        feat_norm = F.normalize(feat_flat, dim=1)

        # Correlation matrix: (B, hw, hw)
        corr = torch.bmm(feat_norm.transpose(1, 2), feat_norm)

        conf_mask = conf >= thresh_global

        for img_idx in range(B):
            high_conf = conf_mask[img_idx].view(-1)
            low_conf = ~high_conf

            if high_conf.sum() == 0 or low_conf.sum() == 0:
                continue

            # Correlation between low-conf and high-conf pixels
            corr_subset = corr[img_idx][low_conf][:, high_conf]

            # Find best matching high-conf pixel for each low-conf pixel
            best_match = corr_subset.argmax(dim=1)
            best_corr = corr_subset.max(dim=1)[0]

            # Only propagate where correlation is high enough
            propagate = best_corr > 0.8

            # Get labels from best-matching anchors
            anchor_labels = pred_label[img_idx].view(-1)[high_conf][best_match[propagate]]

            # Apply propagated labels
            low_conf_indices = torch.where(low_conf)[0][propagate]
            pred_label[img_idx].view(-1)[low_conf_indices] = anchor_labels
            conf_mask[img_idx].view(-1)[low_conf_indices] = True

        # Upsample results back to original resolution
        pred_label_up = F.interpolate(
            pred_label.unsqueeze(1).float(), size=(H, W), mode='nearest'
        ).squeeze(1).long()
        conf_mask_up = F.interpolate(
            conf_mask.unsqueeze(1).float(), size=(H, W), mode='nearest'
        ).squeeze(1).bool()

        return pred_label_up, conf_mask_up

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
            feat_weak_inv = self._inverse_aug(feat_weak, aug_type)

            # Update dynamic threshold (per-class max confidence)
            self.thresh_controller.thresh_update(pred_weak.detach(), None, update_g=True)
            thresh_global = self.thresh_controller.get_thresh_global()

            # Correlation-based label propagation (returns upsampled results)
            corr_proj = self.corr_head(feat_weak_inv.detach())
            pseudo_label, conf_mask = self._label_propagation(
                pred_weak, corr_proj, thresh_global
            )

        # Strong augmentation on unlabeled data
        img_s1, pseudo_s1, mask_s1 = self._strong_augment(
            unlabeled_volume, pseudo_label.clone(), conf_mask.float().clone()
        )
        img_s2, pseudo_s2, mask_s2 = self._strong_augment(
            unlabeled_volume, pseudo_label.clone(), conf_mask.float().clone()
        )

        # CutMix between weak and strong
        if np.random.random() < self.cutmix_prob:
            weak_s = self._weak_augment(unlabeled_volume)[0]
            img_s1, img_s2, pseudo_s1, pseudo_s2, mask_s1, mask_s2 = self._cutmix(
                weak_s, img_s1, pseudo_label, pseudo_s1, conf_mask.float(), mask_s1
            )

        # Supervised loss on labeled data
        output_labeled = self.model(labeled_volume)
        if isinstance(output_labeled, tuple):
            output_labeled = output_labeled[0]
        output_labeled_soft = torch.softmax(output_labeled, dim=1)

        loss_ce = self.ce_loss(output_labeled, labeled_label.long())
        loss_dice = self.dice_loss(output_labeled_soft, labeled_label.unsqueeze(1))
        sup_loss = 0.5 * (loss_dice + loss_ce)

        # Student forward via encoder/decoder for feature perturbation
        features_s1 = self.model.encoder(img_s1)
        output_s1 = self.model.decoder(features_s1)
        feat_s1 = features_s1[4]

        features_s2 = self.model.encoder(img_s2)
        output_s2 = self.model.decoder(features_s2)

        # Feature perturbation: dropout on bottleneck, then re-decode
        features_s1_list = list(features_s1)
        features_s1_list[4] = F.dropout2d(feat_s1, p=0.5, training=True)
        output_fp = self.model.decoder(features_s1_list)

        # Consistency loss
        total_mask = mask_s1 + mask_s2
        consistency_loss = torch.tensor(0.0, device=self.device)

        if total_mask.sum() > 0:
            loss_s1 = self.ce_loss_none(output_s1, pseudo_s1.long())
            loss_s2 = self.ce_loss_none(output_s2, pseudo_s2.long())
            loss_fp = self.ce_loss_none(output_fp, pseudo_s1.long())

            # KL divergence between strong branches
            softmax_s1 = F.softmax(output_s1.detach(), dim=1)
            logsoftmax_s2 = F.log_softmax(output_s2, dim=1)
            loss_kl = self.kl_loss(logsoftmax_s2, softmax_s1).sum(dim=1) * mask_s1.float()

            consistency_weight = self._get_consistency_weight(iter_num)

            # CorrMatch loss: CE + KL + feature perturbation (equal weights)
            consistency_loss = (
                (loss_s1 * mask_s1.float()).sum() / mask_s1.float().sum().clamp(min=1) +
                (loss_s2 * mask_s2.float()).sum() / mask_s2.float().sum().clamp(min=1) +
                loss_kl.sum() / mask_s1.float().sum().clamp(min=1) +
                (loss_fp * mask_s1.float()).sum() / mask_s1.float().sum().clamp(min=1)
            ) / 4
        else:
            consistency_weight = self._get_consistency_weight(iter_num)

        total_loss = sup_loss + consistency_weight * consistency_loss

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': consistency_loss,
            'consistency_weight': consistency_weight,
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
