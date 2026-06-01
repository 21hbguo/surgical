import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .base_strategy import BaseTrainingStrategy


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning (from U2PL)."""
    def __init__(self, in_channels, mid_channels=256, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class U2PLStrategy(BaseTrainingStrategy):
    """U2PL: Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels (CVPR 2022).

    Faithful reproduction based on official code:
    - EMA teacher generates pseudo-labels
    - Entropy-based percentile thresholding separates reliable/unreliable pixels
    - Unsupervised loss: CE on reliable pixels (dropping high-entropy ones)
    - Contrastive loss: memory bank with class-wise negative samples, cosine similarity
    - Projection head maps bottleneck features to contrastive space
    - Momentum prototype for positive anchors
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()

        num_classes = getattr(args, 'num_classes', 2)
        bottleneck_channels = 256  # filter_num * 16 for default UNet

        # U2PL hyperparameters (matching original config for binary segmentation)
        self.drop_percent = getattr(args, 'u2pl_drop_percent', 80)
        self.contrast_weight = getattr(args, 'u2pl_contrast_weight', 0.1)
        self.low_entropy_threshold = getattr(args, 'u2pl_low_entropy_threshold', 20.0)
        self.num_queries = getattr(args, 'u2pl_num_queries', 256)
        self.num_negatives = getattr(args, 'u2pl_num_negatives', 50)
        self.temperature = getattr(args, 'u2pl_temperature', 0.5)
        self.current_class_threshold = getattr(args, 'u2pl_class_threshold', 0.3)
        self.current_class_negative_threshold = getattr(args, 'u2pl_class_neg_threshold', 1.0)
        self.low_rank = getattr(args, 'u2pl_low_rank', 1)
        self.high_rank = getattr(args, 'u2pl_high_rank', 2)
        self.sup_only_epoch = getattr(args, 'u2pl_sup_only_epoch', 0)
        self.max_epochs = getattr(args, 'max_iterations', 30000)

        # Projection head maps bottleneck features (256 channels) to contrastive space
        self.projector = ProjectionHead(bottleneck_channels, 256, 256).to(device)

        # Class-wise memory bank (from original U2PL)
        self.memobank = []
        self.queue_ptrlis = []
        self.queue_size = []
        for i in range(num_classes):
            self.memobank.append(deque(maxlen=30000))
            self.queue_size.append(30000)
            self.queue_ptrlis.append(0)
        self.queue_size[0] = 50000

        # Momentum prototype (from original U2PL)
        self.num_classes = num_classes
        self.prototype = torch.zeros(
            (num_classes, self.num_queries, 1, 256)
        ).to(device)

        # Update optimizer to include projector
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001
        )

    def _extract_features(self, volume):
        """Extract logits and bottleneck features from model."""
        result = self.model(volume, return_features=True)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        # Fallback: extract features via encoder directly
        features = self.model.encoder(volume)
        logits = self.model.decoder(features)
        return logits, features[4]

    def _label_onehot(self, label, num_classes):
        """Convert label to one-hot encoding (from U2PL)."""
        B, H, W = label.shape
        onehot = torch.zeros(B, num_classes, H, W, device=label.device)
        label_clamped = label.long().clamp(0, num_classes - 1)
        onehot.scatter_(1, label_clamped.unsqueeze(1), 1.0)
        return onehot

    def _compute_unsupervised_loss(self, pred, target, percent, pred_teacher):
        """Unsupervised loss with entropy-based percentile thresholding (from U2PL)."""
        batch_size, num_class, h, w = pred.shape

        with torch.no_grad():
            prob = torch.softmax(pred_teacher, dim=1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

            valid_mask = target != 255
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)

            thresh = np.percentile(
                entropy[valid_mask].detach().cpu().numpy().flatten(), percent
            )
            thresh_mask = entropy.ge(thresh).bool() * valid_mask.bool()
            target = target.clone()
            target[thresh_mask] = 255
            weight = batch_size * h * w / torch.sum(target != 255).clamp(min=1)

        loss = weight * F.cross_entropy(pred, target.long(), ignore_index=255)
        return loss

    def _compute_contrastive_loss(self, rep, label_l, label_u, prob_l, prob_u,
                                   low_mask, high_mask, rep_teacher):
        """Contrastive loss using memory bank (from U2PL)."""
        num_feat = rep.shape[1]

        low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
        high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

        rep = rep.permute(0, 2, 3, 1)
        rep_teacher = rep_teacher.permute(0, 2, 3, 1)

        prob = torch.cat((prob_l, prob_u), dim=0)

        _, prob_indices_l = torch.sort(prob_l, 1, True)
        prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)

        _, prob_indices_u = torch.sort(prob_u, 1, True)
        prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)

        seg_feat_low_entropy_list = []
        seg_num_list = []
        seg_proto_list = []
        valid_classes = []

        for i in range(self.num_classes):
            low_valid_pixel_seg = low_valid_pixel[:, i]
            high_valid_pixel_seg = high_valid_pixel[:, i]

            prob_seg = prob[:, i, :, :]
            rep_mask_low_entropy = (
                prob_seg > self.current_class_threshold
            ) * low_valid_pixel_seg.bool()
            rep_mask_high_entropy = (
                prob_seg < self.current_class_negative_threshold
            ) * high_valid_pixel_seg.bool()

            seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

            # Positive sample: class center from teacher
            if low_valid_pixel_seg.sum() > 0:
                seg_proto_list.append(
                    torch.mean(
                        rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
                    )
                )
            else:
                seg_proto_list.append(torch.zeros(1, num_feat, device=rep.device))

            # Generate negative mask for unlabeled data
            class_mask_u = torch.sum(
                prob_indices_u[:, :, :, self.low_rank:self.high_rank].eq(i), dim=3
            ).bool()

            # Generate negative mask for labeled data
            class_mask_l = torch.sum(
                prob_indices_l[:, :, :, :self.low_rank].eq(i), dim=3
            ).bool()

            class_mask = torch.cat(
                (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
            )
            negative_mask = rep_mask_high_entropy * class_mask

            keys = rep_teacher[negative_mask].detach()
            # Update memory bank
            if keys.shape[0] > 0:
                for key in keys:
                    if len(self.memobank[i]) < self.queue_size[i]:
                        self.memobank[i].append(key.cpu())
                    else:
                        self.memobank[i][self.queue_ptrlis[i] % self.queue_size[i]] = key.cpu()
                        self.queue_ptrlis[i] += 1

            if low_valid_pixel_seg.sum() > 0:
                seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
                valid_classes.append(i)

        if len(seg_num_list) <= 1:
            return torch.tensor(0.0, device=rep.device) * rep.sum()

        reco_loss = torch.tensor(0.0, device=rep.device)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[valid_classes[i]]) > 0
                and len(self.memobank[valid_classes[i]]) > 10
            ):
                # Select anchor pixels
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[valid_classes[i]]),
                    size=(min(self.num_queries, len(seg_feat_low_entropy_list[valid_classes[i]])),)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[valid_classes[i]][seg_low_entropy_idx].clone().to(rep.device)
                )

                with torch.no_grad():
                    negative_feat = torch.stack(list(self.memobank[valid_classes[i]])).to(rep.device)
                    # Sample negatives with replacement (from official U2PL)
                    num_q = min(self.num_queries, anchor_feat.shape[0])
                    high_entropy_idx = torch.randint(
                        len(negative_feat),
                        size=(num_q * self.num_negatives,)
                    )
                    negative_feat = negative_feat[high_entropy_idx]
                    negative_feat = negative_feat.reshape(num_q, self.num_negatives, num_feat)

                    positive_feat = (
                        seg_proto[valid_classes[i]]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(num_q, 1, 1)
                        .to(rep.device)
                    )

                    # Momentum prototype update
                    ema_decay = min(1 - 1 / max(1, self.queue_ptrlis[valid_classes[i]]), 0.999)
                    if not (self.prototype[valid_classes[i]] == 0).all():
                        positive_feat = (
                            (1 - ema_decay) * positive_feat
                            + ema_decay * self.prototype[valid_classes[i]][:num_q]
                        )
                    self.prototype[valid_classes[i]][:num_q] = positive_feat.clone()

                    all_feat = torch.cat(
                        (positive_feat, negative_feat), dim=1
                    )

                anchor_feat = anchor_feat[:num_q]
                seg_logits = torch.cosine_similarity(
                    anchor_feat.unsqueeze(1), all_feat, dim=2
                )
                reco_loss = reco_loss + F.cross_entropy(
                    seg_logits / self.temperature,
                    torch.zeros(num_q, dtype=torch.long, device=rep.device)
                )

        return reco_loss / valid_seg

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data["label"].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        # Supervised loss on labeled data (extract features for contrastive)
        output_labeled, feat_labeled = self._extract_features(labeled_volume)
        output_labeled_soft = torch.softmax(output_labeled, dim=1)

        loss_ce = self.ce_loss(output_labeled, labeled_label.long())
        loss_dice = self.dice_loss(output_labeled_soft, labeled_label.unsqueeze(1))
        sup_loss = 0.5 * (loss_dice + loss_ce)

        consistency_weight = self._get_consistency_weight(iter_num)
        unsup_loss = torch.tensor(0.0, device=self.device)
        contra_loss = torch.tensor(0.0, device=self.device)

        if unlabeled_volume.shape[0] > 0 and epoch >= self.sup_only_epoch:
            # Teacher generates pseudo-labels
            with torch.no_grad():
                pred_u_teacher, feat_teacher_u = self._extract_features(unlabeled_volume)
                # EMA teacher features
                ema_result = self.ema_model(unlabeled_volume, return_features=True)
                if isinstance(ema_result, tuple) and len(ema_result) == 2:
                    pred_u_teacher = ema_result[0]
                    feat_teacher_u = ema_result[1]
                else:
                    pred_u_teacher = ema_result

                pred_u_teacher_soft = torch.softmax(pred_u_teacher, dim=1)
                logits_u_aug, label_u_aug = torch.max(pred_u_teacher_soft, dim=1)

            # Student forward (features already extracted above for labeled)
            student_output, feat_student_u = self._extract_features(unlabeled_volume)

            # Unsupervised loss with entropy-based thresholding
            # alpha_t decays with epoch (from official U2PL)
            alpha_t = self.low_entropy_threshold * (1 - epoch / max(1, self.max_epochs))
            drop_percent = 100 - alpha_t

            unsup_loss = self._compute_unsupervised_loss(
                student_output,
                label_u_aug.clone(),
                drop_percent,
                pred_u_teacher.detach()
            )

            # Contrastive loss using bottleneck features
            # Upsample features to match prediction spatial size
            target_size = student_output.shape[2:]  # (H, W)
            feat_student_up = F.interpolate(feat_student_u, size=target_size, mode='bilinear', align_corners=False)
            feat_teacher_up = F.interpolate(feat_teacher_u, size=target_size, mode='bilinear', align_corners=False)

            rep_student = self.projector(feat_student_up)
            rep_teacher = self.projector(feat_teacher_up.detach())

            # Teacher predictions on labeled data
            with torch.no_grad():
                ema_labeled_result = self.ema_model(labeled_volume, return_features=True)
                if isinstance(ema_labeled_result, tuple) and len(ema_labeled_result) == 2:
                    prob_l_teacher = torch.softmax(ema_labeled_result[0], dim=1)
                else:
                    prob_l_teacher = torch.softmax(ema_labeled_result, dim=1)
            prob_u_teacher = pred_u_teacher_soft

            with torch.no_grad():
                prob = torch.softmax(pred_u_teacher.detach(), dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                valid_mask = label_u_aug != 255
                if valid_mask.sum() > 0:
                    low_thresh = np.percentile(
                        entropy[valid_mask].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * valid_mask.bool()
                    )

                    high_thresh = np.percentile(
                        entropy[valid_mask].cpu().numpy().flatten(), 100 - alpha_t
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * valid_mask.bool()
                    )
                else:
                    low_entropy_mask = torch.zeros_like(entropy)
                    high_entropy_mask = torch.zeros_like(entropy)

                low_mask_all = torch.cat(
                    (
                        (labeled_label.unsqueeze(1) != 255).float(),
                        low_entropy_mask.unsqueeze(1),
                    )
                )

                high_mask_all = torch.cat(
                    (
                        (labeled_label.unsqueeze(1) != 255).float(),
                        high_entropy_mask.unsqueeze(1),
                    )
                )

                # Downsample masks to match feature resolution
                target_h, target_w = rep_student.shape[2:]
                low_mask_all = F.interpolate(low_mask_all, size=(target_h, target_w), mode="nearest")
                high_mask_all = F.interpolate(high_mask_all, size=(target_h, target_w), mode="nearest")

                label_l_small = F.interpolate(
                    self._label_onehot(labeled_label, self.num_classes),
                    size=(target_h, target_w), mode="nearest"
                )
                label_u_small = F.interpolate(
                    self._label_onehot(label_u_aug, self.num_classes),
                    size=(target_h, target_w), mode="nearest"
                )

            # Combine labeled and unlabeled representations
            feat_labeled_up = F.interpolate(feat_labeled, size=target_size, mode='bilinear', align_corners=False)
            rep_labeled = self.projector(feat_labeled_up)
            rep_all = torch.cat((rep_labeled, rep_student), dim=0)

            # Teacher features for labeled
            with torch.no_grad():
                ema_feat_l = self.ema_model(labeled_volume, return_features=True)
                if isinstance(ema_feat_l, tuple) and len(ema_feat_l) == 2:
                    feat_ema_l = ema_feat_l[1]
                else:
                    feat_ema_l = self.model.encoder(labeled_volume)[4]
                feat_ema_l_up = F.interpolate(feat_ema_l, size=target_size, mode='bilinear', align_corners=False)
            rep_ema_l = self.projector(feat_ema_l_up.detach())
            rep_all_teacher = torch.cat((rep_ema_l, rep_teacher), dim=0)

            contra_loss = self._compute_contrastive_loss(
                rep_all,
                label_l_small.long(),
                label_u_small.long(),
                prob_l_teacher.detach(),
                prob_u_teacher.detach(),
                low_mask_all,
                high_mask_all,
                rep_all_teacher.detach()
            )

        # U2PL official: sup + unsup + contrast (no extra consistency_weight rampup)
        total_loss = sup_loss + unsup_loss + self.contrast_weight * contra_loss

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': unsup_loss + contra_loss,
            'consistency_weight': 1.0
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        if self.ema_model is not None:
            self.ema_model.eval()
        self.projector.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "ema_model" in state_dict:
                self.ema_model.load_state_dict(state_dict["ema_model"])
            if "projector" in state_dict:
                self.projector.load_state_dict(state_dict["projector"])
            return
        self.model.load_state_dict(state_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--u2pl_drop_percent', type=float, default=80)
        parser.add_argument('--u2pl_contrast_weight', type=float, default=0.1)
        parser.add_argument('--u2pl_low_entropy_threshold', type=float, default=20.0)
        parser.add_argument('--u2pl_num_queries', type=int, default=256)
        parser.add_argument('--u2pl_num_negatives', type=int, default=50)
        parser.add_argument('--u2pl_temperature', type=float, default=0.5)
        parser.add_argument('--u2pl_class_threshold', type=float, default=0.3)
        parser.add_argument('--u2pl_class_neg_threshold', type=float, default=1.0)
        parser.add_argument('--u2pl_low_rank', type=int, default=1)
        parser.add_argument('--u2pl_high_rank', type=int, default=2)
        parser.add_argument('--u2pl_sup_only_epoch', type=int, default=0)
