import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class CWBASStrategy(BaseTrainingStrategy):
    """CW-BASS: Confidence-Weighted Boundary-Aware Learning for Semi-Supervised Semantic Segmentation (IJCNN 2025).

    Faithful reproduction based on official code:
    - EMA teacher generates pseudo-labels
    - Confidence-weighted cross-entropy loss (high-confidence pixels weighted more)
    - Boundary detection using Sobel filters on pseudo-labels
    - Dynamic thresholding based on average confidence
    - Boundary-aware loss: base weighted CE + extra boundary region loss
    - Confidence decay over training
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()

        num_classes = getattr(args, 'num_classes', 2)

        # CW-BASS hyperparameters (matching original code)
        self.gamma = getattr(args, 'cwbass_gamma', 1.0)
        self.decay_factor = getattr(args, 'cwbass_decay_factor', 0.9)
        self.base_threshold = getattr(args, 'cwbass_base_threshold', 0.6)
        self.beta = getattr(args, 'cwbass_beta', 0.5)
        self.min_threshold = getattr(args, 'cwbass_min_threshold', 0.3)
        self.max_threshold = getattr(args, 'cwbass_max_threshold', 0.8)
        self.use_confidence_decay = getattr(args, 'cwbass_use_decay', False)
        self.boundary_weight = getattr(args, 'cwbass_boundary_weight', 0.5)

        # Sobel filters for boundary detection (from original CW-BASS)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(device)

        self.num_classes = num_classes

    def _compute_pixel_confidence(self, prediction):
        """Compute pixel-wise confidence (from CW-BASS)."""
        probs = F.softmax(prediction, dim=1)
        return torch.max(probs, dim=1).values

    def _dynamic_thresholding(self, confidence):
        """Dynamic threshold based on average confidence (from CW-BASS)."""
        avg_conf = confidence.mean()
        threshold = self.base_threshold / (1 + torch.exp(-self.beta * (avg_conf - 0.5)))
        return torch.clamp(threshold, min=self.min_threshold, max=self.max_threshold)

    def _detect_boundaries(self, labels):
        """Detect boundaries using Sobel filters (from CW-BASS)."""
        num_classes = max(labels.max().item() + 1, self.num_classes)
        one_hot_labels = F.one_hot(labels.long().clamp(0, num_classes - 1), num_classes=num_classes)
        one_hot_labels = one_hot_labels.permute(0, 3, 1, 2).float()

        # Apply Sobel filter per class
        edges_x = F.conv2d(one_hot_labels, self.sobel_x.expand(num_classes, -1, -1, -1),
                           padding=1, groups=num_classes)
        edges_y = F.conv2d(one_hot_labels, self.sobel_y.expand(num_classes, -1, -1, -1),
                           padding=1, groups=num_classes)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        boundary_mask = edges.sum(dim=1) > 0

        return boundary_mask.float()

    def _weighted_cross_entropy_loss(self, pred, pseudo_labels, confidence):
        """Confidence-weighted cross-entropy loss (from CW-BASS)."""
        ce_loss = F.cross_entropy(pred, pseudo_labels.long(), reduction='none', ignore_index=255)
        valid_mask = (pseudo_labels != 255).float()
        weighted_loss = (confidence ** self.gamma * ce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return weighted_loss

    def _boundary_loss(self, pred, pseudo_labels, confidence, boundary_mask):
        """Boundary-aware loss (from CW-BASS).

        Combines:
        1. Base confidence-weighted CE loss
        2. Additional CE loss on boundary regions (only valid pixels)
        """
        base_loss = self._weighted_cross_entropy_loss(pred, pseudo_labels, confidence)

        # Boundary region loss (only on valid pixels at boundaries)
        valid_mask = (pseudo_labels != 255).float()
        boundary_valid = boundary_mask * valid_mask
        if boundary_valid.sum() > 0:
            boundary_ce_loss = F.cross_entropy(pred, pseudo_labels.long(), reduction='none', ignore_index=255)
            boundary_ce_loss = (boundary_ce_loss * boundary_valid).sum() / boundary_valid.sum().clamp(min=1)
            return base_loss + self.boundary_weight * boundary_ce_loss
        return base_loss

    def _decay_confidence(self, confidence):
        """Decay confidence values over time (from CW-BASS)."""
        if self.use_confidence_decay:
            return confidence * self.decay_factor
        return confidence

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data["label"].to(self.device)

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
        cw_loss = torch.tensor(0.0, device=self.device)

        if unlabeled_volume.shape[0] > 0:
            # Teacher generates pseudo-labels and confidence (from CW-BASS)
            with torch.no_grad():
                teacher_output = self.ema_model(unlabeled_volume)
                if isinstance(teacher_output, tuple):
                    teacher_output = teacher_output[0]

                pixel_confidence = self._compute_pixel_confidence(teacher_output)
                pseudo_labels = torch.argmax(teacher_output, dim=1)

                # Detect boundaries
                boundary_mask = self._detect_boundaries(pseudo_labels)

                # Dynamic thresholding
                threshold = self._dynamic_thresholding(pixel_confidence)
                mask = pixel_confidence > threshold
                # Set low-confidence pixels to ignore_index=255 (not background class 0)
                pseudo_labels = torch.where(mask, pseudo_labels, torch.full_like(pseudo_labels, 255))

                # Optionally decay confidence
                pixel_confidence = self._decay_confidence(pixel_confidence)

            # Student forward
            student_output = self.model(unlabeled_volume)
            if isinstance(student_output, tuple):
                student_output = student_output[0]

            # Confidence-weighted boundary-aware loss (from CW-BASS)
            cw_loss = self._boundary_loss(
                student_output, pseudo_labels, pixel_confidence, boundary_mask
            )

        total_loss = sup_loss + consistency_weight * cw_loss

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': cw_loss,
            'consistency_weight': consistency_weight
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

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "ema_model" in state_dict:
                self.ema_model.load_state_dict(state_dict["ema_model"])
            return
        self.model.load_state_dict(state_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--cwbass_gamma', type=float, default=1.0,
                            help='Exponent for confidence weighting')
        parser.add_argument('--cwbass_decay_factor', type=float, default=0.9,
                            help='Confidence decay factor')
        parser.add_argument('--cwbass_base_threshold', type=float, default=0.6,
                            help='Base threshold for dynamic thresholding')
        parser.add_argument('--cwbass_beta', type=float, default=0.5,
                            help='Beta for dynamic thresholding sigmoid')
        parser.add_argument('--cwbass_min_threshold', type=float, default=0.3,
                            help='Minimum dynamic threshold')
        parser.add_argument('--cwbass_max_threshold', type=float, default=0.8,
                            help='Maximum dynamic threshold')
        parser.add_argument('--cwbass_use_decay', action='store_true',
                            help='Enable confidence decay')
        parser.add_argument('--cwbass_boundary_weight', type=float, default=0.5,
                            help='Weight for boundary loss')
