import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class CWBASStrategy(BaseTrainingStrategy):
    """CW-BASS: Confidence-Weighted Boundary-Aware Learning (IJCNN 2025).

    ST++ style two-stage training:
    - Stage 1 (warmup): Train on labeled data only
    - Stage 2: Model generates pseudo-labels, train with confidence-weighted boundary-aware loss
    - Confidence-weighted cross-entropy loss
    - Boundary detection using Sobel filters (excluding ignore pixels)
    - Dynamic thresholding based on average confidence
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)

        num_classes = getattr(args, 'num_classes', 2)

        # CW-BASS hyperparameters
        self.gamma = getattr(args, 'cwbass_gamma', 1.0)
        self.decay_factor = getattr(args, 'cwbass_decay_factor', 0.9)
        self.base_threshold = getattr(args, 'cwbass_base_threshold', 0.6)
        self.beta = getattr(args, 'cwbass_beta', 0.5)
        self.min_threshold = getattr(args, 'cwbass_min_threshold', 0.3)
        self.max_threshold = getattr(args, 'cwbass_max_threshold', 0.8)
        self.use_confidence_decay = getattr(args, 'cwbass_use_decay', False)
        self.boundary_weight = getattr(args, 'cwbass_boundary_weight', 0.5)
        self.warmup_iters = getattr(args, 'cwbass_warmup_iters', 1000)

        # Sobel filters for boundary detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(device)

        self.num_classes = num_classes

    def _compute_pixel_confidence(self, prediction):
        probs = F.softmax(prediction, dim=1)
        return torch.max(probs, dim=1).values

    def _dynamic_thresholding(self, confidence):
        avg_conf = confidence.mean()
        threshold = self.base_threshold / (1 + torch.exp(-self.beta * (avg_conf - 0.5)))
        return torch.clamp(threshold, min=self.min_threshold, max=self.max_threshold)

    def _detect_boundaries(self, labels):
        """Detect boundaries using Sobel filters, excluding ignore pixels."""
        valid_mask = (labels != 255).float()
        labels_clamped = labels.long().clamp(0, self.num_classes - 1)
        one_hot_labels = F.one_hot(labels_clamped, num_classes=self.num_classes)
        one_hot_labels = one_hot_labels.permute(0, 3, 1, 2).float()

        edges_x = F.conv2d(one_hot_labels, self.sobel_x.expand(self.num_classes, -1, -1, -1),
                           padding=1, groups=self.num_classes)
        edges_y = F.conv2d(one_hot_labels, self.sobel_y.expand(self.num_classes, -1, -1, -1),
                           padding=1, groups=self.num_classes)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        boundary_mask = (edges.sum(dim=1) > 0).float() * valid_mask

        return boundary_mask

    def _weighted_cross_entropy_loss(self, pred, pseudo_labels, confidence):
        ce_loss = F.cross_entropy(pred, pseudo_labels.long(), reduction='none', ignore_index=255)
        valid_mask = (pseudo_labels != 255).float()
        weighted_loss = (confidence ** self.gamma * ce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return weighted_loss

    def _boundary_loss(self, pred, pseudo_labels, confidence, boundary_mask):
        """Confidence-weighted CE + boundary region CE."""
        base_loss = self._weighted_cross_entropy_loss(pred, pseudo_labels, confidence)

        valid_mask = (pseudo_labels != 255).float()
        boundary_valid = boundary_mask * valid_mask
        if boundary_valid.sum() > 0:
            boundary_ce_loss = F.cross_entropy(pred, pseudo_labels.long(), reduction='none', ignore_index=255)
            boundary_ce_loss = (boundary_ce_loss * boundary_valid).sum() / boundary_valid.sum().clamp(min=1)
            return base_loss + self.boundary_weight * boundary_ce_loss
        return base_loss

    def _decay_confidence(self, confidence):
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

        # ST++ style: only use unlabeled data after warmup
        if unlabeled_volume.shape[0] > 0 and iter_num >= self.warmup_iters:
            # Generate pseudo-labels using model's own predictions (ST++ style)
            with torch.no_grad():
                self.model.eval()
                pseudo_output = self.model(unlabeled_volume)
                if isinstance(pseudo_output, tuple):
                    pseudo_output = pseudo_output[0]
                self.model.train()

                pixel_confidence = self._compute_pixel_confidence(pseudo_output)
                pseudo_labels = torch.argmax(pseudo_output, dim=1)

                boundary_mask = self._detect_boundaries(pseudo_labels)

                threshold = self._dynamic_thresholding(pixel_confidence)
                mask = pixel_confidence > threshold
                pseudo_labels = torch.where(mask, pseudo_labels, torch.full_like(pseudo_labels, 255))

                pixel_confidence = self._decay_confidence(pixel_confidence)

            # Student forward
            student_output = self.model(unlabeled_volume)
            if isinstance(student_output, tuple):
                student_output = student_output[0]

            cw_loss = self._boundary_loss(
                student_output, pseudo_labels, pixel_confidence, boundary_mask
            )

        total_loss = sup_loss + consistency_weight * cw_loss

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': cw_loss,
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

    def get_state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
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
        parser.add_argument('--cwbass_warmup_iters', type=int, default=1000,
                            help='Warmup iterations (labeled data only)')
