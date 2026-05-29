import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class UniMatchStrategy(BaseTrainingStrategy):
    """UniMatch: Revisiting Pseudo-Labeling for Semi-Supervised Semantic Segmentation (CVPR 2023).

    Unified weak-to-strong consistency framework with:
    - Weak augmentation branch for pseudo-label generation
    - Strong augmentation branch for consistency learning
    - Confidence-based pixel selection
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)
        self.pseudo_threshold = getattr(args, 'pseudo_threshold', 0.95)

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
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        consistency_weight = self._get_consistency_weight(iter_num)
        consistency_loss = torch.tensor(0.0, device=self.device)

        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            # Weak augmentation branch (teacher) generates pseudo-labels
            with torch.no_grad():
                weak_output = self.ema_model(unlabeled_volume)
                if isinstance(weak_output, tuple):
                    weak_output = weak_output[0]
                weak_soft = torch.softmax(weak_output, dim=1)
                max_prob, pseudo_label = torch.max(weak_soft, dim=1)

            # Strong augmentation branch (student)
            strong_aug_unlabeled = self._add_noise(unlabeled_volume, strong_flag='s')
            strong_output = self.model(strong_aug_unlabeled)
            if isinstance(strong_output, tuple):
                strong_output = strong_output[0]
            strong_soft = torch.softmax(strong_output, dim=1)

            # Confidence-based mask
            mask = (max_prob >= self.pseudo_threshold).float()

            if mask.sum() > 0:
                # Cross-entropy loss with pseudo-labels
                ce_loss = F.cross_entropy(strong_output, pseudo_label.long(), reduction='none')
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
                            help='Confidence threshold for pseudo-labels in UniMatch')
