import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class SegMatchStrategy(BaseTrainingStrategy):
    """SegMatch: Semi-supervised surgical instrument segmentation (Scientific Reports 2025).

    Based on FixMatch with adversarial strong augmentation (I-FGSM).
    Uses weak augmentation for pseudo-label generation and strong augmentation
    (including adversarial) for consistency learning.
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)
        self.pseudo_threshold = getattr(args, 'pseudo_threshold', 0.95)
        self.adversarial_eps = getattr(args, 'adversarial_eps', 0.01)
        self.adversarial_steps = getattr(args, 'adversarial_steps', 1)

    def _generate_adversarial(self, unlabeled_volume, teacher_soft):
        """Generate adversarial augmentation using I-FGSM on unlabeled data.

        Uses standard I-FGSM: forward through model, backward for input gradient,
        then zero model gradients to avoid polluting the main training step.
        """
        adv_volume = unlabeled_volume.clone().detach().requires_grad_(True)
        pseudo_label = torch.argmax(teacher_soft, dim=1)

        for _ in range(self.adversarial_steps):
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                adv_output = self.model(adv_volume)
                if isinstance(adv_output, tuple):
                    adv_output = adv_output[0]
                adv_loss = self.ce_loss(adv_output, pseudo_label.long())

            adv_loss.backward()

            with torch.no_grad():
                adv_volume = adv_volume + self.adversarial_eps * adv_volume.grad.sign()
                delta = adv_volume - unlabeled_volume
                delta = torch.clamp(delta, -2 * self.adversarial_eps, 2 * self.adversarial_eps)
                adv_volume = (unlabeled_volume + delta).detach().requires_grad_(True)

            # Zero model gradients to avoid accumulation from adversarial generation
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

        # Supervised loss
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
            # Teacher with weak augmentation generates pseudo-labels
            with torch.no_grad():
                weak_aug_unlabeled = self._add_noise(unlabeled_volume, strong_flag='t')
                teacher_output = self.ema_model(weak_aug_unlabeled)
                if isinstance(teacher_output, tuple):
                    teacher_output = teacher_output[0]
                teacher_soft = torch.softmax(teacher_output, dim=1)

            # Generate adversarial strong augmentation
            adv_volume = self._generate_adversarial(unlabeled_volume, teacher_soft)

            # Student with adversarial augmentation
            student_output = self.model(adv_volume)
            if isinstance(student_output, tuple):
                student_output = student_output[0]
            student_soft = torch.softmax(student_output, dim=1)

            # Confidence mask for pseudo-labels
            max_prob, pseudo_label = torch.max(teacher_soft, dim=1)
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
                            help='Confidence threshold for pseudo-labels in SegMatch')
        parser.add_argument('--adversarial_eps', type=float, default=0.01,
                            help='Epsilon for adversarial augmentation (I-FGSM)')
        parser.add_argument('--adversarial_steps', type=int, default=1,
                            help='Number of I-FGSM steps')
