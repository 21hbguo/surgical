import torch
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy
from utils.losses import DepthLoss
# 旧的策略

class DepthGuidedMTStrategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--depth_loss_weight", type=float, default=1.0)
        parser.add_argument("--depth_consistency_weight", type=float, default=0.5)
        parser.add_argument("--depth_l1_weight", type=float, default=1.0)
        parser.add_argument("--depth_gradient_weight", type=float, default=0.1)
        parser.add_argument("--depth_smoothness_weight", type=float, default=0.01)
        parser.add_argument("--depth_ssim_weight", type=float, default=0.5)
        parser.add_argument("--depth_range_weight", type=float, default=0.1)

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)

        self.depth_loss_weight = args.depth_loss_weight
        self.depth_consistency_weight = args.depth_consistency_weight

        self.depth_loss_fn = DepthLoss(l1_weight=args.depth_l1_weight, gradient_weight=args.depth_gradient_weight, smoothness_weight=args.depth_smoothness_weight, ssim_weight=args.depth_ssim_weight, range_weight=args.depth_range_weight)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data['label'].to(self.device)

        stud_volume = self._add_noise(volume, strong_flag='s', unlabeled_only=True)

        student_seg_logits, student_depth = self.model(stud_volume)
        student_seg_soft = F.softmax(student_seg_logits, dim=1)

        unlabeled_volume = volume[self.labeled_bs:]

        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag='t')
            ema_seg_logits, ema_depth = self.ema_model(ema_inputs)
            ema_seg_soft = F.softmax(ema_seg_logits, dim=1)

        batch_data['teacher_pred'] = ema_seg_soft

        loss_ce = self.ce_loss(student_seg_logits[:self.labeled_bs], label[:self.labeled_bs].long())
        loss_dice = self.dice_loss(student_seg_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1))
        loss_sup_seg = 0.5 * (loss_dice + loss_ce)

        loss_dict = {'ce': loss_ce, 'dice': loss_dice, 'sup_seg': loss_sup_seg}

        gt_depth = self._get_depth_tensor(batch_data)
        if gt_depth is not None:
            depth_loss_dict = self.depth_loss_fn(student_depth[:self.labeled_bs], gt_depth[:self.labeled_bs])
            loss_dict.update(depth_loss_dict)
            loss_dict['sup_depth'] = depth_loss_dict['depth_total']
        else:
            loss_dict['sup_depth'] = torch.tensor(0.0, device=self.device)
            loss_dict['depth_l1'] = torch.tensor(0.0, device=self.device)
            loss_dict['depth_gradient'] = torch.tensor(0.0, device=self.device)
            loss_dict['depth_smoothness'] = torch.tensor(0.0, device=self.device)
            loss_dict['depth_ssim'] = torch.tensor(0.0, device=self.device)
            loss_dict['depth_range'] = torch.tensor(0.0, device=self.device)

        consistency_weight = self._get_consistency_weight(iter_num)

        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            loss_seg_cons = torch.mean((student_seg_soft[self.labeled_bs:] - ema_seg_soft) ** 2)
            loss_depth_cons = F.mse_loss(student_depth[self.labeled_bs:], ema_depth)
            loss_cons_total = loss_seg_cons + self.depth_consistency_weight * loss_depth_cons
        else:
            loss_seg_cons = torch.tensor(0.0, device=self.device)
            loss_depth_cons = torch.tensor(0.0, device=self.device)
            loss_cons_total = torch.tensor(0.0, device=self.device)

        loss_dict.update({
            'seg_cons': loss_seg_cons,
            'depth_cons': loss_depth_cons,
            'cons_total': loss_cons_total,
            'consistency_weight': consistency_weight,
        })

        total_loss = loss_sup_seg + self.depth_loss_weight * loss_dict['sup_depth'] + consistency_weight * loss_cons_total

        loss_dict['total'] = total_loss
        return loss_dict

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict
