import torch
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy

class MeanTeacherStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data['label'].to(self.device)

        stud_volume = self._add_noise(volume, strong_flag='s', unlabeled_only=True)

        output = self.model(stud_volume)
        if isinstance(output, tuple):
            output = output[0]
        output_soft = torch.softmax(output, dim=1)

        unlabeled_volume = volume[self.labeled_bs:]

        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag='t')
            ema_output = self.ema_model(ema_inputs)
            if isinstance(ema_output, tuple):
                ema_output = ema_output[0]
            ema_output_soft = torch.softmax(ema_output, dim=1)

        batch_data['teacher_pred'] = ema_output_soft

        loss_ce = self.ce_loss(output[:self.labeled_bs], label[:self.labeled_bs].long())
        loss_dice = self.dice_loss(output_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            consistency_loss = torch.mean((output_soft[self.labeled_bs:] - ema_output_soft) ** 2)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
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
