import torch
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseTrainingStrategy
from utils.common import sigmoid_rampup
from utils.losses import softmax_mse_loss

DEFAULT_T_SAMPLES = 8


class UncertaintyMTStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.labeled_bs = args.labeled_bs
        self.t_samples = DEFAULT_T_SAMPLES
        self.max_iter = args.max_iterations

    def _compute_uncertainty(self, volume, T):
        B, _, w, h = volume.shape
        preds = torch.zeros([B * T, self.dice_loss.n_classes, w, h]).to(self.device)

        for i in range(T):
            noise = torch.clamp(torch.randn_like(volume) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                ema_out = self.ema_model(volume + noise)
                if isinstance(ema_out, tuple):
                    ema_out = ema_out[0]
                preds[i * B : (i + 1) * B] = ema_out

        preds = F.softmax(preds, dim=1).reshape(T, B, self.dice_loss.n_classes, w, h).mean(dim=0)
        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
        return uncertainty

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data['image'].to(self.device)
        label = batch_data['label'].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        unlabeled_volume = volume[self.labeled_bs:]

        stud_volume = self._add_noise(volume, strong_flag='s', unlabeled_only=True)

        output = self.model(stud_volume)
        if isinstance(output, tuple):
            output = output[0]
        output_soft = torch.softmax(output, dim=1)

        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag='t')
            ema_output = self.ema_model(ema_inputs)
            if isinstance(ema_output, tuple):
                ema_output = ema_output[0]

        uncertainty = self._compute_uncertainty(unlabeled_volume, self.t_samples)

        loss_ce = self.ce_loss(output[:self.labeled_bs], label[:self.labeled_bs].long())
        loss_dice = self.dice_loss(output_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        consistency_weight = self._get_consistency_weight(iter_num)
        consistency_dist = softmax_mse_loss(output[self.labeled_bs:], ema_output)
        threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, self.max_iter)) * np.log(2)
        mask = (uncertainty < threshold).float()
        consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
        total_loss = supervised_loss + consistency_weight * consistency_loss

        loss_dict = {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': consistency_loss,
            'consistency_weight': consistency_weight
        }
        return loss_dict

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict
