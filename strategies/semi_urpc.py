import torch
import torch.nn as nn
from .base_strategy import BaseTrainingStrategy


class URPCStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.labeled_bs = args.labeled_bs

    def _get_consistency_loss(self, out_soft, preds):
        variance = torch.sum(self.kl_loss(torch.log(out_soft[self.labeled_bs:]), preds[self.labeled_bs:]), dim=1, keepdim=True)
        exp_variance = torch.exp(-variance)
        dist = (preds[self.labeled_bs:] - out_soft[self.labeled_bs:]) ** 2
        return torch.mean(dist * exp_variance) / (torch.mean(exp_variance) + 1e-8) + torch.mean(variance)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data['image'].to(self.device)
        label = batch_data['label'].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        output, out_aux1, out_aux2, out_aux3 = self.model(volume)
        output_soft = torch.softmax(output, dim=1)
        out_aux1_soft = torch.softmax(out_aux1, dim=1)
        out_aux2_soft = torch.softmax(out_aux2, dim=1)
        out_aux3_soft = torch.softmax(out_aux3, dim=1)

        loss_ce = (self.ce_loss(output[:self.labeled_bs], label[:self.labeled_bs].long()) +
                   self.ce_loss(out_aux1[:self.labeled_bs], label[:self.labeled_bs].long()) +
                   self.ce_loss(out_aux2[:self.labeled_bs], label[:self.labeled_bs].long()) +
                   self.ce_loss(out_aux3[:self.labeled_bs], label[:self.labeled_bs].long())) / 4

        loss_dice = (self.dice_loss(output_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1)) +
                     self.dice_loss(out_aux1_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1)) +
                     self.dice_loss(out_aux2_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1)) +
                     self.dice_loss(out_aux3_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1))) / 4

        supervised_loss = 0.5 * (loss_dice + loss_ce)

        preds = (output_soft + out_aux1_soft + out_aux2_soft + out_aux3_soft) / 4
        consistency_loss = (self._get_consistency_loss(output_soft, preds) +
                           self._get_consistency_loss(out_aux1_soft, preds) +
                           self._get_consistency_loss(out_aux2_soft, preds) +
                           self._get_consistency_loss(out_aux3_soft, preds)) / 4

        consistency_weight = self._get_consistency_weight(iter_num)
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
        loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        return loss_dict
