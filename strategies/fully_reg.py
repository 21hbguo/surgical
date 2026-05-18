import torch
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy

class FullyRegSupervisedStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data['image'].to(self.device)
        label = batch_data['label'].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        output = self.model(volume)
        if isinstance(output, tuple):
            output = output[0]

        loss_ce = self.ce_loss(output, label.long())
        loss_dice = self.dice_loss(F.softmax(output, dim=1), label.unsqueeze(1))
        loss_coord = self.coord_loss(output, label)

        loss = 0.5 * (loss_dice + loss_ce) + 2 * loss_coord
        # loss = loss_dice
        return {'total': loss, 'ce': loss_ce, 'dice': loss_dice, 'coord': loss_coord}
        # return {'total': loss}
