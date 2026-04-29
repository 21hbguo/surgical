import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy

class DFormerv2FullyStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        output = self.model(volume)
        if isinstance(output, tuple):
            output = output[0]

        loss_ce = self.ce_loss(output, label.long())
        loss_dice = self.dice_loss(F.softmax(output, dim=1), label.unsqueeze(1))
        loss = 0.5 * (loss_dice + loss_ce)
        return {"total": loss, "ce": loss_ce, "dice": loss_dice}
