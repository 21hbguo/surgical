import torch
import torch.nn.functional as F
import math
from .base_strategy import BaseTrainingStrategy
from utils.losses import CosineSimilarityContrastiveLoss

DEFAULT_CONTRASTIVE_MARGIN = 0.5
DEFAULT_CONTRASTIVE_TEMP = 0.07
DEFAULT_CONTRASTIVE_WEIGHT = 0.1
DEFAULT_RAMPUP = 200


def gaussian_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = min(current, rampup_length)
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)


class W2SStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self.contrastive_loss = CosineSimilarityContrastiveLoss(margin=DEFAULT_CONTRASTIVE_MARGIN, temperature=DEFAULT_CONTRASTIVE_TEMP)
        self.labeled_bs = args.labeled_bs
        self.max_iterations = args.max_iterations
        self.contrastive_weight = DEFAULT_CONTRASTIVE_WEIGHT
        self.consistency_weight = args.consistency
        self.w2s_rampup = DEFAULT_RAMPUP

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data['image'].to(self.device)
        label = batch_data['label'].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        batch_size = volume.shape[0]

        outputs = self.model(volume)
        (main_seg, main_cont), (w2s1_seg, w2s1_cont), (w2s2_seg, w2s2_cont), (w2s3_seg, w2s3_cont) = outputs

        supervised_loss = 0.0

        main_labeled_seg = main_seg[:self.labeled_bs]
        main_labeled_soft = torch.softmax(main_labeled_seg, dim=1)
        loss_ce = self.ce_loss(main_labeled_seg, labeled_label.long())
        loss_dice = self.dice_loss(main_labeled_soft, labeled_label.unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        rampup_weight = gaussian_rampup(iter_num, self.w2s_rampup)

        cont_labeled_main = main_cont[:self.labeled_bs]
        B, C, H, W = cont_labeled_main.shape

        num_samples = min(1024, B * H * W)
        indices = torch.randperm(B * H * W, device=self.device)[:num_samples]
        cont_main_flat = cont_labeled_main.permute(0, 2, 3, 1).reshape(-1, C)[indices]

        cont_labeled_w2s1 = w2s1_cont[:self.labeled_bs]
        cont_w2s1_flat = cont_labeled_w2s1.permute(0, 2, 3, 1).reshape(-1, C)[indices]

        cont_labeled_w2s2 = w2s2_cont[:self.labeled_bs]
        cont_w2s2_flat = cont_labeled_w2s2.permute(0, 2, 3, 1).reshape(-1, C)[indices]

        cont_labeled_w2s3 = w2s3_cont[:self.labeled_bs]
        cont_w2s3_flat = cont_labeled_w2s3.permute(0, 2, 3, 1).reshape(-1, C)[indices]

        contrastive_loss = 0.0
        contrastive_loss += self.contrastive_loss(cont_main_flat, cont_w2s1_flat)[0]
        contrastive_loss += self.contrastive_loss(cont_main_flat, cont_w2s2_flat)[0]
        contrastive_loss += self.contrastive_loss(cont_main_flat, cont_w2s3_flat)[0]
        contrastive_loss = contrastive_loss / 3.0

        main_soft = torch.softmax(main_seg, dim=1)
        w2s1_soft = torch.softmax(w2s1_seg, dim=1)
        w2s2_soft = torch.softmax(w2s2_seg, dim=1)
        w2s3_soft = torch.softmax(w2s3_seg, dim=1)

        consistency_loss = 0.0
        consistency_loss += F.mse_loss(main_soft[self.labeled_bs:], w2s1_soft[self.labeled_bs:])
        consistency_loss += F.mse_loss(main_soft[self.labeled_bs:], w2s2_soft[self.labeled_bs:])
        consistency_loss += F.mse_loss(main_soft[self.labeled_bs:], w2s3_soft[self.labeled_bs:])
        consistency_loss = consistency_loss / 3.0

        unsupervised_weight = self.consistency_weight * rampup_weight
        contrastive_total_weight = self.contrastive_weight * rampup_weight

        total_loss = supervised_loss + unsupervised_weight * consistency_loss + contrastive_total_weight * contrastive_loss

        loss_dict = {
            'total': total_loss,
            'supervised': supervised_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': consistency_loss,
            'contrastive': contrastive_loss,
            'unsup_weight': unsupervised_weight,
            'cont_weight': contrastive_total_weight,
        }
        return loss_dict

    def validation_step(self, batch_data):
        with torch.no_grad():
            volume = batch_data['image'].to(self.device)
            depth_tensor = self._get_depth_tensor(batch_data)
            if depth_tensor is not None:
                volume = torch.cat([volume, depth_tensor], dim=1)
            output = self.model(volume)
            (main_seg, main_cont), _, _, _ = output

            return main_seg
