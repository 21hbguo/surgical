import torch
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy
from utils.losses import UnCLoss, FeCLoss, adaptive_beta

DEFAULT_FEATURE_SCALER = 2
DEFAULT_GAMMA = 2.0
DEFAULT_BETA_MIN = 0.5
DEFAULT_BETA_MAX = 5.0
DEFAULT_S_BETA = None
DEFAULT_TEMP = 0.6
DEFAULT_L_WEIGHT = 1.0
DEFAULT_U_WEIGHT = 0.5
DEFAULT_USE_FOCAL = True
DEFAULT_USE_TEACHER_LOSS = True
DEFAULT_FECL_RAMPUP = None
DEFAULT_LAMBDA_CROSS = 1.0
DEFAULT_USE_ASPP = False


class DyConStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self._enable_ema_support()
        self.labeled_bs = args.labeled_bs

        self.feature_scaler = DEFAULT_FEATURE_SCALER
        self.gamma = DEFAULT_GAMMA
        self.beta_min = DEFAULT_BETA_MIN
        self.beta_max = DEFAULT_BETA_MAX
        self.s_beta = DEFAULT_S_BETA
        self.temp = DEFAULT_TEMP
        self.l_weight = DEFAULT_L_WEIGHT
        self.u_weight = DEFAULT_U_WEIGHT
        self.use_focal = DEFAULT_USE_FOCAL
        self.use_teacher_loss = DEFAULT_USE_TEACHER_LOSS
        self.lambda_cross = DEFAULT_LAMBDA_CROSS
        self.max_epochs = args.max_iterations
        self.fecl_rampup = DEFAULT_FECL_RAMPUP
        if self.fecl_rampup is None:
            self.fecl_rampup = int(self.max_epochs * 0.15)

        self.uncl_criterion = UnCLoss()
        self.fecl_criterion = FeCLoss(device=device, temperature=self.temp, gamma=self.gamma, use_focal=self.use_focal, rampup_epochs=self.fecl_rampup, lambda_cross=self.lambda_cross)

        self.consistency_criterion = lambda x, y: F.mse_loss(x, y, reduction='none')

    def _get_adaptive_beta(self, epoch):
        if self.s_beta is not None:
            return self.s_beta
        return adaptive_beta(epoch, self.max_epochs, max_beta=self.beta_max, min_beta=self.beta_min)

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data['image'].to(self.device)
        label = batch_data['label'].to(self.device)

        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)

        unlabeled_volume = volume[self.labeled_bs:]

        stud_volume = self._add_noise(volume, strong_flag='s', unlabeled_only=True)

        stud_logits, stud_features = self.model(stud_volume)
        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag='t')
            ema_logits, ema_features = self.ema_model(ema_inputs)

        stud_probs = F.softmax(stud_logits, dim=1)
        ema_probs = F.softmax(ema_logits, dim=1)

        loss_ce = self.ce_loss(stud_logits[:self.labeled_bs], label[:self.labeled_bs].long())
        loss_dice = self.dice_loss(stud_probs[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1))

        B, C, H, W = stud_features.shape
        stud_embedding = stud_features.view(B, C, -1).transpose(1, 2)
        stud_embedding = F.normalize(stud_embedding, dim=-1)

        unlabeled_B = B - self.labeled_bs
        ema_embedding = ema_features.view(unlabeled_B, C, -1).transpose(1, 2)
        ema_embedding = F.normalize(ema_embedding, dim=-1)

        scale = self.feature_scaler * 4
        mask_con = F.interpolate(
            label.unsqueeze(1).float(),
            scale_factor=1/scale,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        mask_con = (mask_con > 0.5).float().reshape(B, -1).unsqueeze(1)

        beta = self._get_adaptive_beta(epoch)
        uncl_loss = self.uncl_criterion(stud_logits[self.labeled_bs:], ema_logits, beta)

        teacher_feat = ema_embedding if self.use_teacher_loss else None
        fecl_loss = self.fecl_criterion(
            feat=stud_embedding[self.labeled_bs:],
            mask=mask_con[self.labeled_bs:],
            teacher_feat=teacher_feat,
            gambling_uncertainty=None,
            epoch=epoch,
        )

        consistency_weight = self._get_consistency_weight(iter_num)
        consistency_loss = self.consistency_criterion(stud_probs[self.labeled_bs:], ema_probs).mean()

        supervised = self.l_weight * (loss_ce + loss_dice)
        unsupervised = self.u_weight * (uncl_loss + fecl_loss)
        total_loss = supervised + consistency_weight * consistency_loss + unsupervised

        loss_dict = {
            'total': total_loss,
            'supervised': supervised,
            'ce': loss_ce,
            'dice': loss_dice,
            'uncl': uncl_loss,
            'fecl': fecl_loss,
            'consistency': consistency_loss,
            'consistency_weight': consistency_weight,
            'beta': beta,
        }
        return loss_dict

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(
            loss_dict['total'],
            optimizer=self.optimizer,
            clip_params=self.model.parameters(),
            clip_max_norm=1.0,
        )
        self._update_ema(iter_num)
        return loss_dict

def kl_div_loss(x, y, reduction='mean'):
    return F.kl_div(torch.log(x.clamp(min=1e-10)), y, reduction=reduction)
