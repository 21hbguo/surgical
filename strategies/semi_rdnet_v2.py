import copy
import torch

from .base_strategy import BaseTrainingStrategy
from utils.loss_rdnet import (
    BceDiceLoss1,
    BceDiceLoss1_D,
    L2_loss,
    cont_loss,
    dpa,
    mse_consistency_loss,
    update_pseudo_labels,
)


class RDNetV2Strategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        assert int(args.use_depth) == 13
        assert self.model.params["in_chns"] == 3
        self.rdnet_thresh = args.rdnet_thresh
        self.rdnet_thresh_depth = args.rdnet_thresh_depth
        self.rdnet_thresh_dpa = args.rdnet_thresh_dpa
        self.rdnet_beta = args.rdnet_beta
        self.rdnet_unsup_rgb_weight = 0.5
        self.rdnet_unsup_depth_weight = 1.2
        self.rdnet_cross_weight = 0.5
        self.rdnet_dpa_weight = 1.0
        self.rdnet_contrast_weight = 1.0
        self.rdnet_mse_weight = 0.5
        self.rdnet_feature_l2_weight = 0.5
        self.rdnet_total_epoch = 1
        self.rdnet_criterion = BceDiceLoss1().to(self.device)
        self.rdnet_criterion_d = BceDiceLoss1_D().to(self.device)
        self.depth_model = copy.deepcopy(self.model).to(self.device)
        self.depth_optimizer = torch.optim.Adam(
            self.depth_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0001,
        )
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_decay = self.args.ema_decay
        self.ema_model.train()

    def _forward_with_feature(self, model, x):
        output = model(x)
        assert isinstance(output, (tuple, list)) and len(output) == 2
        assert output[0].shape[1] == 1
        return output[0], output[1]

    def _target_binary(self, label):
        return (label.unsqueeze(1) > 0).float()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)
        depth3 = batch_data["depth3"].to(self.device)
        depth1 = batch_data["depth1"].to(self.device)
        image_s = self._add_noise(image, strong_flag="s", unlabeled_only=True)
        labeled_bs = self.labeled_bs
        labeled_img = image[:labeled_bs]
        unlabeled_img = image[labeled_bs:]
        unlabeled_img_s = image_s[labeled_bs:]
        labeled_depth = depth3[:labeled_bs]
        unlabeled_depth = depth3[labeled_bs:]
        unlabeled_depth1 = depth1[labeled_bs:]
        labeled_label = self._target_binary(label[:labeled_bs])
        with torch.no_grad():
            ema_pred_u, _ = self._forward_with_feature(self.ema_model, unlabeled_img)
        rgb_pred, rgb_feat = self._forward_with_feature(self.model, torch.cat((unlabeled_img_s, labeled_img), dim=0))
        pred_u, pred_l = torch.split(rgb_pred, [unlabeled_img_s.shape[0], labeled_img.shape[0]], dim=0)
        e5_u = rgb_feat[:unlabeled_img_s.shape[0]]
        depth_pred, depth_feat = self._forward_with_feature(self.depth_model, torch.cat((unlabeled_depth, labeled_depth), dim=0))
        pred_u_d, pred_l_d = torch.split(depth_pred, [unlabeled_depth.shape[0], labeled_depth.shape[0]], dim=0)
        e5_u_d = depth_feat[:unlabeled_depth.shape[0]]
        if epoch > 0:
            self.rdnet_total_epoch = max(self.rdnet_total_epoch, (self.args.max_iterations + max(1, iter_num // epoch) - 1) // max(1, iter_num // epoch))
        unlabeled_img_s_cutmix, ema_pred_u_cutmix = dpa(
            unlabeled_depth1, unlabeled_img_s, ema_pred_u, beta=self.rdnet_beta, t=epoch, T=self.rdnet_total_epoch
        )
        pred_u_cutmix, _ = self._forward_with_feature(self.model, unlabeled_img_s_cutmix)
        loss_supervised_rgb = self.rdnet_criterion(pred_l, labeled_label, threshold=-1)
        loss_supervised_depth = self.rdnet_criterion(pred_l_d, labeled_label, threshold=-1)
        loss_u_rgb = self.rdnet_criterion(pred_u, ema_pred_u, threshold=self.rdnet_thresh)
        loss_u_depth = self.rdnet_criterion(pred_u_d, ema_pred_u, threshold=self.rdnet_thresh)
        if self.args.ud:
            pred_u_up = update_pseudo_labels(ema_pred_u, pred_u_d, gamma=self.rdnet_thresh_depth)
        else:
            pred_u_up = ema_pred_u
        loss_u_rgb_d = self.rdnet_criterion_d(pred_u, pred_u_up, threshold=self.rdnet_thresh_depth)
        loss_u_rgb_dpa = self.rdnet_criterion(pred_u_cutmix, ema_pred_u_cutmix, threshold=self.rdnet_thresh_dpa)
        loss_rgb = (
            loss_supervised_rgb
            + self.rdnet_unsup_rgb_weight * loss_u_rgb
            + self.rdnet_dpa_weight * loss_u_rgb_dpa
            + self.rdnet_cross_weight * loss_u_rgb_d
        ) / 3.0
        loss_depth = (loss_supervised_depth + self.rdnet_unsup_depth_weight * loss_u_depth) / 2.0
        loss_contrast = cont_loss(pred_u, pred_l, pred_u_d, pred_l_d)
        loss_mse = mse_consistency_loss(pred_u, pred_u_d)
        loss_feat_l2 = L2_loss(e5_u, e5_u_d)
        loss_con = (
            self.rdnet_contrast_weight * loss_contrast
            + self.rdnet_mse_weight * loss_mse
            + self.rdnet_feature_l2_weight * loss_feat_l2
        ) / 2.0
        total_loss = (loss_rgb + loss_depth + loss_con) / 3.0
        return {
            "total": total_loss,
            "sup_rgb": loss_supervised_rgb,
            "sup_depth": loss_supervised_depth,
            "unsup_rgb": loss_u_rgb,
            "unsup_depth": loss_u_depth,
            "cross_cons": loss_u_rgb_d,
            "dpa_cons": loss_u_rgb_dpa,
            "contrast": loss_contrast,
            "mse": loss_mse,
            "feat_l2": loss_feat_l2,
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        lr = self.optimizer.param_groups[0]["lr"]
        for param_group in self.depth_optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        self.depth_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss = loss_dict["total"]
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                self.scaler.unscale_(self.depth_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.depth_model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.step(self.depth_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.depth_model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.depth_optimizer.step()
        self._update_ema(iter_num)
        return loss_dict

    def validation_step(self, batch_data):
        image = batch_data["image"].to(self.device)
        pred, _ = self._forward_with_feature(self.model, image)
        return torch.cat((1 - pred, pred), dim=1)

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        self.depth_model.train(mode=training)
        self.ema_model.eval()

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "depth_model": self.depth_model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.depth_model.load_state_dict(state_dict["depth_model"])
        self.ema_model.load_state_dict(state_dict["ema_model"])

    @staticmethod
    def add_args(parser):
        parser.add_argument("--rdnet_thresh", type=float, default=0.85, help="RGB伪标签置信阈值")
        parser.add_argument("--rdnet_thresh_depth", type=float, default=0.85, help="Depth伪标签置信阈值")
        parser.add_argument("--rdnet_thresh_dpa", type=float, default=0.95, help="DPA增强置信阈值")
        parser.add_argument("--rdnet_beta", type=float, default=0.3, help="DPA patch保留比例")
        parser.add_argument("--ud", action="store_true", default=False, help="启用depth分支更新RGB伪标签")
