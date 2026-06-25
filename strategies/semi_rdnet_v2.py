import copy
import random

import torch
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy
from utils.losses import (
    feature_l2_loss,
    mse_consistency_loss,
    rdnet_contrastive_loss,
    update_pseudo_labels,
)


class RDNetV2Strategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        assert int(args.use_depth) == 13
        assert int(args.num_classes) == 2
        assert self.model.params["in_chns"] == 3
        self.num_classes = int(args.num_classes)
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
        return output[0], output[1]

    def _foreground_prob(self, logits):
        assert logits.shape[1] == 2
        return F.softmax(logits, dim=1)[:, 1:2]

    def _target_binary(self, label):
        return (label.unsqueeze(1) > 0).float()

    def _masked_bce_dice_loss(self, pred, target, threshold, clamp_target=False):
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            pred = pred.float()
            target = target.float()
            if clamp_target:
                target = torch.where(target <= 0.0, torch.tensor(1e-3, device=target.device), target)
                target = torch.where(target >= 1.0, torch.tensor(1.0 - 1e-3, device=target.device), target)
                pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
            mask = ((target > threshold) | (target < 1 - threshold)).float()
            bce_loss = F.binary_cross_entropy(pred, target, reduction="none")
            masked_bce_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-8)
            pred_flat = pred.view(pred.size(0), -1)
            target_flat = target.view(target.size(0), -1)
            dice_score = (2 * (pred_flat * target_flat).sum(1) + 1) / (pred_flat.sum(1) + target_flat.sum(1) + 1)
            dice_loss = 1 - dice_score.sum() / pred.size(0)
            dice_loss = dice_loss * mask.sum() / (mask.sum() + 1e-8)
            return masked_bce_loss + dice_loss

    def _dpa_augment(self, depth_map, images_s, pred_u, iter_num):
        N, _, H, W = depth_map.size()
        # h_patch = random.choice([40, 10, 20])
        # w_patch = random.choice([40, 10, 20])
        h_patch = random.choice([28, 7, 14])
        w_patch = random.choice([28, 7, 14])
        depth_patches = depth_map.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
        patch_variance = depth_patches.var(dim=(-2, -1))
        hardness_scores = patch_variance.flatten(1)
        num_patches = hardness_scores.shape[1]
        if iter_num < self.args.max_iterations:
            k = int(self.rdnet_beta * (iter_num / self.args.max_iterations) * num_patches)
        else:
            k = int(self.rdnet_beta * num_patches)
        _, indices = torch.topk(hardness_scores, k, dim=1, largest=True)
        augmented_imgs = []
        augmented_labels = []
        grid_w = W // w_patch
        for i in range(N):
            mask = torch.zeros(num_patches, dtype=torch.bool, device=images_s.device)
            mask[indices[i]] = True
            available = torch.nonzero(~mask, as_tuple=False).flatten().tolist()
            chosen_patch = random.choice(available)
            ph = chosen_patch // grid_w
            pw = chosen_patch % grid_w
            y_start, y_end = ph * h_patch, (ph + 1) * h_patch
            x_start, x_end = pw * w_patch, (pw + 1) * w_patch
            next_idx = (i + 1) % N
            augmented_img = images_s[i].clone()
            augmented_label = pred_u[i].clone()
            augmented_img[:, y_start:y_end, x_start:x_end] = images_s[next_idx, :, y_start:y_end, x_start:x_end]
            augmented_label[:, y_start:y_end, x_start:x_end] = pred_u[next_idx, :, y_start:y_end, x_start:x_end]
            augmented_imgs.append(augmented_img)
            augmented_labels.append(augmented_label)
        return torch.stack(augmented_imgs), torch.stack(augmented_labels)

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
            ema_logits, _ = self._forward_with_feature(self.ema_model, unlabeled_img)
            ema_pred_u = self._foreground_prob(ema_logits)
        rgb_logits, rgb_feat = self._forward_with_feature(self.model, torch.cat((unlabeled_img_s, labeled_img), dim=0))
        rgb_pred = self._foreground_prob(rgb_logits)
        pred_u, pred_l = torch.split(rgb_pred, [unlabeled_img_s.shape[0], labeled_img.shape[0]], dim=0)
        e5_u = rgb_feat[:unlabeled_img_s.shape[0]]
        depth_logits, depth_feat = self._forward_with_feature(self.depth_model, torch.cat((unlabeled_depth, labeled_depth), dim=0))
        depth_pred = self._foreground_prob(depth_logits)
        pred_u_d, pred_l_d = torch.split(depth_pred, [unlabeled_depth.shape[0], labeled_depth.shape[0]], dim=0)
        e5_u_d = depth_feat[:unlabeled_depth.shape[0]]
        unlabeled_img_s_cutmix, ema_pred_u_cutmix = self._dpa_augment(unlabeled_depth1, unlabeled_img_s, ema_pred_u, iter_num)
        pred_u_cutmix_logits, _ = self._forward_with_feature(self.model, unlabeled_img_s_cutmix)
        pred_u_cutmix = self._foreground_prob(pred_u_cutmix_logits)
        loss_supervised_rgb = self._masked_bce_dice_loss(pred_l, labeled_label, threshold=-1)
        loss_supervised_depth = self._masked_bce_dice_loss(pred_l_d, labeled_label, threshold=-1)
        loss_u_rgb = self._masked_bce_dice_loss(pred_u, ema_pred_u, threshold=self.rdnet_thresh)
        loss_u_depth = self._masked_bce_dice_loss(pred_u_d, ema_pred_u, threshold=self.rdnet_thresh)
        pred_u_up = update_pseudo_labels(ema_pred_u, pred_u_d, gamma=self.rdnet_thresh_depth)
        loss_u_rgb_d = self._masked_bce_dice_loss(pred_u, pred_u_up, threshold=self.rdnet_thresh_depth, clamp_target=True)
        loss_u_rgb_dpa = self._masked_bce_dice_loss(pred_u_cutmix, ema_pred_u_cutmix, threshold=self.rdnet_thresh_dpa)
        loss_rgb = (
            loss_supervised_rgb
            + self.rdnet_unsup_rgb_weight * loss_u_rgb
            + self.rdnet_dpa_weight * loss_u_rgb_dpa
            + self.rdnet_cross_weight * loss_u_rgb_d
        ) / 3.0
        loss_depth = (loss_supervised_depth + self.rdnet_unsup_depth_weight * loss_u_depth) / 2.0
        loss_contrast = rdnet_contrastive_loss(pred_l, pred_u, pred_l_d, pred_u_d)
        loss_mse = mse_consistency_loss(pred_u, pred_u_d)
        loss_feat_l2 = feature_l2_loss(e5_u, e5_u_d)
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
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
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
        logits, _ = self._forward_with_feature(self.model, image)
        return logits

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
