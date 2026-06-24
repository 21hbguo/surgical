"""MT-DepthGuider-GAN-v3：
v1 的问题是 GAN 从训练初期就约束 full batch，未标注预测还不稳定时会和 Mean Teacher 一致性目标冲突。
v3 始终只用 labeled 做 GAN，unlabeled 只走 Mean Teacher 一致性分支，并对 GAN 权重做 ramp-up。
"""

import torch
import torch.nn.functional as F

from utils.common import sigmoid_rampup

from .base_strategy import BaseTrainingStrategy
from .fully_supervised_depthGAN import DepthGANDiscriminator


class MTDepthGuiderGANV3Strategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--gan_loss_weight", type=float, default=0.1)
        parser.add_argument("--gan_lr", type=float, default=1e-5)

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self.use_depth = int(args.use_depth)
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)
        self.gan_loss_weight = float(args.gan_loss_weight)
        self.discriminator = DepthGANDiscriminator(args.num_classes).to(device)
        self.gan_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=float(args.gan_lr), betas=(0.9, 0.99), weight_decay=0.0001)
        self.gan_loss = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        depth1 = batch_data["depth1"].to(self.device)
        volume = torch.cat([image, depth1], dim=1)
        label = batch_data["label"].to(self.device)
        stud_volume = self._add_noise(volume, strong_flag="s", unlabeled_only=True)
        output = self.model(stud_volume)
        if isinstance(output, tuple):
            output = output[0]
        output_soft = torch.softmax(output, dim=1)
        unlabeled_volume = volume[self.labeled_bs :]
        with torch.no_grad():
            ema_inputs = self._add_noise(unlabeled_volume, strong_flag="t")
            ema_output = self.ema_model(ema_inputs)
            if isinstance(ema_output, tuple):
                ema_output = ema_output[0]
            ema_output_soft = torch.softmax(ema_output, dim=1)
        batch_data["teacher_pred"] = ema_output_soft
        loss_ce = self.ce_loss(output[: self.labeled_bs], label[: self.labeled_bs].long())
        loss_dice = self.dice_loss(output_soft[: self.labeled_bs], label[: self.labeled_bs].unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)
        consistency_weight = self._get_consistency_weight(iter_num)
        if iter_num >= self.consistency_start_iters and unlabeled_volume.shape[0] > 0:
            consistency_loss = torch.mean((output_soft[self.labeled_bs :] - ema_output_soft) ** 2)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
        fake_depth = output_soft[: self.labeled_bs] * depth1[: self.labeled_bs]
        gan_logits = self.discriminator(fake_depth)
        gan_adv = self.gan_loss(gan_logits, torch.ones_like(gan_logits))
        gan_ramp = sigmoid_rampup(iter_num // self.args.consistency_rampup_div, self.args.consistency_rampup)
        gan_weight = self.gan_loss_weight * gan_ramp
        total_loss = supervised_loss + consistency_weight * consistency_loss + gan_weight * gan_adv
        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "consistency": consistency_loss,
            "consistency_weight": consistency_weight,
            "gan_adv": gan_adv,
            "gan_weight": torch.tensor(gan_weight, device=total_loss.device),
            "gan_fake_samples": torch.tensor(fake_depth.shape[0], device=total_loss.device),
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        image = batch_data["image"].to(self.device)
        depth1 = batch_data["depth1"].to(self.device)
        label = batch_data["label"].to(self.device)
        volume = torch.cat([image, depth1], dim=1)
        self.gan_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = self.model(volume)
            if isinstance(output, tuple):
                output = output[0]
            output_soft = torch.softmax(output, dim=1)
            label_onehot = F.one_hot(label[: self.labeled_bs].long(), num_classes=self.args.num_classes).permute(0, 3, 1, 2).to(image.dtype)
            real_depth = label_onehot * depth1[: self.labeled_bs]
            fake_depth = output_soft[: self.labeled_bs] * depth1[: self.labeled_bs]
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            real_logits = self.discriminator(real_depth)
            fake_logits = self.discriminator(fake_depth)
            gan_real = self.gan_loss(real_logits, torch.ones_like(real_logits))
            gan_fake = self.gan_loss(fake_logits, torch.zeros_like(fake_logits))
            gan_disc = 0.5 * (gan_real + gan_fake)
        self._backward_and_step(gan_disc, optimizer=self.gan_optimizer, clip_params=self.discriminator.parameters())
        self.optimizer.zero_grad(set_to_none=True)
        for param in self.discriminator.parameters():
            param.requires_grad_(False)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict["total"], optimizer=self.optimizer)
        for param in self.discriminator.parameters():
            param.requires_grad_(True)
        self._update_ema(iter_num)
        loss_dict["gan_disc"] = gan_disc.detach()
        loss_dict["gan_real"] = gan_real.detach()
        loss_dict["gan_fake"] = gan_fake.detach()
        return loss_dict

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        self.discriminator.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            self.ema_model.load_state_dict(state_dict["ema_model"])
            self.discriminator.load_state_dict(state_dict["discriminator"])
            return
        self.model.load_state_dict(state_dict)
