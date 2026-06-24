"""全监督 DepthGAN：在全部样本都有真实标签时，用 label-depth 与 pred-depth 对抗约束分割结构。
这个策略不存在未标注伪标签不稳定的问题，因此 GAN 可以直接作用在全 batch。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy


class DepthGANDiscriminator(nn.Module):
    def __init__(self, in_chns):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chns, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class FullySupervisedDepthGANStrategy(BaseTrainingStrategy):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--gan_loss_weight", type=float, default=0.1)
        parser.add_argument("--gan_lr", type=float, default=1e-5)

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self.gan_loss_weight = float(args.gan_loss_weight)
        depth_chns = 1 if int(args.use_depth) == 13 else int(args.use_depth)
        self.discriminator = DepthGANDiscriminator(args.num_classes * depth_chns).to(device)
        self.gan_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=float(args.gan_lr), betas=(0.9, 0.99), weight_decay=0.0001)
        self.gan_loss = nn.BCEWithLogitsLoss()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)
        depth = self._get_depth_tensor(batch_data)
        output = self.model(image)
        if isinstance(output, tuple):
            output = output[0]
        output_soft = torch.softmax(output, dim=1)
        loss_ce = self.ce_loss(output, label.long())
        loss_dice = self.dice_loss(output_soft, label.unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)
        if depth.shape[1] == 1:
            pred_depth = output_soft * depth
        else:
            pred_depth = (output_soft.unsqueeze(2) * depth.unsqueeze(1)).flatten(1, 2)
        gan_logits = self.discriminator(pred_depth)
        gan_target = torch.ones_like(gan_logits)
        gan_adv = self.gan_loss(gan_logits, gan_target)
        total_loss = supervised_loss + self.gan_loss_weight * gan_adv
        return {
            "total": total_loss,
            "ce": loss_ce,
            "dice": loss_dice,
            "gan_adv": gan_adv,
            "gan_weight": torch.tensor(self.gan_loss_weight, device=total_loss.device),
        }

    def training_step(self, batch_data, iter_num=0, epoch=0):
        image = batch_data["image"].to(self.device)
        label = batch_data["label"].to(self.device)
        depth = self._get_depth_tensor(batch_data)
        self.gan_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = self.model(image)
            if isinstance(output, tuple):
                output = output[0]
            output_soft = torch.softmax(output, dim=1)
            label_onehot = F.one_hot(label.long(), num_classes=self.args.num_classes).permute(0, 3, 1, 2).to(image.dtype)
            if depth.shape[1] == 1:
                real_depth = label_onehot * depth
                fake_depth = output_soft * depth
            else:
                real_depth = (label_onehot.unsqueeze(2) * depth.unsqueeze(1)).flatten(1, 2)
                fake_depth = (output_soft.unsqueeze(2) * depth.unsqueeze(1)).flatten(1, 2)
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
        loss_dict["gan_disc"] = gan_disc.detach()
        loss_dict["gan_real"] = gan_real.detach()
        loss_dict["gan_fake"] = gan_fake.detach()
        return loss_dict

    def validation_step(self, batch_data):
        image = batch_data["image"].to(self.device)
        return self.model(image)

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        self.discriminator.train(mode=training)

    def get_state_dict(self):
        return {"model": self.model.state_dict(), "discriminator": self.discriminator.state_dict()}

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            self.discriminator.load_state_dict(state_dict["discriminator"])
            return
        self.model.load_state_dict(state_dict)
