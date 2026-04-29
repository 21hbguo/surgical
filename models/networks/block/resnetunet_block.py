import os

import torch
import torch.nn as nn
import torchvision.models as models

from .common_block import FeaturePerturbation

def _resolve_group_count(num_channels, max_groups=32):
    upper = min(max_groups, num_channels)
    for groups in range(upper, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1

def build_group_norm(num_channels, max_groups=32):
    return nn.GroupNorm(_resolve_group_count(num_channels, max_groups=max_groups), num_channels)

def replace_batchnorm2d_with_groupnorm(module, max_groups=32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            gn = build_group_norm(child.num_features, max_groups=max_groups)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            replace_batchnorm2d_with_groupnorm(child, max_groups=max_groups)
    return module

def _load_resnet_pretrained(variant, pretrain_root, load_pretrained=True):
    if variant == "resnet34":
        default_weights_path = os.path.join(pretrain_root, "resnet34-333f7ec4.pth")
        model_builder = models.resnet34
    else:
        default_weights_path = os.path.join(pretrain_root, "resnet18-5c106cde.pth")
        model_builder = models.resnet18

    if not load_pretrained:
        return model_builder(weights=None)

    if isinstance(pretrain_root, str) and pretrain_root.endswith(".pth"):
        weights_path = pretrain_root
    else:
        weights_path = default_weights_path

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[ResNetEncoder] Expected local pretrained weights not found: {weights_path}. "
            "Please place the file under pretrain_root or disable pretrained loading."
        )

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "model_state" in state and isinstance(state["model_state"], dict):
            state = state["model_state"]
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    resnet = model_builder(weights=None)
    load_result = resnet.load_state_dict(state, strict=False)
    print(f"[ResNetEncoder] Loaded pretrained weights from {weights_path}, missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
    return replace_batchnorm2d_with_groupnorm(resnet)

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        variant="resnet34",
        in_chns=3,
        pretrain_root="../pre_train_ckp/",
        load_pretrained=True,
    ):
        super().__init__()
        resnet = _load_resnet_pretrained(variant, pretrain_root, load_pretrained=load_pretrained)

        if in_chns != 3:
            self.encoder1_conv = nn.Conv2d(in_chns, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.data
                if in_chns == 6:
                    # Keep RGB pretrained filters intact for both RGB and depth triplets.
                    self.encoder1_conv.weight.data = torch.cat([pretrained_weight, pretrained_weight], dim=1)
                else:
                    new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(1, in_chns, 1, 1)
                    self.encoder1_conv.weight.data = new_weight * (3.0 / in_chns)
        else:
            self.encoder1_conv = resnet.conv1

        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = build_group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)

class DecoderBlock_W2S(nn.Module):
    def __init__(self, in_channels, out_channels, kap=None, use_perturb=False):
        super().__init__()
        self.decoder = DecoderBlock(in_channels=in_channels, out_channels=out_channels)
        self.perturb = FeaturePerturbation(kap=kap) if use_perturb and kap is not None else None

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.decoder(x)
        if self.perturb is not None:
            x = self.perturb(x)
        return x

class ResNetDecoder_W2S(nn.Module):
    def __init__(self, class_num, kap=None):
        super().__init__()
        self.decoder5 = DecoderBlock_W2S(in_channels=512, out_channels=512, kap=kap, use_perturb=True)
        self.decoder4 = DecoderBlock_W2S(in_channels=512 + 256, out_channels=256, kap=kap, use_perturb=True)
        self.decoder3 = DecoderBlock_W2S(in_channels=256 + 128, out_channels=128, kap=kap, use_perturb=True)
        self.decoder2 = DecoderBlock_W2S(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock_W2S(in_channels=64 + 64, out_channels=64)
        self.out_conv = nn.Conv2d(64, class_num, kernel_size=3, padding=1)
        self.cont_conv_a = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.cont_conv_b = nn.Conv2d(64, 64, kernel_size=1, padding=0)

    def forward(self, features):
        e1, e2, e3, e4, e5 = features
        d5 = self.decoder5(e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        seg_output = self.out_conv(d1)
        cont = self.cont_conv_a(d1)
        cont_output = self.cont_conv_b(cont)
        return seg_output, cont_output
