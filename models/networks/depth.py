import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from .block.resnetunet_block import ConvBlock, DecoderBlock, DecoderBlock_W2S, ResNetDecoder_W2S, ResNetEncoder
from .resnet import DepthGuider
# 提取当前模型的层级特征。

_DEPTH_HEAD_PREFIXES = (
    "outconv.",
    "out_conv.",
    "out_conv_dp",
    "seg_head.",
    "depth_head.",
    "projector.",
    "cont_conv_",
)


def _unwrap_checkpoint_state(state):
    if not isinstance(state, dict):
        return state
    while isinstance(state, dict):
        next_state = None
        for key in ("model", "state_dict", "model_state"):
            value = state.get(key)
            if isinstance(value, dict):
                next_state = value
                break
        if next_state is None:
            break
        state = next_state
    return state


def _clean_state_keys(state):
    if not isinstance(state, dict):
        return {}
    cleaned = {}
    for k, v in state.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        key = key.replace("model.", "", 1) if key.startswith("model.") else key
        cleaned[key] = v
    return cleaned


def _should_skip_head_key(key):
    return key.startswith(_DEPTH_HEAD_PREFIXES)


def _load_checkpoint_state(path):
    if not (isinstance(path, str) and os.path.exists(path)):
        return {}
    raw_state = torch.load(path, map_location="cpu")
    return _clean_state_keys(_unwrap_checkpoint_state(raw_state))


def _has_decoder_key(state):
    return any(k.startswith("decoder") or ".decoder" in k for k in state.keys())


def _maybe_load_depth_full_pretrain(model, pretrain_root, enabled=True):
    if (not enabled) or (not isinstance(pretrain_root, str)) or (not pretrain_root.endswith(".pth")):
        return

    state = _load_checkpoint_state(pretrain_root)
    source_path = pretrain_root

    # 兼容之前把主文件标准化为纯 encoder 的情况：回退到原始备份拿 decoder。
    if state and (not _has_decoder_key(state)):
        fallback_path = pretrain_root + ".bak_raw"
        fallback_state = _load_checkpoint_state(fallback_path)
        if fallback_state and _has_decoder_key(fallback_state):
            state = fallback_state
            source_path = fallback_path

    if not state:
        return

    model_state = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if _should_skip_head_key(key):
            continue
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value

    if not filtered:
        return
    load_result = model.load_state_dict(filtered, strict=False)
    print(
        f"[DepthUNet] Loaded encoder+decoder pretrained from {source_path}, "
        f"matched={len(filtered)}, missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
    )

class DepthUNet_Base(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, class_num, 1),
        )

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out = self.outconv(d1)
        return out


class DepthUNet_RDNet(DepthUNet_Base):
    def __init__(self, in_chns, class_num=1, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained)
        self.params['class_num'] = 1
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out = torch.sigmoid(self.outconv(d1))
        return out, e5


class DepthUNet_DepthGuiderV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.depth_guiders = nn.ModuleList([DepthGuider(c, depth_channels=1) for c in [64, 64, 128, 256, 512]])

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, class_num, 1),
        )
        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        e1, e2, e3, e4, e5 = self.encoder(rgb)
        e1, e2, e3, e4, e5 = [g(f, depth) for g, f in zip(self.depth_guiders, [e1, e2, e3, e4, e5])]
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        return self.outconv(d1)


class ProjectionHeadContrastV1(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.layers(x)

class DepthUNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.out_conv = nn.Conv2d(64, class_num, kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(512, class_num, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(256, class_num, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(128, class_num, kernel_size=3, padding=1)

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        shape = x.shape[2:]

        d5 = self.decoder5(e5)
        dp4_out = self.out_conv_dp4(d5)
        dp4_out = F.interpolate(dp4_out, shape, mode='bilinear', align_corners=True)

        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        dp3_out = self.out_conv_dp3(d4)
        dp3_out = F.interpolate(dp3_out, shape, mode='bilinear', align_corners=True)

        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        dp2_out = self.out_conv_dp2(d3)
        dp2_out = F.interpolate(dp2_out, shape, mode='bilinear', align_corners=True)

        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        dp0_out = self.out_conv(d1)

        return dp0_out, dp2_out, dp3_out, dp4_out


class DepthUNet_ContrastV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 feature_dim=256, dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'feature_dim': feature_dim,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.feature_dim = feature_dim
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.out_conv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, class_num, 1),
        )
        self.projector = ProjectionHeadContrastV1(in_channels=512, feature_dim=feature_dim)

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        contrast_feat = self.projector(e5)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        seg_output = self.out_conv(d1)
        return seg_output, contrast_feat

class DepthUNet_proto(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 feature_dim=256, dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'feature_dim': feature_dim,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.feature_dim = feature_dim
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.out_conv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, class_num, 1),
        )
        self.projector = ProjectionHeadContrastV1(in_channels=512, feature_dim=feature_dim)

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        contrast_feat = self.projector(e5)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        seg_output = self.out_conv(d1)
        return seg_output, contrast_feat


DepthUNet_PROTO = DepthUNet_proto


class DepthUNet_DepthGuiderProtoV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 feature_dim=256, dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'feature_dim': feature_dim,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.feature_dim = feature_dim
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.depth_guiders = nn.ModuleList([DepthGuider(c, depth_channels=1) for c in [64, 64, 128, 256, 512]])

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.out_conv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, class_num, 1),
        )
        self.projector = ProjectionHeadContrastV1(in_channels=512, feature_dim=feature_dim)
        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        e1, e2, e3, e4, e5 = self.encoder(rgb)
        e1, e2, e3, e4, e5 = [g(f, depth) for g, f in zip(self.depth_guiders, [e1, e2, e3, e4, e5])]
        contrast_feat = self.projector(e5)
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        seg_output = self.out_conv(d1)
        return seg_output, contrast_feat

class DepthUNet_DyCON(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 scale_factor=2, use_aspp=False, dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'scale_factor': scale_factor,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.scale_factor = scale_factor

        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.out_conv = nn.Conv2d(64, class_num, kernel_size=3, padding=1)
        self.cont_conv_a = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.cont_conv_b = nn.Conv2d(64, 64, kernel_size=1, padding=0)

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        seg_output = self.out_conv(d1)
        cont = self.cont_conv_a(d1)
        cont_output = self.cont_conv_b(cont)

        return seg_output, cont_output

class DepthUNet_W2S(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }

        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        kaps = [0.067, 0.134, 0.2]
        self.decoder = ResNetDecoder_W2S(class_num=class_num, kap=None)
        self.decoder_1 = ResNetDecoder_W2S(class_num=class_num, kap=kaps[0])
        self.decoder_2 = ResNetDecoder_W2S(class_num=class_num, kap=kaps[1])
        self.decoder_3 = ResNetDecoder_W2S(class_num=class_num, kap=kaps[2])

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        output_1 = self.decoder_1(features)
        output_2 = self.decoder_2(features)
        output_3 = self.decoder_3(features)
        return output, output_1, output_2, output_3

class DepthUNet_Depth(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }

        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.seg_head = nn.Conv2d(64, class_num, kernel_size=3, padding=1)
        self.depth_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        seg_output = self.seg_head(d1)
        depth_output = self.depth_head(d1)

        return seg_output, depth_output


class DepthUNet_DepthPretrain(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.1, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }

        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.depth_head = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        _maybe_load_depth_full_pretrain(self, pretrain_root, enabled=load_encoder_pretrained)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        depth_output = self.depth_head(d1)
        return depth_output
