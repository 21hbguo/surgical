# -*- coding: utf-8 -*-
"""实现参考自：https://github.com/HiLab-git/PyMIC"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from .common_block import FeaturePerturbation

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

class ConvBlock(nn.Module):
    """由两层卷积、归一化和激活组成的基础块。"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """先下采样再做卷积特征提取。"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """先上采样再融合跳连特征。"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 提取当前模型的层级特征。
class Encoder(nn.Module):
    def __init__(self, params, filter_num=16):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = [filter_num * (2 ** i) for i in range(5)]
        self.params['feature_chns'] = self.ft_chns
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params, filter_num=16):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params.get('feature_chns', [filter_num * (2 ** i) for i in range(5)])
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class Decoder_DS(nn.Module):
    def __init__(self, params, filter_num=16):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params.get('feature_chns', [filter_num * (2 ** i) for i in range(5)])
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg

class Decoder_URPC(nn.Module):
    def __init__(self, params, filter_num=16):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params.get('feature_chns', [filter_num * (2 ** i) for i in range(5)])
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

def expectation(distribution, dim=-1, keepdim=False):
    length = distribution.shape[dim]
    rng = torch.arange(length, dtype=distribution.dtype, device=distribution.device)

    # 到
    shape = [1] * distribution.dim()
    shape[dim] = length
    rng_view = rng.view(*shape)
    result = (distribution * rng_view).sum(dim=dim, keepdim=keepdim)
    return result

class UNet(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns, 'dropout': [0, 0, 0, 0, 0], 'class_num': class_num, 'bilinear': False, 'acti_func': 'relu', 'filter_num': filter_num}

        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)

    def forward(self, x, return_features=False):
        feature = self.encoder(x)
        output = self.decoder(feature)
        if return_features:
            return output, feature[4]
        return output

class UNet_DYCON(nn.Module):

    def __init__(self, in_chns, class_num, proj_dim=256, filter_num=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'proj_dim': proj_dim,
            'filter_num': filter_num,
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)

        # 到用于学习(B,dim,H/16,W/16).
        bottleneck_ch = self.encoder.ft_chns[-1]
        self.proj = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
        )

    def forward(self, x):
        feats = self.encoder(x)
        seg_logits = self.decoder(feats)
        bottleneck = feats[-1]
        proj_feat = self.proj(bottleneck)
        return seg_logits, proj_feat

class UNet_Point(nn.Module):

    def __init__(self, in_chns, class_num, num_points=5, coord_temperature: float = 0.01, filter_num=16):
        super().__init__()

        base_params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
        }

        self.num_points = num_points
        self.coord_temperature = coord_temperature
        self.encoder = Encoder(base_params, filter_num)
        self.seg_decoder = Decoder(base_params, filter_num)

        # (FCNstyle)到输入stride.
        self.point_channels = self.encoder.ft_chns
        bottleneck_channels = self.point_channels[4]
        fuse_in = sum(self.point_channels)
        self.point_fuse = nn.Conv2d(fuse_in, bottleneck_channels, kernel_size=1)
        self.point_refine = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.point_heatmap = nn.Conv2d(bottleneck_channels, num_points, kernel_size=1)

    def _spatial_expectation(self, heatmap: torch.Tensor):
        b, p, h, w = heatmap.shape
        flat = heatmap.view(b, p, -1)
        prob = F.softmax(flat / max(1e-6, self.coord_temperature), dim=-1)

        ys = torch.linspace(0, 1, steps=h, device=heatmap.device)
        xs = torch.linspace(0, 1, steps=w, device=heatmap.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).view(1, 1, -1, 2)

        expected = (prob.unsqueeze(-1) * grid).sum(dim=2)
        return expected  # [B, P, 2] normalized coords

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        seg_logits = self.seg_decoder([x0, x1, x2, x3, x4])

        # head:(x0x4),到输入
        h, w = x.shape[-2:]
        up_feats = [F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
                for feat in (x0, x1, x2, x3, x4)]
        fused = torch.cat(up_feats, dim=1)
        point_feat = F.relu(self.point_fuse(fused))
        point_feat = F.relu(self.point_refine(point_feat))
        heatmap = self.point_heatmap(point_feat)

        point_coords = self._spatial_expectation(heatmap)

        return seg_logits, point_coords, heatmap

class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'dropout': [0, 0, 0, 0, 0],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'filter_num': filter_num}
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder_URPC(params, filter_num)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg
