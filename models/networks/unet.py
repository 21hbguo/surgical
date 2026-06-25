
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block.common_block import DepthGuider, FeaturePerturbation, RiskGuidedPerturbation
from .block.unet_block import ConvBlock, DownBlock, Encoder, Decoder, Decoder_URPC, UpBlock, build_group_norm, replace_batchnorm2d_with_groupnorm

class DepthGuiderV3(nn.Module):
    def __init__(self, in_channels, depth_channels=1):
        super().__init__()
        mid = max(16, in_channels // 2)
        self.global_encoder = nn.Sequential(nn.Conv2d(depth_channels, mid, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.local_encoder = nn.Sequential(nn.Conv2d(depth_channels, mid, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.global_gamma = nn.Linear(mid, in_channels)
        self.global_beta = nn.Linear(mid, in_channels)
        self.local_gamma = nn.Conv2d(mid, in_channels, kernel_size=1)
        self.local_beta = nn.Conv2d(mid, in_channels, kernel_size=1)
        self.local_gate = nn.Sequential(nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, in_channels, kernel_size=1))
        nn.init.constant_(self.global_gamma.weight, 0)
        nn.init.constant_(self.global_gamma.bias, 1)
        nn.init.constant_(self.global_beta.weight, 0)
        nn.init.constant_(self.global_beta.bias, 0)
        nn.init.constant_(self.local_gamma.weight, 0)
        nn.init.constant_(self.local_gamma.bias, 0)
        nn.init.constant_(self.local_beta.weight, 0)
        nn.init.constant_(self.local_beta.bias, 0)
        nn.init.constant_(self.local_gate[3].weight, 0)
        nn.init.constant_(self.local_gate[3].bias, 0)
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, rgb_feat, depth):
        b, c, h, w = rgb_feat.shape
        if depth.shape[2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        global_feat = self.global_encoder(depth).view(b, -1)
        local_feat = self.local_encoder(depth)
        gamma_global = self.global_gamma(global_feat).view(b, c, 1, 1)
        beta_global = self.global_beta(global_feat).view(b, c, 1, 1)
        gamma_local = self.local_gamma(local_feat)
        beta_local = self.local_beta(local_feat)
        gate = torch.sigmoid(self.local_gate(local_feat))
        gamma = gamma_global * (1 + gamma_local)
        beta = beta_global + beta_local
        delta = gate * (gamma * rgb_feat + beta)
        return rgb_feat + delta

class DepthGuiderV4(nn.Module):
    def __init__(self, in_channels, depth_channels=1, pool_size=8):
        super().__init__()
        mid = max(16, in_channels // 2)
        self.pool_size = pool_size
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(depth_channels * 3, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
        )
        self.depth_mixer = nn.Sequential(
            nn.Conv2d(mid * 2, mid, kernel_size=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
        )
        self.rgb_proj = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.scale_router = nn.Sequential(
            nn.Conv2d(mid * 3, mid, kernel_size=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3, kernel_size=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 4, mid, kernel_size=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(mid * 2, mid, kernel_size=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1),
        )
        self.delta_proj = nn.Conv2d(mid, in_channels, kernel_size=1)
        nn.init.constant_(self.gate[3].weight, 0)
        nn.init.constant_(self.gate[3].bias, 0)
        nn.init.constant_(self.delta_proj.weight, 0)
        nn.init.constant_(self.delta_proj.bias, 0)

    def _resize_depth(self, depth, h, w):
        if depth.shape[2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth

    def _compute_geometry(self, depth):
        grad_x = F.pad(depth[:, :, :, 1:] - depth[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(depth[:, :, 1:, :] - depth[:, :, :-1, :], (0, 0, 0, 1))
        grad_m = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return torch.cat([grad_x, grad_y, grad_m], dim=1)

    def _encode_depth(self, depth):
        geom_feat = self.geometry_encoder(self._compute_geometry(depth))
        depth_feat = self.depth_encoder(depth)
        depth_feat = self.depth_mixer(torch.cat([depth_feat, geom_feat], dim=1))
        return depth_feat, geom_feat

    def _build_scale_context(self, depth_feat):
        h, w = depth_feat.shape[2:]
        pooled_h = min(h, self.pool_size)
        pooled_w = min(w, self.pool_size)
        local_ctx = depth_feat
        pooled_ctx = F.adaptive_avg_pool2d(depth_feat, (pooled_h, pooled_w))
        if pooled_ctx.shape[2:] != (h, w):
            pooled_ctx = F.interpolate(pooled_ctx, size=(h, w), mode="bilinear", align_corners=False)
        global_ctx = F.adaptive_avg_pool2d(depth_feat, 1).expand(-1, -1, h, w)
        return local_ctx, pooled_ctx, global_ctx

    def _compute_scale_weight(self, rgb_ctx, depth_feat, geom_feat):
        return torch.softmax(self.scale_router(torch.cat([rgb_ctx, depth_feat, geom_feat], dim=1)), dim=1)

    def compute_delta(self, rgb_feat, depth):
        _, _, h, w = rgb_feat.shape
        depth = self._resize_depth(depth, h, w)
        rgb_ctx = self.rgb_proj(rgb_feat)
        depth_feat, geom_feat = self._encode_depth(depth)
        local_ctx, pooled_ctx, global_ctx = self._build_scale_context(depth_feat)
        scale_weight = self._compute_scale_weight(rgb_ctx, depth_feat, geom_feat)
        depth_ctx = scale_weight[:, 0:1] * local_ctx + scale_weight[:, 1:2] * pooled_ctx + scale_weight[:, 2:3] * global_ctx
        interact_ctx = rgb_ctx * torch.sigmoid(depth_ctx)
        diff_ctx = torch.abs(rgb_ctx - depth_ctx)
        fused = self.fusion(torch.cat([rgb_ctx, depth_ctx, interact_ctx, diff_ctx], dim=1))
        gate = torch.sigmoid(self.gate(torch.cat([fused, depth_ctx], dim=1)))
        return gate * self.delta_proj(fused)

    def compute_gamma_beta(self, rgb_feat, depth):
        delta = self.compute_delta(rgb_feat, depth)
        gamma = torch.zeros_like(delta)
        beta = delta
        return gamma, beta

    def forward(self, rgb_feat, depth):
        return rgb_feat + self.compute_delta(rgb_feat, depth)

class Decoder_NP(nn.Module):
    def __init__(self, params, filter_num=16):
        super(Decoder_NP, self).__init__()
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
        self.cont_conv_a = nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=1, padding=0)
        self.cont_conv_b = nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=1, padding=0)

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
        seg_output = self.out_conv(x)
        cont = self.cont_conv_a(x)
        cont_output = self.cont_conv_b(cont)
        return seg_output, cont_output

class Decoder_W2S(nn.Module):
    def __init__(self, params, kap, filter_num=16):
        super(Decoder_W2S, self).__init__()
        self.params = params
        self.kap = kap
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params.get('feature_chns', [filter_num * (2 ** i) for i in range(5)])
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.perturbe = FeaturePerturbation(kap=kap)
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.cont_conv_a = nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=1, padding=0)
        self.cont_conv_b = nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=1, padding=0)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.perturbe(x)
        x = self.up2(x, x2)
        x = self.perturbe(x)
        x = self.up3(x, x1)
        x = self.perturbe(x)
        x = self.up4(x, x0)
        seg_output = self.out_conv(x)
        cont = self.cont_conv_a(x)
        cont_output = self.cont_conv_b(cont)
        return seg_output, cont_output

class Decoder_PROTO(nn.Module):
    def __init__(self, params, filter_num=16, cont_dim=16):
        super(Decoder_PROTO, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params.get('feature_chns', [filter_num * (2 ** i) for i in range(5)])
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.cont_dim = cont_dim
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

        self.cont_conv_a = nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=1, padding=0)
        self.cont_conv_b = nn.Conv2d(self.ft_chns[0], self.cont_dim, kernel_size=1, padding=0)

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

        seg_output = self.out_conv(x)
        # cont = self.cont_conv_a(x)
        # cont_output = self.cont_conv_b(cont)
        return seg_output

class DepthDecoder(nn.Module):
    def __init__(self, params, filter_num=16):
        super(DepthDecoder, self).__init__()
        self.params = params
        self.ft_chns = params.get('feature_chns', [filter_num * (2 ** i) for i in range(5)])

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

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
        return x

class UNet_Base(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super(UNet_Base, self).__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)

    def forward(self, x, return_features=False):
        feature = self.encoder(x)
        output = self.decoder(feature)
        if return_features:
            return output, feature[4]
        return output


class UNet_RDNet(UNet_Base):
    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output, feature[4]


class UNet_DepthGuiderV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)
        self.depth_guiders = nn.ModuleList(
            [DepthGuider(c, depth_channels=1) for c in [filter_num * (2 ** i) for i in range(5)]]
        )

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        feature = self.encoder(rgb)
        feature = [g(f, depth) for g, f in zip(self.depth_guiders, feature)]
        return self.decoder(feature)

class DepthGuidedEncoder(nn.Module):
    def __init__(self, params, filter_num=16):
        super().__init__()
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
        self.depth_guiders = nn.ModuleList([DepthGuider(c, depth_channels=1) for c in self.ft_chns])

    def forward(self, x, depth):
        x0 = self.depth_guiders[0](self.in_conv(x), depth)
        x1 = self.depth_guiders[1](self.down1(x0), depth)
        x2 = self.depth_guiders[2](self.down2(x1), depth)
        x3 = self.depth_guiders[3](self.down3(x2), depth)
        x4 = self.depth_guiders[4](self.down4(x3), depth)
        return [x0, x1, x2, x3, x4]

class UNet_DepthGuiderV1_2(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = DepthGuidedEncoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        feature = self.encoder(rgb, depth)
        return self.decoder(feature)

class DepthGuidedEncoderV4(nn.Module):
    def __init__(self, params, filter_num=16):
        super().__init__()
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
        self.depth_guiders = nn.ModuleList([DepthGuiderV4(c, depth_channels=1, pool_size=s) for c, s in zip(self.ft_chns, [16, 16, 8, 4, 2])])

    def forward(self, x, depth):
        x0 = self.depth_guiders[0](self.in_conv(x), depth)
        x1 = self.depth_guiders[1](self.down1(x0), depth)
        x2 = self.depth_guiders[2](self.down2(x1), depth)
        x3 = self.depth_guiders[3](self.down3(x2), depth)
        x4 = self.depth_guiders[4](self.down4(x3), depth)
        return [x0, x1, x2, x3, x4]

class UNet_DepthGuiderV4(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = DepthGuidedEncoderV4(params, filter_num)
        self.decoder = Decoder(params, filter_num)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        feature = self.encoder(rgb, depth)
        return self.decoder(feature)

class UNet_DepthGuiderV3(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super().__init__()
        params = {'in_chns': in_chns, 'dropout': [0, 0, 0, 0, 0], 'class_num': class_num, 'bilinear': False, 'acti_func': 'relu', 'filter_num': filter_num}
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)
        self.depth_guiders = nn.ModuleList([DepthGuiderV3(c, depth_channels=1) for c in [filter_num * (2 ** i) for i in range(5)]])

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        feature = self.encoder(rgb)
        feature = [g(f, depth) for g, f in zip(self.depth_guiders, feature)]
        return self.decoder(feature)


class UNet_GeoRiskSPC(nn.Module):
    """Plain UNet with dual decoder for risk-guided perturbation consistency.

    Accepts in_chns input channels (may include depth), but encoder processes
    only the first 3 channels (RGB). Depth is used externally for risk maps.
    """

    def __init__(self, in_chns, class_num, filter_num=16,
                 dropout_rate=0.3, noise_std=0.1):
        super().__init__()
        self.in_chns = in_chns
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)
        self.decoder_pert = Decoder(params, filter_num)
        self.perturbation = RiskGuidedPerturbation(dropout_rate, noise_std)

    def forward(self, x, M_r=None):
        feature = self.encoder(x)
        p_clean = self.decoder(feature)
        if M_r is not None:
            feat_pert = self.perturbation(feature[4], M_r)
            feature_pert = list(feature)
            feature_pert[4] = feat_pert
            p_pert = self.decoder_pert(feature_pert)
            return p_clean, p_pert
        return p_clean


class UNet_DepthGuiderV4_GeoRiskSPC(nn.Module):
    """DepthGuiderV4 UNet with dual decoder for risk-guided perturbation consistency.

    Accepts in_chns input (may include depth). Encoder processes RGB only (3ch),
    depth is passed to DepthGuiderV4 modules at each encoder level.
    """

    def __init__(self, in_chns, class_num, filter_num=16,
                 dropout_rate=0.3, noise_std=0.1):
        super().__init__()
        self.in_chns = in_chns
        # Encoder always uses 3ch RGB; depth handled by DepthGuiderV4 modules
        rgb_params = {
            'in_chns': 3,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = rgb_params
        self.encoder = DepthGuidedEncoderV4(rgb_params, filter_num)
        self.decoder = Decoder(rgb_params, filter_num)
        self.decoder_pert = Decoder(rgb_params, filter_num)
        self.perturbation = RiskGuidedPerturbation(dropout_rate, noise_std)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x, M_r=None):
        rgb, depth = self._split_rgb_depth(x)
        feature = self.encoder(rgb, depth)
        p_clean = self.decoder(feature)
        if M_r is not None:
            feat_pert = self.perturbation(feature[4], M_r)
            feature_pert = list(feature)
            feature_pert[4] = feat_pert
            p_pert = self.decoder_pert(feature_pert)
            return p_clean, p_pert
        return p_clean


class ProjectionHeadContrastV1(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            build_group_norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet_ContrastV1(nn.Module):
    def __init__(self, in_chns, class_num, feature_dim=256, filter_num=16):
        super(UNet_ContrastV1, self).__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num,
            'feature_dim': feature_dim,
        }
        self.params = params
        self.feature_dim = feature_dim
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder(params, filter_num)
        self.projector = ProjectionHeadContrastV1(filter_num * 16, feature_dim)

    def forward(self, x):
        features = self.encoder(x)
        seg_logits = self.decoder(features)
        contrast_feat = self.projector(features[4])
        return seg_logits, contrast_feat


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super(UNet_URPC, self).__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder_URPC(params, filter_num)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg

class UNet_proto(nn.Module):
    def __init__(self, in_chns, class_num, feature_dim=256, filter_num=16, cont_dim=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num,
            'feature_dim': feature_dim,
        }
        self.params = params
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.filter_num = filter_num
        self.cont_dim = cont_dim
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder_PROTO(params, filter_num, cont_dim)
        self.projector = ProjectionHeadContrastV1(filter_num * 16, feature_dim)

    def forward(self, x):
        feats = self.encoder(x)
        seg_logits = self.decoder(feats)
        proto_feat = self.projector(feats[4])
        return seg_logits, proto_feat


UNet_PROTO = UNet_proto


class UNet_DepthGuiderProtoV1(nn.Module):
    def __init__(self, in_chns, class_num, feature_dim=256, filter_num=16, cont_dim=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num,
            'feature_dim': feature_dim,
        }
        self.params = params
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.filter_num = filter_num
        self.cont_dim = cont_dim
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder_PROTO(params, filter_num, cont_dim)
        self.projector = ProjectionHeadContrastV1(filter_num * 16, feature_dim)
        self.depth_guiders = nn.ModuleList(
            [DepthGuider(c, depth_channels=1) for c in [filter_num * (2 ** i) for i in range(5)]]
        )

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        feats = self.encoder(rgb)
        feats = [g(f, depth) for g, f in zip(self.depth_guiders, feats)]
        seg_logits = self.decoder(feats)
        proto_feat = self.projector(feats[4])
        return seg_logits, proto_feat

class UNet_W2S(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super(UNet_W2S, self).__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.decoder = Decoder_NP(params, filter_num)
        kaps = [0.067, 0.134, 0.2]
        self.decoder_1 = Decoder_W2S(params, kaps[0], filter_num)
        self.decoder_2 = Decoder_W2S(params, kaps[1], filter_num)
        self.decoder_3 = Decoder_W2S(params, kaps[2], filter_num)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        output_1 = self.decoder_1(feature)
        output_2 = self.decoder_2(feature)
        output_3 = self.decoder_3(feature)
        return output, output_1, output_2, output_3

class UNet_Depth(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super(UNet_Depth, self).__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.seg_decoder = Decoder(params, filter_num)
        self.depth_decoder = DepthDecoder(params, filter_num)
        self.depth_head = nn.Conv2d(filter_num, 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        seg_logits = self.seg_decoder(features)
        depth_feat = self.depth_decoder(features)
        depth_output = self.depth_head(depth_feat)
        return seg_logits, depth_output

class ASPP2D(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP2D, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class UNet_DyCON(nn.Module):
    def __init__(self, in_chns=1, class_num=2, scale_factor=2, use_aspp=False, filter_num=16):
        super(UNet_DyCON, self).__init__()
        self.use_aspp = use_aspp
        self.scale_factor = scale_factor
        filters = [filter_num * (2 ** i) for i in range(5)]
        dropout = [0, 0, 0, 0, 0]
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'scale_factor': scale_factor,
            'use_aspp': use_aspp
        }
        self.conv1 = ConvBlock(in_chns, filters[0], dropout[0])
        self.down1 = DownBlock(filters[0], filters[1], dropout[1])
        self.down2 = DownBlock(filters[1], filters[2], dropout[2])
        self.down3 = DownBlock(filters[2], filters[3], dropout[3])
        self.down4 = DownBlock(filters[3], filters[4], dropout[4])
        self.up4 = UpBlock(filters[4], filters[3], filters[3], dropout[3], bilinear=True)
        self.up3 = UpBlock(filters[3], filters[2], filters[2], dropout[2], bilinear=True)
        self.up2 = UpBlock(filters[2], filters[1], filters[1], dropout[1], bilinear=True)
        self.up1 = UpBlock(filters[1], filters[0], filters[0], dropout[0], bilinear=True)
        self.out_seg = nn.Conv2d(filters[0], class_num, 1)
        if self.use_aspp:
            self.aspp = ASPP2D(filters[4], filters[4], output_stride=16)
        self.projection = nn.Sequential(
            nn.Conv2d(filters[4], 512, kernel_size=1),
            build_group_norm(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            build_group_norm(256),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, x):
        conv1 = self.conv1(x)
        conv2 = self.down1(conv1)
        conv3 = self.down2(conv2)
        conv4 = self.down3(conv3)
        center = self.down4(conv4)
        return conv1, conv2, conv3, conv4, center

    def forward(self, x):
        conv1, conv2, conv3, conv4, center = self.encode(x)
        up4 = self.up4(center, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        center_processed = center
        if self.use_aspp:
            center_processed = self.aspp(center_processed)
        center_upsampled = F.interpolate(center_processed, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        features = self.projection(center_upsampled)
        out_seg = self.out_seg(up1)
        return out_seg, features


class UNet_DepthPretrain(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=16):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'dropout': [0, 0, 0, 0, 0],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'filter_num': filter_num,
        }
        self.params = params
        self.encoder = Encoder(params, filter_num)
        self.depth_decoder = DepthDecoder(params, filter_num)
        self.depth_head = nn.Conv2d(filter_num, 3, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        depth_feat = self.depth_decoder(features)
        depth_output = self.depth_head(depth_feat)
        return depth_output
