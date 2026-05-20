
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block.common_block import FeaturePerturbation
from .block.unet_block import ConvBlock, DownBlock, Encoder, Decoder, Decoder_URPC, UpBlock, build_group_norm, replace_batchnorm2d_with_groupnorm


class DepthGuider(nn.Module):
    def __init__(self, in_channels, depth_channels=1):
        super().__init__()
        mid = max(16, in_channels // 2)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, mid, kernel_size=3, padding=1),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1) # 全局池化，得到全局深度向量
        )
        
        self.fc_gamma = nn.Linear(mid, in_channels)
        self.fc_beta = nn.Linear(mid, in_channels)
        
        nn.init.constant_(self.fc_gamma.weight, 0)
        nn.init.constant_(self.fc_gamma.bias, 1)
        nn.init.constant_(self.fc_beta.weight, 0)
        nn.init.constant_(self.fc_beta.bias, 0)

    def forward(self, rgb_feat, depth):
        B, C, H, W = rgb_feat.shape
        if depth.shape[2:] != rgb_feat.shape[2:]:
            depth = F.interpolate(depth, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)
        depth_feat = self.depth_encoder(depth).view(B, -1)
        gamma = self.fc_gamma(depth_feat).view(B, C, 1, 1)
        beta = self.fc_beta(depth_feat).view(B, C, 1, 1)
        modulated_feat = gamma * rgb_feat + beta
        return modulated_feat + rgb_feat


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
