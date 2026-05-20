
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block.common_block import DepthGuider
from .block.resnetunet_block import ConvBlock, DecoderBlock, DecoderBlock_W2S, ResNetDecoder_W2S, ResNetEncoder, build_group_norm, replace_batchnorm2d_with_groupnorm
from .block.unet_block import Dropout, FeatureDropout, FeatureNoise

class DepthGuiderV2(nn.Module):
    def __init__(self, in_channels, depth_channels=1):
        super().__init__()
        mid = max(16, in_channels // 2)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, mid, kernel_size=3, padding=1),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1)
            )
        self.fc_gamma = nn.Conv2d(mid, in_channels, kernel_size=1)
        self.fc_beta = nn.Conv2d(mid, in_channels, kernel_size=1)
        
        nn.init.constant_(self.fc_gamma.weight, 0)
        nn.init.constant_(self.fc_gamma.bias, 1)
        nn.init.constant_(self.fc_beta.weight, 0)
        nn.init.constant_(self.fc_beta.bias, 0)

    def forward(self, rgb_feat, depth):
        B, C, H, W = rgb_feat.shape
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        depth_feat = self.depth_encoder(depth)
        gamma = self.fc_gamma(depth_feat)
        beta = self.fc_beta(depth_feat)
        modulated = gamma * rgb_feat + beta
        return modulated + rgb_feat

class DepthGuiderV3(nn.Module):
    def __init__(self, in_channels, depth_channels=1):
        super().__init__()
        mid=max(16,in_channels//2)
        self.global_encoder=nn.Sequential(nn.Conv2d(depth_channels,mid,kernel_size=3,padding=1,bias=False),build_group_norm(mid),nn.ReLU(inplace=True),nn.Conv2d(mid,mid,kernel_size=3,padding=1,bias=False),build_group_norm(mid),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1))
        self.local_encoder=nn.Sequential(nn.Conv2d(depth_channels,mid,kernel_size=3,padding=1,bias=False),build_group_norm(mid),nn.ReLU(inplace=True),nn.Conv2d(mid,mid,kernel_size=3,padding=1,bias=False),build_group_norm(mid),nn.ReLU(inplace=True))
        self.global_gamma=nn.Linear(mid,in_channels)
        self.global_beta=nn.Linear(mid,in_channels)
        self.local_gamma=nn.Conv2d(mid,in_channels,kernel_size=1)
        self.local_beta=nn.Conv2d(mid,in_channels,kernel_size=1)
        self.local_gate=nn.Sequential(nn.Conv2d(mid,mid,kernel_size=3,padding=1,groups=mid,bias=False),build_group_norm(mid),nn.ReLU(inplace=True),nn.Conv2d(mid,in_channels,kernel_size=1))
        nn.init.constant_(self.global_gamma.weight,0)
        nn.init.constant_(self.global_gamma.bias,1)
        nn.init.constant_(self.global_beta.weight,0)
        nn.init.constant_(self.global_beta.bias,0)
        nn.init.constant_(self.local_gamma.weight,0)
        nn.init.constant_(self.local_gamma.bias,0)
        nn.init.constant_(self.local_beta.weight,0)
        nn.init.constant_(self.local_beta.bias,0)
        nn.init.constant_(self.local_gate[3].weight,0)
        nn.init.constant_(self.local_gate[3].bias,0)

    def forward(self, rgb_feat, depth):
        b,c,h,w=rgb_feat.shape
        if depth.shape[2:]!=(h,w):depth=F.interpolate(depth,size=(h,w),mode="bilinear",align_corners=False)
        global_feat=self.global_encoder(depth).view(b,-1)
        local_feat=self.local_encoder(depth)
        gamma_global=self.global_gamma(global_feat).view(b,c,1,1)
        beta_global=self.global_beta(global_feat).view(b,c,1,1)
        gamma_local=self.local_gamma(local_feat)
        beta_local=self.local_beta(local_feat)
        gate=torch.sigmoid(self.local_gate(local_feat))
        gamma=gamma_global*(1+gamma_local)
        beta=beta_global+beta_local
        delta=gate*(gamma*rgb_feat+beta)
        return rgb_feat+delta


class ResNetUNet_Base(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out = self.outconv(d1)
        return out


class ResNetUNet_DepthGuiderV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

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


class ResNetUNet_DepthGuiderV2(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'filter_num': filter_num,
            'variant': variant,
            'pretrain_root': pretrain_root,
        }
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.depth_guiders = nn.ModuleList([DepthGuiderV2(c, depth_channels=1) for c in [64, 64, 128, 256, 512]])

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
        replace_batchnorm2d_with_groupnorm(self)

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

class ResNetUNet_DepthGuiderV3(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params={'in_chns':in_chns,'class_num':class_num,'filter_num':filter_num,'variant':variant,'pretrain_root':pretrain_root}
        self.encoder=ResNetEncoder(variant,in_chns,pretrain_root,load_pretrained=load_encoder_pretrained)
        self.depth_guiders=nn.ModuleList([DepthGuiderV3(c,depth_channels=1) for c in [64,64,128,256,512]])
        self.decoder5=DecoderBlock(in_channels=512,out_channels=512)
        self.decoder4=DecoderBlock(in_channels=512+256,out_channels=256)
        self.decoder3=DecoderBlock(in_channels=256+128,out_channels=128)
        self.decoder2=DecoderBlock(in_channels=128+64,out_channels=64)
        self.decoder1=DecoderBlock(in_channels=64+64,out_channels=64)
        self.outconv=nn.Sequential(ConvBlock(64,32,kernel_size=3,stride=1,padding=1),nn.Dropout2d(dropout),nn.Conv2d(32,class_num,1))
        replace_batchnorm2d_with_groupnorm(self)

    def _split_rgb_depth(self, x):
        rgb=x[:,:3,:,:]
        depth=x[:,3:4,:,:] if x.shape[1]>=4 else rgb[:,:1,:,:]
        return rgb,depth

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
            build_group_norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.layers(x)

class ResNetUNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        self.feature_noise = FeatureNoise()
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        shape = x.shape[2:]

        d5 = self.decoder5(e5)
        dp4_out = self.out_conv_dp4(Dropout(d5, p=0.5) if self.training else d5)
        dp4_out = F.interpolate(dp4_out, shape, mode='bilinear', align_corners=True)

        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        dp3_out = self.out_conv_dp3(FeatureDropout(d4) if self.training else d4)
        dp3_out = F.interpolate(dp3_out, shape, mode='bilinear', align_corners=True)

        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        dp2_out = self.out_conv_dp2(self.feature_noise(d3) if self.training else d3)
        dp2_out = F.interpolate(dp2_out, shape, mode='bilinear', align_corners=True)

        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        dp0_out = self.out_conv(d1)

        return dp0_out, dp2_out, dp3_out, dp4_out


class ResNetUNet_ContrastV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 feature_dim=256, dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

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

class ResNetUNet_proto(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 feature_dim=256, dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

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


ResNetUNet_PROTO = ResNetUNet_proto


class ResNetUNet_DepthGuiderProtoV1(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 feature_dim=256, dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

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


class ResNetUNet_DyCON(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34',
                 scale_factor=2, use_aspp=False, dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

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

class ResNetUNet_W2S(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        output_1 = self.decoder_1(features)
        output_2 = self.decoder_2(features)
        output_3 = self.decoder_3(features)
        return output, output_1, output_2, output_3

class ResNetUNet_Depth(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

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


class ResNetUNet_DepthPretrain(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
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
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        depth_output = self.depth_head(d1)
        return depth_output
