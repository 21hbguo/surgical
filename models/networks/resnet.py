
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block.common_block import DepthGuider, RiskGuidedPerturbation
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

class DepthGuidedResNetEncoderV1_2(nn.Module):
    def __init__(self, variant="resnet34", in_chns=3, pretrain_root='../pre_train_ckp/', load_pretrained=True):
        super().__init__()
        resnet = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_pretrained)
        self.encoder1_conv = resnet.encoder1_conv
        self.encoder1_bn = resnet.encoder1_bn
        self.encoder1_relu = resnet.encoder1_relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.encoder2
        self.encoder3 = resnet.encoder3
        self.encoder4 = resnet.encoder4
        self.encoder5 = resnet.encoder5
        self.depth_guiders = nn.ModuleList([DepthGuider(c, depth_channels=1) for c in [64, 64, 128, 256, 512]])

    def forward(self, x, depth):
        e1 = self.encoder1_relu(self.encoder1_bn(self.encoder1_conv(x)))
        e1 = self.depth_guiders[0](e1, depth)
        e2 = self.encoder2(self.maxpool(e1))
        e2 = self.depth_guiders[1](e2, depth)
        e3 = self.encoder3(e2)
        e3 = self.depth_guiders[2](e3, depth)
        e4 = self.encoder4(e3)
        e4 = self.depth_guiders[3](e4, depth)
        e5 = self.encoder5(e4)
        e5 = self.depth_guiders[4](e5, depth)
        return e1, e2, e3, e4, e5

class DepthGuidedResNetEncoderV4(nn.Module):
    def __init__(self, variant="resnet34", in_chns=3, pretrain_root='../pre_train_ckp/', load_pretrained=True):
        super().__init__()
        resnet = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_pretrained)
        self.encoder1_conv = resnet.encoder1_conv
        self.encoder1_bn = resnet.encoder1_bn
        self.encoder1_relu = resnet.encoder1_relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.encoder2
        self.encoder3 = resnet.encoder3
        self.encoder4 = resnet.encoder4
        self.encoder5 = resnet.encoder5
        self.depth_guiders = nn.ModuleList([DepthGuiderV4(c, depth_channels=1, pool_size=s) for c, s in zip([64, 64, 128, 256, 512], [16, 16, 8, 4, 2])])

    def forward(self, x, depth):
        e1 = self.encoder1_relu(self.encoder1_bn(self.encoder1_conv(x)))
        e1 = self.depth_guiders[0](e1, depth)
        e2 = self.encoder2(self.maxpool(e1))
        e2 = self.depth_guiders[1](e2, depth)
        e3 = self.encoder3(e2)
        e3 = self.depth_guiders[2](e3, depth)
        e4 = self.encoder4(e3)
        e4 = self.depth_guiders[3](e4, depth)
        e5 = self.encoder5(e4)
        e5 = self.depth_guiders[4](e5, depth)
        return e1, e2, e3, e4, e5

class ResNetSegDecoder(nn.Module):
    def __init__(self, class_num, dropout=0.0):
        super().__init__()
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)
        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1), nn.Dropout2d(dropout), nn.Conv2d(32, class_num, 1))

    def forward(self, features):
        e1, e2, e3, e4, e5 = features
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        return self.outconv(d1)


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


class ResNetUNet_RDNet(ResNetUNet_Base):
    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out = self.outconv(d1)
        return out, e5


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

class ResNetUNet_DepthGuiderV1_2(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {'in_chns': in_chns, 'class_num': class_num, 'filter_num': filter_num, 'variant': variant, 'pretrain_root': pretrain_root}
        self.encoder = DepthGuidedResNetEncoderV1_2(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.decoder = ResNetSegDecoder(class_num, dropout=dropout)
        replace_batchnorm2d_with_groupnorm(self)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        return self.decoder(self.encoder(rgb, depth))

class ResNetUNet_DepthGuiderV4(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', load_encoder_pretrained=True):
        super().__init__()
        self.params = {'in_chns': in_chns, 'class_num': class_num, 'filter_num': filter_num, 'variant': variant, 'pretrain_root': pretrain_root}
        self.encoder = DepthGuidedResNetEncoderV4(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.decoder = ResNetSegDecoder(class_num, dropout=dropout)
        replace_batchnorm2d_with_groupnorm(self)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x):
        rgb, depth = self._split_rgb_depth(x)
        return self.decoder(self.encoder(rgb, depth))

class ResNetUNet_GeoRiskSPC(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', dropout_rate=0.3, noise_std=0.1, load_encoder_pretrained=True):
        super().__init__()
        self.params = {'in_chns': in_chns, 'class_num': class_num, 'filter_num': filter_num, 'variant': variant, 'pretrain_root': pretrain_root}
        self.encoder = ResNetEncoder(variant, in_chns, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.decoder = ResNetSegDecoder(class_num, dropout=dropout)
        self.decoder_pert = ResNetSegDecoder(class_num, dropout=dropout)
        self.perturbation = RiskGuidedPerturbation(dropout_rate, noise_std)
        replace_batchnorm2d_with_groupnorm(self)

    def forward(self, x, risk_mask=None):
        features = self.encoder(x)
        p_clean = self.decoder(features)
        if risk_mask is not None:
            feature_pert = list(features)
            feature_pert[4] = self.perturbation(feature_pert[4], risk_mask)
            p_pert = self.decoder_pert(feature_pert)
            return p_clean, p_pert
        return p_clean

class ResNetUNet_DepthGuiderV4_GeoRiskSPC(nn.Module):
    def __init__(self, in_chns, class_num, filter_num=32, variant='resnet34', dropout=0.0, pretrain_root='../pre_train_ckp/', dropout_rate=0.3, noise_std=0.1, load_encoder_pretrained=True):
        super().__init__()
        self.params = {'in_chns': in_chns, 'class_num': class_num, 'filter_num': filter_num, 'variant': variant, 'pretrain_root': pretrain_root}
        self.encoder = DepthGuidedResNetEncoderV4(variant, 3, pretrain_root, load_pretrained=load_encoder_pretrained)
        self.decoder = ResNetSegDecoder(class_num, dropout=dropout)
        self.decoder_pert = ResNetSegDecoder(class_num, dropout=dropout)
        self.perturbation = RiskGuidedPerturbation(dropout_rate, noise_std)
        replace_batchnorm2d_with_groupnorm(self)

    def _split_rgb_depth(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :] if x.shape[1] >= 4 else rgb[:, :1, :, :]
        return rgb, depth

    def forward(self, x, risk_mask=None):
        rgb, depth = self._split_rgb_depth(x)
        features = self.encoder(rgb, depth)
        p_clean = self.decoder(features)
        if risk_mask is not None:
            feature_pert = list(features)
            feature_pert[4] = self.perturbation(feature_pert[4], risk_mask)
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
