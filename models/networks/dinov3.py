import torch.nn as nn
from .block.dinov3unet_block import _DINOv3BackboneMixin
from .resnet import (
    ResNetUNet_Base,
    ResNetUNet_ContrastV1,
    ResNetUNet_DepthGuiderV1,
    ResNetUNet_DepthGuiderProtoV1,
    ResNetUNet_DyCON,
    ResNetUNet_Depth,
    ResNetUNet_DepthPretrain,
    ResNetUNet_RDNet,
    ResNetUNet_proto,
    ResNetUNet_URPC,
    ResNetUNet_W2S,
)

class DINOv3UNet_Base(_DINOv3BackboneMixin, ResNetUNet_Base):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)


class DINOv3UNet_RDNet(_DINOv3BackboneMixin, ResNetUNet_RDNet):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)


class DINOv3UNet_DepthGuiderV1(_DINOv3BackboneMixin, ResNetUNet_DepthGuiderV1):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)


class DINOv3UNet_DepthGuiderProtoV1(_DINOv3BackboneMixin, ResNetUNet_DepthGuiderProtoV1):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", feature_dim=256, cont_dim=16, dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, feature_dim, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)

class DINOv3UNet_ContrastV1(_DINOv3BackboneMixin, ResNetUNet_ContrastV1):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", feature_dim=256, dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, feature_dim, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)

class DINOv3UNet_URPC(_DINOv3BackboneMixin, ResNetUNet_URPC):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)

class DINOv3UNet_proto(_DINOv3BackboneMixin, ResNetUNet_proto):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", feature_dim=256, cont_dim=16, dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, feature_dim, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)


DINOv3UNet_PROTO = DINOv3UNet_proto

class DINOv3UNet_DyCON(_DINOv3BackboneMixin, ResNetUNet_DyCON):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", scale_factor=2, use_aspp=False, dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, scale_factor, use_aspp, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)

class DINOv3UNet_W2S(_DINOv3BackboneMixin, ResNetUNet_W2S):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)

class DINOv3UNet_Depth(_DINOv3BackboneMixin, ResNetUNet_Depth):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)


class DINOv3UNet_DepthPretrain(_DINOv3BackboneMixin, ResNetUNet_DepthPretrain):
    def __init__(self, in_chns, class_num, filter_num=32, variant="resnet34", dropout=0.0, pretrain_root="../pre_train_ckp/", dinov3_repo_dir="../dinov3", dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__(in_chns, class_num, filter_num, variant, dropout, pretrain_root, load_encoder_pretrained=False)
        self._swap_encoder(in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights)
