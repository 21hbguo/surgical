import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

def _import_dinov3_vits16(repo_dir):
    repo_dir = Path(repo_dir).expanduser().resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return importlib.import_module("dinov3.hub.backbones").dinov3_vits16

class DINOv3ResLikeEncoder(nn.Module):
    def __init__(
        self,
        in_chns,
        pretrain_root="../pre_train_ckp/",
        dinov3_repo_dir="../dinov3",
        dinov3_weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ):
        super().__init__()
        self.input_adapter = nn.Identity() if in_chns == 3 else nn.Conv2d(in_chns, 3, kernel_size=1, bias=False)

        weights_path = os.path.join(pretrain_root, dinov3_weights)

        dinov3_vits16 = _import_dinov3_vits16(dinov3_repo_dir)
        self.backbone = dinov3_vits16(pretrained=True, weights=weights_path)
        self.fuse = nn.Sequential(
            nn.Conv2d(384 * 4, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.to_e1 = nn.Conv2d(384, 64, kernel_size=1)
        self.to_e2 = nn.Conv2d(384, 64, kernel_size=1)
        self.to_e3 = nn.Conv2d(384, 128, kernel_size=1)
        self.to_e4 = nn.Conv2d(384, 256, kernel_size=1)
        self.to_e5 = nn.Conv2d(384, 512, kernel_size=1)

    def forward(self, x):
        x = self.input_adapter(x)
        f2, f3, f4, f5 = self.backbone.get_intermediate_layers(
            x,
            n=[2, 5, 8, 11],
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        fused = self.fuse(torch.cat([f2, f3, f4, f5], dim=1))
        e4 = self.to_e4(fused)
        e5 = self.to_e5(F.interpolate(fused, scale_factor=0.5, mode="bilinear", align_corners=False))
        e3 = self.to_e3(F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False))
        e2 = self.to_e2(F.interpolate(fused, scale_factor=4.0, mode="bilinear", align_corners=False))
        e1 = self.to_e1(F.interpolate(fused, scale_factor=8.0, mode="bilinear", align_corners=False))
        return e1, e2, e3, e4, e5

class _DINOv3BackboneMixin:
    def _swap_encoder(self, in_chns, pretrain_root, dinov3_repo_dir, dinov3_weights):
        self.encoder = DINOv3ResLikeEncoder(
            in_chns=in_chns,
            pretrain_root=pretrain_root,
            dinov3_repo_dir=dinov3_repo_dir,
            dinov3_weights=dinov3_weights,
        )
        self.params["dinov3_repo_dir"] = dinov3_repo_dir
        self.params["dinov3_weights"] = dinov3_weights
        self.params["backbone"] = "dinov3_vits16"
