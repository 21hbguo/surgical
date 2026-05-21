#!/bin/bash
# RDNet Training Scripts
# 支持多种pretrained backbone: unet, resnet, depth, dinov3
# use_depth=13 同时加载 depth3（3通道）和 depth1（1通道）

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"

# ==================== ResNet Backbone（最接近原始RDNet） ====================
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode resnet --exp kvasir_SEG/RDNet_ResNet --labeled_num 10 --use_depth 13
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode resnet --exp kvasir_SEG/RDNet_ResNet --labeled_num 20 --use_depth 13
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode resnet --exp kvasir_SEG/RDNet_ResNet --labeled_num 40 --use_depth 13

# ==================== UNet Backbone ====================
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode none --exp kvasir_SEG/RDNet_UNet --labeled_num 10 --use_depth 13
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode none --exp kvasir_SEG/RDNet_UNet --labeled_num 40 --use_depth 13

# ==================== Depth Backbone ====================
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode depth --exp kvasir_SEG/RDNet_Depth --labeled_num 10 --use_depth 13
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode depth --exp kvasir_SEG/RDNet_Depth --labeled_num 40 --use_depth 13

# ==================== DINOv3 Backbone ====================
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode dinov3 --exp kvasir_SEG/RDNet_DINOv3 --labeled_num 10 --use_depth 13
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way rdnet --pretrain_mode dinov3 --exp kvasir_SEG/RDNet_DINOv3 --labeled_num 40 --use_depth 13
