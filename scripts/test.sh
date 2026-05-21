#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"
TMPDIR="$DIR/outputs/.tmp"
mkdir -p "$TMPDIR"
export TMPDIR

# c && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way dformerv2_fully --exp endovis2017/DFormerv2Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way dformerv2_fully --exp endovis2017/DFormerv2Fully --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 10 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 40 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 10 --use_depth 3
# # Additional commented strategies (parity block)
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way w2s --exp endovis2017/W2S --labeled_num 40 --use_depth 3 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way w2s --exp endovis2017/W2S --labeled_num 10 --use_depth 3 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1 --exp endovis2017/FullyContrastV1 --labeled_num 100 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1 --exp endovis2017/FullyContrastV1 --labeled_num 40 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1 --exp endovis2017/FullyContrastV1 --labeled_num 20 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1 --exp endovis2017/FullyContrastV1 --labeled_num 10 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1 --exp endovis2017/FullyContrastV1None --labeled_num 20 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1 --exp endovis2017/FullyContrastV1None --labeled_num 10 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1_1 --exp endovis2017/FullyContrastV1_1 --labeled_num 20 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1_1 --exp endovis2017/FullyContrastV1_1 --labeled_num 10 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1_1 --exp endovis2017/FullyContrastV1_1None --labeled_num 20 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_contrast_v1_1 --exp endovis2017/FullyContrastV1_1None --labeled_num 10 --contrast_loss_weight 0.05 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way dycon --exp endovis2017/DyCON --labeled_num 40 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way dycon --exp endovis2017/DyCON --labeled_num 10 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 40 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 10 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 40 --use_depth 3 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 10 --use_depth 3 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way depth_mt --exp endovis2017/DepthMT --labeled_num 40 --use_depth 3 && \
# # CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way depth_mt --exp endovis2017/DepthMT --labeled_num 10 --use_depth 3 && \

# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_teacher_v1 --exp endovis2017/MT_depth_teacher_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_teacher_v1 --exp endovis2017/MT_depth_teacher2_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v1 --exp endovis2017/MT_depth_guider_v1 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v2 --exp endovis2017/MT_depth_guider_v2 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_proto_v1 --exp endovis2017/MT_depth_guider_proto_v1 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_proto_teacher_v1 --exp endovis2017/mt_depth_guider_proto_teacher_v1 --labeled_num 40 --use_depth 13 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_proto_teacher_v2 --exp endovis2017/mt_depth_guider_proto_teacher_v2 --labeled_num 40 --use_depth 13 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way semi_mean_teacher_contrast_v1 --exp endovis2017/MT_contrast_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/MT_proto_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_depth_pretest_v1 --exp endovis2017/FullyDepthPretest_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_rgb_masking_depth_v1 --exp endovis2017/FullyRGBMaskingDepth_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 20 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 10


# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_teacher_v1 --exp endovis2017/MT_depth_teacher_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_teacher_v1 --exp endovis2017/MT_depth_teacher2_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v1 --exp endovis2017/MT_depth_guider_v1 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v2 --exp endovis2017/MT_depth_guider_v2 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_proto_v1 --exp endovis2017/MT_depth_guider_proto_v1 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_proto_teacher_v1 --exp endovis2017/mt_depth_guider_proto_teacher_v1 --labeled_num 40 --use_depth 13 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_proto_teacher_v2 --exp endovis2017/mt_depth_guider_proto_teacher_v2 --labeled_num 40 --use_depth 13 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way semi_mean_teacher_contrast_v1 --exp endovis2017/MT_contrast_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way proto_v1 --exp endovis2017/MT_proto_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_depth_pretest_v1 --exp endovis2017/FullyDepthPretest_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully_rgb_masking_depth_v1 --exp endovis2017/FullyRGBMaskingDepth_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 20 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 10
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 100 &
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 40
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 20 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 20 && \



## 要挂的实验(work的)
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way semi_mean_teacher_text_v1 --exp endovis2017/MTTextV1 --labeled_num 40 --proto_feature_dim 512
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold 0 --task 1 --way mt_depth_guider_v1_2 --exp endovis2017/MT_depth_guider_v12 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold 0 --task 1 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold 0 --task 1 --way mt --exp endovis2017/MT2 --labeled_num 40
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold 0 --task 1 --way fully --exp endovis2017/Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold 0 --task 1 --way fully --exp endovis2017/Fully --labeled_num 40
## 测试验证可不可以
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_teacher_v1 --exp endovis2017/MT_depth_teacher_v1 --labeled_num 40 --use_depth 3

# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way only_depth_input --exp endovis2017/OnlyDepthInput --labeled_num 40 --use_depth 1
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way only_depth_input --exp endovis2017/OnlyDepthInput --labeled_num 10 --use_depth 1


## 计划跑的实验
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 40 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 20 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 10 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 5 --use_depth 1 && \
## 计划跑的实验
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt --exp endovis2017/MT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way uamt --exp endovis2017/UAMT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way urpc --exp endovis2017/URPC --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 40 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt --exp endovis2017/MT --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way uamt --exp endovis2017/UAMT --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way urpc --exp endovis2017/URPC --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 20 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt --exp endovis2017/MT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way uamt --exp endovis2017/UAMT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way urpc --exp endovis2017/URPC --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 10 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt --exp endovis2017/MT --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way uamt --exp endovis2017/UAMT --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way urpc --exp endovis2017/URPC --labeled_num 5 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 5 --use_depth 1