#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way ternaus --exp kvasir_SEG/TernausNet16 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way ternaus --exp kvasir_SEG/TernausNet16 --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way fully --exp kvasir_SEG/Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way fully --exp kvasir_SEG/Fully --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way dformerv2_fully --exp kvasir_SEG/DFormerv2Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way dformerv2_fully --exp kvasir_SEG/DFormerv2Fully --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way urpc --exp kvasir_SEG/URPC --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way urpc --exp kvasir_SEG/URPC --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way mt --exp kvasir_SEG/MT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way mt --exp kvasir_SEG/MT --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way uamt --exp kvasir_SEG/UAMT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way uamt --exp kvasir_SEG/UAMT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way proto_v1 --exp kvasir_SEG/Proto_v1 --labeled_num 40 --use_depth 3 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way proto_v1 --exp kvasir_SEG/Proto_v1 --labeled_num 10 --use_depth 3
# Additional commented strategies (parity block)
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way w2s --exp kvasir_SEG/W2S --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way w2s --exp kvasir_SEG/W2S --labeled_num 10 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way dycon --exp kvasir_SEG/DyCON --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way dycon --exp kvasir_SEG/DyCON --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way proto_v1 --exp kvasir_SEG/Proto_v1 --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way proto_v1 --exp kvasir_SEG/Proto_v1 --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way proto_v1 --exp kvasir_SEG/Proto_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way proto_v1 --exp kvasir_SEG/Proto_v1 --labeled_num 10 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way depth_mt --exp kvasir_SEG/DepthMT --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --task 1 --way depth_mt --exp kvasir_SEG/DepthMT --labeled_num 10 --use_depth 3 && \
