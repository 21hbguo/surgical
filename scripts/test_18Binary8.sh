#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way ternaus --exp endovis2018ISINet/TernausNet16 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way ternaus --exp endovis2018ISINet/TernausNet16 --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way fully --exp endovis2018ISINet/Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way fully --exp endovis2018ISINet/Fully --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way dformerv2_fully --exp endovis2018ISINet/DFormerv2Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way dformerv2_fully --exp endovis2018ISINet/DFormerv2Fully --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way urpc --exp endovis2018ISINet/URPC --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way urpc --exp endovis2018ISINet/URPC --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way mt --exp endovis2018ISINet/MT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way mt --exp endovis2018ISINet/MT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way uamt --exp endovis2018ISINet/UAMT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way uamt --exp endovis2018ISINet/UAMT --labeled_num 10 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way proto_v1 --exp endovis2018ISINet/Proto_v1 --labeled_num 40 --use_depth 3 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way proto_v1 --exp endovis2018ISINet/Proto_v1 --labeled_num 10 --use_depth 3
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way w2s --exp endovis2018ISINet/W2S --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way w2s --exp endovis2018ISINet/W2S --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way dycon --exp endovis2018ISINet/DyCON --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way dycon --exp endovis2018ISINet/DyCON --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way proto_v1 --exp endovis2018ISINet/Proto_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way proto_v1 --exp endovis2018ISINet/Proto_v1 --labeled_num 10 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way proto_v1 --exp endovis2018ISINet/Proto_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way proto_v1 --exp endovis2018ISINet/Proto_v1 --labeled_num 10 --use_depth 3
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way depth_mt --exp endovis2018ISINet/DepthMT --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --task 1 --way depth_mt --exp endovis2018ISINet/DepthMT --labeled_num 10 --use_depth 3 && \
