#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"

# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way ternaus --exp endovis2017/TernausNet16 --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way dformerv2_fully --exp endovis2017/DFormerv2Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way dformerv2_fully --exp endovis2017/DFormerv2Fully --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way urpc --exp endovis2017/URPC --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way urpc --exp endovis2017/URPC --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt --exp endovis2017/MT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way mt --exp endovis2017/MT --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way uamt --exp endovis2017/UAMT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way uamt --exp endovis2017/UAMT --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way proto_v1 --exp endovis2017/Proto_v1 --labeled_num 10 --use_depth 3
# Additional commented strategies (parity block)
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way w2s --exp endovis2017/W2S  --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way w2s --exp endovis2017/W2S  --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way dycon --exp endovis2017/DyCON  --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way dycon --exp endovis2017/DyCON  --labeled_num 10 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way proto_v1 --exp endovis2017/Proto_v1  --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way proto_v1 --exp endovis2017/Proto_v1  --labeled_num 10 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way proto_v1 --exp endovis2017/Proto_v1  --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way proto_v1 --exp endovis2017/Proto_v1  --labeled_num 10 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way depth_mt --exp endovis2017/DepthMT  --labeled_num 40 --use_depth 3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way depth_mt --exp endovis2017/DepthMT  --labeled_num 10 --use_depth 3 && \

# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 100 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 3 --way fully --exp endovis2017/Fully --labeled_num 100

CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 1 --way only_depth_input --exp endovis2017/OnlyDepthInput --labeled_num 40 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 20 && \
CUDA_VISIBLE_DEVICES=0 python -m core.test --fold -1 --task 2 --way fully --exp endovis2017/Fully --labeled_num 10
