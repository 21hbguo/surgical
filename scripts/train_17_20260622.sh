#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"
TMPDIR="$DIR/outputs/.tmp"
mkdir -p "$TMPDIR"
export TMPDIR

## task 1 对比实验
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way fully --exp endovis2017/Fully --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way mt --exp endovis2017/MT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way uamt --exp endovis2017/UAMT --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way urpc --exp endovis2017/URPC --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way cps --exp endovis2017/CPS --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way u2pl --exp endovis2017/U2PL --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way segmatch --exp endovis2017/SegMatch --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way unimatch --exp endovis2017/UniMatch --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way corrmatch --exp endovis2017/CorrMatch --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way cwbass --exp endovis2017/CW-BASS --labeled_num 40 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way mt_depth_guider_v4 --exp endovis2017/MT_depth_guider_v4 --labeled_num 40 --use_depth 1 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/GeoRiskSPC_DGv4 --labeled_num 40 --use_depth 1

## task 1 消融实验
# 组件移除
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_no_cons --labeled_num 20 --use_depth 1 --risk_cons_weight 0.0 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_no_bd --labeled_num 20 --use_depth 1 --risk_bd_weight 0.0 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_random_mask --labeled_num 20 --use_depth 1 --risk_tau_r 1.0 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_dg_no_risk --labeled_num 20 --use_depth 1 --risk_no_supervision && \
# 风险源分解
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_no_risk --labeled_num 20 --use_depth 1 --risk_no_supervision && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_depth_only --labeled_num 20 --use_depth 1 --risk_source depth && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_uncertainty_only --labeled_num 20 --use_depth 1 --risk_source uncertainty && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_conflict_only --labeled_num 20 --use_depth 1 --risk_source conflict && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_depth_uncertainty --labeled_num 20 --use_depth 1 --risk_source depth_uncertainty && \
# 阈值敏感性
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_tau_r_0.3 --labeled_num 20 --use_depth 1 --risk_tau_r 0.3 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_tau_r_0.7 --labeled_num 20 --use_depth 1 --risk_tau_r 0.7 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_tau_c_0.7 --labeled_num 20 --use_depth 1 --risk_tau_c 0.7 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_tau_c_0.95 --labeled_num 20 --use_depth 1 --risk_tau_c 0.95 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp endovis2017/Ablation_tau_r03_c07 --labeled_num 20 --use_depth 1 --risk_tau_r 0.3 --risk_tau_c 0.7
