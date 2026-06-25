#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$DIR"
TMPDIR="$DIR/outputs/.tmp"
mkdir -p "$TMPDIR"
export TMPDIR

## task 1 对比实验
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way fully --exp bibm_kvasir_SEG_h5_224_224/Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way fully --exp bibm_endovis2017_h5_224_224/Fully --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way fully --exp bibm_endovis2018ISINet_h5_224_224/Fully --labeled_num 100
CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way rdnet --exp bibm_kvasir_SEG_h5_224_224/RDNet --labeled_num 40 --use_depth 13
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way mt --exp bibm_kvasir_SEG_h5_224_224/MT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way uamt --exp bibm_kvasir_SEG_h5_224_224/UAMT --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way urpc --exp bibm_kvasir_SEG_h5_224_224/URPC --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way cps --exp bibm_kvasir_SEG_h5_224_224/CPS --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way u2pl --exp bibm_kvasir_SEG_h5_224_224/U2PL_official --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way segmatch --exp bibm_kvasir_SEG_h5_224_224/SegMatch_official --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way unimatch --exp bibm_kvasir_SEG_h5_224_224/UniMatch_official --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way corrmatch --exp bibm_kvasir_SEG_h5_224_224/CorrMatch_official --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way cwbass --exp bibm_kvasir_SEG_h5_224_224/CW-BASS_official --labeled_num 40 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way mt_depth_guider_v4 --exp bibm_kvasir_SEG_h5_224_224/MT_depth_guider_v4 --labeled_num 40 --use_depth 1 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/GeoRiskSPC_DGv4 --labeled_num 40 --use_depth 1

# ## task 1 消融实验
# # 组件移除
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_no_cons --labeled_num 20 --use_depth 1 --risk_cons_weight 0.0 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_no_bd --labeled_num 20 --use_depth 1 --risk_bd_weight 0.0 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_random_mask --labeled_num 20 --use_depth 1 --risk_tau_r 1.0 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_dg_no_risk --labeled_num 20 --use_depth 1 --risk_no_supervision && \
# # 风险源分解
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_no_risk --labeled_num 20 --use_depth 1 --risk_no_supervision && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_depth_only --labeled_num 20 --use_depth 1 --risk_source depth && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_uncertainty_only --labeled_num 20 --use_depth 1 --risk_source uncertainty && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_conflict_only --labeled_num 20 --use_depth 1 --risk_source conflict && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_depth_uncertainty --labeled_num 20 --use_depth 1 --risk_source depth_uncertainty && \
# # 阈值敏感性
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_tau_r_0.3 --labeled_num 20 --use_depth 1 --risk_tau_r 0.3 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_tau_r_0.7 --labeled_num 20 --use_depth 1 --risk_tau_r 0.7 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_tau_c_0.7 --labeled_num 20 --use_depth 1 --risk_tau_c 0.7 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_tau_c_0.95 --labeled_num 20 --use_depth 1 --risk_tau_c 0.95 && \
# CUDA_VISIBLE_DEVICES=0 python -m core.train --fold -1 --task 1 --way georisk_spc_dgv4 --exp bibm_kvasir_SEG_h5_224_224/Ablation_tau_r03_c07 --labeled_num 20 --use_depth 1 --risk_tau_r 0.3 --risk_tau_c 0.7
