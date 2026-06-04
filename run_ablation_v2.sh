#!/bin/bash
# Run ALL 13 ablation experiments for GeoRisk-SPC (DGv4)
# Corrected directory naming: _depth1C suffix

set -e
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/guo/project/ssl4mis/code_all_vibe_v2"
RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
PRED_BASE="/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/task1"
LOG_DIR="/tmp"

wait_for_gpu_memory() {
    local threshold=${1:-8000}
    while true; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')
        if [ "$used" -lt "$threshold" ]; then
            break
        fi
        sleep 120
    done
}

is_fold_done() {
    local exp_name=$1
    local fold=$2
    local result_dir="${RESULT_BASE}/${exp_name}/20_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C/f${fold}"
    [ -f "$result_dir/model_final.pth" ]
}

is_test_done() {
    local exp_name=$1
    local fold=$2
    local result_dir="${PRED_BASE}/${exp_name}/20_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C/f${fold}"
    [ -f "$result_dir/test_metrics.csv" ]
}

run_fold() {
    local exp_name=$1
    local fold=$2
    shift 2
    local extra_args="$@"

    # Training
    if is_fold_done "$exp_name" "$fold"; then
        echo "[$(date)] Skipping training ${exp_name} f${fold} (already done)"
    else
        echo "[$(date)] Starting training: ${exp_name} f${fold}"
        cd "$BASE_DIR"
        python -m core.train --task 1 --way georisk_spc_dgv4 \
            --exp "endovis2017/${exp_name}" --labeled_num 20 --fold "$fold" \
            --use_depth 1 $extra_args \
            > "${LOG_DIR}/${exp_name}_20_f${fold}.log" 2>&1
        echo "[$(date)] Completed training: ${exp_name} f${fold}"
    fi

    # Testing
    if is_test_done "$exp_name" "$fold"; then
        echo "[$(date)] Skipping test ${exp_name} f${fold} (already done)"
    else
        echo "[$(date)] Starting test: ${exp_name} f${fold}"
        cd "$BASE_DIR"
        python -m core.test --task 1 --way georisk_spc_dgv4 \
            --exp "endovis2017/${exp_name}" --labeled_num 20 --fold "$fold" \
            --use_depth 1 $extra_args \
            > "${LOG_DIR}/${exp_name}_20_f${fold}_test.log" 2>&1
        echo "[$(date)] Completed test: ${exp_name} f${fold}"
    fi
}

run_ablation() {
    local exp_name=$1
    shift
    local extra_args="$@"

    for fold in 0 1 2 3; do
        wait_for_gpu_memory 8000
        run_fold "$exp_name" "$fold" $extra_args
    done
}

echo "[$(date)] Starting ALL ablation experiments (13 configs)"

# === Existing 7 ablations ===

echo "[$(date)] === 1/13: w/o L_cons ==="
run_ablation "Ablation_no_cons" --risk_cons_weight 0.0

echo "[$(date)] === 2/13: w/o L_bd ==="
run_ablation "Ablation_no_bd" --risk_bd_weight 0.0

echo "[$(date)] === 3/13: random mask (tau_r=1.0) ==="
run_ablation "Ablation_random_mask" --risk_tau_r 1.0

echo "[$(date)] === 4/13: tau_r=0.3 ==="
run_ablation "Ablation_tau_r_0.3" --risk_tau_r 0.3

echo "[$(date)] === 5/13: tau_r=0.7 ==="
run_ablation "Ablation_tau_r_0.7" --risk_tau_r 0.7

echo "[$(date)] === 6/13: tau_c=0.7 ==="
run_ablation "Ablation_tau_c_0.7" --risk_tau_c 0.7

echo "[$(date)] === 7/13: tau_c=0.95 ==="
run_ablation "Ablation_tau_c_0.95" --risk_tau_c 0.95

# === New 6 ablations (risk source decomposition) ===

echo "[$(date)] === 8/13: w/o risk localization ==="
run_ablation "Ablation_no_risk" --risk_no_supervision

echo "[$(date)] === 9/13: depth only ==="
run_ablation "Ablation_depth_only" --risk_source depth

echo "[$(date)] === 10/13: uncertainty only ==="
run_ablation "Ablation_uncertainty_only" --risk_source uncertainty

echo "[$(date)] === 11/13: conflict only ==="
run_ablation "Ablation_conflict_only" --risk_source conflict

echo "[$(date)] === 12/13: depth + uncertainty ==="
run_ablation "Ablation_depth_uncertainty" --risk_source depth_uncertainty

echo "[$(date)] === 13/13: DG encoder w/o risk supervision ==="
run_ablation "Ablation_dg_no_risk" --risk_no_supervision

echo "[$(date)] All 13 ablation experiments completed"
