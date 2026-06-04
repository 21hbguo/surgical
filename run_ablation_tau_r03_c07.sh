#!/bin/bash
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

echo "[$(date)] === Ablation: tau_r=0.3 + tau_c=0.7 ==="

for fold in 0 1 2 3; do
    wait_for_gpu_memory 8000
    run_fold "Ablation_tau_r03_c07" "$fold" --risk_tau_r 0.3 --risk_tau_c 0.7
done

echo "[$(date)] Done: tau_r=0.3 + tau_c=0.7"
