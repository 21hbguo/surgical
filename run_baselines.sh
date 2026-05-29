#!/bin/bash
# Run baseline experiments: SegMatch, UniMatch, CPS at 5/10/20/40%
# Usage: bash run_baselines.sh [start_from]
# start_from: "segmatch_5", "unimatch_20", "cps_5", etc. (default: segmatch_5)

set -e
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/guo/project/ssl4mis/code_all_vibe_v2"
RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
LOG_DIR="/tmp"
LABEL_RATIOS=(20 5 10 40)  # 20 first for SegMatch (already running)
METHODS=("segmatch:SegMatch" "unimatch:UniMatch" "cps:CPS")

wait_for_gpu() {
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
        echo "[$(date)] Waiting for GPU to be free..."
        sleep 60
    done
}

is_experiment_done() {
    local exp_name=$1
    local pct=$2
    local result_dir="${RESULT_BASE}/${exp_name}/${pct}_labeled_lr1e-4_s_unet"
    local done_count=0
    for fold in 0 1 2 3; do
        if [ -f "$result_dir/f${fold}/model_final.pth" ]; then
            ((done_count++))
        fi
    done
    [ $done_count -eq 4 ]
}

run_experiment() {
    local way=$1
    local exp_name=$2
    local pct=$3

    if is_experiment_done "$exp_name" "$pct"; then
        echo "[$(date)] Skipping ${way} ${pct}% (already done)"
        return 0
    fi

    echo "[$(date)] Starting: ${way} ${pct}%"
    cd "$BASE_DIR"

    # Train all folds
    python -m core.train --fold -1 --task 1 --way "$way" \
        --exp "endovis2017/${exp_name}" --labeled_num "$pct" \
        > "${LOG_DIR}/train_${way}_${pct}.log" 2>&1

    # Test (no --use_depth for these methods)
    python -m core.test --task 1 --way "$way" \
        --exp "endovis2017/${exp_name}" --labeled_num "$pct" \
        --fold -1 \
        > "${LOG_DIR}/test_${way}_${pct}.log" 2>&1

    echo "[$(date)] Completed: ${way} ${pct}%"
}

# Parse start point
START_FROM="${1:-segmatch_20}"
STARTED=false

for method_entry in "${METHODS[@]}"; do
    IFS=':' read -r way exp_name <<< "$method_entry"
    for pct in "${LABEL_RATIOS[@]}"; do
        key="${way}_${pct}"
        if [ "$key" = "$START_FROM" ]; then
            STARTED=true
        fi
        if [ "$STARTED" = true ]; then
            run_experiment "$way" "$exp_name" "$pct"
        fi
    done
done

echo "[$(date)] All baseline experiments completed!"
