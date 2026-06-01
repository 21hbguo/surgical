#!/bin/bash
# Queue remaining MMS and UniMatch official experiments
# Waits for current 20% runs to finish, then runs remaining folds + ratios

set -e
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/guo/project/ssl4mis/code_all_vibe_v2"
RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
LOG_DIR="/tmp"

wait_for_gpu_memory() {
    # Wait until GPU memory usage drops below threshold (in MB)
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
    local pct=$2
    local fold=$3
    local result_dir="${RESULT_BASE}/${exp_name}/${pct}_labeled_lr1e-4_s_unet"
    [ -f "$result_dir/f${fold}/model_final.pth" ]
}

run_fold() {
    local way=$1
    local exp_name=$2
    local pct=$3
    local fold=$4

    if is_fold_done "$exp_name" "$pct" "$fold"; then
        echo "[$(date)] Skipping ${way} ${pct}% f${fold} (already done)"
        return 0
    fi

    echo "[$(date)] Starting: ${way} ${pct}% f${fold}"
    cd "$BASE_DIR"

    python -m core.train --task 1 --way "$way" \
        --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
        > "${LOG_DIR}/${way}_${pct}_f${fold}.log" 2>&1

    echo "[$(date)] Completed: ${way} ${pct}% f${fold}"
}

run_all_folds() {
    local way=$1
    local exp_name=$2
    local pct=$3

    for fold in 0 1 2 3; do
        if ! is_fold_done "$exp_name" "$pct" "$fold"; then
            wait_for_gpu_memory 6000
            run_fold "$way" "$exp_name" "$pct" "$fold"
        fi
    done
}

run_test() {
    local way=$1
    local exp_name=$2
    local pct=$3

    echo "[$(date)] Testing: ${way} ${pct}%"
    cd "$BASE_DIR"

    python -m core.test --task 1 --way "$way" \
        --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold -1 \
        > "${LOG_DIR}/test_${way}_${pct}.log" 2>&1

    echo "[$(date)] Test done: ${way} ${pct}%"
}

# Wait for current 20% runs to finish
echo "[$(date)] Waiting for current 20% experiments to complete..."
while true; do
    mms_20_done=true
    unimatch_20_done=true

    for fold in 0 1 2 3; do
        if ! is_fold_done "MMS" "20" "$fold"; then
            mms_20_done=false
        fi
        if ! is_fold_done "UniMatch_official" "20" "$fold"; then
            unimatch_20_done=false
        fi
    done

    if $mms_20_done && $unimatch_20_done; then
        break
    fi
    sleep 300
done

# Test 20%
run_test "mms" "MMS" "20"
run_test "unimatch_official" "UniMatch_official" "20"

# Run remaining ratios
for pct in 5 10 40; do
    run_all_folds "mms" "MMS" "$pct"
    run_test "mms" "MMS" "$pct"

    run_all_folds "unimatch_official" "UniMatch_official" "$pct"
    run_test "unimatch_official" "UniMatch_official" "$pct"
done

echo "[$(date)] All new baseline experiments completed!"
