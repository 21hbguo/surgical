#!/bin/bash
# Run official baseline experiments: U2PL, CorrMatch, CW-BASS
# Runs one experiment at a time, with test evaluation after each fold

set -e
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/guo/project/ssl4mis/code_all_vibe_v2"
RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
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
    local pct=$2
    local fold=$3
    local result_dir="${RESULT_BASE}/${exp_name}/${pct}_labeled_lr1e-4_s_unet"
    [ -f "$result_dir/f${fold}/model_final.pth" ]
}

is_test_done() {
    local exp_name=$1
    local pct=$2
    local fold=$3
    local result_dir="/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/task1/${exp_name}/${pct}_labeled_lr1e-4_s_unet/f${fold}"
    [ -f "$result_dir/test_metrics.csv" ]
}

run_fold() {
    local way=$1
    local exp_name=$2
    local pct=$3
    local fold=$4

    # Training
    if is_fold_done "$exp_name" "$pct" "$fold"; then
        echo "[$(date)] Skipping training ${way} ${pct}% f${fold} (already done)"
    else
        echo "[$(date)] Starting training: ${way} ${pct}% f${fold}"
        cd "$BASE_DIR"
        python -m core.train --task 1 --way "$way" \
            --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
            > "${LOG_DIR}/${way}_${pct}_f${fold}.log" 2>&1
        echo "[$(date)] Completed training: ${way} ${pct}% f${fold}"
    fi

    # Testing
    if is_test_done "$exp_name" "$pct" "$fold"; then
        echo "[$(date)] Skipping test ${way} ${pct}% f${fold} (already done)"
    else
        echo "[$(date)] Starting test: ${way} ${pct}% f${fold}"
        cd "$BASE_DIR"
        python -m core.test --task 1 --way "$way" \
            --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
            > "${LOG_DIR}/${way}_${pct}_f${fold}_test.log" 2>&1
        echo "[$(date)] Completed test: ${way} ${pct}% f${fold}"
    fi
}

run_all_folds() {
    local way=$1
    local exp_name=$2

    for pct in 5 10 20 40; do
        for fold in 0 1 2 3; do
            wait_for_gpu_memory 8000
            run_fold "$way" "$exp_name" "$pct" "$fold"
        done
    done
}

echo "[$(date)] Starting official baseline experiments"

# Check if previous run is still running
if pgrep -f "run_official_baselines.sh" | grep -v $$ > /dev/null; then
    echo "[$(date)] Warning: Another instance is running. Waiting..."
    sleep 60
fi

# U2PL official
echo "[$(date)] === U2PL (official) ==="
run_all_folds "u2pl" "U2PL_official"

# CorrMatch official
echo "[$(date)] === CorrMatch (official) ==="
run_all_folds "corrmatch" "CorrMatch_official"

# CW-BASS official
echo "[$(date)] === CW-BASS (official) ==="
run_all_folds "cwbass" "CW-BASS_official"

echo "[$(date)] All official baseline experiments completed"
