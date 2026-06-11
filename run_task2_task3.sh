#!/bin/bash
# Run Task 2 (part segmentation, 4 classes) and Task 3 (type segmentation, 7 classes)
# Priority: Task 2 GeoRisk-SPC first, then baselines, then Task 3

set -e
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/guo/project/ssl4mis/code_all_vibe_v2"
RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval"
PRED_BASE="/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval"
LOG_DIR="/tmp"
LABEL_RATIOS=(20 5 10 40)

wait_for_gpu_memory() {
    local threshold=${1:-6000}
    while true; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')
        if [ "$used" -lt "$threshold" ]; then
            break
        fi
        sleep 120
    done
}

# Check if fold is done by looking for model_best.pth in any matching directory
is_fold_done() {
    local task=$1
    local exp_name=$2
    local pct=$3
    local fold=$4
    local base="${RESULT_BASE}/task${task}/${exp_name}"
    # Find directory matching pattern
    local dir=$(ls -d ${base}/${pct}_labeled_* 2>/dev/null | head -1)
    [ -n "$dir" ] && [ -f "$dir/f${fold}/model_best.pth" ]
}

is_test_done() {
    local task=$1
    local exp_name=$2
    local pct=$3
    local fold=$4
    local base="${PRED_BASE}/task${task}/${exp_name}"
    local dir=$(ls -d ${base}/${pct}_labeled_* 2>/dev/null | head -1)
    [ -n "$dir" ] && [ -f "$dir/f${fold}/test_metrics.csv" ]
}

run_experiment() {
    local task=$1
    local way=$2
    local exp_name=$3
    shift 3
    local extra_args="$@"

    for pct in "${LABEL_RATIOS[@]}"; do
        for fold in 0 1 2 3; do
            # Training
            if is_fold_done "$task" "$exp_name" "$pct" "$fold"; then
                echo "[$(date)] Skip train task${task} ${way} ${pct}% f${fold}"
            else
                wait_for_gpu_memory 6000
                echo "[$(date)] Train task${task} ${way} ${pct}% f${fold}"
                cd "$BASE_DIR"
                python -m core.train --task "$task" --way "$way" \
                    --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
                    $extra_args \
                    > "${LOG_DIR}/task${task}_${way}_${pct}_f${fold}.log" 2>&1
                echo "[$(date)] Done train task${task} ${way} ${pct}% f${fold}"
            fi

            # Testing
            if is_test_done "$task" "$exp_name" "$pct" "$fold"; then
                echo "[$(date)] Skip test task${task} ${way} ${pct}% f${fold}"
            else
                wait_for_gpu_memory 6000
                echo "[$(date)] Test task${task} ${way} ${pct}% f${fold}"
                cd "$BASE_DIR"
                python -m core.test --task "$task" --way "$way" \
                    --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
                    $extra_args \
                    > "${LOG_DIR}/task${task}_${way}_${pct}_f${fold}_test.log" 2>&1
                echo "[$(date)] Done test task${task} ${way} ${pct}% f${fold}"
            fi
        done
    done
}

echo "[$(date)] ========== Starting Task 2 & Task 3 experiments =========="

# ==================== TASK 2 ====================
echo "[$(date)] ========== TASK 2: Part Segmentation (4 classes) =========="

# GeoRisk-SPC DGv4 (our method - highest priority)
echo "[$(date)] === Task 2: GeoRisk-SPC-DGv4 ==="
run_experiment 2 georisk_spc_dgv4 GeoRiskSPC_DGv4 --use_depth 1

# SegMatch
echo "[$(date)] === Task 2: SegMatch ==="
run_experiment 2 segmatch_official SegMatch

# UniMatch
echo "[$(date)] === Task 2: UniMatch ==="
run_experiment 2 unimatch_official UniMatch

# CPS
echo "[$(date)] === Task 2: CPS ==="
run_experiment 2 cps CPS

# U2PL
echo "[$(date)] === Task 2: U2PL ==="
run_experiment 2 u2pl U2PL

# CorrMatch
echo "[$(date)] === Task 2: CorrMatch ==="
run_experiment 2 corrmatch CorrMatch

# CW-BASS
echo "[$(date)] === Task 2: CW-BASS ==="
run_experiment 2 cwbass CW-BASS

# ==================== TASK 3 ====================
echo "[$(date)] ========== TASK 3: Type Segmentation (7 classes) =========="

# Fully supervised (upper bound)
echo "[$(date)] === Task 3: Fully ==="
run_experiment 3 fully Fully

# MT
echo "[$(date)] === Task 3: MT ==="
run_experiment 3 mt MT

# UAMT
echo "[$(date)] === Task 3: UAMT ==="
run_experiment 3 uamt UAMT

# URPC
echo "[$(date)] === Task 3: URPC ==="
run_experiment 3 urpc URPC

# MT-DGv4
echo "[$(date)] === Task 3: MT-DGv4 ==="
run_experiment 3 mt_depth_guider_v4 MT_depth_guider_v4 --use_depth 1

# GeoRisk-SPC DGv4
echo "[$(date)] === Task 3: GeoRisk-SPC-DGv4 ==="
run_experiment 3 georisk_spc_dgv4 GeoRiskSPC_DGv4 --use_depth 1

# SegMatch
echo "[$(date)] === Task 3: SegMatch ==="
run_experiment 3 segmatch_official SegMatch

# UniMatch
echo "[$(date)] === Task 3: UniMatch ==="
run_experiment 3 unimatch_official UniMatch

# CPS
echo "[$(date)] === Task 3: CPS ==="
run_experiment 3 cps CPS

# U2PL
echo "[$(date)] === Task 3: U2PL ==="
run_experiment 3 u2pl U2PL

# CorrMatch
echo "[$(date)] === Task 3: CorrMatch ==="
run_experiment 3 corrmatch CorrMatch

echo "[$(date)] ========== All Task 2 & Task 3 experiments completed =========="
