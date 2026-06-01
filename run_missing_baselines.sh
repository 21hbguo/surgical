#!/bin/bash
# Run all missing baseline experiments sequentially
# Started: 2026-05-30
set -e
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/guo/project/ssl4mis/code_all_vibe_v2"
RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
LOG_DIR="/tmp/baseline_logs"
mkdir -p "$LOG_DIR"

cd "$BASE_DIR"

is_fold_done() {
    local exp_name=$1
    local pct=$2
    local fold=$3
    local result_dir="${RESULT_BASE}/${exp_name}/${pct}_labeled_lr1e-4_s_unet"
    [ -f "$result_dir/f${fold}/model_best.pth" ]
}

run_fold() {
    local way=$1
    local exp_name=$2
    local pct=$3
    local fold=$4

    if is_fold_done "$exp_name" "$pct" "$fold"; then
        echo "[$(date)] SKIP ${way} ${pct}% f${fold} (done)"
        return 0
    fi

    echo "[$(date)] TRAIN ${way} ${pct}% f${fold}"
    python -m core.train --task 1 --way "$way" \
        --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
        2>&1 | tee "${LOG_DIR}/${way}_${pct}_f${fold}.log"
    echo "[$(date)] DONE ${way} ${pct}% f${fold}"
}

run_all_folds() {
    local way=$1
    local exp_name=$2
    local pct=$3
    for fold in 0 1 2 3; do
        run_fold "$way" "$exp_name" "$pct" "$fold"
    done
}

run_test() {
    local way=$1
    local exp_name=$2
    local pct=$3
    echo "[$(date)] TEST ${way} ${pct}%"
    python -m core.test --task 1 --way "$way" \
        --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold -1 \
        2>&1 | tee "${LOG_DIR}/test_${way}_${pct}.log"
    echo "[$(date)] TEST DONE ${way} ${pct}%"
}

# ============================================================
# Phase 1: CPS (5% f3 already running separately)
# ============================================================
echo "===== Phase 1: CPS ====="

# Wait for CPS 5% f3 (started separately) by checking checkpoint file
while ! is_fold_done "CPS" "5" "3"; do
    if ! pgrep -f "core.train.*cps.*fold 3" > /dev/null 2>&1; then
        echo "[$(date)] ERROR: CPS 5% f3 process died without producing checkpoint"
        exit 1
    fi
    echo "[$(date)] Waiting for CPS 5% f3..."
    sleep 120
done

run_test "cps" "CPS" "5"

run_all_folds "cps" "CPS" "10"
run_test "cps" "CPS" "10"

run_all_folds "cps" "CPS" "40"
run_test "cps" "CPS" "40"

# ============================================================
# Phase 2: MMS
# ============================================================
echo "===== Phase 2: MMS ====="

run_all_folds "mms" "MMS" "20"
run_test "mms" "MMS" "20"

run_all_folds "mms" "MMS" "5"
run_test "mms" "MMS" "5"

run_all_folds "mms" "MMS" "10"
run_test "mms" "MMS" "10"

run_all_folds "mms" "MMS" "40"
run_test "mms" "MMS" "40"

# ============================================================
# Phase 3: UniMatch official
# ============================================================
echo "===== Phase 3: UniMatch official ====="

# 20% f2, f3 (f0, f1 already done)
run_fold "unimatch_official" "UniMatch_official" "20" "2"
run_fold "unimatch_official" "UniMatch_official" "20" "3"
run_test "unimatch_official" "UniMatch_official" "20"

run_all_folds "unimatch_official" "UniMatch_official" "5"
run_test "unimatch_official" "UniMatch_official" "5"

run_all_folds "unimatch_official" "UniMatch_official" "10"
run_test "unimatch_official" "UniMatch_official" "10"

run_all_folds "unimatch_official" "UniMatch_official" "40"
run_test "unimatch_official" "UniMatch_official" "40"

# ============================================================
# Phase 4: SegMatch official
# ============================================================
echo "===== Phase 4: SegMatch official ====="

for pct in 5 10 20 40; do
    run_all_folds "segmatch_official" "SegMatch_official" "$pct"
    run_test "segmatch_official" "SegMatch_official" "$pct"
done

echo "===== ALL BASELINE EXPERIMENTS COMPLETE ====="
echo "[$(date)] Done!"
