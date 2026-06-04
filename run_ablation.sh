#!/bin/bash
# Run ablation experiments for GeoRisk-SPC
# Ablations: remove L_cons, remove L_bd, random mask, tau_r sweep, tau_c sweep

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
    local exp_name=$1
    local pct=$2
    local fold=$3
    shift 3
    local extra_args="$@"

    # Training
    if is_fold_done "$exp_name" "$pct" "$fold"; then
        echo "[$(date)] Skipping training ${exp_name} ${pct}% f${fold} (already done)"
    else
        echo "[$(date)] Starting training: ${exp_name} ${pct}% f${fold}"
        cd "$BASE_DIR"
        python -m core.train --task 1 --way georisk_spc \
            --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
            $extra_args \
            > "${LOG_DIR}/${exp_name}_${pct}_f${fold}.log" 2>&1
        echo "[$(date)] Completed training: ${exp_name} ${pct}% f${fold}"
    fi

    # Testing
    if is_test_done "$exp_name" "$pct" "$fold"; then
        echo "[$(date)] Skipping test ${exp_name} ${pct}% f${fold} (already done)"
    else
        echo "[$(date)] Starting test: ${exp_name} ${pct}% f${fold}"
        cd "$BASE_DIR"
        python -m core.test --task 1 --way georisk_spc \
            --exp "endovis2017/${exp_name}" --labeled_num "$pct" --fold "$fold" \
            $extra_args \
            > "${LOG_DIR}/${exp_name}_${pct}_f${fold}_test.log" 2>&1
        echo "[$(date)] Completed test: ${exp_name} ${pct}% f${fold}"
    fi
}

run_ablation() {
    local exp_name=$1
    local pct=$2
    shift 2
    local extra_args="$@"

    for fold in 0 1 2 3; do
        wait_for_gpu_memory 8000
        run_fold "$exp_name" "$pct" "$fold" $extra_args
    done
}

echo "[$(date)] Starting ablation experiments"

# Wait for CW-BASS to complete
while pgrep -f "cwbass" > /dev/null 2>&1; do
    echo "[$(date)] Waiting for CW-BASS to complete..."
    sleep 300
done

echo "[$(date)] CW-BASS completed, starting ablations"

# Ablation 1: Remove L_cons (consistency loss)
echo "[$(date)] === Ablation: w/o L_cons ==="
for pct in 20; do
    run_ablation "Ablation_no_cons" "$pct" --risk_cons_weight 0.0
done

# Ablation 2: Remove L_bd (boundary loss)
echo "[$(date)] === Ablation: w/o L_bd ==="
for pct in 20; do
    run_ablation "Ablation_no_bd" "$pct" --risk_bd_weight 0.0
done

# Ablation 3: Random mask (disable risk map by setting tau_r=1.0)
echo "[$(date)] === Ablation: random mask (tau_r=1.0) ==="
for pct in 20; do
    run_ablation "Ablation_random_mask" "$pct" --risk_tau_r 1.0
done

# Ablation 4: tau_r sweep
echo "[$(date)] === Ablation: tau_r sweep ==="
for tau in 0.3 0.7; do
    for pct in 20; do
        run_ablation "Ablation_tau_r_${tau}" "$pct" --risk_tau_r "$tau"
    done
done

# Ablation 5: tau_c sweep
echo "[$(date)] === Ablation: tau_c sweep ==="
for tau in 0.7 0.95; do
    for pct in 20; do
        run_ablation "Ablation_tau_c_${tau}" "$pct" --risk_tau_c "$tau"
    done
done

echo "[$(date)] All ablation experiments completed"
