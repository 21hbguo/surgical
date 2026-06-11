#!/bin/bash
# Check Task 2/3 training progress
# Usage: bash check_training_progress.sh

RESULT_BASE="/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval"
CSV_PATH="/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/all_experiments_results_best.csv"
LABEL_RATIOS=(5 10 20 40)
FOLDS=(0 1 2 3)

echo "=== Task 2/3 Training Progress ==="
echo "Time: $(date)"
echo ""

for task in 2 3; do
    echo "--- Task $task ---"
    for exp_dir in ${RESULT_BASE}/task${task}/*/; do
        [ -d "$exp_dir" ] || continue
        method=$(basename "$exp_dir")
        total=0
        trained=0
        tested=0

        for pct in "${LABEL_RATIOS[@]}"; do
            labeled_dir=$(ls -d ${exp_dir}/${pct}_labeled_* 2>/dev/null | head -1)
            [ -z "$labeled_dir" ] && continue
            for fold in "${FOLDS[@]}"; do
                total=$((total + 1))
                if [ -f "$labeled_dir/f${fold}/model_best.pth" ]; then
                    trained=$((trained + 1))
                fi
            done
        done

        # Count tested folds from CSV
        tested=$(grep "^task${task}_${method}" "$CSV_PATH" 2>/dev/null | grep -v "ALL_Folds" | wc -l)

        if [ $total -gt 0 ]; then
            pct_train=$((trained * 100 / total))
            pct_test=$((tested * 100 / total))
            echo "  $method: train $trained/$total ($pct_train%), test $tested/$total ($pct_test%)"
        fi
    done
    echo ""
done

# Show current running process
echo "--- Current Process ---"
ps aux | grep "core.train" | grep -v grep | awk '{for(i=NF;i>=1;i--) if($i ~ /--/) break; for(i=i;i<=NF;i++) printf "%s ", $i; print ""}' | head -3
echo ""

# Show latest log
echo "--- Latest Training Log ---"
latest_log=$(ls -t /tmp/task*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "Log: $latest_log"
    tail -1 "$latest_log" 2>/dev/null
fi
