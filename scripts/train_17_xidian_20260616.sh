#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
TMPDIR="$DIR/outputs/.tmp"
mkdir -p "$TMPDIR"
export TMPDIR
MODE="${MODE:-train}"
GPU_LIST="${GPU_LIST:-[0,1,2,3,4,5,6,7]}"
GPU_LIST="${GPU_LIST//[[:space:]]/}"
GPU_LIST="${GPU_LIST#[}"
GPU_LIST="${GPU_LIST%]}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
QUEUE_FILE="$TMPDIR/train_17_xidian_20260616.queue"
LOCK_FILE="$TMPDIR/train_17_xidian_20260616.lock"
LOG_DIR="${LOG_DIR:-$TMPDIR/train_17_xidian_20260616_logs}"
mkdir -p "$LOG_DIR"
TRAIN_RESULT_ROOT="${TRAIN_RESULT_ROOT:-$DIR/../result_train}"
PREDICT_RESULT_ROOT="${PREDICT_RESULT_ROOT:-$DIR/../result_predict}"
PRETRAIN="${PRETRAIN:-resnet}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_LABELED_NUMS="${DEFAULT_LABELED_NUMS:-100,40}"
JOB_SPECS=(
"task=1|fold=0|way=ternaus|exp=endovis2017_h5_224_224/TernausNet16|labeled_nums=100|extra="
"task=1|fold=0|way=fully|exp=endovis2017_h5_224_224/Fully|labeled_nums=|extra="
"task=1|fold=0|way=mt|exp=endovis2017_h5_224_224/MT|labeled_nums=40|extra="
"task=1|fold=0|way=uamt|exp=endovis2017_h5_224_224/UAMT|labeled_nums=40|extra="
"task=1|fold=0|way=urpc|exp=endovis2017_h5_224_224/URPC|labeled_nums=40|extra="
"task=1|fold=0|way=georisk_spc_dgv4|exp=endovis2017_h5_224_224/GeoRiskSPC_DGv4|labeled_nums=40|extra=--use_depth 1"
"task=2|fold=0|way=ternaus|exp=endovis2017_h5_224_224/TernausNet16|labeled_nums=100|extra="
"task=2|fold=0|way=fully|exp=endovis2017_h5_224_224/Fully|labeled_nums=|extra="
"task=2|fold=0|way=mt|exp=endovis2017_h5_224_224/MT|labeled_nums=40|extra="
"task=2|fold=0|way=uamt|exp=endovis2017_h5_224_224/UAMT|labeled_nums=40|extra="
"task=2|fold=0|way=urpc|exp=endovis2017_h5_224_224/URPC|labeled_nums=40|extra="
"task=2|fold=0|way=georisk_spc_dgv4|exp=endovis2017_h5_224_224/GeoRiskSPC_DGv4|labeled_nums=40|extra=--use_depth 1"
)
JOBS=()
for spec in "${JOB_SPECS[@]}"; do
    IFS='|' read -r task_part fold_part way_part exp_part labeled_nums_part extra_part <<< "$spec"
    task="${task_part#task=}"
    fold="${fold_part#fold=}"
    way="${way_part#way=}"
    exp="${exp_part#exp=}"
    labeled_nums="${labeled_nums_part#labeled_nums=}"
    extra="${extra_part#extra=}"
    if [ -z "$labeled_nums" ]; then
        labeled_nums="$DEFAULT_LABELED_NUMS"
    fi
    IFS=',' read -r -a labeled_num_list <<< "$labeled_nums"
    for labeled_num in "${labeled_num_list[@]}"; do
        JOBS+=("task=$task|fold=$fold|way=$way|exp=$exp|labeled_num=$labeled_num|extra=$extra")
    done
done
echo 0 > "$QUEUE_FILE"
build_output_dir(){
python - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" <<'PY'
import os, sys
from types import SimpleNamespace
from utils.common import build_run_output_dir
from models.factory import resolve_default_model_name
mode,root,task,exp,labeled_num,fold,way,use_depth,pretrain=sys.argv[1:10]
model=resolve_default_model_name(way, pretrain)
args=SimpleNamespace(
    exp=exp,
    task=int(task),
    sampling="interval",
    model=model,
    filter_num=16,
    use_depth=None if use_depth=="" else int(use_depth),
    strong="s",
    pretrain=pretrain,
    way=way,
    labeled_num=float(labeled_num),
    lr=1e-4,
    fold=int(fold),
    train_result_root=root,
    predict_result_root=root,
)
print(build_run_output_dir(args, mode=mode, fold=None if int(fold) == -1 else int(fold)))
PY
}
run_job(){
    local gpu="$1"
    local task="$2"
    local fold="$3"
    local way="$4"
    local exp="$5"
    local labeled_num="$6"
    local extra="$7"
    local use_depth=""
    if [[ "$extra" =~ --use_depth[[:space:]]+([0-9]+) ]]; then
        use_depth="${BASH_REMATCH[1]}"
    fi
    local train_dir
    local test_dir
    train_dir="$(build_output_dir train "$TRAIN_RESULT_ROOT" "$task" "$exp" "$labeled_num" "$fold" "$way" "$use_depth" "$PRETRAIN")"
    test_dir="$(build_output_dir test "$PREDICT_RESULT_ROOT" "$task" "$exp" "$labeled_num" "$fold" "$way" "$use_depth" "$PRETRAIN")"
    local train_log="$LOG_DIR/${task}_${fold}_${way}_${labeled_num}_train.log"
    local test_log="$LOG_DIR/${task}_${fold}_${way}_${labeled_num}_test.log"
    if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
        if [ -f "$train_dir/model_best.pth" ] && [ -f "$train_dir/model_final.pth" ]; then
            echo "[$(date)] Skip train task${task} fold${fold} ${way} labeled${labeled_num}"
        else
            echo "[$(date)] Train gpu${gpu} task${task} fold${fold} ${way} labeled${labeled_num}"
            if [ "$DRY_RUN" = "1" ]; then
                echo "CUDA_VISIBLE_DEVICES=$gpu python -m core.train --task $task --fold $fold --way $way --exp $exp --labeled_num $labeled_num $extra"
            else
                CUDA_VISIBLE_DEVICES="$gpu" python -m core.train --task "$task" --fold "$fold" --way "$way" --exp "$exp" --labeled_num "$labeled_num" --pretrain "$PRETRAIN" $extra > "$train_log" 2>&1
            fi
            echo "[$(date)] Done train gpu${gpu} task${task} fold${fold} ${way} labeled${labeled_num}"
        fi
    fi
    if [ "$MODE" = "test" ] || [ "$MODE" = "both" ]; then
        if [ -f "$test_dir/test_metrics.csv" ]; then
            echo "[$(date)] Skip test task${task} fold${fold} ${way} labeled${labeled_num}"
        else
            echo "[$(date)] Test gpu${gpu} task${task} fold${fold} ${way} labeled${labeled_num}"
            if [ "$DRY_RUN" = "1" ]; then
                echo "CUDA_VISIBLE_DEVICES=$gpu python -m core.test --task $task --fold $fold --way $way --exp $exp --labeled_num $labeled_num $extra"
            else
                CUDA_VISIBLE_DEVICES="$gpu" python -m core.test --task "$task" --fold "$fold" --way "$way" --exp "$exp" --labeled_num "$labeled_num" --pretrain "$PRETRAIN" $extra > "$test_log" 2>&1
            fi
            echo "[$(date)] Done test gpu${gpu} task${task} fold${fold} ${way} labeled${labeled_num}"
        fi
    fi
}
worker(){
    local gpu="$1"
    local idx
    local line
    local task
    local fold
    local way
    local exp
    local labeled_num
    local extra
    while true; do
        exec 9>"$LOCK_FILE"
        flock 9
        idx=$(cat "$QUEUE_FILE")
        if [ "$idx" -ge "${#JOBS[@]}" ]; then
            flock -u 9
            exec 9>&-
            break
        fi
        echo $((idx + 1)) > "$QUEUE_FILE"
        flock -u 9
        exec 9>&-
        line="${JOBS[$idx]}"
        IFS='|' read -r task_part fold_part way_part exp_part labeled_num_part extra_part <<< "$line"
        task="${task_part#task=}"
        fold="${fold_part#fold=}"
        way="${way_part#way=}"
        exp="${exp_part#exp=}"
        labeled_num="${labeled_num_part#labeled_num=}"
        extra="${extra_part#extra=}"
        run_job "$gpu" "$task" "$fold" "$way" "$exp" "$labeled_num" "$extra"
    done
}
for gpu in "${GPUS[@]}"; do
    worker "$gpu" &
done
wait
