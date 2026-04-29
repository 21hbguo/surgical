#!/usr/bin/env python3
import json
import os
import subprocess
import time
from datetime import datetime

PROJECT_DIR = "/home/guo/project/ssl4mis/code_all"
PROGRESS_JSON = "/home/guo/.openclaw/workspace/memory/v2_progress.json"
HISTORY_JSONL = "/home/guo/.openclaw/workspace/memory/v2_iteration_history.jsonl"
LOCK_FILE = os.path.join(PROJECT_DIR, ".oc_state", "v2_auto_iterate.lock")
THRESHOLD = 0.77
CHECK_SECONDS = 60


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_history(event):
    os.makedirs(os.path.dirname(HISTORY_JSONL), exist_ok=True)
    with open(HISTORY_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def pid_alive(pid):
    return isinstance(pid, int) and os.path.exists(f"/proc/{pid}")


def acquire_lock():
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    if os.path.exists(LOCK_FILE):
        old = read_json(LOCK_FILE, {})
        old_pid = old.get("pid")
        if pid_alive(old_pid):
            return False
    write_json(LOCK_FILE, {"pid": os.getpid(), "started_at": now()})
    return True


def choose_config(best, round_idx):
    # 简单启发式：best 越低，原型约束越强；后续逐步拉长训练
    ladder = [
        {"lr": 2e-5, "proto_pixel_weight": 0.15, "max_iterations": 30000},
        {"lr": 2e-5, "proto_pixel_weight": 0.20, "max_iterations": 35000},
        {"lr": 1.5e-5, "proto_pixel_weight": 0.20, "max_iterations": 40000},
        {"lr": 1.5e-5, "proto_pixel_weight": 0.25, "max_iterations": 40000},
    ]
    idx = min(max(round_idx - 1, 0), len(ladder) - 1)

    cfg = dict(ladder[idx])
    if best is not None:
        if best < 0.74:
            cfg["proto_pixel_weight"] = 0.25
            cfg["lr"] = 1.5e-5
        elif best < 0.75:
            cfg["proto_pixel_weight"] = max(cfg["proto_pixel_weight"], 0.20)
        elif best < 0.765:
            cfg["max_iterations"] = max(cfg["max_iterations"], 40000)
    return cfg


def launch_round(round_idx, cfg):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(PROJECT_DIR, ".oc_logs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, ".oc_state"), exist_ok=True)

    log_path = os.path.join(PROJECT_DIR, ".oc_logs", f"fully_proto_v2_round{round_idx}_{ts}.log")
    watcher_log = os.path.join(PROJECT_DIR, ".oc_logs", f"fully_proto_v2_round{round_idx}_watcher_{ts}.log")

    run_cmd = (
        "CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -m core.train "
        "--task 2 --fold 0 --way fully_proto_v2 "
        "--exp endovis2017/FullyProto_v2_auto "
        "--labeled_num 20 --normalize 255 --pretrain resnet --model resnet "
        f"--lr {cfg['lr']} --max_iterations {cfg['max_iterations']} --val_iter 300 "
        f"--proto_pixel_weight {cfg['proto_pixel_weight']}"
    )

    train_proc = subprocess.Popen(
        ["bash", "-lc", f"cd {PROJECT_DIR} && nohup bash -lc \"{run_cmd}\" > '{log_path}' 2>&1 & echo $!"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = train_proc.communicate(timeout=30)
    if train_proc.returncode != 0:
        raise RuntimeError(f"launch train failed: {err.strip()}")

    train_pid = int(out.strip().splitlines()[-1])

    watch_cmd = (
        f"cd {PROJECT_DIR} && nohup python3 tools/v2_watchdog.py "
        f"--pid {train_pid} --log '{log_path}' --progress-json '{PROGRESS_JSON}' "
        f"--threshold {THRESHOLD} --check-interval 20 --heartbeat-seconds 600 "
        f"> '{watcher_log}' 2>&1 & echo $!"
    )

    watch_proc = subprocess.Popen(
        ["bash", "-lc", watch_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wout, werr = watch_proc.communicate(timeout=30)
    if watch_proc.returncode != 0:
        raise RuntimeError(f"launch watcher failed: {werr.strip()}")
    watch_pid = int(wout.strip().splitlines()[-1])

    state = {
        "started_at": now(),
        "project_path": PROJECT_DIR,
        "run_cmd": run_cmd,
        "log_path": log_path,
        "latest_iter": None,
        "latest_dice": None,
        "best_dice": None,
        "status": "running",
        "note": f"auto iteration round {round_idx} launched",
        "stop_threshold": THRESHOLD,
        "stop_triggered": False,
        "train_pid": train_pid,
        "watcher_log": watcher_log,
        "watch_pid": watch_pid,
        "auto_round": round_idx,
        "cfg": cfg,
        "updated_at": now(),
    }
    write_json(PROGRESS_JSON, state)

    append_history({
        "time": now(),
        "event": "round_launched",
        "round": round_idx,
        "cfg": cfg,
        "train_pid": train_pid,
        "watch_pid": watch_pid,
        "log_path": log_path,
    })


def main_loop():
    append_history({"time": now(), "event": "auto_iterate_started", "threshold": THRESHOLD})
    while True:
        state = read_json(PROGRESS_JSON, {})
        status = state.get("status")
        best = state.get("best_dice")
        round_idx = int(state.get("auto_round") or 0)

        if isinstance(best, (int, float)) and best >= THRESHOLD:
            state["status"] = "done"
            state["stop_triggered"] = True
            state["note"] = f"target reached in auto-iterate: best_dice={best:.4f} >= {THRESHOLD:.2f}"
            state["updated_at"] = now()
            write_json(PROGRESS_JSON, state)
            append_history({"time": now(), "event": "target_reached", "best_dice": best})
            break

        if status == "running":
            train_pid = state.get("train_pid")
            if pid_alive(train_pid):
                time.sleep(CHECK_SECONDS)
                continue
            # 进程已不在但状态还没切换，强制改为 done 后进入下一轮
            state["status"] = "done"
            state["note"] = f"detected stale running state: train_pid {train_pid} not alive; mark done and iterate"
            state["updated_at"] = now()
            write_json(PROGRESS_JSON, state)

        # done/error 且没达到阈值 -> 自动开下一轮
        next_round = round_idx + 1 if round_idx >= 1 else 1
        cfg = choose_config(best if isinstance(best, (int, float)) else None, next_round)

        # 记录“自动分析”结论
        append_history({
            "time": now(),
            "event": "auto_analysis",
            "prev_best": best,
            "prev_status": status,
            "decision": "launch_next_round",
            "next_round": next_round,
            "next_cfg": cfg,
        })

        try:
            launch_round(next_round, cfg)
        except Exception as e:
            fail = read_json(PROGRESS_JSON, {})
            fail["status"] = "error"
            fail["note"] = f"auto launch failed: {e}"
            fail["updated_at"] = now()
            write_json(PROGRESS_JSON, fail)
            append_history({"time": now(), "event": "launch_failed", "error": str(e)})
            time.sleep(120)


if __name__ == "__main__":
    ok = acquire_lock()
    if not ok:
        print("another auto iterator is running; exit")
        raise SystemExit(0)
    main_loop()
