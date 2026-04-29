#!/usr/bin/env python3
import argparse
import json
import os
import re
import signal
import time
from datetime import datetime

DICE_RE = re.compile(r"Iter\s+(\d+):\s+Dice=([0-9.]+)\s+Best=([0-9.]+)")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def proc_alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def parse_latest_metrics(log_path):
    latest_iter = None
    latest_dice = None
    best_dice = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = DICE_RE.search(line)
                if m:
                    latest_iter = int(m.group(1))
                    latest_dice = float(m.group(2))
                    best_dice = float(m.group(3))
    except FileNotFoundError:
        return None, None, None
    return latest_iter, latest_dice, best_dice


def terminate(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return
    for _ in range(15):
        if not proc_alive(pid):
            return
        time.sleep(1)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--progress-json", required=True)
    p.add_argument("--threshold", type=float, default=0.77)
    p.add_argument("--check-interval", type=int, default=20)
    p.add_argument("--heartbeat-seconds", type=int, default=600)
    args = p.parse_args()

    pid = args.pid
    threshold = float(args.threshold)
    last_iter = None
    last_best = None
    last_heartbeat = 0

    while True:
        latest_iter, latest_dice, best_dice = parse_latest_metrics(args.log)
        state = read_json(args.progress_json)
        changed = False

        if latest_iter is not None and latest_iter != state.get("latest_iter"):
            state["latest_iter"] = latest_iter
            changed = True
        if latest_dice is not None and latest_dice != state.get("latest_dice"):
            state["latest_dice"] = latest_dice
            changed = True
        if best_dice is not None and best_dice != state.get("best_dice"):
            state["best_dice"] = best_dice
            changed = True

        # 阈值触发自动停止
        if best_dice is not None and best_dice > threshold and proc_alive(pid):
            terminate(pid)
            state["status"] = "done"
            state["stop_triggered"] = True
            state["note"] = f"threshold reached: best_dice={best_dice:.4f} > {threshold:.2f}, training stopped"
            state["latest_iter"] = latest_iter
            state["latest_dice"] = latest_dice
            state["best_dice"] = best_dice
            state["updated_at"] = now_str()
            write_json(args.progress_json, state)
            break

        alive = proc_alive(pid)
        now_ts = time.time()

        if not alive:
            state["status"] = "done"
            state["stop_triggered"] = bool(state.get("stop_triggered", False))
            if state["stop_triggered"]:
                state["note"] = state.get("note", "stopped after threshold triggered")
            else:
                bd = best_dice if best_dice is not None else state.get("best_dice")
                if bd is not None:
                    state["note"] = f"process exited before threshold, best_dice={bd:.4f}"
                else:
                    state["note"] = "process exited before threshold, no Dice metric parsed"
            state["updated_at"] = now_str()
            write_json(args.progress_json, state)
            break

        # 每10分钟心跳汇报（写入 progress.json note）
        if now_ts - last_heartbeat >= args.heartbeat_seconds:
            if latest_iter is not None and best_dice is not None:
                state["note"] = (
                    f"heartbeat: running, iter={latest_iter}, latest_dice={latest_dice:.4f}, "
                    f"best_dice={best_dice:.4f}, threshold={threshold:.2f}"
                )
            else:
                state["note"] = f"heartbeat: running, waiting first Dice log, threshold={threshold:.2f}"
            changed = True
            last_heartbeat = now_ts

        if changed:
            state["status"] = "running"
            state["stop_threshold"] = threshold
            state["stop_triggered"] = bool(state.get("stop_triggered", False))
            if latest_iter is not None and best_dice is not None:
                state["note"] = f"running: iter={latest_iter}, latest_dice={latest_dice:.4f}, best_dice={best_dice:.4f}, threshold={threshold:.2f}"
            else:
                state["note"] = f"running: waiting first Dice log, threshold={threshold:.2f}"
            state["updated_at"] = now_str()
            write_json(args.progress_json, state)

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
