"""Compute boundary-aware segmentation metrics with fold-level aggregation.

Evaluates on per-fold held-out sequences (from task JSON), not shared test set.
Reports fold-level mean±sample std (ddof=1) for consistency with Dice tables.

Metrics: Dice, IoU, Boundary F1, HD95, ASSD.

Usage:
    python paper_lab/compute_boundary_metrics.py --task 1 --ratio 20
    python paper_lab/compute_boundary_metrics.py --task 1 --ratio 20 --methods GeoRisk-SPC MT
"""

import argparse
import csv
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.transforms import _normalize_array, _resize_numpy_array
from models.networks.unet import UNet_Base, UNet_DepthGuiderV4_GeoRiskSPC

# ============ Configuration ============
DATA_ROOT = "/home/guo/project/ssl4mis/data/endovis2017"
RESULT_ROOT_TPL = "/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task{}"
RESIZE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Method definitions: name -> (model_type, checkpoint_subpath_template)
# {ratio} is replaced at runtime
METHODS = {
    "GeoRisk-SPC": {
        "type": "georisk_dgv4",
        "path": "GeoRiskSPC_DGv4/{ratio}_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C",
    },
    "MT": {
        "type": "unet_base",
        "path": "MT/{ratio}_labeled_lr1e-4_s_unet",
    },
    "MT-DGv4": {
        "type": "unet_base",
        "path": "MT_depth_guider_v4/{ratio}_labeled_lr1e-4_s_unet_depth_guider_v4_depth1C",
    },
    "UAMT": {
        "type": "unet_base",
        "path": "UAMT/{ratio}_labeled_lr1e-4_s_unet",
    },
    "URPC": {
        "type": "unet_base",
        "path": "URPC/{ratio}_labeled_lr1e-4_s_unet_urpc",
    },
    "CPS": {
        "type": "unet_base",
        "path": "CPS/{ratio}_labeled_lr1e-4_s_unet",
    },
    "UniMatch": {
        "type": "unet_base",
        "path": "UniMatch/{ratio}_labeled_lr1e-4_s_unet",
    },
    "SegMatch": {
        "type": "unet_base",
        "path": "SegMatch/{ratio}_labeled_lr1e-4_s_unet",
    },
    "U2PL": {
        "type": "unet_base",
        "path": "U2PL/{ratio}_labeled_lr1e-4_s_unet",
    },
    "CorrMatch": {
        "type": "unet_base",
        "path": "CorrMatch/{ratio}_labeled_lr1e-4_s_unet",
    },
    "CW-BASS": {
        "type": "unet_base",
        "path": "CW-BASS/{ratio}_labeled_lr1e-4_s_unet",
    },
}

TASK_CONFIG = {
    1: {"num_classes": 2, "label_dir": "labels_task1_binary", "filter_num": 16},
    2: {"num_classes": 4, "label_dir": "labels_task2_part", "filter_num": 16},
    3: {"num_classes": 7, "label_dir": "labels_task3_type", "filter_num": 16},
}


# ============ Data loading ============
def load_fold_map(task):
    path = os.path.join(DATA_ROOT, f"task{task}.json")
    with open(path) as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["fold_seqs"].items()}


def load_test_list():
    path = os.path.join(DATA_ROOT, "test_slices.list")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_seq_id(case_name):
    """Extract sequence ID from case name like 'seq_1_frame225' -> 1"""
    m = re.match(r"seq_(\d+)_frame", case_name)
    return int(m.group(1)) if m else None


def load_sample(case, task):
    cfg = TASK_CONFIG[task]
    img_path = os.path.join(DATA_ROOT, "data", "images", f"{case}.png")
    lab_path = os.path.join(DATA_ROOT, "data", cfg["label_dir"], f"{case}.png")
    depth_path = os.path.join(DATA_ROOT, "data", "depth1c_slices_uint16", f"{case}.png")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_numpy_array(img, RESIZE_SIZE)
    img_norm = _normalize_array(img.copy(), method="255")

    lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    lab = _resize_numpy_array(lab, RESIZE_SIZE)

    depth = None
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / 65535.0 if depth.max() > 255 else depth / 255.0
        depth = _resize_numpy_array(depth, RESIZE_SIZE)

    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return {"image": img_t, "label": lab, "depth": depth, "case": case}


# ============ Model loading ============
def load_checkpoint(checkpoint_path, model_type, num_classes, filter_num):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    if model_type == "georisk_dgv4":
        model = UNet_DepthGuiderV4_GeoRiskSPC(
            in_chns=4, class_num=num_classes, filter_num=filter_num,
            dropout_rate=0.3, noise_std=0.1,
        )
    else:
        model = UNet_Base(in_chns=3, class_num=num_classes, filter_num=filter_num)

    model.load_state_dict(cleaned, strict=False)
    model.to(DEVICE).eval()
    return model


def run_inference(model, sample, model_type):
    image = sample["image"].to(DEVICE)
    if model_type == "georisk_dgv4":
        depth = sample["depth"]
        if depth is not None:
            depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            volume = torch.cat([image, depth_t], dim=1)
        else:
            volume = image
        with torch.no_grad():
            output = model(volume)
            if isinstance(output, tuple):
                output = output[0]
    else:
        with torch.no_grad():
            output = model(image)

    pred = torch.softmax(output, dim=1)[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
    return pred


# ============ Metrics ============
def compute_dice(pred, gt, smooth=1e-6):
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def compute_iou(pred, gt, smooth=1e-6):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + smooth) / (union + smooth)


def _get_boundary(mask, thickness=0):
    if thickness > 0:
        dilated = binary_dilation(mask, iterations=thickness)
        eroded = binary_erosion(mask, iterations=thickness)
        return dilated & ~eroded
    return mask ^ binary_erosion(mask)


def compute_boundary_f1(pred, gt, tolerance=2):
    pred_boundary = _get_boundary(pred, 0)
    gt_boundary = _get_boundary(gt, 0)

    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    gt_dt = distance_transform_edt(~gt)
    pred_dt = distance_transform_edt(~pred)

    precision_hits = pred_boundary & (gt_dt <= tolerance)
    precision = precision_hits.sum() / (pred_boundary.sum() + 1e-8)

    recall_hits = gt_boundary & (pred_dt <= tolerance)
    recall = recall_hits.sum() / (gt_boundary.sum() + 1e-8)

    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hd95(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_dt = distance_transform_edt(~pred)
    gt_dt = distance_transform_edt(~gt)

    pred_boundary = _get_boundary(pred, 0)
    gt_boundary = _get_boundary(gt, 0)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    d_pred_to_gt = gt_dt[pred_boundary]
    d_gt_to_pred = pred_dt[gt_boundary]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_assd(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_dt = distance_transform_edt(~pred)
    gt_dt = distance_transform_edt(~gt)

    pred_boundary = _get_boundary(pred, 0)
    gt_boundary = _get_boundary(gt, 0)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    d_pred_to_gt = gt_dt[pred_boundary].mean()
    d_gt_to_pred = pred_dt[gt_boundary].mean()
    return float((d_pred_to_gt + d_gt_to_pred) / 2.0)


def compute_all_metrics(pred, gt):
    """Compute all metrics for a single sample (foreground class)."""
    pred_fg = pred > 0
    gt_fg = gt > 0
    return {
        "Dice": compute_dice(pred_fg, gt_fg),
        "IoU": compute_iou(pred_fg, gt_fg),
        "BF1": compute_boundary_f1(pred_fg, gt_fg, tolerance=2),
        "HD95": compute_hd95(pred_fg, gt_fg),
        "ASSD": compute_assd(pred_fg, gt_fg),
    }


# ============ Main ============
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Compute boundary metrics with fold-level aggregation")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--ratio", type=int, default=20, choices=[5, 10, 20, 40])
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit test samples per fold (0=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto)")
    args = parser.parse_args()

    set_seed(42)
    task_cfg = TASK_CONFIG[args.task]
    fold_map = load_fold_map(args.task)
    all_test = load_test_list()
    result_root = RESULT_ROOT_TPL.format(args.task)

    # Group test cases by sequence
    test_by_seq = {}
    for case in all_test:
        seq_id = get_seq_id(case)
        if seq_id is not None:
            test_by_seq.setdefault(seq_id, []).append(case)

    methods_to_eval = args.methods if args.methods else list(METHODS.keys())
    print(f"Task: {args.task}, Ratio: {args.ratio}%, Methods: {methods_to_eval}")
    print(f"Fold map: {fold_map}")

    # CSV output
    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__),
                                   f"boundary_metrics_task{args.task}_{args.ratio}pct.csv")

    csv_rows = []

    for method_name in methods_to_eval:
        if method_name not in METHODS:
            print(f"Unknown method: {method_name}, skipping")
            continue

        method_cfg = METHODS[method_name]
        ckpt_subpath = method_cfg["path"].format(ratio=args.ratio)

        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        fold_results = {}  # fold_id -> mean metrics over that fold's test cases

        for fold_id, seqs in fold_map.items():
            fold_tag = f"f{fold_id}"
            ckpt_path = os.path.join(result_root, ckpt_subpath, fold_tag, "model_best.pth")
            if not os.path.exists(ckpt_path):
                print(f"  {fold_tag}: checkpoint not found ({ckpt_path})")
                continue

            # Get test cases for this fold's held-out sequences
            fold_cases = []
            for seq_id in seqs:
                fold_cases.extend(test_by_seq.get(seq_id, []))
            if not fold_cases:
                print(f"  {fold_tag}: no test cases for sequences {seqs}")
                continue
            if args.max_samples > 0:
                fold_cases = fold_cases[:args.max_samples]

            print(f"  {fold_tag}: {len(fold_cases)} test cases (seqs {seqs})")
            model = load_checkpoint(ckpt_path, method_cfg["type"],
                                    task_cfg["num_classes"], task_cfg["filter_num"])

            case_metrics = []
            for j, case in enumerate(fold_cases):
                sample = load_sample(case, args.task)
                pred = run_inference(model, sample, method_cfg["type"])
                metrics = compute_all_metrics(pred, sample["label"])
                case_metrics.append(metrics)

                if (j + 1) % 50 == 0:
                    avg_dice = np.nanmean([m["Dice"] for m in case_metrics])
                    print(f"    [{j+1}/{len(fold_cases)}] running avg Dice={avg_dice:.4f}")

            # Fold-level mean over cases
            fold_mean = {k: np.nanmean([m[k] for m in case_metrics])
                         for k in case_metrics[0]}
            fold_results[fold_id] = fold_mean

            avg_dice = fold_mean["Dice"]
            avg_hd95 = fold_mean["HD95"]
            print(f"  {fold_tag} done: Dice={avg_dice:.4f}, HD95={avg_hd95:.2f}")

            del model
            torch.cuda.empty_cache()

        if not fold_results:
            print(f"  No valid folds for {method_name}")
            continue

        # Fold-level mean±std (sample std, ddof=1)
        metric_keys = list(next(iter(fold_results.values())).keys())
        fold_ids = sorted(fold_results.keys())
        n_folds = len(fold_ids)

        final_mean = {}
        final_std = {}
        for k in metric_keys:
            vals = [fold_results[f][k] for f in fold_ids if not np.isnan(fold_results[f][k])]
            if vals:
                final_mean[k] = np.mean(vals)
                final_std[k] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            else:
                final_mean[k] = float("nan")
                final_std[k] = float("nan")

        print(f"\n  {method_name} (fold-level, n={n_folds}):")
        print(f"    Dice={final_mean['Dice']*100:.2f}% +/- {final_std['Dice']*100:.2f}%")
        print(f"    BF1 ={final_mean['BF1']*100:.2f}% +/- {final_std['BF1']*100:.2f}%")
        print(f"    HD95={final_mean['HD95']:.2f} +/- {final_std['HD95']:.2f}")
        print(f"    ASSD={final_mean['ASSD']:.2f} +/- {final_std['ASSD']:.2f}")

        csv_rows.append({
            "task": args.task,
            "ratio": args.ratio,
            "method": method_name,
            "n_folds": n_folds,
            **{f"{k}_mean": final_mean[k] for k in metric_keys},
            **{f"{k}_std": final_std[k] for k in metric_keys},
        })

    # Write CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults saved to {args.output}")

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table")
    print("=" * 80)
    metric_order = ["Dice", "BF1", "HD95", "ASSD"]
    print("\\begin{tabular}{l" + "c" * len(metric_order) + "}")
    print("\\toprule")
    print("Method & " + " & ".join(metric_order) + " \\\\")
    print("\\midrule")
    for row in csv_rows:
        vals = []
        for k in metric_order:
            m = row[f"{k}_mean"]
            s = row[f"{k}_std"]
            if k in ("HD95", "ASSD"):
                vals.append(f"{m:.2f}{{\\tiny$\\pm${s:.2f}}}")
            else:
                vals.append(f"{m*100:.2f}{{\\tiny$\\pm${s*100:.2f}}}")
        print(f"{row['method']} & {' & '.join(vals)} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
