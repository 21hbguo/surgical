"""Aggregate Task 2/3 test results into paper-ready tables.

Reads all_experiments_results_best.csv and generates LaTeX tables
for Task 2 (4-class part segmentation) and Task 3 (7-class type segmentation).

Usage:
    python aggregate_task2_task3.py [--task 2|3] [--ratio 20]
"""
import csv
import argparse
from collections import defaultdict
import re

CSV_PATH = "/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/all_experiments_results_best.csv"

# Method display names mapped to CSV name prefixes
TASK2_METHODS = {
    "Fully": "task2_Fully",
    "MT": "task2_MT_",
    "UAMT": "task2_UAMT_",
    "URPC": "task2_URPC_",
    "MT-DGv4": "task2_MT_depth_guider_v4_",
    "CPS": "task2_CPS_",
    "UniMatch": "task2_UniMatch_",
    "SegMatch": "task2_SegMatch_",
    "U2PL": "task2_U2PL_",
    "CorrMatch": "task2_CorrMatch_",
    "CW-BASS": "task2_CW-BASS_",
    "GeoRisk-SPC-DG": "task2_GeoRiskSPC_DGv4_",
}

TASK3_METHODS = {
    "Fully": "task3_Fully",
    "MT": "task3_MT_",
    "UAMT": "task3_UAMT_",
    "URPC": "task3_URPC_",
    "MT-DGv4": "task3_MT_depth_guider_v4_",
    "CPS": "task3_CPS_",
    "UniMatch": "task3_UniMatch_",
    "SegMatch": "task3_SegMatch_",
    "U2PL": "task3_U2PL_",
    "CorrMatch": "task3_CorrMatch_",
    "GeoRisk-SPC-DG": "task3_GeoRiskSPC_DGv4_",
}

RATIOS = [5, 10, 20, 40]

# Task 2 class names (4 classes: bg + 3 parts)
TASK2_CLASSES = ["Shaft", "Wrist", "Claspers"]
# Task 3 class names (7 classes: bg + 6 types)
TASK3_CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
                 "Vessel Sealer", "Grasping Retractor", "Monopolar Curved Scissors"]


def parse_metric(s):
    """Parse '85.23 ± 2.10' -> (85.23, 2.10) or return None."""
    if not s or s.strip() == "":
        return None
    parts = s.split("±")
    if len(parts) == 2:
        try:
            return (float(parts[0].strip()), float(parts[1].strip()))
        except ValueError:
            return None
    try:
        return (float(s.strip()), 0.0)
    except ValueError:
        return None


def load_results(csv_path, task_num, method_prefix, ratio):
    """Load per-fold results for a method at a given ratio."""
    folds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if not name.startswith(method_prefix):
                continue
            if "ALL_Folds" in name:
                continue
            # Check ratio in name
            if f"{ratio}_labeled" not in name:
                continue
            # Extract fold
            fold_match = re.search(r'_f(\d+)$', name)
            if not fold_match:
                continue
            fold = int(fold_match.group(1))

            avg_dice = parse_metric(row.get("Avg_dice", ""))
            if avg_dice is None:
                continue

            # Per-class dice
            per_class = {}
            if task_num == 2:
                for i, cls_name in enumerate(TASK2_CLASSES, 1):
                    val = parse_metric(row.get(f"C{i}_dice", ""))
                    if val:
                        per_class[cls_name] = val
            elif task_num == 3:
                for i, cls_name in enumerate(TASK3_CLASSES, 1):
                    val = parse_metric(row.get(f"C{i}_dice", ""))
                    if val:
                        per_class[cls_name] = val

            folds[fold] = {
                "avg_dice": avg_dice,
                "per_class": per_class,
            }
    return folds


def format_val(mean, std=None, bold=False):
    """Format as LaTeX: 85.23{\tiny$\pm$2.10} or \\textbf{85.23}{\\tiny$\\pm$2.10}."""
    s = f"{mean:.2f}"
    if std is not None:
        s += f"{{\\tiny$\\pm${std:.2f}}}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=2, choices=[2, 3])
    parser.add_argument("--ratio", type=int, default=20, choices=RATIOS)
    parser.add_argument("--per_class", action="store_true", help="Show per-class breakdown")
    args = parser.parse_args()

    methods = TASK2_METHODS if args.task == 2 else TASK3_METHODS
    class_names = TASK2_CLASSES if args.task == 2 else TASK3_CLASSES

    print(f"Task {args.task} Results at {args.ratio}% Labels")
    print("=" * 80)

    all_results = {}
    for display_name, prefix in sorted(methods.items()):
        folds = load_results(CSV_PATH, args.task, prefix, args.ratio)
        if not folds:
            print(f"{display_name}: no data")
            continue

        dice_values = [f["avg_dice"][0] for f in folds.values()]
        mean_dice = sum(dice_values) / len(dice_values)
        std_dice = (sum((d - mean_dice)**2 for d in dice_values) / max(len(dice_values) - 1, 1)) ** 0.5

        all_results[display_name] = {
            "mean": mean_dice,
            "std": std_dice,
            "n_folds": len(folds),
            "folds": folds,
        }

    if not all_results:
        print("No results found.")
        return

    # Find best
    best_name = max(all_results, key=lambda k: all_results[k]["mean"])
    best_val = all_results[best_name]["mean"]

    # Print summary table
    print(f"\n{'Method':<25s} {'Dice':>10s} {'Folds':>6s} {'vs Best':>8s}")
    print("-" * 55)
    for name in sorted(all_results, key=lambda k: -all_results[k]["mean"]):
        r = all_results[name]
        delta = r["mean"] - best_val
        bold = name == best_name
        marker = " *" if bold else ""
        print(f"{name:<25s} {r['mean']:.2f}±{r['std']:.2f} {r['n_folds']:>4d}f {delta:>+7.2f}{marker}")

    # Generate LaTeX table
    print("\n\n% LaTeX Table")
    n_cols = 1 + len(class_names) if args.per_class else 1
    col_spec = "l" + "c" * n_cols
    print(f"\\begin{{table}}[t]")
    print(f"\\caption{{Test Dice on EndoVis 2017 Task {args.task} at {args.ratio}\\% labels (4-fold mean$\\pm$std).}}")
    print(f"\\label{{tab:task{args.task}_results}}")
    print(f"\\centering")
    print(f"\\scriptsize")
    if args.per_class:
        header = "Method & Avg Dice"
        for cls in class_names:
            header += f" & {cls}"
        header += " \\\\"
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print(f"\\toprule")
        print(header)
    else:
        print(f"\\begin{{tabular}}{{lc}}")
        print(f"\\toprule")
        print(f"Method & Test Dice \\\\")

    print(f"\\midrule")

    for name in sorted(all_results, key=lambda k: -all_results[k]["mean"]):
        r = all_results[name]
        bold = name == best_name
        dice_str = format_val(r["mean"], r["std"], bold)

        if args.per_class:
            row = f"{name} & {dice_str}"
            # Average per-class across folds
            for cls in class_names:
                cls_vals = []
                for fold_data in r["folds"].values():
                    if cls in fold_data["per_class"]:
                        cls_vals.append(fold_data["per_class"][cls][0])
                if cls_vals:
                    cls_mean = sum(cls_vals) / len(cls_vals)
                    cls_std = (sum((v - cls_mean)**2 for v in cls_vals) / max(len(cls_vals) - 1, 1)) ** 0.5
                    is_best_cls = all(
                        all_results[n]["folds"][f]["per_class"].get(cls, (0,))[0]
                        for n in all_results for f in all_results[n]["folds"]
                    )  # simplified - just format
                    row += f" & {format_val(cls_mean, cls_std)}"
                else:
                    row += " & --"
            row += " \\\\"
            print(row)
        else:
            print(f"{name} & {dice_str} \\\\")

    print(f"\\bottomrule")
    print(f"\\end{{tabular}}")
    print(f"\\end{{table}}")

    # Per-class breakdown if requested
    if args.per_class:
        print(f"\n\n% Per-class breakdown for {best_name}")
        for cls in class_names:
            vals = []
            for fold_data in all_results[best_name]["folds"].values():
                if cls in fold_data["per_class"]:
                    vals.append(fold_data["per_class"][cls][0])
            if vals:
                m = sum(vals) / len(vals)
                s = (sum((v - m)**2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                print(f"  {cls}: {m:.2f} ± {s:.2f}")


if __name__ == "__main__":
    main()
