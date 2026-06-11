"""Generate all LaTeX tables for the paper from CSV data.

Unified statistical standard: fold-level mean ± sample std (ddof=1).

Usage:
    python paper_lab/generate_paper_tables.py
    python paper_lab/generate_paper_tables.py --output paper/tables.tex
"""

import argparse
import csv
import os
import re
from collections import defaultdict

CSV_PATH = "/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/all_experiments_results_best.csv"
BOUNDARY_CSV_TPL = os.path.join(os.path.dirname(__file__),
                                "boundary_metrics_task{}_{}pct.csv")

# ============ Method definitions ============
# Task 1 methods: display_name -> CSV prefix pattern
TASK1_METHODS = [
    ("Sup-only", "task1_Sup-only", False),
    ("MT", "task1_MT_", False),
    ("MT-DGv4", "task1_MT_depth_guider_v4_", False),
    ("UAMT", "task1_UAMT_", False),
    ("URPC", "task1_URPC_", False),
    ("CPS", "task1_CPS_", False),
    ("MMS", "task1_MMS_", False),
    ("UniMatch (reimpl.)", "task1_UniMatch_", False),
    ("SegMatch (reimpl.)", "task1_SegMatch_", False),
    ("U2PL (official)", "task1_U2PL_", False),
    ("CorrMatch (official)", "task1_CorrMatch_", False),
    ("CW-BASS (official)", "task1_CW-BASS_", False),
    ("GeoRisk-SPC (UNet)", "task1_GeoRiskSPC_UNet_", False),
    ("GeoRisk-SPC-DG", "task1_GeoRiskSPC_DGv4_", True),
]

TASK1_ABLATION = [
    ("Depth + Uncertainty", "ablation_depth_uncertainty"),
    ("w/o risk localization", "ablation_no_risk"),
    ("DG encoder w/o risk", "ablation_dg_no_risk"),
    ("w/o $\\mathcal{L}_{cons}$", "ablation_no_cons"),
    ("$\\tau_r=0.7$", "ablation_tr07"),
    ("Random mask ($\\tau_r=1.0$)", "ablation_random"),
    ("Depth only", "ablation_depth_only"),
    ("Conflict only", "ablation_conflict_only"),
    ("$\\tau_c=0.7$", "ablation_tc07"),
    ("Uncertainty only", "ablation_uncertainty_only"),
    ("Full method", "ablation_full"),
    ("$\\tau_r=0.3,\\tau_c=0.7$", "ablation_tr03_tc07"),
    ("$\\tau_r=0.3$", "ablation_tr03"),
    ("$\\tau_c=0.95$", "ablation_tc095"),
    ("w/o $\\mathcal{L}_{bd}$", "ablation_no_bd"),
]

TASK2_METHODS = [
    ("Fully", "task2_Fully_"),
    ("MT", "task2_MT_"),
    ("MT-DGv4", "task2_MT_depth_guider_v4_"),
    ("UAMT", "task2_UAMT_"),
    ("URPC", "task2_URPC_"),
    ("CPS", "task2_CPS_"),
    ("UniMatch", "task2_UniMatch_"),
    ("SegMatch", "task2_SegMatch_"),
    ("U2PL", "task2_U2PL_"),
    ("CorrMatch", "task2_CorrMatch_"),
    ("CW-BASS", "task2_CW-BASS_"),
    ("GeoRisk-SPC-DG", "task2_GeoRiskSPC_DGv4_"),
]

TASK3_METHODS = [
    ("Fully", "task3_Fully_"),
    ("MT", "task3_MT_"),
    ("MT-DGv4", "task3_MT_depth_guider_v4_"),
    ("UAMT", "task3_UAMT_"),
    ("URPC", "task3_URPC_"),
    ("CPS", "task3_CPS_"),
    ("UniMatch", "task3_UniMatch_"),
    ("SegMatch", "task3_SegMatch_"),
    ("U2PL", "task3_U2PL_"),
    ("CorrMatch", "task3_CorrMatch_"),
    ("GeoRisk-SPC-DG", "task3_GeoRiskSPC_DGv4_"),
]

RATIOS = [5, 10, 20, 40]


# ============ CSV parsing ============
def parse_metric(s):
    """Parse '85.23 ± 2.10' -> (85.23, 2.10)."""
    if not s or s.strip() == "":
        return None
    # Handle both ± and +/- separators
    s = s.replace("+/-", "±")
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


def load_fold_dice(csv_path, method_prefix, ratio):
    """Load per-fold Avg_dice for a method at a given ratio. Returns {fold_id: dice_value}."""
    folds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if not name.startswith(method_prefix):
                continue
            if "ALL_Folds" in name:
                continue
            if f"{ratio}_labeled" not in name:
                continue
            # Ensure the ratio comes right after the prefix (avoid prefix collisions)
            # e.g., prefix="task2_MT_" should not match "task2_MT_depth_guider_v4_..."
            remainder = name[len(method_prefix):]
            if not re.match(rf'_?{ratio}_labeled', remainder):
                continue
            fold_match = re.search(r'_f(\d+)$', name)
            if not fold_match:
                continue
            fold = int(fold_match.group(1))
            avg_dice = parse_metric(row.get("Avg_dice", ""))
            if avg_dice is None:
                continue
            # Keep first occurrence per fold (CSV may have duplicates)
            if fold not in folds:
                folds[fold] = avg_dice[0]
    return folds


def fold_mean_std(values):
    """Compute fold-level mean and sample std (ddof=1)."""
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    s = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return m, s


# ============ Formatting ============
def fmt(mean, std=None, bold=False):
    """Format as LaTeX. Input values are already in percentage (e.g. 85.23)."""
    s = f"{mean:.2f}"
    if std is not None:
        s += f"{{\\tiny$\\pm${std:.2f}}}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def fmt_boundary(mean, std=None, bold=False, is_distance=False):
    """Format boundary metrics. Values already in percentage unless is_distance."""
    s = f"{mean:.2f}"
    if std is not None:
        s += f"{{\\tiny$\\pm${std:.2f}}}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


# ============ Table generators ============
def gen_task1_main(csv_path, ratios=None):
    """Generate Task 1 main results table (multi-ratio)."""
    if ratios is None:
        ratios = RATIOS

    rows = []
    for display_name, prefix, is_ours in TASK1_METHODS:
        ratio_vals = {}
        for r in ratios:
            folds = load_fold_dice(csv_path, prefix, r)
            if folds:
                m, s = fold_mean_std(list(folds.values()))
                ratio_vals[r] = (m, s, len(folds))
            else:
                ratio_vals[r] = None
        rows.append((display_name, ratio_vals, is_ours))

    # Find best per ratio
    best_per_ratio = {}
    for r in ratios:
        best = 0
        for name, vals, _ in rows:
            if vals.get(r) and vals[r][0] > best:
                best = vals[r][0]
        best_per_ratio[r] = best

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\caption{Test Dice on EndoVis 2017 Task 1 with 4-fold cross-validation. Values are fold-level mean$\\pm$std.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{l" + "c" * len(ratios) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(f"{r}\\%" for r in ratios) + " \\\\")
    lines.append("\\midrule")

    for name, vals, is_ours in rows:
        cols = []
        for r in ratios:
            if vals.get(r) is None:
                cols.append("--")
            else:
                m, s, n = vals[r]
                is_best = abs(m - best_per_ratio[r]) < 0.005
                cols.append(fmt(m, s, bold=is_best))
        line = f"{name} & {' & '.join(cols)} \\\\"
        if is_ours:
            line = "\\midrule\n" + line
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def gen_task_table(csv_path, task_num, methods, ratios=None, caption="", label=""):
    """Generate a multi-ratio table for Task 2 or Task 3."""
    if ratios is None:
        ratios = RATIOS

    rows = []
    for display_name, prefix in methods:
        ratio_vals = {}
        for r in ratios:
            folds = load_fold_dice(csv_path, prefix, r)
            if folds:
                m, s = fold_mean_std(list(folds.values()))
                ratio_vals[r] = (m, s, len(folds))
            else:
                ratio_vals[r] = None
        rows.append((display_name, ratio_vals))

    # Find best per ratio
    best_per_ratio = {}
    for r in ratios:
        best = 0
        for name, vals in rows:
            if vals.get(r) and vals[r][0] > best:
                best = vals[r][0]
        best_per_ratio[r] = best

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{l" + "c" * len(ratios) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(f"{r}\\%" for r in ratios) + " \\\\")
    lines.append("\\midrule")

    for name, vals in rows:
        cols = []
        for r in ratios:
            if vals.get(r) is None:
                cols.append("--{\\tiny$\\dagger$}")
            else:
                m, s, n = vals[r]
                is_best = abs(m - best_per_ratio[r]) < 0.005
                cols.append(fmt(m, s, bold=is_best))
        lines.append(f"{name} & {' & '.join(cols)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def gen_ablation_table(csv_path):
    """Generate ablation table (20% labels)."""
    rows = []
    for display_name, prefix in TASK1_ABLATION:
        folds = load_fold_dice(csv_path, prefix, 20)
        if folds:
            m, s = fold_mean_std(list(folds.values()))
            rows.append((display_name, m, s))

    if not rows:
        return "% No ablation data found"

    full_val = None
    for name, m, s in rows:
        if name == "Full method":
            full_val = m
            break

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Ablation results at 20\\% labels. Values are fold-level test Dice.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Configuration & Test Dice & $\\Delta$ vs Full \\\\")
    lines.append("\\midrule")

    for name, m, s in rows:
        delta = m - full_val if full_val else 0
        is_full = name == "Full method"
        dice_str = fmt(m, s, bold=is_full)
        delta_str = f"{delta:+.2f}" if not is_full else "--"
        lines.append(f"{name} & {dice_str} & {delta_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def gen_boundary_table(boundary_csv, task=1, ratio=20):
    """Generate boundary metrics table from pre-computed CSV."""
    if not os.path.exists(boundary_csv):
        return f"% Boundary metrics CSV not found: {boundary_csv}"

    rows = []
    with open(boundary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return "% No boundary metrics data"

    metric_order = ["Dice", "BF1", "HD95", "ASSD"]
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{Boundary-aware metrics at {ratio}\\% labels (fold-level mean$\\pm$std).}}")
    lines.append("\\label{tab:boundary}")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{l" + "c" * len(metric_order) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(metric_order) + " \\\\")
    lines.append("\\midrule")

    # Find best Dice
    best_dice = max(float(r.get("Dice_mean", 0)) for r in rows)

    for row in rows:
        name = row["method"]
        vals = []
        for k in metric_order:
            m = float(row.get(f"{k}_mean", 0))
            s = float(row.get(f"{k}_std", 0))
            is_dist = k in ("HD95", "ASSD")
            # Convert non-distance metrics from decimal to percentage
            if not is_dist:
                m *= 100
                s *= 100
            is_best = (k == "Dice" and abs(m - best_dice * 100) < 0.01)
            vals.append(fmt_boundary(m, s, bold=is_best, is_distance=is_dist))
        lines.append(f"{name} & {' & '.join(vals)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ============ Main ============
def main():
    parser = argparse.ArgumentParser(description="Generate all paper tables from CSV")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to results CSV")
    parser.add_argument("--output", default=None, help="Output .tex file")
    parser.add_argument("--boundary-dir", default=os.path.dirname(__file__),
                        help="Directory containing boundary metrics CSVs")
    args = parser.parse_args()

    sections = []

    # Task 1 main results
    sections.append("%" + "=" * 60)
    sections.append("% TABLE 1: Task 1 Main Results")
    sections.append("%" + "=" * 60)
    sections.append(gen_task1_main(args.csv))

    # Task 1 ablation
    sections.append("")
    sections.append("%" + "=" * 60)
    sections.append("% TABLE 2: Ablation (Task 1, 20%)")
    sections.append("%" + "=" * 60)
    sections.append(gen_ablation_table(args.csv))

    # Task 2 results
    sections.append("")
    sections.append("%" + "=" * 60)
    sections.append("% TABLE 3: Task 2 (4-class part segmentation)")
    sections.append("%" + "=" * 60)
    sections.append(gen_task_table(
        args.csv, 2, TASK2_METHODS,
        caption="Test Dice on EndoVis 2017 Task 2 (4-class part segmentation) with 4-fold cross-validation. Values are fold-level mean$\\pm$std.",
        label="tab:task2_results",
    ))

    # Task 3 results
    sections.append("")
    sections.append("%" + "=" * 60)
    sections.append("% TABLE 4: Task 3 (7-class type segmentation)")
    sections.append("%" + "=" * 60)
    sections.append(gen_task_table(
        args.csv, 3, TASK3_METHODS,
        caption="Test Dice on EndoVis 2017 Task 3 (7-class type segmentation) with 4-fold cross-validation. Values are fold-level mean$\\pm$std. Methods marked $\\dagger$ are pending.",
        label="tab:task3_results",
    ))

    # Boundary metrics (if available)
    for task, ratio in [(1, 20)]:
        bcsv = os.path.join(args.boundary_dir, f"boundary_metrics_task{task}_{ratio}pct.csv")
        sections.append("")
        sections.append("%" + "=" * 60)
        sections.append(f"% TABLE 5: Boundary Metrics (Task {task}, {ratio}%)")
        sections.append("%" + "=" * 60)
        sections.append(gen_boundary_table(bcsv, task, ratio))

    output = "\n\n".join(sections)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Tables written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
