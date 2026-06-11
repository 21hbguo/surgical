"""Run Wilcoxon signed-rank tests comparing GeoRisk-SPC-DGv4 vs baselines.

Uses per-sample Dice from test predictions to compute paired statistical tests.
"""
import csv
import os
import sys
from collections import defaultdict
from scipy import stats
import numpy as np

CSV_PATH = "/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/all_experiments_results_best.csv"

# Methods to compare at 20% label ratio
METHODS_20 = {
    "MT": "task1_MT_20_labeled_lr1e-4_s_unet",
    "UAMT": "task1_UAMT_20_labeled_lr1e-4_s_unet",
    "URPC": "task1_URPC_20_labeled_lr1e-4_s_unet",
    "SegMatch": "task1_SegMatch_20_labeled_lr1e-4_s_unet",
    "UniMatch": "task1_UniMatch_20_labeled_lr1e-4_s_unet",
    "UniMatch_official": "task1_UniMatch_official_20_labeled_lr1e-4_s_unet",
    "U2PL_official": "task1_U2PL_official_20_labeled_lr1e-4_s_unet",
    "CPS": "task1_CPS_20_labeled_lr1e-4_s_unet",
    "CorrMatch_official": "task1_CorrMatch_official_20_labeled_lr1e-4_s_unet",
    "MT-DGv4": "task1_MT_depth_guider_v4_20_labeled_lr1e-4_s_unet",
}
PROPOSED = "task1_GeoRiskSPC_DGv4_20_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C"


def load_fold_dice(csv_path, method_prefix):
    """Load per-fold average Dice for a method (deduplicated, no ALL_Folds)."""
    dice_by_fold = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if not name.startswith(method_prefix):
                continue
            if "ALL_Folds" in name:
                continue
            # Extract fold number
            if "_f" not in name:
                continue
            fold_str = name.split("_f")[-1]
            if not fold_str.isdigit():
                continue
            fold = int(fold_str)
            dice_str = row.get("Avg_dice", "")
            if dice_str:
                dice_val = float(dice_str.split("±")[0].strip())
                # Keep last occurrence per fold (in case of duplicates)
                dice_by_fold[fold] = dice_val
    return [dice_by_fold[f] for f in sorted(dice_by_fold.keys())]


def main():
    print("Statistical Tests: GeoRisk-SPC-DGv4 vs Baselines (20% labels)")
    print("=" * 70)

    proposed_dice = load_fold_dice(CSV_PATH, PROPOSED)
    print(f"GeoRisk-SPC-DGv4: {len(proposed_dice)} folds, mean={np.mean(proposed_dice):.2f}")
    print()

    results = []
    for method_name, prefix in sorted(METHODS_20.items()):
        baseline_dice = load_fold_dice(CSV_PATH, prefix)
        if len(baseline_dice) < 2 or len(proposed_dice) < 2:
            print(f"{method_name}: insufficient data ({len(baseline_dice)} folds)")
            continue

        # Ensure same number of folds
        n = min(len(proposed_dice), len(baseline_dice))
        p = proposed_dice[:n]
        b = baseline_dice[:n]

        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(p, b, alternative='greater')
        except ValueError:
            stat, p_value = float('nan'), float('nan')

        diff = np.mean(p) - np.mean(b)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"{method_name:25s}: {np.mean(b):.2f} → {np.mean(p):.2f} (Δ={diff:+.2f}), p={p_value:.4f} {sig}")
        results.append((method_name, np.mean(b), np.mean(p), diff, p_value, sig))

    print()
    print("=" * 70)
    print("Summary: p < 0.05 (*) = significant, p < 0.01 (**) = highly significant")


if __name__ == "__main__":
    main()
