"""Generate convergence.pdf: Dice curves during training."""
import csv
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RESULT_BASE = "/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
OUT_PATH = "/home/guo/project/ssl4mis/code_all_vibe_v2/paper/figures/convergence.pdf"

# Methods to plot: (display_name, dir_name, subdir_pattern, color, linestyle)
METHODS = [
    ("MT", "MT", "20_labeled_lr1e-4_s_unet", "#1f77b4", "--"),
    ("URPC", "URPC", "20_labeled_lr1e-4_s_unet_urpc", "#ff7f0e", "--"),
    ("UniMatch†", "UniMatch", "20_labeled_lr1e-4_s_unet", "#2ca02c", "--"),
    ("SegMatch†", "SegMatch", "20_labeled_lr1e-4_s_unet", "#d62728", "--"),
    ("GeoRisk-SPC", "GeoRiskSPC", "20_labeled_lr1e-4_s_unet_georisk_spc_depth1C", "#9467bd", "-"),
    ("GeoRisk-SPC+DGv4", "GeoRiskSPC_DGv4", "20_labeled_lr1e-4_s_unet_georisk_spc_depth1C", "#e377c2", "-"),
]

def load_dice_curve(csv_path):
    """Load iteration and best_dice from metrics.csv."""
    iters, dice = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row['epoch']))
            dice.append(float(row['best_dice']))
    return iters, dice

def main():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for label, method_dir, subdir, color, ls in METHODS:
        # Average across folds
        fold_curves = []
        max_len = 0
        for fold in range(4):
            csv_path = os.path.join(RESULT_BASE, method_dir, subdir, f"f{fold}", "metrics.csv")
            if os.path.exists(csv_path):
                iters, dice = load_dice_curve(csv_path)
                fold_curves.append((iters, dice))
                max_len = max(max_len, len(iters))

        if not fold_curves:
            print(f"Warning: No data for {label}")
            continue

        # Use fold 0 for the main curve (most representative)
        iters, dice = fold_curves[0]
        ax.plot(iters, dice, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Dice", fontsize=12)
    ax.set_title("Convergence Comparison (20% Labeled)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30000)
    ax.set_ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
