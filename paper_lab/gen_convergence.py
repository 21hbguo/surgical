"""从真实训练日志生成收敛曲线 (paper 级)"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_ROOT = "/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1"
OUTPUT_DIR = "/home/guo/project/ssl4mis/code_all_vibe_v2/paper/figures"

EXPERIMENTS = {
    "GeoRisk-SPC (Ours)": os.path.join(RESULT_ROOT, "GeoRiskSPC_DGv4/40_labeled_lr1e-4_s_unet_georisk_spc_dgv4_depth1C"),
    "Mean Teacher": os.path.join(RESULT_ROOT, "MT/40_labeled_lr1e-4_s_unet"),
}


def load_folds(exp_dir):
    """加载所有 fold 的 metrics.csv，返回 {epoch: [dice_per_fold]}"""
    fold_data = {}
    for fold in ["f0", "f1", "f2", "f3"]:
        csv_path = os.path.join(exp_dir, fold, "metrics.csv")
        if not os.path.exists(csv_path):
            continue
        epochs, dices = [], []
        with open(csv_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                epochs.append(int(parts[0]))
                dices.append(float(parts[3]))
        fold_data[fold] = (np.array(epochs), np.array(dices))
    return fold_data


def interpolate_to_common(fold_data):
    """将所有 fold 插值到公共 epoch 轴"""
    all_epochs = set()
    for epochs, _ in fold_data.values():
        all_epochs.update(epochs.tolist())
    common_epochs = np.array(sorted(all_epochs))

    interpolated = []
    for epochs, dices in fold_data.values():
        interp = np.interp(common_epochs, epochs, dices)
        interpolated.append(interp)

    stacked = np.stack(interpolated)  # [n_folds, n_epochs]
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return common_epochs, mean, std


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    colors = {"GeoRisk-SPC (Ours)": "#e74c3c", "Mean Teacher": "#3498db"}
    labels = {"GeoRisk-SPC (Ours)": "GeoRisk-SPC (Ours)", "Mean Teacher": "Mean Teacher"}

    for name, exp_dir in EXPERIMENTS.items():
        fold_data = load_folds(exp_dir)
        if not fold_data:
            print(f"  No data for {name}")
            continue

        epochs, mean, std = interpolate_to_common(fold_data)
        epochs_k = epochs / 1000  # 转换为 k-iters

        color = colors[name]
        ax.plot(epochs_k, mean, color=color, linewidth=1.8, label=labels[name])
        ax.fill_between(epochs_k, mean - std, mean + std, color=color, alpha=0.15)
        print(f"  {name}: {len(fold_data)} folds, "
              f"final Dice={mean[-1]:.4f} ± {std[-1]:.4f}")

    ax.set_xlabel("Iterations (k)", fontsize=10)
    ax.set_ylabel("Validation Dice", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "convergence.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
