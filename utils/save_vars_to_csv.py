import pandas as pd
import os
from pathlib import Path
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: 'matplotlib' package not found. Plotting will be disabled.")
    print("Please run 'pip install matplotlib' to enable metric plot generation.")

COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_metrics_from_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    plot_save_path = csv_path.with_name(f"{csv_path.stem}_plot.png")
    fig = None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        metrics_to_plot = []
        all_metric_names = [col for col in df.columns if col != 'epoch']
        for metric_name in all_metric_names:
            if df[metric_name].notna().any():
                metric_values = df[metric_name].dropna()
                if len(metric_values) > 1 and metric_values.nunique() > 1:
                    metrics_to_plot.append(metric_name)
        if not metrics_to_plot:
            return
        num_metrics = len(metrics_to_plot)
        num_cols = min(num_metrics, 3)
        num_rows = (num_metrics + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), squeeze=False)
        axes = axes.flatten()
        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i]
            metric_data = df[df[metric_name].notna()]
            values = metric_data[metric_name]
            epochs = metric_data['epoch']
            color = COLOR_LIST[i % len(COLOR_LIST)]
            display_name = metric_name.replace('_', ' ').capitalize()
            ax.plot(epochs, values, marker='o', markersize=2, linestyle='-', color=color, label=display_name)
            if not values.empty:
                idx_min = values.idxmin()
                idx_max = values.idxmax()
                min_epoch = epochs.loc[idx_min]
                min_val = values.loc[idx_min]
                max_epoch = epochs.loc[idx_max]
                max_val = values.loc[idx_max]
                ax.plot(min_epoch, min_val, marker='x', color=color, markersize=10, markeredgewidth=2.5)
                ax.plot(max_epoch, max_val, marker='x', color=color, markersize=10, markeredgewidth=2.5)
            ax.set_title(f'Curve for {display_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(display_name)
            ax.legend()
            ax.grid(True)
        for j in range(num_metrics, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        plt.savefig(str(plot_save_path), dpi=100)
    except pd.errors.EmptyDataError:
        pass
    except Exception as e:
        pass
    finally:
        if fig is not None:
            plt.close(fig)
    return plot_save_path

def save_vars_to_csv(save_path, epoch, **kwargs):
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    data = {"epoch": epoch}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.detach().cpu().item()
        else:
            data[key] = value
    current_df = pd.DataFrame([data])
    file_exists = os.path.exists(save_path)
    if epoch == 1 or not file_exists:
        current_df.to_csv(save_path, mode="w", header=True, index=False, encoding="utf-8")
    else:
        try:
            existing_df = pd.read_csv(save_path)
            combined_df = pd.concat([existing_df, current_df], ignore_index=True)
            combined_df.to_csv(save_path, mode="w", header=True, index=False, encoding="utf-8")
        except pd.errors.EmptyDataError:
            current_df.to_csv(save_path, mode="w", header=True, index=False, encoding="utf-8")
    png_path = plot_metrics_from_csv(save_path)
    return png_path
