import logging
import os
from contextlib import nullcontext

import pandas as pd

from utils.common import (
    format_lr,
    get_dataset_result_dir_name,
    get_labeled_num_str,
    get_model_name,
)

try:
    import portalocker
except ImportError:  # pragma: no cover - optional dependency
    portalocker = None


DISPLAY_METRIC_PAIRS = (
    ("Dice", "dice"),
    ("IoU", "iou"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Acc", "acc"),
)


def build_export_experiment_name(args, model_name: str, total_folds: int, effective_lr: float) -> str:
    clean_exp_name = args.exp.split("/")[-1] if "/" in args.exp else args.exp
    parts = [
        clean_exp_name,
        model_name,
        args.labeled_num,
        total_folds,
        args.optimizer,
        f"{effective_lr:.1e}",
        f"task{args.task}",
    ]
    return "_".join(str(part) for part in parts)


def build_result_name(args, model_name: str, suffix: str) -> str:
    clean_exp_name = args.exp.split("/")[-1] if "/" in args.exp else args.exp
    labeled_num = get_labeled_num_str(args.labeled_num)
    lr = format_lr(args.lr)
    return f"task{args.task}_{clean_exp_name}_{labeled_num}_labeled_lr{lr}_{model_name}_{suffix}"


def format_metric_percentage(val_str: str) -> str:
    if "±" in val_str:
        mean_text, std_text = val_str.split("±", 1)
        mean_val = float(mean_text.strip())
        std_val = float(std_text.strip())
    else:
        mean_val, std_val = float(val_str), 0.0
    return f"{mean_val * 100:.2f} ± {std_val * 100:.2f}"


def _summary_key_to_column_key(key: str) -> str:
    if key.startswith("C"):
        col_key = key
        for summary_metric, column_metric in DISPLAY_METRIC_PAIRS:
            col_key = col_key.replace(f"_{summary_metric}", f"_{column_metric}")
        return col_key
    if key.startswith("Avg_"):
        return "Avg_" + key.split("_", 1)[1].lower()
    return "Avg_" + key.lower()


def append_summary_metrics_to_row(row: dict, summary: dict):
    for key, value in summary.items():
        if key in {"Total_Samples", "Valid_Samples"}:
            continue
        row[_summary_key_to_column_key(key)] = format_metric_percentage(value)


def build_summary_row(args, summary, fold_label, train_best_dice=None, total_samples=None):
    model_name = get_model_name(
        args.model,
        args.filter_num,
        args.use_depth,
        args.strong,
        args.pretrain,
        include_filter_num=False,
        way=args.way,
    )
    per_class_metrics = []
    for cls in range(1, args.num_classes):
        for summary_metric, _ in DISPLAY_METRIC_PAIRS:
            key = f"C{cls}_{summary_metric}"
            if key in summary:
                per_class_metrics.append(key)

    row = {
        "Experiment": build_export_experiment_name(
            args,
            model_name,
            args.num_folds if args.num_folds is not None else 1,
            args.lr,
        ),
        "Fold": fold_label,
        "Seq": "",
    }
    if train_best_dice is not None:
        row["train_best_dice"] = f"{train_best_dice:.4f}"
    if total_samples is not None:
        row["Total_Samples"] = total_samples
    for metric in [key for key in summary.keys() if key.startswith("Avg_")]:
        row[_summary_key_to_column_key(metric)] = format_metric_percentage(summary[metric])
    for key in per_class_metrics:
        row[_summary_key_to_column_key(key)] = format_metric_percentage(summary[key])
    return row


def append_csv_with_lock(df: pd.DataFrame, path: str, timeout: int = 60):
    if df is None or df.empty:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = f"{path}.lock"
    logger = logging.getLogger(__name__)

    try:
        lock_context = nullcontext()
        if portalocker is None:
            logger.warning("portalocker not installed, falling back to non-locked write")
        else:
            lock_context = portalocker.Lock(lock_path, mode="w", timeout=timeout)

        with lock_context:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                existing_df = pd.DataFrame()
            else:
                try:
                    existing_df = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    existing_df = pd.DataFrame()
            all_columns = list(dict.fromkeys(list(existing_df.columns) + list(df.columns)))
            pd.concat(
                [existing_df.reindex(columns=all_columns), df.reindex(columns=all_columns)],
                ignore_index=True,
            ).to_csv(path, index=False)
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.warning("Error saving CSV %s: %s", path, exc)


def _format_train_best_metric(fold_label, train_best_by_fold):
    if fold_label == "ALL_Folds":
        if not train_best_by_fold:
            return ""
        values = list(train_best_by_fold.values())
        return f"{sum(values) / len(values) * 100:.2f} ± {pd.Series(values).std(ddof=0) * 100:.2f}"
    if fold_label in train_best_by_fold:
        return f"{train_best_by_fold[fold_label] * 100:.2f}"
    return ""


def _build_named_result_row(args, model_name, dataset_name, experiment_name, row, suffix, train_best_dice=""):
    export_row = {
        "name": build_result_name(args, model_name, suffix),
        "Dataset": dataset_name,
        "Experiment": experiment_name,
        "train_best_dice": train_best_dice,
    }
    for key, value in row.items():
        if key in {"Dataset", "Experiment", "train_best_dice"}:
            continue
        export_row[key] = value
    return export_row


def build_result_export_rows(args, context, fold_rows, seq_rows, train_best_by_fold, total_folds: int, case_records=None):
    dataset_name = get_dataset_result_dir_name(
        args.exp,
        args.root_path,
        args.task,
        sampling=args.sampling,
    )
    experiment_name = build_export_experiment_name(
        args,
        context.model_name,
        total_folds,
        context.effective_lr,
    )
    fold_output_rows = [
        _build_named_result_row(
            args,
            context.model_name,
            dataset_name,
            experiment_name,
            row,
            suffix=row["Fold"],
            train_best_dice=_format_train_best_metric(row["Fold"], train_best_by_fold),
        )
        for row in fold_rows
    ]
    seq_output_rows = [
        _build_named_result_row(
            args,
            context.model_name,
            dataset_name,
            experiment_name,
            row,
            suffix=f"{row['Fold']}_seq{row['Seq']}",
            train_best_dice=_format_train_best_metric(row["Fold"], train_best_by_fold),
        )
        for row in seq_rows
    ]
    if case_records is None:
        return fold_output_rows, seq_output_rows, [dict(fold_output_rows[-1])]
    case_output_rows = [
        _build_named_result_row(
            args,
            context.model_name,
            dataset_name,
            experiment_name,
            record,
            suffix=f"{record.get('Fold', '')}_{record.get('Case', '')}_c{record.get('Class', '')}",
            train_best_dice=_format_train_best_metric(record.get("Fold", ""), train_best_by_fold),
        )
        for record in case_records
    ]
    return fold_output_rows, seq_output_rows, [dict(fold_output_rows[-1])], case_output_rows


def persist_result_tables(context, fold_output_rows, seq_output_rows, all_folds_summary_rows, case_output_rows=None):
    os.makedirs(context.dataset_result_path, exist_ok=True)
    global_results_path = os.path.join(
        context.dataset_result_path,
        f"all_experiments_results_{context.checkpoint_type}.csv",
    )
    seq_results_path = os.path.join(
        context.dataset_result_path,
        f"all_experiments_results_seq_{context.checkpoint_type}.csv",
    )
    all_folds_summary_path = os.path.join(
        context.dataset_result_path,
        f"all_folds_summary_{context.checkpoint_type}.csv",
    )
    case_results_path = os.path.join(
        context.dataset_result_path,
        f"all_experiments_results_case_{context.checkpoint_type}.csv",
    )
    case_summary_path = os.path.join(
        context.dataset_result_path,
        f"all_experiments_results_case_summary_{context.checkpoint_type}.csv",
    )

    append_csv_with_lock(pd.DataFrame(fold_output_rows), global_results_path)
    append_csv_with_lock(pd.DataFrame(fold_output_rows), case_summary_path)
    append_csv_with_lock(pd.DataFrame(all_folds_summary_rows), all_folds_summary_path)
    if seq_output_rows:
        append_csv_with_lock(pd.DataFrame(seq_output_rows), seq_results_path)
    elif not os.path.exists(seq_results_path):
        pd.DataFrame(columns=["Dataset", "Experiment", "Fold", "Seq"]).to_csv(seq_results_path, index=False)
    if case_output_rows is not None:
        append_csv_with_lock(pd.DataFrame(case_output_rows), case_results_path)

    if case_output_rows is not None:
        return global_results_path, seq_results_path, all_folds_summary_path, case_results_path, case_summary_path
    return global_results_path, seq_results_path, all_folds_summary_path, case_summary_path
