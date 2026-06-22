import logging
import os
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.args import build_test_feature_parser, finalize_test_args, format_args_for_logging
from core.runtime import (
    build_test_run_context,
    resolve_device,
)
from core.testing import (
    GradCAMHookManager,
    append_csv_with_lock,
    append_summary_metrics_to_row,
    build_result_export_rows,
    build_result_name,
    persist_result_tables,
    prepare_visual_output_dirs,
    save_confidence_visualization,
    save_multiclass_gradcam_visualization,
    save_test_rgb_visualization,
)
from data import BaseDataSets
from models.factory import create_model
from strategies import create_strategy
from utils.common import CHECKPOINT_INFO_KEYS, build_run_output_dir, get_fold_seqs_from_path

warnings.filterwarnings('ignore', message='.*timm.models.layers.*')

METRIC_PAIRS = (
    ('Dice', 'dice'),
    ('IoU', 'iou'),
    ('Precision', 'precision'),
    ('Recall', 'recall'),
    ('Acc', 'acc'),
    ('HD95', 'hd95'),
    ('ASD', 'asd'),
)
PER_CLASS_METRIC_PAIRS = METRIC_PAIRS
SUMMARY_METRIC_PAIRS = METRIC_PAIRS
TEST_NUM_WORKERS = 4


def parse_requested_folds(raw_folds, num_folds):
    if num_folds is None or num_folds <= 1:
        return [None]
    if raw_folds in (None, [], ()):
        return list(range(num_folds))
    if isinstance(raw_folds, (int, str)):
        raw_folds = [raw_folds]
    tokens = []
    for value in raw_folds:
        text = str(value).strip()
        if not text:
            continue
        tokens.extend(part.strip() for part in text.split(',') if part.strip())
    if not tokens or '-1' in tokens:
        return list(range(num_folds))
    folds = []
    for token in tokens:
        try:
            fold = int(token)
        except ValueError as exc:
            raise ValueError(f'Invalid fold value: {token!r}') from exc
        if fold < 0 or fold >= num_folds:
            raise ValueError(f'Fold {fold} is out of range for num_folds={num_folds}')
        folds.append(fold)
    return sorted(set(folds))


def _predict_logits(args, model, strategy, sample_batch, device, use_grad=False):
    if strategy is not None:
        if use_grad:
            outputs = strategy.validation_step(sample_batch)
        else:
            with torch.no_grad():
                outputs = strategy.validation_step(sample_batch)
    else:
        volume = sample_batch["image"].to(device)
        if use_grad:
            outputs = model(volume)
        else:
            with torch.no_grad():
                outputs = model(volume)
    if not isinstance(outputs, tuple):
        return outputs
    if args.way == "urpc":
        return outputs[0]
    if args.way == "w2s":
        return outputs[0][0]
    return outputs[0]


def _normalize_heatmap(heat: np.ndarray) -> np.ndarray:
    heat = heat.astype(np.float32)
    heat -= heat.min()
    denom = heat.max()
    if denom > 1e-8:
        return heat / denom
    return np.zeros_like(heat)


def _compute_gradcam_heatmap(args, model, strategy, sample_batch, device, hook_manager, target_class=None):
    strategy_model = strategy.model if strategy is not None else None
    if strategy is not None and strategy_model is None:
        return None
    target_model = strategy_model if strategy_model is not None else model
    hook_manager.clear_cache()
    target_model.zero_grad(set_to_none=True)
    logits = _predict_logits(args, model, strategy, sample_batch, device, use_grad=True)
    if logits.ndim != 4 or logits.shape[0] < 1:
        return None
    num_classes = int(logits.shape[1])
    if target_class is None:
        requested_class = int(args.feat_vis_target_class)
        if 0 <= requested_class < num_classes:
            target_class = requested_class
        elif num_classes == 1:
            target_class = 0
        else:
            target_class = int(torch.argmax(logits[0].mean(dim=(1, 2))[1:]).item() + 1)
    target_class = int(target_class)
    if target_class < 0 or target_class >= num_classes:
        return None
    logits[:, target_class, :, :].mean().backward()

    cams = []
    out_size = tuple(int(v) for v in logits.shape[2:])
    for layer_name in hook_manager.layer_names:
        activation = hook_manager.activations.get(layer_name)
        gradient = hook_manager.gradients.get(layer_name)
        if activation is None or gradient is None:
            continue
        if activation.ndim != 4 or gradient.ndim != 4:
            continue
        cam = torch.relu(torch.sum(torch.mean(gradient, dim=(2, 3), keepdim=True) * activation, dim=1, keepdim=True))
        if cam.shape[2:] != out_size:
            cam = torch.nn.functional.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        cams.append(_normalize_heatmap(cam[0, 0].detach().cpu().numpy()))
    if not cams:
        return None
    return _normalize_heatmap(np.mean(np.stack(cams, axis=0), axis=0))


def _collate_test_batch(batch):
    images, labels, cases, original_labels, original_shapes, original_images = [], [], [], [], [], []
    depth3_list, depth1_list = [], []
    has_depth3 = False
    has_depth1 = False
    for item in batch:
        images.append(item['image'])
        labels.append(item['label'])
        cases.append(item['case'])
        original_labels.append(item.get('original_label'))
        original_shapes.append(item['original_shape'])
        original_images.append(item['original_image'])
        depth3 = item.get('depth3')
        depth1 = item.get('depth1')
        depth3_list.append(depth3)
        depth1_list.append(depth1)
        has_depth3 = has_depth3 or depth3 is not None
        has_depth1 = has_depth1 or depth1 is not None
    collated = {
        'image': images,
        'label': labels,
        'case': cases,
        'original_label': original_labels,
        'original_shape': original_shapes,
        'original_image': original_images,
    }
    if has_depth3:
        collated['depth3'] = depth3_list
    if has_depth1:
        collated['depth1'] = depth1_list
    return collated


def _build_test_loader(args, fold, fold_map):
    depth_channels = args.use_depth if args.use_depth else None
    is_depth = bool(depth_channels)
    test_dataset = BaseDataSets(
        base_dir=args.root_path,
        split='test',
        fold=fold,
        resize_size=tuple(args.resize_size),
        load_mode='path',
        depth_channels=depth_channels if is_depth else None,
        depth_uint=int(args.depth_uint),
        normalize_method=args.normalize,
        fold_map=fold_map,
        use_val=False,
        for_inference=True,
        task=args.task,
    )

    return test_dataset, DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TEST_NUM_WORKERS,
        pin_memory=True,
        collate_fn=_collate_test_batch,
    ), is_depth, depth_channels


def calculate_metric_percase(pred, gt, smooth=1e-6):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    if pred.sum() == 0 and gt.sum() == 0:
        return {
            'Dice': float('nan'),
            'IoU': float('nan'),
            'TP': 0.0,
            'FP': 0.0,
            'FN': 0.0,
            'Acc': float((pred == gt).mean()),
            'HD95': float('nan'),
            'ASD': float('nan'),
            'Valid': False,
        }
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    tp = float(intersection)
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    acc = float((pred == gt).mean())
    hd95 = float('nan')
    asd = float('nan')
    if pred.sum() > 0 and gt.sum() > 0:
        pred_mask = pred.astype(bool)
        gt_mask = gt.astype(bool)
        structure = generate_binary_structure(pred_mask.ndim, 1)
        pred_surface = np.logical_xor(pred_mask, binary_erosion(pred_mask, structure=structure, border_value=0))
        gt_surface = np.logical_xor(gt_mask, binary_erosion(gt_mask, structure=structure, border_value=0))
        if pred_surface.any() and gt_surface.any():
            pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
            gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
            surface_distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64)
            hd95 = float(np.percentile(surface_distances, 95))
            asd = float(surface_distances.mean())
    return {'Dice': dice, 'IoU': iou, 'TP': tp, 'FP': fp, 'FN': fn, 'Acc': acc, 'HD95': hd95, 'ASD': asd, 'Valid': True}


def _resize_mask_to_shape(mask: np.ndarray, target_shape) -> np.ndarray:
    mask = np.asarray(mask)
    target_h, target_w = tuple(int(v) for v in target_shape[:2])
    if mask.shape == (target_h, target_w):
        return mask
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def inference(
    args,
    model,
    test_loader,
    device,
    fold_label='f0',
    rgb_output_dir=None,
    strategy=None,
    feat_output_dir=None,
    conf_output_dir=None,
):
    logger = logging.getLogger(__name__)
    if strategy is None:
        model.eval()
    elif hasattr(strategy, "eval"):
        strategy.eval()
    all_metrics = []
    gradcam_hooks = None
    if args.feat_vis:
        target_model = strategy.model if strategy is not None else model
        if target_model is None:
            target_model = model
        gradcam_hooks = GradCAMHookManager(
            target_model,
            layer_filter=args.feat_vis_layer,
            all_layers=bool(args.feat_vis_all_layers),
            max_layers=int(args.feat_vis_max_layers),
        )
        if not gradcam_hooks.layer_names:
            logger.warning("Grad-CAM found no Conv2d layers. Feature visualization will be skipped.")
    saved_feat_cases = 0
    for batch_idx, batch in enumerate(tqdm(test_loader, desc='Inference')):
        if batch_idx == 0:
            logger.info('First batch received, processing %s samples...', len(batch['case']))
        labels = batch['label']
        original_labels = batch.get('original_label', labels)
        cases = batch['case']
        original_images = batch.get('original_image', [None] * len(cases))
        for index, case in enumerate(cases):
            sample_batch = {"image": batch["image"][index].unsqueeze(0)}
            for depth_key in ("depth3", "depth1"):
                depth_values = batch.get(depth_key)
                if depth_values is None:
                    continue
                depth_value = depth_values[index]
                if depth_value is not None:
                    sample_batch[depth_key] = depth_value.unsqueeze(0)
            sample_batch["label"] = labels[index]
            label_value = original_labels[index] if original_labels[index] is not None else labels[index]
            label_np = label_value.cpu().numpy() if torch.is_tensor(label_value) else np.asarray(label_value)
            outputs = _predict_logits(args, model, strategy, sample_batch, device, use_grad=False)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze().cpu().numpy()
            if args.rgb and rgb_output_dir is not None and original_images[index] is not None:
                label_vis_np = label_np
                if args.rgb == 2:
                    label_vis_np = _resize_mask_to_shape(label_np, pred.shape)
                save_test_rgb_visualization(
                    image=original_images[index],
                    label=label_vis_np,
                    pred=pred,
                    output_dir=rgb_output_dir,
                    case_name=case,
                    mode=args.rgb,
                    num_classes=args.num_classes,
                )

            if (
                args.feat_vis
                and feat_output_dir is not None
                and original_images[index] is not None
                and saved_feat_cases < max(0, int(args.feat_vis_max_cases))
                and gradcam_hooks is not None
                and gradcam_hooks.layer_names
            ):
                class_heats = []
                has_any_heat = False
                for class_idx in range(int(args.num_classes)):
                    heat = _compute_gradcam_heatmap(
                        args,
                        model,
                        strategy,
                        sample_batch,
                        device,
                        gradcam_hooks,
                        target_class=class_idx,
                    )
                    class_heats.append(heat)
                    has_any_heat = has_any_heat or heat is not None
                if has_any_heat:
                    save_multiclass_gradcam_visualization(
                        image=original_images[index],
                        class_heats=class_heats,
                        output_dir=feat_output_dir,
                        case_name=case,
                        alpha=float(args.feat_vis_alpha),
                    )
                    saved_feat_cases += 1

            if args.conf_vis and conf_output_dir is not None and original_images[index] is not None:
                save_confidence_visualization(
                    image=original_images[index],
                    logits=outputs[0].detach().cpu().numpy(),
                    output_dir=conf_output_dir,
                    case_name=case,
                    num_classes=args.num_classes,
                    alpha=float(args.conf_vis_alpha),
                )

            pred_for_metrics = _resize_mask_to_shape(pred, label_np.shape)
            seq = None
            for token in re.split(r'[_-]', str(case)):
                if token.isdigit():
                    seq = int(token)
                    break
                match = re.search(r'\d+', token)
                if match:
                    seq = int(match.group())
                    break
            for cls in range(1, args.num_classes):
                metrics = calculate_metric_percase(pred_for_metrics == cls, label_np == cls)
                metrics['Case'] = case
                metrics['Class'] = cls
                metrics['Fold'] = fold_label
                metrics['Seq'] = seq
                all_metrics.append(metrics)
    if gradcam_hooks is not None:
        gradcam_hooks.remove()
    return all_metrics


def summarize_metrics(all_metrics, num_classes=2):
    valid_metrics = [metric for metric in all_metrics if metric.get('Valid', True)]
    if not valid_metrics:
        return {}
    summary = {}
    derived_metric_values = {
        'Precision': [item['TP'] / (item['TP'] + item['FP']) if (item['TP'] + item['FP']) > 0 else 0.0 for item in valid_metrics],
        'Recall': [item['TP'] / (item['TP'] + item['FN']) if (item['TP'] + item['FN']) > 0 else 0.0 for item in valid_metrics],
    }
    def format_metric_values(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return 'nan ± nan'
        return f'{values.mean():.4f} ± {values.std():.4f}'
    for summary_metric, _ in SUMMARY_METRIC_PAIRS:
        if summary_metric in derived_metric_values:
            values = derived_metric_values[summary_metric]
        else:
            values = [item.get(summary_metric, float('nan')) for item in valid_metrics]
        summary[f'Avg_{summary_metric}'] = format_metric_values(values)
    for cls in range(1, num_classes):
        class_metrics = [item for item in valid_metrics if item['Class'] == cls]
        if not class_metrics:
            continue
        derived_class_metric_values = {
            'Precision': [item['TP'] / (item['TP'] + item['FP']) if (item['TP'] + item['FP']) > 0 else 0.0 for item in class_metrics],
            'Recall': [item['TP'] / (item['TP'] + item['FN']) if (item['TP'] + item['FN']) > 0 else 0.0 for item in class_metrics],
        }
        for per_class_metric, _ in PER_CLASS_METRIC_PAIRS:
            if per_class_metric in derived_class_metric_values:
                values = derived_class_metric_values[per_class_metric]
            else:
                values = [item.get(per_class_metric, float('nan')) for item in class_metrics]
            summary[f'C{cls}_{per_class_metric}'] = format_metric_values(values)
    return summary


def summarize_records(records, num_classes=2):
    if not records:
        return {}
    case_validity = {}
    for index, record in enumerate(records):
        case_key = record.get('Case')
        if case_key is None:
            case_key = f'__record_{index}'
        is_valid = _is_valid_metric_record(record)
        if case_key not in case_validity:
            case_validity[case_key] = is_valid
        else:
            case_validity[case_key] = case_validity[case_key] and is_valid
    summary = {
        'Total_Samples': len(case_validity),
        'Valid_Samples': sum(1 for is_valid in case_validity.values() if is_valid),
    }
    summary.update(summarize_metrics(records, num_classes=num_classes))
    return summary


def _is_valid_metric_record(record: dict) -> bool:
    value = record.get('Valid', True)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def build_seq_rows(records, num_classes=2):
    if not records:
        return []
    df = pd.DataFrame(records)
    if 'Seq' not in df.columns:
        return []
    df = df[df['Seq'].notna()].copy()
    if df.empty:
        return []

    rows = []
    grouped = df.groupby(['Fold', 'Seq'], dropna=False)
    keys = sorted(
        grouped.groups.keys(),
        key=lambda item: (
            float('inf')
            if str(item[0]) == 'ALL_Folds'
            else (int(match.group()) if (match := re.search(r'\d+', str(item[0]))) else -1),
            int(item[1]),
        ),
    )
    for fold, seq in keys:
        summary = summarize_records(grouped.get_group((fold, seq)).to_dict(orient='records'), num_classes=num_classes)
        row = {
            'Fold': fold,
            'Seq': int(seq),
            'Total_Samples': summary.get('Total_Samples', 0),
            'Valid_Samples': summary.get('Valid_Samples', 0),
        }
        append_summary_metrics_to_row(row, summary)
        rows.append(row)
    return rows


def build_fold_rows(records, num_classes=2):
    if not records:
        return []
    df = pd.DataFrame(records)
    rows = []
    fold_labels = sorted(
        df['Fold'].dropna().unique().tolist(),
        key=lambda label: (
            float('inf')
            if str(label) == 'ALL_Folds'
            else (int(match.group()) if (match := re.search(r'\d+', str(label))) else -1)
        ),
    )
    for fold_label in fold_labels:
        summary = summarize_records(df[df['Fold'] == fold_label].to_dict(orient='records'), num_classes=num_classes)
        row = {
            'Fold': fold_label,
            'Seq': '',
            'Total_Samples': summary.get('Total_Samples', 0),
            'Valid_Samples': summary.get('Valid_Samples', 0),
        }
        append_summary_metrics_to_row(row, summary)
        rows.append(row)

    all_summary = summarize_records(records, num_classes=num_classes)
    all_row = {
        'Fold': 'ALL_Folds',
        'Seq': '',
        'Total_Samples': all_summary.get('Total_Samples', 0),
        'Valid_Samples': all_summary.get('Valid_Samples', 0),
    }
    append_summary_metrics_to_row(all_row, all_summary)
    rows.append(all_row)
    return rows


def run_one_fold(args, fold, fold_map, context):
    logger = logging.getLogger(__name__)
    device = resolve_device(args)
    snapshot_path = args.snapshot_path or build_run_output_dir(args, mode='train', fold=fold)
    fold_label = f"f{0 if fold is None else fold}"

    logger.info('\n%s', '=' * 60)
    logger.info('Testing %s', fold_label)
    logger.info('Snapshot path: %s', snapshot_path)
    logger.info('%s', '=' * 60)

    model = create_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001)
    strategy = create_strategy(args.way, args, model, optimizer, device)
    checkpoint_name = 'model_best.pth' if context.checkpoint_type == 'best' else 'model_final.pth'
    checkpoint_path = os.path.join(snapshot_path, checkpoint_name)
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) == 0:
        raise ValueError(f'{checkpoint_path} is an empty checkpoint marker. Use --checkpoint-type best for validation runs or retrain with --no_val for final checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    prefix = "_orig_mod."
    if isinstance(state_dict, dict) and "model" in state_dict:
        model_state = state_dict["model"]
        if any(key.startswith(prefix) for key in model_state):
            state_dict = dict(state_dict)
            state_dict["model"] = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in model_state.items()}
    elif isinstance(state_dict, dict) and any(key.startswith(prefix) for key in state_dict):
        state_dict = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}
    strategy.load_state_dict(state_dict)
    info = {key: checkpoint[key] for key in CHECKPOINT_INFO_KEYS if isinstance(checkpoint, dict) and key in checkpoint}
    train_best_dice = info.get('best_performance', None)
    logger.info('Loaded %s checkpoint from %s', context.checkpoint_type, checkpoint_path)
    if train_best_dice is not None:
        logger.info('Training best dice: %.4f', train_best_dice)

    test_dataset, test_loader, is_depth, depth_channels = _build_test_loader(args, fold, fold_map)
    logger.info('Testing %s images...', len(test_dataset))
    logger.info('DataLoader: batch_size=%s, num_workers=%s, pin_memory=True', args.batch_size, TEST_NUM_WORKERS)
    logger.info('Dataset: is_depth=%s, depth_channels=%s', is_depth, depth_channels)

    rgb_output_dir, feat_output_dir, conf_output_dir = prepare_visual_output_dirs(args, fold, logger)

    records = inference(
        args,
        model,
        test_loader,
        device,
        fold_label,
        rgb_output_dir=rgb_output_dir,
        strategy=strategy,
        feat_output_dir=feat_output_dir,
        conf_output_dir=conf_output_dir,
    )
    summary = summarize_records(records, num_classes=args.num_classes)
    if summary:
        logger.info('Fold %s summary:', fold_label)
        for key, value in summary.items():
            logger.info('  %s: %s', key, value)
    else:
        logger.warning('No valid metrics produced for %s', fold_label)
    return train_best_dice, records


def main():
    args = finalize_test_args(build_test_feature_parser().parse_args())
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    logger.info('Args: %s', format_args_for_logging(args))
    logger.info('Strategy: %s', args.way)
    logger.info('Model: %s', args.model)
    logger.info('Device: %s', args.device)
    if args.no_val:
        logger.info('No-val test mode enabled: final checkpoint will be used')

    requested_folds = parse_requested_folds(args.fold, args.num_folds)
    logger.info('Requested folds: %s', requested_folds)

    args.fold = None
    context = build_test_run_context(args)
    fold_map = get_fold_seqs_from_path(args.root_path, args.task) if args.num_folds is not None and args.num_folds > 1 else {}

    train_best_by_fold = {}
    all_records = []
    for fold in requested_folds:
        train_best_dice, fold_records = run_one_fold(args, fold, fold_map, context)
        if not fold_records:
            continue
        fold_label = f"f{0 if fold is None else fold}"
        if train_best_dice is not None:
            train_best_by_fold[fold_label] = float(train_best_dice)
        all_records.extend(fold_records)

    if not all_records:
        logger.warning('No fold results produced.')
        return

    fold_rows = build_fold_rows(all_records, num_classes=args.num_classes)
    seq_rows = build_seq_rows(all_records, num_classes=args.num_classes)
    fold_output_rows, seq_output_rows, all_folds_summary_rows, case_output_rows = build_result_export_rows(
        args,
        context,
        fold_rows,
        seq_rows,
        train_best_by_fold,
        total_folds=len(requested_folds),
        case_records=all_records,
    )

    global_results_path, seq_results_path, all_folds_summary_path, case_results_path, case_summary_path = persist_result_tables(
        context,
        fold_output_rows,
        seq_output_rows,
        all_folds_summary_rows,
        case_output_rows,
    )

    logger.info('Global fold results saved to: %s', global_results_path)
    logger.info('Global sequence results saved to: %s', seq_results_path)
    logger.info('Global all-fold summary saved to: %s', all_folds_summary_path)
    logger.info('Global case results saved to: %s', case_results_path)
    logger.info('Global case summary saved to: %s', case_summary_path)
    logger.info('Final ALL_Folds summary:')
    for line in pd.DataFrame(all_folds_summary_rows).to_string(index=False).split('\n'):
        logger.info(line)


if __name__ == '__main__':
    main()
