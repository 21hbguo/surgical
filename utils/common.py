import glob
import json
import logging
import math
import os
import random

import numpy as np
import torch

try:
    from yacs.config import CfgNode
except ImportError:  # pragma: no cover - optional during args-only migration
    CfgNode = None

if CfgNode is not None:
    torch.serialization.add_safe_globals([CfgNode])

DEFAULT_COLORMAP = [
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [128, 128, 128], [128, 0, 0], [0, 128, 0],
    [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 192, 192],
    [255, 128, 0]
]
VALID_TASK_IDS = (1, 2, 3)
CHECKPOINT_INFO_KEYS = ("epoch", "best_metric", "iter_num", "best_dice", "best_performance")
VALID_DEPTH_CHANNELS = {1, 3, 13}


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def _resolve_train_slice_files(root_path, fold=None):
    if fold is None or fold < 0:
        list_files = sorted(glob.glob(os.path.join(root_path, "train_slices*.list")))
        if list_files:
            return list_files
        fallback = os.path.join(root_path, "train.list")
        return [fallback] if os.path.exists(fallback) else []

    preferred = os.path.join(root_path, f"train_slices_f{fold}.list")
    fallback = os.path.join(root_path, "train_slices.list")
    generic = os.path.join(root_path, "train.list")
    if os.path.exists(preferred):
        return [preferred]
    if os.path.exists(fallback):
        return [fallback]
    if os.path.exists(generic):
        return [generic]
    return []


def get_task_label_dir(root_path, task):
    data_dir = os.path.join(root_path, "data")
    task_id = int(task) if isinstance(task, str) else task
    prefix = f"labels_task{task_id}_"
    matches = sorted(
        entry for entry in os.listdir(data_dir)
        if entry.startswith(prefix) and os.path.isdir(os.path.join(data_dir, entry))
    )
    return matches[0]

def add_sampling_suffix(name, sampling):
    if not name or sampling in (None, ""):
        return name
    suffix = f"_Sampling{sampling}"
    return name if str(name).endswith(suffix) else f"{name}{suffix}"


def build_task_scoped_exp_name(exp_name, task, sampling=None):
    task_id = int(task) if isinstance(task, str) else task
    task_segment = f"task{task_id}"
    if not exp_name:
        return task_segment
    normalized = exp_name.replace("\\", "/")
    if "/" not in normalized:
        return os.path.join(add_sampling_suffix(normalized, sampling), task_segment)
    dataset_name, remainder = normalized.split("/", 1)
    dataset_name = add_sampling_suffix(dataset_name, sampling)
    if remainder == task_segment or remainder.startswith(f"{task_segment}/"):
        return os.path.join(dataset_name, remainder)
    return os.path.join(dataset_name, task_segment, remainder)


def resolve_runtime_path(path, cwd=None):
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    base_cwd = cwd or os.getcwd()
    return os.path.normpath(os.path.abspath(os.path.join(base_cwd, path)))


def get_labeled_num_str(labeled_num):
    if labeled_num == int(labeled_num):
        return str(int(labeled_num))
    return f"{labeled_num}".replace(".", "p")


def format_lr(lr):
    if lr == 0:
        return "0"
    mantissa, exp = f"{lr:.10e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


def get_model_name(model_type, filter_num, use_depth=False, strong=None, pretrain_mode="none", include_filter_num=True, way=None):
    from strategies.specs import PRETRAINED_UNET_PREFIXES,SEMI_STRATEGY_NAMES,SPECIAL_MODEL_NAMES
    if model_type in SPECIAL_MODEL_NAMES:
        base_name = model_type
    elif pretrain_mode != "none" and model_type.startswith(PRETRAINED_UNET_PREFIXES):
        base_name = model_type.split("_", 1)[0]
    else:
        base_name = f"{model_type}{filter_num}" if include_filter_num else model_type

    # For custom depth-pretrained ResNet backbone, expose "depth_*" in run names.
    if pretrain_mode == "depth" and str(base_name).startswith("resnet"):
        base_name = "depth" + str(base_name)[len("resnet"):]

    depth_suffix = f"_depth{use_depth}C" if use_depth else ""
    proto_suffix = "_proto" if model_type.endswith("_proto") and not str(base_name).endswith("_proto") else ""
    include_strong_prefix = not way or str(way).lower() in SEMI_STRATEGY_NAMES
    strong_prefix = f"{strong}_" if strong and include_strong_prefix else ""
    return f"{strong_prefix}{base_name}{depth_suffix}{proto_suffix}"


def build_shared_run_rel_dir(args, fold=None):
    scoped_exp = build_task_scoped_exp_name(args.exp, args.task, sampling=args.sampling)
    model_name = get_model_name(args.model, args.filter_num, args.use_depth, args.strong, args.pretrain, include_filter_num=False, way=args.way)
    run_name = f"{get_labeled_num_str(args.labeled_num)}_labeled_lr{format_lr(args.lr)}_{model_name}"
    parts = [scoped_exp, run_name]
    effective_fold = args.fold if fold is None else fold
    if effective_fold not in (None, -1):
        parts.append(f"f{effective_fold}")
    return os.path.join(*parts)


def build_run_output_dir(args, mode, fold=None, root_override=None, cwd=None):
    attr_name = "train_result_root" if mode == "train" else "predict_result_root"
    raw_root = root_override if root_override is not None else getattr(args, attr_name)
    return os.path.join(
        resolve_runtime_path(raw_root, cwd=cwd),
        build_shared_run_rel_dir(args, fold=fold),
    )


def get_dataset_result_dir_name(exp, root_path, task=None, sampling=None):
    if exp and "/" in exp:
        dataset_name, _ = exp.split("/", 1)
        return add_sampling_suffix(dataset_name, sampling)
    if exp:
        return add_sampling_suffix(exp, sampling)
    if root_path:
        return add_sampling_suffix(os.path.basename(root_path.rstrip("/")), sampling)
    return "unknown_dataset"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def patients_to_slices(root_path, labeled_num, fold=None):
    labeled_num = float(labeled_num)
    list_files = _resolve_train_slice_files(root_path, fold)
    if not list_files:
        return 1 if labeled_num > 0 else 0

    total_train_slices = 0
    for list_file in list_files:
        with open(list_file, "r", encoding="utf-8") as f:
            total_train_slices += len([line.strip() for line in f if line.strip()])

    if labeled_num <= 0 or total_train_slices <= 0:
        return 0

    ratio = min(labeled_num, 100.0) / 100.0
    labeled_slices = math.ceil(total_train_slices * ratio)
    return max(1, min(total_train_slices, labeled_slices))


def load_classes_info(root_path, task):
    task_id = int(task) if isinstance(task, str) else task
    json_path = os.path.join(root_path, f"task{task_id}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("classes", [])

    for cls in classes:
        if "color" not in cls and "label_id" in cls:
            cls["color"] = DEFAULT_COLORMAP[cls["label_id"] % len(DEFAULT_COLORMAP)]

    return json_path, data


def get_num_classes_from_path(root_path, task):
    _, info = load_classes_info(root_path, task)
    return info["num_classes"]


def get_n_folds_from_path(root_path, task):
    _, info = load_classes_info(root_path, task)
    return info["n_folds"]


def get_fold_seqs_from_path(root_path, task):
    _, info = load_classes_info(root_path, task)
    n_folds = info.get("n_folds", 1)
    raw = info.get("fold_seqs")
    if n_folds > 1 and raw is None:
        logging.warning("n_folds=%d but no fold_seqs in %s/task%s.json, no per-fold filtering", n_folds, root_path, task)
    if not raw:
        return {}
    return {int(k): v for k, v in raw.items()}


def _extract_state_dict_from_checkpoint(checkpoint):
    state_dict = checkpoint
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    if isinstance(state_dict, dict) and "model" in state_dict and "model_state" in checkpoint:
        state_dict = state_dict["model"]
    return state_dict


def load_checkpoint(model, checkpoint_path):
    logger = logging.getLogger(__name__)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state_dict)
    info = {key: checkpoint[key] for key in CHECKPOINT_INFO_KEYS if key in checkpoint}

    logger.info("Loaded checkpoint from %s", checkpoint_path)
    if info:
        logger.info("Checkpoint info: %s", info)
    return info
