import copy
import logging
import os
from dataclasses import dataclass

import torch
from utils.common import (
    build_run_output_dir,
    get_dataset_result_dir_name,
    get_model_name,
)


@dataclass(frozen=True)
class TestRunContext:
    model_name: str
    effective_lr: float
    checkpoint_type: str
    parent_snapshot_path: str
    parent_output_path: str
    dataset_result_path: str


def resolve_device(args):
    if not torch.cuda.is_available() and "cuda" in args.device.lower():
        logging.warning("CUDA requested but not available. Using CPU.")
    if torch.cuda.is_available() and "cuda" in args.device.lower():
        return torch.device(args.device)
    return torch.device("cpu")


def resolve_train_folds(args):
    if args.fold != -1:
        return [args.fold]
    if args.num_folds is None:
        return [None]
    return list(range(args.num_folds))


def resolve_train_run_args(args, fold):
    run_args = copy.deepcopy(args)
    run_args.fold = fold
    run_args.snapshot_path = args.snapshot_path if args.snapshot_path else build_run_output_dir(run_args, mode="train", fold=fold)
    return run_args


def existing_train_checkpoints(snapshot_path, no_val):
    checkpoint_files = ["model_final.pth"] if no_val else ["model_best.pth", "model_final.pth"]
    return [filename for filename in checkpoint_files if os.path.exists(os.path.join(snapshot_path, filename))]

def build_test_run_context(args):
    return TestRunContext(
        model_name=get_model_name(args.model, args.filter_num, args.use_depth, args.strong, args.pretrain, include_filter_num=False, way=args.way),
        effective_lr=args.lr,
        checkpoint_type=args.checkpoint_type,
        parent_snapshot_path=build_run_output_dir(args, mode="train"),
        parent_output_path=build_run_output_dir(args, mode="test"),
        dataset_result_path=os.path.join(
            args.predict_result_root,
            get_dataset_result_dir_name(args.exp, args.root_path, args.task, sampling=args.sampling),
        ),
    )
