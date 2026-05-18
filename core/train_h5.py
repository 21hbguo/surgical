import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.args import build_train_parser, finalize_train_args, format_args_for_logging
from core.runtime import (
    existing_train_checkpoints,
    resolve_device,
    resolve_train_folds,
    resolve_train_run_args,
)
from core.train import Trainer, build_train_components
from data import H5DataSets, OneStreamBatchSampler, RandomGenerator, TwoStreamBatchSampler
from utils.common import patients_to_slices, setup_seed

warnings.filterwarnings("ignore", message=".*timm.models.layers.*")


def _repeat_to_min_length(indices, min_length):
    values = list(indices)
    if min_length <= 0 or not values or len(values) >= min_length:
        return values
    repeat_factor = (min_length + len(values) - 1) // len(values)
    return (values * repeat_factor)[:min_length]


def create_dataloaders_h5(args):
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num, args.fold)
    if args.debug:
        train_num, val_num, debug_labeled_slice = 100, 50, min(20, labeled_slice)
    elif args.is_semi_supervised:
        train_num, val_num, debug_labeled_slice = None, None, None
    else:
        train_num, val_num, debug_labeled_slice = labeled_slice, None, None

    train_dataset = H5DataSets(
        base_dir=args.root_path,
        split="train",
        fold=args.fold,
        num=train_num,
        resize_size=tuple(args.resize_size),
        num_classes=args.num_classes,
        normalize_method=args.normalize,
        sampling=args.sampling,
        task=args.task,
        transform=RandomGenerator(
            resize_size=tuple(args.resize_size),
            is_val=False,
            root_path=args.root_path,
        ),
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    val_dataset = None
    if not args.no_val:
        val_dataset = H5DataSets(
            base_dir=args.root_path,
            split="val",
            fold=args.fold,
            num=val_num,
            resize_size=tuple(args.resize_size),
            num_classes=args.num_classes,
            normalize_method=args.normalize,
            sampling=args.sampling,
            task=args.task,
            transform=RandomGenerator(
                resize_size=tuple(args.resize_size),
                is_val=True,
                root_path=args.root_path,
            ),
        )

    def worker_init_fn(worker_id):
        seed = args.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    actual_labeled_slice = debug_labeled_slice if args.debug else labeled_slice
    count = max(0, min(len(train_dataset), int(actual_labeled_slice)))
    if count == 0 and float(args.labeled_num) > 0 and len(train_dataset) > 0:
        count = 1
    if count == 0:
        labeled_indices = []
    elif args.sampling == "interval" and count < len(train_dataset):
        labeled_indices = [(i * len(train_dataset)) // count for i in range(count)]
    else:
        labeled_indices = list(range(count))
    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(train_dataset)) if i not in labeled_set]

    if args.is_semi_supervised:
        if not labeled_indices:
            raise ValueError("Semi-supervised requires at least one labeled sample.")
        if not unlabeled_indices:
            raise ValueError("Semi-supervised requires at least one unlabeled sample.")

        labeled_indices = _repeat_to_min_length(labeled_indices, args.labeled_bs)
        unlabeled_indices = _repeat_to_min_length(unlabeled_indices, args.unlabeled_bs)
        batch_sampler = TwoStreamBatchSampler(
            labeled_indices, unlabeled_indices,
            batch_size=args.labeled_bs + args.unlabeled_bs,
            secondary_batch_size=args.unlabeled_bs, seed=args.seed,
        )
    else:
        if not labeled_indices:
            raise ValueError("Training requires at least one labeled sample.")
        labeled_indices = _repeat_to_min_length(labeled_indices, args.labeled_bs)
        batch_sampler = OneStreamBatchSampler(labeled_indices, batch_size=args.labeled_bs, seed=args.seed)

    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
    )
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader, labeled_slice


def setup_logging(snapshot_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return logging.getLogger(__name__)


def main():
    args = finalize_train_args(build_train_parser().parse_args())
    setup_seed(args.seed)
    device = resolve_device(args)

    for fold in resolve_train_folds(args):
        run_args = resolve_train_run_args(args, fold)

        if os.path.exists(run_args.snapshot_path):
            existing = existing_train_checkpoints(run_args.snapshot_path, run_args.no_val)
            expected = 1 if run_args.no_val else 2
            if len(existing) == expected:
                logging.warning("Checkpoints exist: %s, skip fold %s", ", ".join(existing), run_args.fold)
                continue

        os.makedirs(run_args.snapshot_path, exist_ok=True)
        logger = setup_logging(run_args.snapshot_path)
        logger.info("Args: %s", format_args_for_logging(run_args))
        logger.info("Device: %s", device)

        # Monkey-patch create_dataloaders for this scope
        import core.train
        original_create = core.train.create_dataloaders
        core.train.create_dataloaders = create_dataloaders_h5
        try:
            components = build_train_components(run_args, device, logger)
        finally:
            core.train.create_dataloaders = original_create

        total_slices = len(components.train_loader.dataset)
        logger.info("Total slices: %s, labeled: %s, unlabeled: %s",
                     total_slices, components.labeled_slice, total_slices - components.labeled_slice)

        components.strategy.model.train()
        trainer = Trainer(
            run_args, components.strategy,
            components.train_loader, components.val_loader, device,
            labeled_slice=components.labeled_slice if run_args.is_semi_supervised else None,
        )
        trainer.train()
        logger.info("Training completed. Model saved at: %s", run_args.snapshot_path)


if __name__ == "__main__":
    main()
