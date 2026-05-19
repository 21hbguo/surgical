import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field

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
from data import BaseDataSets, OneStreamBatchSampler, RandomGenerator, TwoStreamBatchSampler
from models.factory import create_model
from strategies import create_strategy
from utils.lr_scheduler import build_lr_scheduler
from utils.metrics import compute_dice_per_class, compute_depth_psnr_ssim
from utils.common import patients_to_slices, setup_seed
from utils.save_vars_to_csv import save_vars_to_csv

warnings.filterwarnings("ignore", message=".*timm.models.layers.*")


@dataclass(frozen=True)
class TrainComponents:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    strategy: object
    train_loader: DataLoader
    val_loader: DataLoader | None
    labeled_slice: int | float
    scaler: torch.amp.GradScaler | None = None


def create_dataloaders(args):
    def repeat_to_min_length(indices, min_length):
        values = list(indices)
        if min_length <= 0 or not values or len(values) >= min_length:
            return values
        repeat_factor = (min_length + len(values) - 1) // len(values)
        return (values * repeat_factor)[:min_length]

    depth_channels = args.use_depth if args.use_depth else None
    depth_uint = int(args.depth_uint)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num, args.fold)
    if args.debug:
        train_num, val_num, debug_labeled_slice = 100, 50, min(20, labeled_slice)
    elif args.is_semi_supervised:
        train_num, val_num, debug_labeled_slice = None, None, None
    else:
        train_num, val_num, debug_labeled_slice = labeled_slice, None, None

    train_dataset = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        fold=args.fold,
        num=train_num,
        resize_size=tuple(args.resize_size),
        load_mode=args.load_mode,
        num_classes=args.num_classes,
        depth_channels=depth_channels,
        depth_uint=depth_uint,
        strategy=args.way if depth_channels else None,
        normalize_method=args.normalize,
        sampling=args.sampling,
        task=args.task,
        transform=RandomGenerator(
            resize_size=tuple(args.resize_size),
            is_val=False,
            root_path=args.root_path,
            depth_channels=depth_channels,
        ),
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty. "
            f"root_path={args.root_path}, fold={args.fold}, task={args.task}. "
            "Expected one of train_slices*.list or train.list under root_path."
        )

    val_dataset = None
    if not args.no_val:
        val_dataset = BaseDataSets(
            base_dir=args.root_path,
            split="val",
            fold=args.fold,
            num=val_num,
            resize_size=tuple(args.resize_size),
            load_mode=args.load_mode,
            num_classes=args.num_classes,
            depth_channels=depth_channels,
            depth_uint=depth_uint,
            strategy=args.way if depth_channels else None,
            normalize_method=args.normalize,
            sampling=args.sampling,
            task=args.task,
            transform=RandomGenerator(
                resize_size=tuple(args.resize_size),
                is_val=True,
                root_path=args.root_path,
                depth_channels=depth_channels,
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
    sampling = args.sampling
    count = max(0, min(len(train_dataset), int(actual_labeled_slice)))
    if count == 0 and float(args.labeled_num) > 0 and len(train_dataset) > 0:
        count = 1
    if count == 0:
        labeled_indices = []
    elif sampling == "interval" and count < len(train_dataset):
        labeled_indices = [(index * len(train_dataset)) // count for index in range(count)]
    else:
        labeled_indices = list(range(count))
    labeled_set = set(labeled_indices)
    unlabeled_indices = [index for index in range(len(train_dataset)) if index not in labeled_set]

    if args.is_semi_supervised:
        if not labeled_indices:
            raise ValueError(
                "Semi-supervised training requires at least one labeled sample. "
                "Increase --labeled_num or reduce the dataset split."
            )
        if not unlabeled_indices:
            raise ValueError(
                "Semi-supervised training requires at least one unlabeled sample. "
                "Reduce --labeled_num or switch to a fully supervised strategy."
            )
        labeled_indices = repeat_to_min_length(labeled_indices, args.labeled_bs)
        unlabeled_indices = repeat_to_min_length(unlabeled_indices, args.unlabeled_bs)
        batch_sampler = TwoStreamBatchSampler(
            labeled_indices,
            unlabeled_indices,
            batch_size=args.labeled_bs + args.unlabeled_bs,
            secondary_batch_size=args.unlabeled_bs,
            seed=args.seed,
        )
    else:
        if not labeled_indices:
            raise ValueError(
                "Training requires at least one labeled sample. "
                "Increase --labeled_num or check the dataset split files."
            )
        labeled_indices = repeat_to_min_length(labeled_indices, args.labeled_bs)
        batch_sampler = OneStreamBatchSampler(
            labeled_indices,
            batch_size=args.labeled_bs,
            seed=args.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader, labeled_slice


def _freeze_dinov3_backbone_if_needed(model, args, logger=None):
    if not args.freeze:
        return 0
    if args.pretrain != "dinov3":
        if logger is not None:
            logger.info("--freeze enabled but --pretrain is not dinov3; skip freezing.")
        return 0

    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", None) if encoder is not None else None
    if backbone is None:
        if logger is not None:
            logger.warning("--freeze enabled but model.encoder.backbone not found; skip freezing.")
        return 0

    frozen = 0
    for param in backbone.parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen += param.numel()
    if logger is not None:
        logger.info("DINOv3 backbone frozen: %d parameters.", frozen)
    return frozen


def build_train_components(args, device, logger=None):
    model = create_model(args).to(device)
    _freeze_dinov3_backbone_if_needed(model, args, logger)
    if args.compile:
        model = torch.compile(model)
        if logger:
            logger.info("torch.compile enabled")
    scaler = torch.amp.GradScaler(device) if args.amp else None
    if scaler and logger:
        logger.info("AMP enabled: GradScaler created")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001)
    strategy = create_strategy(args.way, args, model, optimizer, device, scaler=scaler)
    train_loader, val_loader, labeled_slice = create_dataloaders(args)
    return TrainComponents(
        model=model,
        optimizer=optimizer,
        strategy=strategy,
        train_loader=train_loader,
        val_loader=val_loader,
        labeled_slice=labeled_slice,
        scaler=scaler,
    )


class Trainer:
    def __init__(self, args, strategy, train_loader, val_loader, device, labeled_slice=None):
        self.args = args
        self.strategy = strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.labeled_slice = labeled_slice
        self.iter_num = 0
        self.best_performance = 0.0
        self.snapshot_path = args.snapshot_path
        self.max_iterations = args.max_iterations
        self.has_val = val_loader is not None
        self.val_iter = args.val_iter if self.has_val else 0
        self.last_val_iter = -1
        self.best_iter = 0
        self.patience = int(args.max_iterations * args.early_stopping) if args.early_stopping > 0 else 0
        os.makedirs(self.snapshot_path, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.lr_scheduler = build_lr_scheduler(self.strategy.optimizer, args)
        self.logger.info(f"LR Scheduler: {args.lr_scheduler}, Warmup: {args.lr_warmup_iters} iters")
        if not self.has_val:
            self.logger.info("Trainer running without validation loader")
        self._save_dataset_lists()

    def _save_dataset_lists(self):
        train_list = self.train_loader.dataset.sample_list
        if self.labeled_slice is not None:
            batch_sampler = getattr(self.train_loader, "batch_sampler", None)
            labeled_indices = getattr(batch_sampler, "primary_indices", None)
            unlabeled_indices = getattr(batch_sampler, "secondary_indices", None)
            if labeled_indices is not None and unlabeled_indices is not None:
                labeled_list = [train_list[int(idx)] for idx in labeled_indices]
                unlabeled_list = [train_list[int(idx)] for idx in unlabeled_indices]
            else:
                labeled_list = train_list[:self.labeled_slice]
                unlabeled_list = train_list[self.labeled_slice:]
            with open(os.path.join(self.snapshot_path, "data_train_labeled.list"), "w") as handle:
                for item in labeled_list:
                    handle.write(f"{item}\n")
            with open(os.path.join(self.snapshot_path, "data_train_unlabeled.list"), "w") as handle:
                for item in unlabeled_list:
                    handle.write(f"{item}\n")
            self.logger.info("Saved labeled (%d) and unlabeled (%d) lists", len(labeled_list), len(unlabeled_list))
        else:
            with open(os.path.join(self.snapshot_path, "data_train.list"), "w") as handle:
                for item in train_list:
                    handle.write(f"{item}\n")
        if self.has_val:
            val_list = self.val_loader.dataset.sample_list
            with open(os.path.join(self.snapshot_path, "data_val.list"), "w") as handle:
                for item in val_list:
                    handle.write(f"{item}\n")
        else:
            self.logger.info("No validation dataset list saved because no_val mode is enabled")
        self.logger.info("Saved dataset lists to %s", self.snapshot_path)

    def train(self):
        self.logger.info("Starting training for %d iterations", self.max_iterations)
        if self.patience > 0:
            self.logger.info("Early stopping enabled: patience=%d iterations (%.0f%% of max)", self.patience, self.args.early_stopping * 100)
        try:
            if len(self.train_loader) == 0:
                raise RuntimeError(
                    "Training loader produced zero batches. "
                    "Check --labeled_num, batch sizes, and dataset split files."
                )
            self.max_epoch = self.max_iterations // len(self.train_loader) + 1
            self.pbar = tqdm(total=self.max_epoch, ncols=70)
            for epoch in range(self.max_epoch):
                if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                    self.train_loader.batch_sampler.set_epoch(epoch)

                epoch_loss = self._train_epoch(epoch)
                self.pbar.update(1)
                self.pbar.set_postfix_str(f"loss {epoch_loss:.4f}")
                if self.iter_num >= self.max_iterations:
                    break
            self.pbar.close()
            if self.has_val and self.last_val_iter != self.iter_num:
                self._validate_and_save(self.max_epoch - 1)
            self._save_model("final")
            if self.has_val:
                if self._is_depth_pretrain_strategy():
                    self.logger.info("Training complete. Best PSNR: %.4f", self.best_performance)
                else:
                    self.logger.info("Training complete. Best Dice: %.4f", self.best_performance)
            else:
                self.logger.info("Training complete without validation. Final checkpoint saved.")
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
            raise
        except Exception:
            self.logger.exception("Training failed unexpectedly.")
            raise

    def _train_epoch(self, epoch):
        epoch_loss = 0.0
        num_batches = 0
        for _, batch_data in enumerate(self.train_loader):
            self.lr_scheduler.step(self.iter_num)
            batch_data = self._process_batch(batch_data)
            loss_dict = self.strategy.training_step(batch_data, self.iter_num, epoch)
            last_loss = loss_dict.get("total", 0.0)
            if isinstance(last_loss, torch.Tensor):
                last_loss = float(last_loss.detach().item())
            epoch_loss += last_loss
            num_batches += 1
            self.iter_num += 1
            if self.has_val and self.val_iter > 0 and self.iter_num % self.val_iter == 0:
                self._validate_and_save(epoch, loss_dict)
                if self.patience > 0 and self.iter_num - self.best_iter >= self.patience:
                    self.logger.info(
                        "Early stopping: no improvement for %d iterations (patience=%d, best at iter %d).",
                        self.iter_num - self.best_iter, self.patience, self.best_iter,
                    )
                    self.iter_num = self.max_iterations
                    return epoch_loss / num_batches if num_batches > 0 else 0.0
            if self.iter_num >= self.max_iterations:
                break
        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _is_depth_pretrain_strategy(self):
        return str(self.args.way).lower() == "fully_depth_pretrain_v1"

    def _process_batch(self, batch_data):
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
        return batch_data

    def _validate_and_save(self, epoch, loss_dict=None):
        if not self.has_val:
            self.logger.info("Skip validation because no validation loader is configured")
            return

        is_depth_pretrain = self._is_depth_pretrain_strategy()
        metric_list = []
        self._set_eval_mode()
        amp_enabled = self.args.amp
        try:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                for _, batch_data in enumerate(self.val_loader):
                    self._set_eval_mode()
                    batch_data = self._process_batch(batch_data)
                    output = self.strategy.validation_step(batch_data)
                    val_output = output[0] if isinstance(output, (list, tuple)) else output

                    if is_depth_pretrain:
                        depth_target = batch_data.get("depth3")
                        if depth_target is None:
                            depth_target = batch_data.get("depth1")
                        if depth_target is None:
                            raise KeyError("Validation for fully_depth_pretrain_v1 requires depth3 or depth1 in batch_data")
                        metrics = compute_depth_psnr_ssim(val_output, depth_target)
                    else:
                        label = batch_data["label"]
                        dice_scores = compute_dice_per_class(val_output, label, self.args.num_classes)
                        metrics = {"dice": float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0}

                    metric_list.append(metrics)
                    del output, val_output, batch_data
        finally:
            self._set_train_mode()

        if is_depth_pretrain:
            psnr_values = [m["psnr"] for m in metric_list]
            ssim_values = [m["ssim"] for m in metric_list]
            mean_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
            mean_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
            current_metric = mean_psnr
            metric_name = "PSNR"
            log_extra = f", SSIM={mean_ssim:.4f}"
        else:
            dice_values = [m["dice"] for m in metric_list]
            current_metric = float(np.mean(dice_values)) if dice_values else 0.0
            metric_name = "Dice"
            log_extra = ""

        if current_metric >= self.best_performance and not np.isnan(current_metric):
            self.best_performance = current_metric
            self.best_iter = self.iter_num
            self._save_model("best")

        self.logger.info("Iter %d: %s=%.4f%s Best=%.4f", self.iter_num, metric_name, current_metric, log_extra, self.best_performance)
        self.last_val_iter = self.iter_num

        csv_path = os.path.join(self.snapshot_path, "metrics.csv")
        train_loss = loss_dict.get("total", 0.0) if loss_dict else 0.0
        if isinstance(train_loss, torch.Tensor):
            train_loss = float(train_loss.detach().item())
        current_lr = self.lr_scheduler.get_last_lr() if hasattr(self.lr_scheduler, 'get_last_lr') else [pg['lr'] for pg in self.strategy.optimizer.param_groups]
        csv_vars = {"train_loss": train_loss, "lr": current_lr[0] if isinstance(current_lr, list) else current_lr}
        if is_depth_pretrain:
            csv_vars["psnr"] = mean_psnr
            csv_vars["ssim"] = mean_ssim
        else:
            csv_vars["dice"] = current_metric
            csv_vars["best_dice"] = self.best_performance
        save_vars_to_csv(csv_path, self.iter_num, **csv_vars)

    def _set_eval_mode(self):
        if hasattr(self.strategy, "eval"):
            self.strategy.eval()
        else:
            self.strategy.model.eval()

    def _set_train_mode(self):
        if hasattr(self.strategy, "train"):
            self.strategy.train()
        else:
            self.strategy.model.train()

    def _save_model(self, suffix):
        path = os.path.join(self.snapshot_path, f"model_{suffix}.pth")
        if suffix == "final" and self.has_val:
            with open(path, "wb"):
                pass
            return
        state = {
            "model_state": self.strategy.get_state_dict(),
            "args": vars(self.args).copy(),
            "iter_num": self.iter_num,
            "best_performance": self.best_performance,
        }
        self._atomic_torch_save(state, path)

    def _atomic_torch_save(self, state, path):
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "wb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


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
            existing_checkpoints = existing_train_checkpoints(run_args.snapshot_path, run_args.no_val)
            expected_count = 1 if run_args.no_val else 2
            if len(existing_checkpoints) == expected_count:
                fold_info = f"f{run_args.fold}" if run_args.fold is not None else "no fold"
                logging.warning("Checkpoints already exist: %s", ", ".join(existing_checkpoints))
                logging.warning("Fold: %s, Skip training.", fold_info)
                continue
            if existing_checkpoints:
                fold_info = f"f{run_args.fold}" if run_args.fold is not None else "no fold"
                logging.info(
                    "Partial checkpoints found: %s, continuing training on %s",
                    ", ".join(existing_checkpoints),
                    fold_info,
                )

        os.makedirs(run_args.snapshot_path, exist_ok=True)
        logger = setup_logging(run_args.snapshot_path)
        logger.info("Args: %s", format_args_for_logging(run_args))
        logger.info("Strategy: %s", run_args.way)
        logger.info("Device: %s", device)
        if run_args.no_val:
            logger.info("Validation disabled by --no_val")
            if run_args.val_iter > 0:
                logger.info("--val_iter ignored because --no_val is enabled")

        logger.info("Using Adam: LR=%s, WD=0.0001, betas=(0.9, 0.99)", run_args.lr)
        components = build_train_components(run_args, device, logger)
        total_slices = len(components.train_loader.dataset)
        logger.info(
            "Total slices is: %s, labeled slices is: %s, unlabeled slices is: %s",
            total_slices,
            components.labeled_slice,
            total_slices - components.labeled_slice,
        )

        components.strategy.model.train()
        trainer = Trainer(
            run_args,
            components.strategy,
            components.train_loader,
            components.val_loader,
            device,
            labeled_slice=components.labeled_slice if run_args.is_semi_supervised else None,
        )
        trainer.train()
        if run_args.no_val:
            logger.info("Training completed. Final model saved at: %s", run_args.snapshot_path)
        else:
            logger.info("Training completed. Best model saved at: %s", run_args.snapshot_path)


if __name__ == "__main__":
    main()
