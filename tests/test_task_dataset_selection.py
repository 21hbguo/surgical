import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from core.args import build_test_parser, build_train_parser, finalize_test_args
import core.test as test_core
import core.train as train_core
from core.testing.export import build_summary_row
from core.testing.visualization import colorize_test_mask
from data import BaseDataSets
from strategies import create_strategy
from strategies import semi_dycon, semi_uncertainty_mt, semi_w2s
from strategies.specs import resolve_strategy_input_settings
import utils.common as common
from utils.common import get_n_folds_from_path, get_num_classes_from_path


def _write_task_json(root_path: str, task: int, num_classes: int, n_folds: int, input_channels: int = 3) -> None:
    with open(os.path.join(root_path, f"task{task}.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_classes": num_classes,
                "input_channels": input_channels,
                "n_folds": n_folds,
                "classes": [{"name": "class0", "label_id": 0}],
            },
            handle,
        )


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)


class TaskDatasetSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_path = self.tmpdir.name
        data_dir = Path(self.root_path) / "data"
        (data_dir / "images").mkdir(parents=True)
        (data_dir / "labels_task1_binary").mkdir(parents=True)
        (data_dir / "labels_task2_part").mkdir(parents=True)
        (data_dir / "labels_task3_class").mkdir(parents=True)
        (data_dir / "images" / "case_a.png").touch()
        (data_dir / "labels_task1_binary" / "case_a.png").touch()
        (data_dir / "labels_task2_part" / "case_a.png").touch()
        (data_dir / "labels_task3_class" / "case_a.png").touch()
        _write_task_json(self.root_path, task=1, num_classes=2, n_folds=4)
        _write_task_json(self.root_path, task=2, num_classes=4, n_folds=5)
        _write_task_json(self.root_path, task=3, num_classes=7, n_folds=6)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_metadata_functions_read_task_json(self):
        self.assertEqual(get_num_classes_from_path(self.root_path, task=2), 4)
        self.assertEqual(get_n_folds_from_path(self.root_path, task=3), 6)

        args = build_train_parser().parse_args(["--task", "1", "--exp", "toy/Fully", "--use_depth", "1"])
        resolved = resolve_strategy_input_settings(way=args.way, root_path=self.root_path, task=1, use_depth=1)
        args.in_chns = resolved["in_chns"]
        args.use_depth = resolved["use_depth"]
        self.assertEqual(args.in_chns, 4)

    def test_resolve_input_channel_settings_shares_depth_rules(self):
        resolved = resolve_strategy_input_settings(way="fully", root_path=self.root_path, task=1, use_depth=13)
        self.assertEqual(resolved["metadata_in_chns"], 3)
        self.assertEqual(resolved["use_depth"], 13)
        self.assertEqual(resolved["in_chns"], 4)

    def test_set_input_channels_keeps_train_semantics_for_mt_depth_teacher_strategy_path(self):
        resolved = resolve_strategy_input_settings(
            way="/home/guo/project/ssl4mis/code_all/strategies/semi_mt_depth_teacher_v1.py",
            root_path=self.root_path,
            task=1,
            use_depth=1,
        )
        self.assertEqual(resolved["use_depth"], 1)
        self.assertEqual(resolved["in_chns"], 3)

    def test_set_input_channels_keeps_rgb_only_model_for_fully_rgb_masking_depth_strategy(self):
        resolved = resolve_strategy_input_settings(
            way="fully_rgb_masking_depth_v1",
            root_path=self.root_path,
            task=1,
            use_depth=3,
        )
        self.assertEqual(resolved["use_depth"], 3)
        self.assertEqual(resolved["in_chns"], 3)

    def test_common_no_longer_exposes_input_channel_resolution_wrappers(self):
        self.assertFalse(hasattr(common, "get_channel_number"))
        self.assertFalse(hasattr(common, "resolve_input_channel_settings"))
        self.assertFalse(hasattr(common, "set_input_channels"))

    def test_dataset_uses_task_specific_label_directory(self):
        image_path, label_path = BaseDataSets._find_png_paths_static("case_a", self.root_path, "train", task=2)
        self.assertTrue(image_path.endswith(os.path.join("images", "case_a.png")))
        self.assertTrue(label_path.endswith(os.path.join("labels_task2_part", "case_a.png")))

    def test_dataset_train_split_accepts_case_names_with_extension(self):
        image = np.full((2, 2, 3), 64, dtype=np.uint8)
        label = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        _write_png(Path(self.root_path) / "data" / "images" / "case_ext.png", image)
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "case_ext.png", label)
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_ext.png\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(2, 2),
            load_mode="path",
            num_classes=2,
            task=1,
        )

        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 2, 2))
        self.assertEqual(sorted(sample["label"].unique().tolist()), [0, 1])

    def test_dataset_test_split_keeps_dotted_case_name_stem(self):
        image = np.full((2, 2, 3), 64, dtype=np.uint8)
        label = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        _write_png(Path(self.root_path) / "data" / "images" / "patient.001.png", image)
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "patient.001.png", label)
        with open(os.path.join(self.root_path, "test_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("patient.001.png\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="test",
            resize_size=(2, 2),
            load_mode="path",
            num_classes=2,
            normalize_method="255",
            for_inference=True,
            task=1,
        )

        sample = dataset[0]
        self.assertEqual(sample["case"], "patient.001")

    def test_package_import_exposes_dataset_symbols(self):
        from data import BaseDataSets as PackageBaseDataSets

        image_path, label_path = PackageBaseDataSets._find_png_paths_static(
            "case_a",
            self.root_path,
            "train",
            task=2,
        )
        self.assertTrue(image_path.endswith(os.path.join("images", "case_a.png")))
        self.assertTrue(label_path.endswith(os.path.join("labels_task2_part", "case_a.png")))

    def test_task_segment_is_inserted_into_model_exp_path(self):
        scoped_exp = common.build_task_scoped_exp_name("endovis2018Binary8_imagenet/MT", task=2)
        self.assertEqual(scoped_exp, os.path.join("endovis2018Binary8_imagenet", "task2", "MT"))

    def test_sampling_suffix_is_inserted_into_output_dataset_segment(self):
        scoped_exp = common.build_task_scoped_exp_name(
            "endovis2018Binary8_imagenet/MT",
            task=2,
            sampling="none",
        )
        self.assertEqual(scoped_exp, os.path.join("endovis2018Binary8_imagenet_Samplingnone", "task2", "MT"))

    def test_test_context_builds_task_scoped_artifact_paths(self):
        args = build_test_parser().parse_args(
            [
                "--task",
                "2",
                "--exp",
                "endovis2018Binary8/MT",
                "--way",
                "mt",
                "--labeled_num",
                "40",
                "--normalize",
                "imagenet",
                "--lr",
                "1e-4",
                "--sampling",
                "interval",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)
        context = test_core.build_test_run_context(args)
        self.assertEqual(
            context.dataset_result_path,
            os.path.join("/home/guo/project/ssl4mis/result_predict", "endovis2018Binary8_imagenet_Samplinginterval"),
        )
        self.assertIn(
            os.path.join("endovis2018Binary8_imagenet_Samplinginterval", "task2", "MT", "40_labeled_lr1e-4_s_resnet"),
            context.parent_snapshot_path,
        )

    def test_metric_summary_includes_average_and_per_class_fields(self):
        metrics = [
            {"Dice": 0.8, "IoU": 0.7, "TP": 8.0, "FP": 2.0, "FN": 1.0, "Acc": 0.9, "Class": 1},
            {"Dice": 0.6, "IoU": 0.5, "TP": 6.0, "FP": 4.0, "FN": 3.0, "Acc": 0.8, "Class": 1},
        ]

        summary = test_core.summarize_metrics(metrics, num_classes=2)

        self.assertEqual(summary["Avg_Dice"], "0.7000 ± 0.1000")
        self.assertEqual(summary["Avg_IoU"], "0.6000 ± 0.1000")
        self.assertIn("Avg_Precision", summary)
        self.assertIn("Avg_Recall", summary)
        self.assertIn("C1_Dice", summary)
        self.assertIn("C1_Acc", summary)

    def test_build_summary_row_uses_expected_display_columns(self):
        args = build_test_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/MT",
                "--way",
                "mt",
                "--labeled_num",
                "10",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)
        row = build_summary_row(
            args,
            {
                "Avg_Dice": "0.7000 ± 0.1000",
                "Avg_Precision": "0.8000 ± 0.0500",
                "C1_Dice": "0.7000 ± 0.1000",
            },
            fold_label="f0",
            train_best_dice=0.85,
            total_samples=12,
        )

        self.assertEqual(
            row["Experiment"],
            "MT_s_resnet_10.0_4_adam_3.0e-05_task1",
        )
        self.assertNotIn("Model", row)
        self.assertNotIn("Labeled_Ratio", row)
        self.assertNotIn("Strategy", row)
        self.assertNotIn("Optimizer", row)
        self.assertNotIn("LR", row)
        self.assertEqual(row["Fold"], "f0")
        self.assertEqual(row["train_best_dice"], "0.8500")
        self.assertEqual(row["Total_Samples"], 12)
        self.assertEqual(row["Avg_dice"], "70.00 ± 10.00")
        self.assertEqual(row["Avg_precision"], "80.00 ± 5.00")
        self.assertEqual(row["C1_dice"], "70.00 ± 10.00")

    def test_strategy_defaults_are_flattened_on_args(self):
        args = build_train_parser().parse_args(["--task", "1"])
        self.assertFalse(hasattr(args, "STRATEGY"))
        self.assertEqual(args.consistency, 0.1)
        self.assertEqual(args.ema_decay, 0.99)
        self.assertFalse(hasattr(args, "proto_feature_dim"))
        proto_args = build_train_parser().parse_args(["--task", "1", "--way", "proto"])
        self.assertEqual(proto_args.proto_feature_dim, 256)
        self.assertEqual(semi_uncertainty_mt.DEFAULT_T_SAMPLES, 8)
        self.assertEqual(semi_dycon.DEFAULT_BETA_MIN, 0.5)
        self.assertEqual(semi_dycon.DEFAULT_FEATURE_SCALER, 2)
        self.assertEqual(semi_w2s.DEFAULT_CONTRASTIVE_MARGIN, 0.5)

    def test_dataset_test_inference_keeps_original_shape_and_separates_depth(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_depth.png", np.full((2, 3, 3), 128, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth.png", np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices_uint16" / "case_depth.png", np.full((2, 3), 32, dtype=np.uint8))
        with open(os.path.join(self.root_path, "test_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth.png\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="test",
            resize_size=(4, 5),
            load_mode="path",
            num_classes=2,
            depth_channels=1,
            normalize_method="minmax",
            for_inference=True,
            is_depth=True,
            task=1,
        )

        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 4, 5))
        self.assertEqual(tuple(sample["depth1"].shape), (1, 4, 5))
        self.assertEqual(tuple(sample["label"].shape), (2, 3))
        self.assertEqual(sample["case"], "case_depth")
        self.assertEqual(sample["original_shape"].tolist(), [2, 3])
        self.assertEqual(sample["original_image"].shape, (4, 5, 3))

    def test_build_test_loader_uses_original_label_only(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_loader.png", np.full((2, 3, 3), 128, dtype=np.uint8))
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_loader.png",
            np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8),
        )
        with open(os.path.join(self.root_path, "test_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_loader.png\n")

        args = Namespace(
            root_path=self.root_path,
            resize_size=(4, 5),
            num_classes=2,
            use_depth=0,
            depth_uint=16,
            normalize="minmax",
            task=1,
            batch_size=1,
        )

        with patch.object(test_core, "TEST_NUM_WORKERS", 0):
            _, test_loader, _, _ = test_core._build_test_loader(args, fold=None, fold_map={})
            batch = next(iter(test_loader))

        self.assertIn("label", batch)
        self.assertNotIn("metric_label", batch)
        self.assertEqual(batch["label"][0].shape, (2, 3))

    def test_train_dataset_disk_cache_writes_expected_paths(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_cache.png", np.full((2, 3, 3), 128, dtype=np.uint8))
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_cache.png",
            np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8),
        )
        _write_png(Path(self.root_path) / "data" / "depth1c_slices_uint16" / "case_cache.png", np.full((2, 3), 32, dtype=np.uint8))
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_cache.png\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(4, 5),
            load_mode="data",
            num_classes=2,
            depth_channels=1,
            depth_uint=16,
            normalize_method="255",
            is_depth=True,
            task=1,
            max_workers=1,
            cache_mode="disk",
        )

        self.assertEqual(len(dataset.data_cache), 1)
        self.assertTrue((Path(self.root_path) / "data_cache" / "4_5" / "images" / "case_cache.npy").exists())
        self.assertTrue((Path(self.root_path) / "data_cache" / "4_5" / "labels_task1_binary" / "case_cache.npy").exists())
        self.assertTrue((Path(self.root_path) / "data_cache" / "4_5" / "depth1c_slices_uint16" / "case_cache.npy").exists())

    def test_train_dataset_disk_cache_reuses_cached_arrays(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_cache_hit.png", np.full((2, 3, 3), 128, dtype=np.uint8))
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_cache_hit.png",
            np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8),
        )
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_cache_hit.png\n")

        BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(4, 5),
            load_mode="data",
            num_classes=2,
            normalize_method="255",
            task=1,
            max_workers=1,
            cache_mode="disk",
        )
        os.remove(Path(self.root_path) / "data" / "images" / "case_cache_hit.png")
        os.remove(Path(self.root_path) / "data" / "labels_task1_binary" / "case_cache_hit.png")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(4, 5),
            load_mode="data",
            num_classes=2,
            normalize_method="255",
            task=1,
            max_workers=1,
            cache_mode="disk",
        )

        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 4, 5))
        self.assertEqual(tuple(sample["label"].shape), (4, 5))

    def test_build_test_loader_keeps_depth_separate_for_mt_depth_teacher(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_mt_depth.png", np.full((2, 3, 3), 128, dtype=np.uint8))
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_mt_depth.png",
            np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8),
        )
        _write_png(Path(self.root_path) / "data" / "depth3c_slices_uint16" / "case_mt_depth.png", np.full((2, 3, 3), 32, dtype=np.uint8))
        with open(os.path.join(self.root_path, "test_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_mt_depth.png\n")

        args = Namespace(
            root_path=self.root_path,
            resize_size=(4, 5),
            num_classes=2,
            use_depth=3,
            depth_uint=16,
            normalize="255",
            task=1,
            batch_size=1,
        )

        with patch.object(test_core, "TEST_NUM_WORKERS", 0):
            _, test_loader, _, _ = test_core._build_test_loader(args, fold=None, fold_map={})
            batch = next(iter(test_loader))

        self.assertEqual(tuple(batch["image"][0].shape), (3, 4, 5))
        self.assertIn("depth3", batch)
        self.assertEqual(tuple(batch["depth3"][0].shape), (3, 4, 5))

    def test_patients_to_slices_preserves_at_least_one_positive_labeled_sample(self):
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            for index in range(5):
                handle.write(f"case_{index}\n")

        self.assertEqual(common.patients_to_slices(self.root_path, 10), 1)
        self.assertEqual(common.patients_to_slices(self.root_path, 0.1), 1)

    def test_patients_to_slices_always_interprets_labeled_num_as_percent(self):
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            for index in range(200):
                handle.write(f"case_{index}\n")

        self.assertEqual(common.patients_to_slices(self.root_path, 0.1), 1)
        self.assertEqual(common.patients_to_slices(self.root_path, 1), 2)
        self.assertEqual(common.patients_to_slices(self.root_path, 40), 80)
        self.assertEqual(common.patients_to_slices(self.root_path, 100), 200)

    def test_patients_to_slices_uses_train_list_when_train_slices_list_is_missing(self):
        with open(os.path.join(self.root_path, "train.list"), "w", encoding="utf-8") as handle:
            for index in range(200):
                handle.write(f"case_{index}\n")

        self.assertEqual(common.patients_to_slices(self.root_path, 1), 2)
        self.assertEqual(common.patients_to_slices(self.root_path, 40), 80)

    def test_create_dataloaders_keeps_small_positive_semi_supervised_run_trainable(self):
        for index in range(5):
            image = np.full((4, 4, 3), 32 + index, dtype=np.uint8)
            label = np.zeros((4, 4), dtype=np.uint8)
            _write_png(Path(self.root_path) / "data" / "images" / f"tiny_{index}.png", image)
            _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / f"tiny_{index}.png", label)
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            for index in range(5):
                handle.write(f"tiny_{index}\n")
        with open(os.path.join(self.root_path, "val_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("tiny_0\n")

        args = Namespace(
            root_path=self.root_path,
            labeled_num=10,
            fold=None,
            debug=False,
            is_semi_supervised=True,
            resize_size=(4, 4),
            load_mode="path",
            num_classes=2,
            use_depth=0,
            depth_uint=16,
            way="mt",
            normalize="255",
            sampling="interval",
            task=1,
            no_val=False,
            seed=42,
            labeled_bs=2,
            unlabeled_bs=2,
            num_workers=0,
        )

        train_loader, val_loader, labeled_slice = train_core.create_dataloaders(args)

        self.assertEqual(labeled_slice, 1)
        self.assertGreaterEqual(len(train_loader), 1)
        self.assertIsNotNone(val_loader)
        batch = next(iter(train_loader))
        self.assertEqual(batch["image"].shape[0], 4)

    def test_depth3_loading_keeps_raw_range_and_skips_depth1(self):
        _write_png(
            Path(self.root_path) / "data" / "images" / "case_depth3.png",
            np.full((2, 2, 3), 100, dtype=np.uint8),
        )
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth3.png",
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
        )
        _write_png(
            Path(self.root_path) / "data" / "depth3c_slices_uint16" / "case_depth3.png",
            np.full((2, 2, 3), 128, dtype=np.uint8),
        )
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth3\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(2, 2),
            load_mode="path",
            num_classes=2,
            depth_channels=3,
            is_depth=True,
            normalize_method="255",
            task=1,
        )

        raw = dataset._get_sample(0)
        self.assertIn("depth3", raw)
        self.assertNotIn("depth1", raw)
        self.assertEqual(float(raw["depth3"].max()), 128.0)

        sample = dataset[0]
        self.assertEqual(tuple(sample["depth3"].shape), (3, 2, 2))
        self.assertAlmostEqual(float(sample["depth3"].max().item()), 128.0 / 255.0, places=6)

    def test_depth_uint_selects_uint_folder_without_legacy_fallback(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_depth_uint.png", np.full((2, 2, 3), 100, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth_uint.png", np.array([[0, 255], [255, 0]], dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices" / "case_depth_uint.png", np.full((2, 2), 200, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices_uint8" / "case_depth_uint.png", np.full((2, 2), 80, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices_uint16" / "case_depth_uint.png", np.full((2, 2), 16000, dtype=np.uint16))
        _write_png(Path(self.root_path) / "data" / "images" / "case_depth_only_legacy.png", np.full((2, 2, 3), 100, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth_only_legacy.png", np.array([[0, 255], [255, 0]], dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices" / "case_depth_only_legacy.png", np.full((2, 2), 123, dtype=np.uint8))
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth_uint\n")
            handle.write("case_depth_only_legacy\n")
        dataset_u8 = BaseDataSets(base_dir=self.root_path, split="train", resize_size=(2, 2), load_mode="path", num_classes=2, depth_channels=1, depth_uint=8, normalize_method="255", is_depth=True, task=1)
        dataset_u16 = BaseDataSets(base_dir=self.root_path, split="train", resize_size=(2, 2), load_mode="path", num_classes=2, depth_channels=1, depth_uint=16, normalize_method="255", is_depth=True, task=1)
        self.assertEqual(float(dataset_u8._get_sample(0)["depth1"].max()), 80.0)
        self.assertEqual(float(dataset_u16._get_sample(0)["depth1"].max()), 16000.0)
        self.assertNotIn("depth1", dataset_u8._get_sample(1))
        self.assertNotIn("depth1", dataset_u16._get_sample(1))

    def test_depth13_loads_both_depth_inputs_and_keeps_image_rgb_only_in_inference(self):
        _write_png(
            Path(self.root_path) / "data" / "images" / "case_depth13.png",
            np.full((2, 2, 3), 100, dtype=np.uint8),
        )
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth13.png",
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
        )
        _write_png(
            Path(self.root_path) / "data" / "depth3c_slices_uint16" / "case_depth13.png",
            np.full((2, 2, 3), 128, dtype=np.uint8),
        )
        _write_png(
            Path(self.root_path) / "data" / "depth1c_slices_uint16" / "case_depth13.png",
            np.full((2, 2), 64, dtype=np.uint8),
        )
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth13\n")
        with open(os.path.join(self.root_path, "test_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth13.png\n")

        train_dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(2, 2),
            load_mode="path",
            num_classes=2,
            depth_channels=13,
            is_depth=True,
            normalize_method="255",
            task=1,
        )
        raw = train_dataset._get_sample(0)
        self.assertIn("depth1", raw)
        self.assertIn("depth3", raw)

        test_dataset = BaseDataSets(
            base_dir=self.root_path,
            split="test",
            resize_size=(2, 2),
            load_mode="path",
            num_classes=2,
            depth_channels=13,
            normalize_method="255",
            for_inference=True,
            is_depth=True,
            task=1,
        )
        item = test_dataset[0]
        self.assertEqual(tuple(item["image"].shape), (3, 2, 2))
        self.assertEqual(tuple(item["depth1"].shape), (1, 2, 2))
        self.assertEqual(tuple(item["depth3"].shape), (3, 2, 2))
    def test_depth1_255_normalization_distinguishes_uint8_and_uint16_in_train_and_test(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_depth_u8.png", np.full((2, 2, 3), 100, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth_u8.png", np.array([[0, 255], [255, 0]], dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices_uint16" / "case_depth_u8.png", np.full((2, 2), 128, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "images" / "case_depth_u16.png", np.full((2, 2, 3), 100, dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "labels_task1_binary" / "case_depth_u16.png", np.array([[0, 255], [255, 0]], dtype=np.uint8))
        _write_png(Path(self.root_path) / "data" / "depth1c_slices_uint16" / "case_depth_u16.png", np.full((2, 2), 32768, dtype=np.uint16))
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth_u8\n")
            handle.write("case_depth_u16\n")
        with open(os.path.join(self.root_path, "test_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_depth_u8.png\n")
            handle.write("case_depth_u16.png\n")
        train_dataset = BaseDataSets(base_dir=self.root_path, split="train", resize_size=(2, 2), load_mode="path", num_classes=2, depth_channels=1, normalize_method="255", is_depth=True, task=1)
        train_u8 = train_dataset[0]["depth1"].numpy()
        train_u16 = train_dataset[1]["depth1"].numpy()
        self.assertAlmostEqual(float(train_u8.max()), 128.0 / 255.0, places=6)
        self.assertAlmostEqual(float(train_u16.max()), 32768.0 / 65535.0, places=6)
        test_dataset = BaseDataSets(base_dir=self.root_path, split="test", resize_size=(2, 2), load_mode="path", num_classes=2, depth_channels=1, normalize_method="255", for_inference=True, is_depth=True, task=1)
        test_u8 = test_dataset[0]["depth1"].numpy()
        test_u16 = test_dataset[1]["depth1"].numpy()
        self.assertAlmostEqual(float(test_u8.max()), 128.0 / 255.0, places=6)
        self.assertAlmostEqual(float(test_u16.max()), 32768.0 / 65535.0, places=6)

    def test_binary_label_values_are_normalized_to_zero_one(self):
        _write_png(Path(self.root_path) / "data" / "images" / "case_binary.png", np.full((3, 3, 3), 64, dtype=np.uint8))
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_binary.png",
            np.array([[0, 255, 0], [255, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_binary\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(3, 3),
            load_mode="path",
            num_classes=2,
            task=1,
        )

        sample = dataset[0]
        self.assertEqual(sorted(sample["label"].unique().tolist()), [0, 1])

    def test_dataset_without_transform_still_normalizes_image(self):
        image = np.array(
            [
                [[0, 128, 255], [255, 128, 0]],
                [[64, 64, 64], [32, 16, 8]],
            ],
            dtype=np.uint8,
        )
        _write_png(Path(self.root_path) / "data" / "images" / "case_norm.png", image)
        _write_png(
            Path(self.root_path) / "data" / "labels_task1_binary" / "case_norm.png",
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
        )
        with open(os.path.join(self.root_path, "train_slices.list"), "w", encoding="utf-8") as handle:
            handle.write("case_norm\n")

        dataset = BaseDataSets(
            base_dir=self.root_path,
            split="train",
            resize_size=(2, 2),
            load_mode="path",
            num_classes=2,
            normalize_method="255",
            task=1,
        )

        sample = dataset[0]
        expected = dataset._get_sample(0)["image"].astype(np.float32) / 255.0
        np.testing.assert_allclose(sample["image"].numpy(), np.transpose(expected, (2, 0, 1)), atol=1e-6)

    def test_colorize_mask_uses_fixed_ten_class_mapping(self):
        mask = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.uint8)

        rgb = colorize_test_mask(mask, num_classes=10)

        expected = np.array(
            [
                [0, 0, 0],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [255, 128, 0],
                [128, 128, 128],
                [128, 0, 255],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(rgb[0], expected)

    def test_save_test_rgb_visualization_mode2_writes_side_by_side_panel(self):
        output_dir = os.path.join(self.root_path, "rgb")
        image = np.full((2, 3, 3), 100, dtype=np.uint8)
        label = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        pred = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)

        saved_path = test_core.save_test_rgb_visualization(
            image=image,
            label=label,
            pred=pred,
            output_dir=output_dir,
            case_name="case_panel",
            mode=2,
            num_classes=2,
        )

        self.assertEqual(saved_path, os.path.join(output_dir, "case_panel.png"))
        self.assertTrue(os.path.exists(saved_path))
        loaded = cv2.imread(saved_path, cv2.IMREAD_COLOR)
        self.assertEqual(loaded.shape, (2, 6, 3))

    def test_save_multiclass_gradcam_visualization_writes_two_row_grid(self):
        output_dir = os.path.join(self.root_path, "feature_gradcam")
        image = np.array(
            [
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
            ],
            dtype=np.uint8,
        )
        heats = [
            np.full((2, 3), 0.1, dtype=np.float32),
            np.full((2, 3), 0.5, dtype=np.float32),
            np.full((2, 3), 0.9, dtype=np.float32),
        ]

        saved_path = test_core.save_multiclass_gradcam_visualization(
            image=image,
            class_heats=heats,
            output_dir=output_dir,
            case_name="case_gradcam_grid",
            alpha=0.6,
        )

        self.assertEqual(saved_path, os.path.join(output_dir, "case_gradcam_grid.png"))
        loaded = cv2.imread(saved_path, cv2.IMREAD_COLOR)
        self.assertEqual(loaded.shape, (4, 12, 3))

        loaded_rgb = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(loaded_rgb[:2, :3], image)
        np.testing.assert_array_equal(loaded_rgb[2:, :3], image)

    def test_inference_feat_vis_exports_multiclass_gradcam_panel(self):
        args = test_core.build_test_feature_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/MT",
                "--feat_vis",
                "1",
                "--feat_vis_max_cases",
                "1",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)
        args.num_classes = 3

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=1, bias=False)

            def forward(self, x):
                return self.conv(x)

        test_loader = [
            {
                "image": [torch.zeros((3, 2, 2), dtype=torch.float32)],
                "label": [np.zeros((2, 2), dtype=np.uint8)],
                "case": ["case_feat_vis"],
                "original_image": [np.full((2, 2, 3), 80, dtype=np.uint8)],
            }
        ]
        feat_dir = os.path.join(self.root_path, "predict_feat")
        heatmaps = [
            np.full((2, 2), 0.1, dtype=np.float32),
            np.full((2, 2), 0.5, dtype=np.float32),
            np.full((2, 2), 0.9, dtype=np.float32),
        ]

        with patch.object(test_core, "_compute_gradcam_heatmap", side_effect=heatmaps) as heat_mock:
            metrics = test_core.inference(
                args,
                DummyModel(),
                test_loader,
                torch.device("cpu"),
                feat_output_dir=feat_dir,
            )

        self.assertEqual(len(metrics), 2)
        self.assertEqual(heat_mock.call_count, 3)
        saved_path = os.path.join(feat_dir, "case_feat_vis.png")
        self.assertTrue(os.path.exists(saved_path))
        loaded = cv2.imread(saved_path, cv2.IMREAD_COLOR)
        self.assertEqual(loaded.shape, (4, 8, 3))

    def test_inference_rgb_mode1_exports_overlay_into_rgb_dir(self):
        args = test_core.build_test_feature_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/MT",
                "--rgb",
                "1",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros((x.shape[0], 2, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device)
                logits[:, 1] = 1.0
                return logits

        test_loader = [
            {
                "image": [torch.zeros((3, 2, 2), dtype=torch.float32)],
                "label": [np.zeros((2, 2), dtype=np.uint8)],
                "case": ["case_rgb"],
                "original_image": [np.full((2, 2, 3), 80, dtype=np.uint8)],
            }
        ]
        rgb_dir = os.path.join(self.root_path, "predict_rgb")

        metrics = test_core.inference(
            args,
            DummyModel(),
            test_loader,
            torch.device("cpu"),
            rgb_output_dir=rgb_dir,
        )

        self.assertEqual(len(metrics), 1)
        self.assertTrue(os.path.exists(os.path.join(rgb_dir, "case_rgb.png")))

    def test_inference_uses_strategy_validation_step_for_depth_batches(self):
        args = test_core.build_test_feature_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/MT_depth_teacher_v1",
                "--way",
                "mt_depth_teacher_v1",
                "--use_depth",
                "3",
                "--feat_vis",
                "0",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)

        class DirectModelCallFails(torch.nn.Module):
            def forward(self, _x):
                raise AssertionError("inference should route through strategy.validation_step")

        class DummyStrategy:
            def __init__(self):
                self.seen_image_shape = None
                self.seen_depth_shape = None
                self.model = None

            def validation_step(self, batch_data):
                self.seen_image_shape = tuple(batch_data["image"].shape)
                self.seen_depth_shape = tuple(batch_data["depth3"].shape)
                logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
                logits[:, 1] = 1.0
                return logits

        strategy = DummyStrategy()
        test_loader = [
            {
                "image": [torch.zeros((3, 2, 2), dtype=torch.float32)],
                "depth3": [torch.ones((3, 2, 2), dtype=torch.float32)],
                "label": [np.zeros((2, 2), dtype=np.uint8)],
                "case": ["case_strategy"],
                "original_image": [np.full((2, 2, 3), 80, dtype=np.uint8)],
            }
        ]

        metrics = test_core.inference(
            args,
            DirectModelCallFails(),
            test_loader,
            torch.device("cpu"),
            strategy=strategy,
        )

        self.assertEqual(len(metrics), 1)
        self.assertEqual(strategy.seen_image_shape, (1, 3, 2, 2))
        self.assertEqual(strategy.seen_depth_shape, (1, 3, 2, 2))

    def test_test_args_hydrate_strategy_defaults_for_mt_depth_teacher(self):
        args = build_test_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/MT_depth_teacher_v1",
                "--way",
                "mt_depth_teacher_v1",
                "--use_depth",
                "3",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)

        class DummyModel(torch.nn.Module):
            def __init__(self, in_chns=3, class_num=2):
                super().__init__()
                self.params = {"in_chns": in_chns, "class_num": class_num}
                self.conv = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)

            def forward(self, x):
                return self.conv(x)

        model = DummyModel(in_chns=args.in_chns, class_num=args.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001)
        strategy = create_strategy(args.way, args, model, optimizer, torch.device("cpu"))

        self.assertEqual(strategy.labeled_bs, 2)
        self.assertEqual(strategy.grad_clip, 0.0)
        self.assertEqual(strategy.consistency_start_iters, 1000)


if __name__ == "__main__":
    unittest.main()
