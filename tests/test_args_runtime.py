import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.args import build_test_parser, build_train_parser, finalize_test_args, finalize_train_args, format_args_for_logging
import core.runtime as runtime
import utils.common as common


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


class ArgsRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_path = self.tmpdir.name
        _write_task_json(self.root_path, task=1, num_classes=2, n_folds=4)
        _write_task_json(self.root_path, task=2, num_classes=4, n_folds=5)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_runtime_no_longer_exposes_duplicate_depth_resolver(self):
        self.assertFalse(hasattr(runtime, "resolve_depth_settings"))
        self.assertFalse(hasattr(common, "set_input_channels"))

    def test_train_module_owns_train_runtime_assembly(self):
        train_source = Path(__file__).resolve().parents[1] / "core" / "train.py"
        runtime_source = Path(__file__).resolve().parents[1] / "core" / "runtime.py"
        train_text = train_source.read_text(encoding="utf-8")
        runtime_text = runtime_source.read_text(encoding="utf-8")

        self.assertNotIn("assemble_train_components,", train_text)
        self.assertNotIn("create_dataloaders,", train_text)
        self.assertNotIn("maybe_freeze_dinov3_backbone,", train_text)
        self.assertNotIn("def create_dataloaders(", runtime_text)
        self.assertNotIn("def maybe_freeze_dinov3_backbone(", runtime_text)
        self.assertNotIn("def assemble_train_components(", runtime_text)

    def test_core_test_module_no_longer_keeps_runtime_wrapper_defs(self):
        test_source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "test.py")
        )
        text = open(test_source, "r", encoding="utf-8").read()
        self.assertNotIn("def build_test_run_context(", text)
        self.assertNotIn("def resolve_fold_snapshot_path(", text)
        self.assertNotIn("def resolve_test_checkpoint_path(", text)

    def test_common_module_no_longer_keeps_thin_private_runtime_helpers(self):
        common_source = (
            os.path.join(os.path.dirname(__file__), "..", "utils", "common.py")
        )
        text = open(common_source, "r", encoding="utf-8").read()
        self.assertNotIn("def _resolve_result_root(", text)
        self.assertNotIn("def should_include_strong_prefix(", text)
        self.assertNotIn("def _require_positive_int_field(", text)
        self.assertNotIn("def build_dataset_metric_dir(", text)

    def test_train_parser_uses_config_defaults_as_args_defaults(self):
        args = build_train_parser().parse_args(["--task", "1"])
        self.assertEqual(args.exp, "endovis2017/default_exp")
        self.assertEqual(args.result_root, "../result_train")
        self.assertEqual(args.labeled_num, 10)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.optimizer, "adam")
        self.assertEqual(args.filter_num, 16)
        self.assertEqual(args.depth_uint, 16)
        self.assertEqual(args.consistency, 0.1)
        self.assertEqual(args.ema_decay, 0.99)
        self.assertFalse(hasattr(args, "proto_feature_dim"))
        self.assertFalse(hasattr(args, "contrast_feature_dim"))
        self.assertFalse(hasattr(args, "depth_loss_weight"))

    def test_strategy_private_args_are_registered_for_selected_strategy(self):
        args = build_train_parser().parse_args(["--task", "1", "--way", "proto", "--proto_feature_dim", "128"])
        self.assertEqual(args.proto_feature_dim, 128)
        self.assertFalse(hasattr(args, "contrast_loss_weight"))

    def test_strategy_private_args_are_rejected_for_other_strategies(self):
        with self.assertRaises(SystemExit):
            build_train_parser().parse_args(["--task", "1", "--way", "fully", "--proto_feature_dim", "128"])

    def test_exp_name_does_not_infer_strategy(self):
        args = build_train_parser().parse_args(["--task", "1", "--exp", "toy/proto"])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.way, "fully")
        with self.assertRaises(SystemExit):
            build_train_parser().parse_args(["--task", "1", "--exp", "toy/proto", "--proto_feature_dim", "128"])

    def test_test_parser_defaults_labeled_num_to_ten_percent(self):
        args = build_test_parser().parse_args(["--task", "1"])
        self.assertEqual(args.labeled_num, 10)

    def test_finalize_train_args_pins_resnet_contrast_model_for_fully_contrast_v1(self):
        args = build_train_parser().parse_args([
            "--task", "2", "--way", "fully_contrast_v1", "--pretrain", "resnet", "--exp", "toy/FullyContrastV1"
        ])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.model, "resnet_contrast_v1")

    def test_finalize_train_args_reuses_contrast_models_for_fully_contrast_v1_1(self):
        args = build_train_parser().parse_args([
            "--task", "2", "--way", "fully_contrast_v1_1", "--pretrain", "resnet", "--exp", "toy/FullyContrastV11"
        ])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.model, "resnet_contrast_v1")

    def test_finalize_train_args_reuses_unet_contrast_model_for_fully_contrast_v1_1_with_none_pretrain(self):
        args = build_train_parser().parse_args([
            "--task", "2", "--way", "fully_contrast_v1_1", "--pretrain", "none", "--exp", "toy/FullyContrastV11"
        ])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.model, "unet_contrast_v1")

    def test_finalize_train_args_pins_unet_contrast_model_for_fully_contrast_v1_with_none_pretrain(self):
        args = build_train_parser().parse_args([
            "--task", "2", "--way", "fully_contrast_v1", "--pretrain", "none", "--exp", "toy/FullyContrastV1"
        ])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.model, "unet_contrast_v1")

    def test_removed_config_flags_are_rejected(self):
        train_parser = build_train_parser()
        test_parser = build_test_parser()
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--config", "foo.yaml"])
        with self.assertRaises(SystemExit):
            test_parser.parse_args(["--task", "1", "--opts", "TRAIN.BASE_LR", "1e-4"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--adam_base_lr", "1e-4"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--sgd_momentum", "0.95"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--adamw_weight_decay", "0.01"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--opt", "adam"])
        with self.assertRaises(SystemExit):
            test_parser.parse_args(["--task", "1", "--optimizer", "sgd"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--proto"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--resume", "checkpoint_dir"])
        with self.assertRaises(SystemExit):
            test_parser.parse_args(["--task", "1", "--resume", "checkpoint_dir"])
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--task", "1", "--depth_uint", "12"])

    def test_parser_accepts_canonical_and_legacy_strategy_flags(self):
        canonical = build_train_parser().parse_args(["--task", "1", "--way", "proto"])
        legacy = build_train_parser().parse_args(["--task", "1", "--way", "proto"])
        self.assertEqual(canonical.way, "proto")
        self.assertEqual(legacy.way, "proto")
        self.assertTrue(hasattr(canonical, "way"))

    def test_finalize_train_args_infers_metadata_and_runtime_fields(self):
        args = build_train_parser().parse_args(
            [
                "--task",
                "2",
                "--way",
                "proto",
                "--pretrain",
                "resnet",
                "--exp",
                "toy/Proto",
                "--use_depth",
                "1",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.num_classes, 4)
        self.assertEqual(finalized.num_folds, 5)
        self.assertEqual(finalized.model, "resnet_proto_v1")
        self.assertEqual(finalized.in_chns, 4)
        self.assertEqual(finalized.use_depth, 1)
        self.assertEqual(finalized.way, "proto")
        self.assertFalse(hasattr(finalized, "base_lr"))
        self.assertFalse(hasattr(finalized, "adam_base_lr"))
        self.assertFalse(hasattr(finalized, "class_num"))
        self.assertFalse(hasattr(finalized, "opt"))
        self.assertFalse(hasattr(finalized, "proto_use"))
        self.assertFalse(hasattr(finalized, "resume"))
        self.assertEqual(finalized.train_result_root, os.path.normpath(os.path.abspath("../result_train")))
        self.assertIsNone(finalized.snapshot_path)

    def test_finalize_train_args_use_depth_13_keeps_input_channels_like_depth1(self):
        args = build_train_parser().parse_args(
            [
                "--task",
                "2",
                "--way",
                "proto",
                "--exp",
                "toy/Proto",
                "--use_depth",
                "13",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.use_depth, 13)
        self.assertEqual(finalized.in_chns, 4)

    def test_finalize_train_args_depth_pretrain_uses_depth_checkpoint_path(self):
        args = build_train_parser().parse_args([
            "--task", "2", "--way", "proto", "--pretrain", "depth", "--exp", "toy/Proto"
        ])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.model, "depth_proto_v1")
        self.assertEqual(
            finalized.pretrain_root,
            common.resolve_runtime_path("../pre_train_ckp/resnet34-depth-pretrain.pth"),
        )

    def test_finalize_test_args_uses_effective_checkpoint_mode(self):
        args = build_test_parser().parse_args(
            [
                "--task",
                "1",
                "--way",
                "mt",
                "--pretrain",
                "resnet",
                "--exp",
                "toy/MT",
                "--no_val",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_test_args(args)
        self.assertEqual(finalized.model, "resnet")
        self.assertEqual(finalized.num_classes, 2)
        self.assertEqual(finalized.num_folds, 4)
        self.assertEqual(finalized.way, "mt")
        self.assertEqual(finalized.requested_checkpoint_type, "final")
        self.assertEqual(finalized.checkpoint_type, "final")
        self.assertFalse(hasattr(finalized, "base_lr"))
        self.assertFalse(hasattr(finalized, "adam_base_lr"))
        self.assertFalse(hasattr(finalized, "resume"))
        self.assertEqual(finalized.train_result_root, os.path.normpath(os.path.abspath("../result_train")))
        self.assertEqual(finalized.predict_result_root, os.path.normpath(os.path.abspath("../result_predict")))

    def test_format_args_for_logging_train_args_does_not_require_test_fields(self):
        args = build_train_parser().parse_args(["--task", "1", "--exp", "toy/Fully"])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        text = format_args_for_logging(finalized)
        self.assertIn("'common':", text)
        self.assertIn("'train':", text)
        self.assertIn("'test': {}", text)

    def test_test_parser_defaults_include_relative_result_roots(self):
        args = build_test_parser().parse_args(["--task", "1"])
        self.assertEqual(args.result_root, "../result_predict")
        self.assertEqual(args.train_result_root, "../result_train")
        self.assertEqual(args.optimizer, "adam")

    def test_test_parser_accepts_explicit_adam_optimizer(self):
        args = build_test_parser().parse_args(["--task", "1", "--optimizer", "adam"])
        self.assertEqual(args.optimizer, "adam")

    def test_test_parser_accepts_rgb_export_modes(self):
        args = build_test_parser().parse_args(["--task", "1", "--rgb", "2"])
        self.assertEqual(args.rgb, 2)

    def test_test_parser_accepts_space_separated_multiple_folds(self):
        args = build_test_parser().parse_args(["--task", "1", "--fold", "0", "2", "3"])
        self.assertEqual(args.fold, ["0", "2", "3"])

    def test_test_parser_accepts_comma_separated_multiple_folds(self):
        args = build_test_parser().parse_args(["--task", "1", "--fold", "0,2,3"])
        self.assertEqual(args.fold, ["0,2,3"])


if __name__ == "__main__":
    unittest.main()
