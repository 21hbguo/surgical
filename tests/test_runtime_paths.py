import json
import os
import tempfile
import unittest
from unittest.mock import patch

from core.args import build_test_parser, build_train_parser, finalize_test_args, finalize_train_args
from utils.common import (
    build_run_output_dir,
    build_shared_run_rel_dir,
    get_model_name,
    resolve_runtime_path,
)


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


class RuntimePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_path = self.tmpdir.name
        _write_task_json(self.root_path, task=1, num_classes=2, n_folds=4)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_relative_result_root_is_resolved_from_current_workdir(self):
        with tempfile.TemporaryDirectory() as cwd:
            resolved = resolve_runtime_path("../artifacts", cwd=cwd)
            self.assertEqual(resolved, os.path.normpath(os.path.join(cwd, "../artifacts")))

    def test_train_and_test_modes_share_rel_dir_but_use_different_roots(self):
        args = build_test_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/MT",
                "--way",
                "mt",
                "--pretrain",
                "resnet",
                "--labeled_num",
                "10",
                "--lr",
                "1e-4",
                "--sampling",
                "none",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)
        rel_dir = build_shared_run_rel_dir(args)
        self.assertEqual(
            rel_dir,
            os.path.join("toy_255_Samplingnone", "task1", "MT", "10_labeled_lr1e-4_s_resnet"),
        )
        train_dir = build_run_output_dir(args, mode="train")
        test_dir = build_run_output_dir(args, mode="test")
        self.assertTrue(train_dir.startswith(resolve_runtime_path("../result_train")))
        self.assertTrue(test_dir.startswith(resolve_runtime_path("../result_predict")))
        self.assertTrue(train_dir.endswith(rel_dir))
        self.assertTrue(test_dir.endswith(rel_dir))

    def test_fully_paths_omit_strong_prefix_in_run_name(self):
        args = build_test_parser().parse_args(
            [
                "--task",
                "1",
                "--exp",
                "toy/Fully",
                "--labeled_num",
                "10",
                "--lr",
                "1e-4",
                "--way",
                "fully",
                "--pretrain",
                "resnet",
                "--sampling",
                "interval",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)
        rel_dir = build_shared_run_rel_dir(args)
        self.assertEqual(
            rel_dir,
            os.path.join("toy_255_Samplinginterval", "task1", "Fully", "10_labeled_lr1e-4_resnet"),
        )

    def test_test_context_uses_dataset_scoped_result_dir(self):
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
                "--sampling",
                "none",
            ]
        )
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            args = finalize_test_args(args)

        from core.test import build_test_run_context

        context = build_test_run_context(args)

        self.assertEqual(
            context.dataset_result_path,
            os.path.join(resolve_runtime_path("../result_predict"), "toy_255_Samplingnone"),
        )

    def test_finalize_train_args_sets_default_train_result_root(self):
        args = build_train_parser().parse_args(["--task", "1", "--exp", "toy/Fully"])
        with patch("core.args.infer_root_path_from_exp", return_value=self.root_path):
            finalized = finalize_train_args(args)
        self.assertEqual(finalized.train_result_root, resolve_runtime_path("../result_train"))
        self.assertIsNone(finalized.snapshot_path)

    def test_test_parser_accepts_sampling_for_train_artifact_resolution(self):
        args = build_test_parser().parse_args(["--task", "1", "--sampling", "none"])
        self.assertEqual(args.sampling, "none")

    def test_pretrained_model_name_omits_strategy_suffix_and_filter_num(self):
        self.assertEqual(
            get_model_name("resnet", 16, strong="s", pretrain_mode="resnet"),
            "s_resnet16",
        )
        self.assertEqual(
            get_model_name("unet", 16, strong="s", pretrain_mode="none"),
            "s_unet16",
        )
        self.assertEqual(
            get_model_name("resnet_proto", 16, strong="s", pretrain_mode="resnet", include_filter_num=False),
            "s_resnet_proto",
        )
        self.assertEqual(
            get_model_name("resnet_proto", 16, strong="s", pretrain_mode="depth", include_filter_num=False),
            "s_depth_proto",
        )


if __name__ == "__main__":
    unittest.main()
