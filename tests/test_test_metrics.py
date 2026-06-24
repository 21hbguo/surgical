from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import core.test as test_core
import core.testing.export as test_export
import torch


class TestRequestedFolds(unittest.TestCase):
    def test_parse_requested_folds_defaults_to_all_folds(self):
        self.assertEqual(test_core.parse_requested_folds(None, num_folds=4), [0, 1, 2, 3])

    def test_parse_requested_folds_supports_minus_one(self):
        self.assertEqual(test_core.parse_requested_folds(["-1"], num_folds=4), [0, 1, 2, 3])

    def test_parse_requested_folds_supports_space_separated_values(self):
        self.assertEqual(test_core.parse_requested_folds(["0", "2", "3"], num_folds=4), [0, 2, 3])

    def test_parse_requested_folds_supports_comma_separated_values(self):
        self.assertEqual(test_core.parse_requested_folds(["0,2,3"], num_folds=4), [0, 2, 3])

    def test_parse_requested_folds_sorts_and_deduplicates(self):
        self.assertEqual(test_core.parse_requested_folds(["3", "1", "3", "2"], num_folds=4), [1, 2, 3])

    def test_parse_requested_folds_rejects_invalid_fold(self):
        with self.assertRaises(ValueError):
            test_core.parse_requested_folds(["4"], num_folds=4)

    def test_metric_pairs_use_one_shared_mapping(self):
        self.assertEqual(
            test_core.SUMMARY_METRIC_PAIRS,
            test_core.PER_CLASS_METRIC_PAIRS,
        )

    def test_distance_metrics_default_off(self):
        parser = test_core.build_test_feature_parser()
        args = parser.parse_args(["--task", "1"])
        self.assertEqual(args.distance_metrics, 0)

    def test_post_resize_default_label_to_pred(self):
        parser = test_core.build_test_feature_parser()
        args = parser.parse_args(["--task", "1"])
        self.assertEqual(args.post_resize, "label_to_pred")

    def test_test_module_outsources_visualization_and_export_details(self):
        project_root = Path(__file__).resolve().parents[1]
        test_source = (project_root / "core" / "test.py").read_text(encoding="utf-8")
        export_source = (project_root / "core" / "testing" / "export.py").read_text(encoding="utf-8")

        self.assertNotIn("class GradCAMHookManager:", test_source)
        self.assertNotIn("def append_csv_with_lock(", test_source)
        self.assertNotIn("def save_test_rgb_visualization(", test_source)
        self.assertNotIn("def save_multiclass_gradcam_visualization(", test_source)
        self.assertNotIn("def format_metric_percentage(", test_source)
        self.assertNotIn("def build_summary_row(", test_source)
        self.assertNotIn("def _format_train_best_metric(", test_source)
        self.assertNotIn("def _build_named_result_row(", test_source)
        self.assertIn("def build_summary_row(", export_source)
        self.assertIn("def build_result_export_rows(", export_source)
        self.assertTrue((project_root / "core" / "testing" / "visualization.py").exists())
        self.assertTrue((project_root / "core" / "testing" / "export.py").exists())


class TestInMemoryAggregation(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"Fold": "f0", "Seq": 1, "Case": "case_1_a", "Class": 1, "Dice": 0.0, "IoU": 0.0, "TP": 0.0, "FP": 1.0, "FN": 1.0, "Acc": 0.5, "Valid": True},
            {"Fold": "f0", "Seq": 1, "Case": "case_1_b", "Class": 1, "Dice": 1.0, "IoU": 1.0, "TP": 1.0, "FP": 0.0, "FN": 0.0, "Acc": 1.0, "Valid": True},
            {"Fold": "f1", "Seq": 2, "Case": "case_2_a", "Class": 1, "Dice": 0.0, "IoU": 0.0, "TP": 0.0, "FP": 1.0, "FN": 1.0, "Acc": 0.5, "Valid": True},
            {"Fold": "f1", "Seq": 2, "Case": "case_2_b", "Class": 1, "Dice": 1.0, "IoU": 1.0, "TP": 1.0, "FP": 0.0, "FN": 0.0, "Acc": 1.0, "Valid": True},
        ]

    def test_summarize_records_uses_in_memory_samples(self):
        summary = test_core.summarize_records(self.records, num_classes=2)

        self.assertEqual(summary["Total_Samples"], 4)
        self.assertEqual(summary["Valid_Samples"], 4)
        self.assertEqual(summary["Avg_Dice"], "0.5000 ± 0.5000")
        self.assertEqual(summary["Avg_IoU"], "0.5000 ± 0.5000")
        self.assertEqual(summary["Avg_Precision"], "0.5000 ± 0.5000")
        self.assertEqual(summary["Avg_Recall"], "0.5000 ± 0.5000")

    def test_build_seq_rows_groups_by_fold_and_sequence(self):
        rows = test_core.build_seq_rows(self.records, num_classes=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Fold"], "f0")
        self.assertEqual(rows[0]["Seq"], 1)
        self.assertEqual(rows[0]["Total_Samples"], 2)
        self.assertEqual(rows[0]["Avg_dice"], "50.00 ± 50.00")
        self.assertEqual(rows[1]["Fold"], "f1")
        self.assertEqual(rows[1]["Seq"], 2)
        self.assertEqual(rows[1]["Total_Samples"], 2)
        self.assertEqual(rows[1]["Avg_dice"], "50.00 ± 50.00")

    def test_build_fold_rows_includes_each_fold_and_all_folds(self):
        rows = test_core.build_fold_rows(self.records, num_classes=2)

        self.assertEqual([row["Fold"] for row in rows], ["f0", "f1", "ALL_Folds"])
        self.assertEqual(rows[0]["Total_Samples"], 2)
        self.assertEqual(rows[0]["Avg_dice"], "50.00 ± 50.00")
        self.assertEqual(rows[1]["Total_Samples"], 2)
        self.assertEqual(rows[1]["Avg_dice"], "50.00 ± 50.00")
        self.assertEqual(rows[2]["Total_Samples"], 4)
        self.assertEqual(rows[2]["Avg_dice"], "50.00 ± 50.00")

    def test_build_result_name_matches_new_summary_prefix(self):
        class Args:
            task = 2
            exp = "endovis2017_255/Fully"
            labeled_num = 10
            lr = 3e-5

        name = test_core.build_result_name(Args(), "t_resnet", "f0")

        self.assertEqual(name, "task2_Fully_10_labeled_lr3e-5_t_resnet_f0")

    def test_build_result_export_rows_formats_train_best_metrics(self):
        args = SimpleNamespace(
            exp="toy/MT",
            labeled_num=10,
            optimizer="adam",
            task=1,
            root_path="/tmp/root",
            lr=3e-5,
            sampling="none",
        )
        context = SimpleNamespace(model_name="t_resnet", effective_lr=3e-5)
        fold_rows = [
            {"Fold": "f0", "Seq": "", "Total_Samples": 2, "Valid_Samples": 2, "Avg_dice": "50.00 ± 0.00"},
            {"Fold": "ALL_Folds", "Seq": "", "Total_Samples": 2, "Valid_Samples": 2, "Avg_dice": "50.00 ± 0.00"},
        ]
        seq_rows = [
            {"Fold": "f0", "Seq": 1, "Total_Samples": 2, "Valid_Samples": 2, "Avg_dice": "50.00 ± 0.00"},
        ]

        with patch.object(test_export, "get_dataset_result_dir_name", return_value="dataset"):
            fold_output_rows, seq_output_rows, all_folds_summary_rows = test_export.build_result_export_rows(
                args,
                context,
                fold_rows,
                seq_rows,
                {"f0": 0.82},
                total_folds=1,
            )

        self.assertEqual(fold_output_rows[0]["train_best_dice"], "82.00")
        self.assertEqual(fold_output_rows[1]["train_best_dice"], "82.00 ± 0.00")
        self.assertEqual(seq_output_rows[0]["name"], "task1_MT_10_labeled_lr3e-5_t_resnet_f0_seq1")
        self.assertEqual(all_folds_summary_rows, [dict(fold_output_rows[-1])])

    def test_build_result_export_rows_keeps_case_names(self):
        args = SimpleNamespace(
            exp="toy/MT",
            labeled_num=10,
            optimizer="adam",
            task=1,
            root_path="/tmp/root",
            lr=3e-5,
            sampling="none",
        )
        context = SimpleNamespace(model_name="t_resnet", effective_lr=3e-5)
        fold_rows = [
            {"Fold": "ALL_Folds", "Seq": "", "Total_Samples": 1, "Valid_Samples": 1, "Avg_dice": "50.00 ± 0.00"},
        ]
        records = [
            {"Fold": "f0", "Seq": 1, "Case": "case_1_a", "Class": 1, "Dice": 0.5, "IoU": 0.5, "TP": 1.0, "FP": 1.0, "FN": 1.0, "Acc": 0.75, "Valid": True},
        ]

        with patch.object(test_export, "get_dataset_result_dir_name", return_value="dataset"):
            _, _, _, case_output_rows = test_export.build_result_export_rows(
                args,
                context,
                fold_rows,
                [],
                {"f0": 0.82},
                total_folds=1,
                case_records=records,
            )

        self.assertEqual(case_output_rows[0]["Case"], "case_1_a")
        self.assertEqual(case_output_rows[0]["name"], "task1_MT_10_labeled_lr3e-5_t_resnet_f0_case_1_a_c1")
        self.assertEqual(case_output_rows[0]["train_best_dice"], "82.00")

    def test_persist_result_tables_writes_case_summary_rows(self):
        context = SimpleNamespace(dataset_result_path="/tmp/results", checkpoint_type="best")
        fold_output_rows = [{"Fold": "f0", "Avg_dice": "50.00 ± 0.00"}, {"Fold": "ALL_Folds", "Avg_dice": "50.00 ± 0.00"}]

        with patch.object(test_export, "append_csv_with_lock") as append_mock, \
             patch.object(test_export.os, "makedirs"), \
             patch.object(test_export.os.path, "exists", return_value=True):
            paths = test_export.persist_result_tables(context, fold_output_rows, [], [{"Fold": "ALL_Folds"}], [])

        self.assertEqual(paths[-1], "/tmp/results/all_experiments_results_case_summary_best.csv")
        self.assertEqual(append_mock.call_args_list[1].args[1], "/tmp/results/all_experiments_results_case_summary_best.csv")
        self.assertEqual(append_mock.call_args_list[1].args[0].to_dict(orient="records"), fold_output_rows)

    def test_summarize_records_counts_unique_cases_without_assuming_class_one_exists(self):
        records = [
            {"Fold": "f0", "Seq": 1, "Case": "case_1_a", "Class": 2, "Dice": 0.0, "IoU": 0.0, "TP": 0.0, "FP": 1.0, "FN": 1.0, "Acc": 0.5, "Valid": True},
            {"Fold": "f0", "Seq": 1, "Case": "case_1_b", "Class": 2, "Dice": 1.0, "IoU": 1.0, "TP": 1.0, "FP": 0.0, "FN": 0.0, "Acc": 1.0, "Valid": True},
        ]

        summary = test_core.summarize_records(records, num_classes=3)

        self.assertEqual(summary["Total_Samples"], 2)
        self.assertEqual(summary["Valid_Samples"], 2)
        self.assertEqual(summary["Avg_Dice"], "0.5000 ± 0.5000")

    def test_calculate_metric_percase_includes_distance_metrics(self):
        gt = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ).numpy()
        pred = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ).numpy()

        metrics = test_core.calculate_metric_percase(pred, gt, distance_metrics=True)

        self.assertAlmostEqual(metrics["HD95"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ASD"], 0.5, places=6)

    def test_calculate_metric_percase_skips_distance_metrics_by_default(self):
        gt = torch.tensor([[0, 1], [0, 1]]).numpy()
        pred = torch.tensor([[0, 1], [1, 0]]).numpy()

        metrics = test_core.calculate_metric_percase(pred, gt)

        self.assertNotIn("HD95", metrics)
        self.assertNotIn("ASD", metrics)

    def test_calculate_metric_percase_empty_empty_counts_as_correct(self):
        gt = torch.zeros((2, 2), dtype=torch.int64).numpy()
        pred = torch.zeros((2, 2), dtype=torch.int64).numpy()

        metrics = test_core.calculate_metric_percase(pred, gt, distance_metrics=True)

        self.assertEqual(metrics["Dice"], 1.0)
        self.assertEqual(metrics["IoU"], 1.0)
        self.assertEqual(metrics["Acc"], 1.0)
        self.assertEqual(metrics["HD95"], 0.0)
        self.assertEqual(metrics["ASD"], 0.0)
        self.assertTrue(metrics["Valid"])

    def test_build_fold_rows_keeps_distance_metrics_unscaled(self):
        records = [
            {"Fold": "f0", "Seq": 1, "Case": "case_1_a", "Class": 1, "Dice": 0.5, "IoU": 0.4, "TP": 2.0, "FP": 1.0, "FN": 1.0, "Acc": 0.8, "HD95": 4.0, "ASD": 1.5, "Valid": True},
            {"Fold": "f0", "Seq": 1, "Case": "case_1_b", "Class": 1, "Dice": 0.7, "IoU": 0.6, "TP": 3.0, "FP": 1.0, "FN": 1.0, "Acc": 0.9, "HD95": 6.0, "ASD": 2.5, "Valid": True},
        ]

        rows = test_core.build_fold_rows(records, num_classes=2)

        self.assertEqual(rows[0]["Avg_hd95"], "5.00 ± 1.00")
        self.assertEqual(rows[0]["Avg_asd"], "2.00 ± 0.50")
        self.assertEqual(rows[0]["C1_hd95"], "5.00 ± 1.00")
        self.assertEqual(rows[0]["C1_asd"], "2.00 ± 0.50")

    def test_build_fold_rows_omits_distance_metrics_when_absent(self):
        rows = test_core.build_fold_rows(self.records, num_classes=2)

        self.assertNotIn("Avg_hd95", rows[0])
        self.assertNotIn("Avg_asd", rows[0])
        self.assertNotIn("C1_hd95", rows[0])
        self.assertNotIn("C1_asd", rows[0])


class TestMainOutputs(unittest.TestCase):
    def test_run_one_fold_loads_full_strategy_state(self):
        class Strategy:
            def __init__(self):
                self.loaded = None
            def load_state_dict(self, state_dict):
                self.loaded = state_dict
        class Dataset:
            def __len__(self):
                return 0
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            def forward(self, x):
                return x
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model_best.pth"
            torch.save({"model_state": {"model": {"weight": torch.tensor([1.0])}, "model2": {"weight": torch.tensor([2.0])}}, "best_performance": 0.7}, checkpoint_path)
            strategy = Strategy()
            args = SimpleNamespace(snapshot_path=tmpdir, device="cpu", lr=1e-4, batch_size=1, num_classes=2, way="fully")
            context = SimpleNamespace(checkpoint_type="best")
            with patch.object(test_core, "create_model", return_value=Model()), \
                 patch.object(test_core, "create_strategy", return_value=strategy), \
                 patch.object(test_core, "_build_test_loader", return_value=(Dataset(), [], False, None)), \
                 patch.object(test_core, "prepare_visual_output_dirs", return_value=(None, None, None)), \
                 patch.object(test_core, "inference", return_value=[]):
                train_best_dice, records = test_core.run_one_fold(args, None, {}, context)
        self.assertEqual(train_best_dice, 0.7)
        self.assertEqual(records, [])
        self.assertIn("model2", strategy.loaded)

    def test_main_does_not_write_excel_workbook(self):
        args = SimpleNamespace(
            way="mt",
            model="unet",
            device="cpu",
            no_val=0,
            fold=None,
            num_folds=1,
            num_classes=2,
            exp="toy/MT",
            root_path="/tmp/root",
            task=1,
            labeled_num=10,
            optimizer="adam",
            lr=3e-5,
            sampling="none",
        )
        context = SimpleNamespace(
            model_name="t_resnet",
            effective_lr=3e-5,
            dataset_result_path="/tmp/results",
            checkpoint_type="best",
        )
        parser = MagicMock()
        parser.parse_args.return_value = args
        fold_rows = [{"Fold": "ALL_Folds", "Seq": "", "Total_Samples": 1, "Valid_Samples": 1, "Avg_dice": "50.00 ± 0.00"}]
        seq_rows = [{"Fold": "f0", "Seq": 1, "Total_Samples": 1, "Valid_Samples": 1, "Avg_dice": "50.00 ± 0.00"}]

        with patch.object(test_core, "build_test_feature_parser", return_value=parser), \
             patch.object(test_core, "finalize_test_args", return_value=args), \
             patch.object(test_core, "format_args_for_logging", return_value="args"), \
             patch.object(test_core, "build_test_run_context", return_value=context), \
             patch.object(test_core, "parse_requested_folds", return_value=[None]), \
             patch.object(test_core, "run_one_fold", return_value=(0.8, [{"Fold": "f0"}])), \
             patch.object(test_core, "build_fold_rows", return_value=fold_rows), \
             patch.object(test_core, "build_seq_rows", return_value=seq_rows), \
             patch.object(test_core, "persist_result_tables", return_value=("/tmp/fold.csv", "/tmp/seq.csv", "/tmp/all.csv", "/tmp/case.csv", "/tmp/case_summary.csv")), \
             patch.object(test_core, "append_csv_with_lock"), \
             patch.object(test_core, "os") as os_mock, \
             patch.object(test_core.pd, "ExcelWriter") as excel_writer_mock:
            os_mock.path.join.side_effect = lambda *parts: "/".join(str(part).strip("/") for part in parts)
            os_mock.path.exists.return_value = True
            test_core.main()
            excel_writer_mock.assert_not_called()
            test_core.parse_requested_folds.assert_called_once_with(None, 1)


class TestInferenceConsistency(unittest.TestCase):
    def test_batch_inference_matches_per_sample_outputs(self):
        args = SimpleNamespace(
            feat_vis=0,
            conf_vis=0,
            rgb=0,
            distance_metrics=0,
            num_classes=2,
            feat_vis_max_cases=10,
            feat_vis_layer="",
            feat_vis_all_layers=1,
            feat_vis_max_layers=0,
            feat_vis_alpha=0.45,
            feat_vis_target_class=-1,
            way="fully",
            post_resize="label_to_pred",
        )
        batch = {
            "image": [
                torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
                torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32),
            ],
            "label": [
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                torch.tensor([[1, 0], [1, 0]], dtype=torch.int64),
            ],
            "original_label": [
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                torch.tensor([[1, 0], [1, 0]], dtype=torch.int64),
            ],
            "case": ["case_1_a", "case_2_b"],
            "original_image": [None, None],
        }

        class Loader:
            def __iter__(self):
                yield batch

        class Model(torch.nn.Module):
            def forward(self, x):
                b, _, h, w = x.shape
                logits = torch.zeros((b, 2, h, w), dtype=x.dtype, device=x.device)
                logits[:, 0] = 1.0 - x[:, 0]
                logits[:, 1] = x[:, 0]
                return logits

        model = Model()
        records = test_core.inference(args, model, Loader(), torch.device("cpu"))
        expected = []
        for index, case in enumerate(batch["case"]):
            image = batch["image"][index].unsqueeze(0)
            outputs = model(image)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze().cpu().numpy()
            label_np = batch["original_label"][index].cpu().numpy()
            metrics = test_core.calculate_metric_percase(pred == 1, label_np == 1)
            metrics["Case"] = case
            metrics["Class"] = 1
            metrics["Fold"] = "f0"
            metrics["Seq"] = index + 1
            expected.append(metrics)

        self.assertEqual(len(records), len(expected))
        for record, exp in zip(records, expected):
            self.assertEqual(record["Case"], exp["Case"])
            self.assertEqual(record["Class"], exp["Class"])
            self.assertEqual(record["Fold"], exp["Fold"])
            self.assertEqual(record["Seq"], exp["Seq"])
            self.assertAlmostEqual(record["Dice"], exp["Dice"], places=6)
            self.assertAlmostEqual(record["IoU"], exp["IoU"], places=6)
            self.assertAlmostEqual(record["Acc"], exp["Acc"], places=6)
            self.assertNotIn("HD95", record)
            self.assertNotIn("ASD", record)

    def test_post_resize_controls_metric_resize_direction(self):
        batch = {
            "image": [torch.zeros((1, 2, 2), dtype=torch.float32)],
            "label": [torch.zeros((4, 4), dtype=torch.int64)],
            "original_label": [torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)],
            "case": ["case_resize"],
            "original_image": [None],
        }

        class Loader:
            def __iter__(self):
                yield batch

        class Model(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32, device=x.device)
                logits[:, 1, 0, 0] = 2.0
                return logits

        args = SimpleNamespace(
            feat_vis=0,
            conf_vis=0,
            rgb=0,
            distance_metrics=0,
            num_classes=2,
            feat_vis_max_cases=10,
            feat_vis_layer="",
            feat_vis_all_layers=1,
            feat_vis_max_layers=0,
            feat_vis_alpha=0.45,
            feat_vis_target_class=-1,
            way="fully",
            post_resize="label_to_pred",
        )
        label_to_pred_records = test_core.inference(args, Model(), Loader(), torch.device("cpu"))
        args.post_resize = "pred_to_label"
        pred_to_label_records = test_core.inference(args, Model(), Loader(), torch.device("cpu"))
        self.assertEqual(label_to_pred_records[0]["Dice"], 1.0)
        self.assertLess(pred_to_label_records[0]["Dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
