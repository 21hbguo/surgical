from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import core.test as test_core
import core.testing.export as test_export


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


class TestMainOutputs(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
