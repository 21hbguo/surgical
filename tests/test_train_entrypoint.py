from argparse import Namespace
import os
from pathlib import Path
import tempfile
import unittest

import core.train as train_core
import torch


class TrainEntrypointStructureTest(unittest.TestCase):
    def test_trainer_is_defined_in_train_module(self):
        self.assertEqual(train_core.Trainer.__module__, "core.train")

    def test_train_module_no_longer_depends_on_core_trainer_file(self):
        train_source = Path(train_core.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from core.trainer import Trainer", train_source)
        trainer_path = Path(train_core.__file__).with_name("trainer.py")
        self.assertFalse(trainer_path.exists())

    def test_train_17_script_uses_local_tmpdir_and_no_conda_activate(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_17.sh"
        script_source = script_path.read_text(encoding="utf-8")

        self.assertIn("TMPDIR", script_source)
        self.assertNotIn("conda activate", script_source)

    def test_shipped_train_test_scripts_do_not_require_conda_activation(self):
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        script_names = [
            "train_17.sh",
            "test_17.sh",
            "train_kvasir_all.sh",
            "test_kvasir.sh",
            "train_18Binary8_all.sh",
            "test_18Binary8.sh",
        ]

        for script_name in script_names:
            script_source = (scripts_dir / script_name).read_text(encoding="utf-8")
            self.assertNotIn("conda activate", script_source, script_name)

    def test_trainer_saves_interval_sampled_semi_lists_from_sampler_indices(self):
        class DummySampler:
            primary_indices = [0, 2]
            secondary_indices = [1, 3]

        class DummyDataset:
            sample_list = ["case0", "case1", "case2", "case3"]

        class DummyLoader:
            dataset = DummyDataset()
            batch_sampler = DummySampler()

        class DummyStrategy:
            def __init__(self):
                param = torch.nn.Parameter(torch.zeros(()))
                self.optimizer = torch.optim.SGD([param], lr=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                snapshot_path=tmpdir,
                max_iterations=1,
                val_iter=0,
                lr_scheduler="poly",
                lr_warmup_iters=0,
                lr=0.1,
                lr_warmup_ratio=0.0,
                lr_min_ratio=0.0,
                poly_power=0.9,
                early_stopping=0.0,
            )

            train_core.Trainer(
                args,
                DummyStrategy(),
                DummyLoader(),
                None,
                torch.device("cpu"),
                labeled_slice=2,
            )

            labeled_path = Path(tmpdir) / "data_train_labeled.list"
            unlabeled_path = Path(tmpdir) / "data_train_unlabeled.list"
            self.assertEqual(labeled_path.read_text(encoding="utf-8").splitlines(), ["case0", "case2"])
            self.assertEqual(unlabeled_path.read_text(encoding="utf-8").splitlines(), ["case1", "case3"])
    def test_trainer_writes_zero_byte_final_marker_when_validation_enabled(self):
        class DummySampler:
            primary_indices = [0]
            secondary_indices = [1]
        class DummyDataset:
            sample_list = ["case0", "case1"]
        class DummyLoader:
            dataset = DummyDataset()
            batch_sampler = DummySampler()
        class DummyStrategy:
            def __init__(self):
                param = torch.nn.Parameter(torch.zeros(()))
                self.optimizer = torch.optim.SGD([param], lr=0.1)
            def get_state_dict(self):
                return {"w": torch.tensor([1.0])}
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(snapshot_path=tmpdir, max_iterations=1, val_iter=1, lr_scheduler="poly", lr_warmup_iters=0, lr=0.1, lr_warmup_ratio=0.0, lr_min_ratio=0.0, poly_power=0.9, early_stopping=0.0)
            trainer = train_core.Trainer(args, DummyStrategy(), DummyLoader(), DummyLoader(), torch.device("cpu"), labeled_slice=1)
            trainer._save_model("final")
            final_path = os.path.join(tmpdir, "model_final.pth")
            self.assertTrue(os.path.exists(final_path))
            self.assertEqual(os.path.getsize(final_path), 0)


if __name__ == "__main__":
    unittest.main()
