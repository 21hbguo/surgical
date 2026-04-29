from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import torch

from strategies.fully_depth_pretrain_v1 import FullyDepthPretrainStrategy
from strategies.fully_supervised import FullySupervisedStrategy
from strategies.base_strategy import BaseTrainingStrategy
from strategies.fully_rgb_masking_depth_v1 import FullyRGBMaskingDepthV1Strategy


class TinyModel(torch.nn.Module):
    def __init__(self, in_chns=1, class_num=2):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num}
        self.conv = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EmaEnabledStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        self._enable_ema_support()

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        raise NotImplementedError


class PlainStrategy(BaseTrainingStrategy):
    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        raise NotImplementedError


class CaptureModel(torch.nn.Module):
    def __init__(self, in_chns=3, class_num=2):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num}
        self.last_input = None
        self.conv = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)

    def forward(self, x):
        self.last_input = x.detach().clone()
        return self.conv(x)


class StrategyBaseTest(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            consistency=0.1,
            grad_clip=0.0,
            strong="s",
            labeled_bs=2,
            num_classes=2,
            use_depth=None,
            ema_decay=0.99,
            consistency_rampup=150,
            consistency_rampup_div=200,
            way="mt",
            rgb_masking_ratio=0.75,
            depth_pretrain_mask_ratio=0.75,
            depth_l1_weight=1.0,
            depth_loss_weight=1.0,
        )
        self.model = TinyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device("cpu")

    def test_base_strategy_can_enable_optional_ema_support(self):
        strategy = EmaEnabledStrategy(self.args, self.model, self.optimizer, self.device)
        self.assertIsNotNone(strategy.ema_model)

        strategy.eval()
        self.assertFalse(strategy.model.training)
        self.assertFalse(strategy.ema_model.training)

        strategy.train()
        self.assertTrue(strategy.model.training)
        self.assertTrue(strategy.ema_model.training)

        state_dict = strategy.get_state_dict()
        self.assertIn("model", state_dict)
        self.assertNotIn("ema_model", state_dict)

    def test_base_strategy_without_ema_keeps_plain_model_state(self):
        strategy = PlainStrategy(self.args, self.model, self.optimizer, self.device)
        state_dict = strategy.get_state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertNotIn("ema_model", state_dict)

    def test_depth_inputs_are_concatenated_for_shared_forward_paths(self):
        self.args.use_depth = 1
        model = TinyModel(in_chns=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = PlainStrategy(self.args, model, optimizer, self.device)
        batch = {
            "image": torch.randn(4, 1, 8, 8),
            "depth1": torch.randn(4, 1, 8, 8),
        }

        prepared = torch.cat([batch["image"], batch["depth1"]], dim=1)
        unlabeled = prepared[strategy.labeled_bs :]
        logits = strategy.validation_step(batch)

        self.assertEqual(tuple(prepared.shape), (4, 2, 8, 8))
        self.assertEqual(tuple(unlabeled.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(logits.shape), (4, 2, 8, 8))

    def test_fully_supervised_strategy_computes_loss_without_depth_inputs(self):
        strategy = FullySupervisedStrategy(self.args, self.model, self.optimizer, self.device)
        batch = {
            "image": torch.randn(2, 1, 8, 8),
            "label": torch.randint(0, 2, (2, 8, 8)),
        }

        loss_dict = strategy.compute_loss(batch)

        self.assertEqual(set(loss_dict.keys()), {"total", "ce", "dice"})
        self.assertTrue(torch.is_tensor(loss_dict["total"]))
        self.assertEqual(loss_dict["total"].ndim, 0)

    def test_ssl_base_file_is_removed_and_strategies_import_single_base_file(self):
        strategies_dir = Path(__file__).resolve().parents[1] / "strategies"
        self.assertFalse((strategies_dir / "base_ssl_strategy.py").exists())
        for path in strategies_dir.glob("semi_*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("base_ssl_strategy", source)

    def test_fully_rgb_masking_depth_strategy_requires_depth3(self):
        model = TinyModel(in_chns=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = FullyRGBMaskingDepthV1Strategy(self.args, model, optimizer, self.device)
        batch = {
            "image": torch.randn(2, 3, 8, 8),
            "label": torch.randint(0, 2, (2, 8, 8)),
        }

        with self.assertRaisesRegex(KeyError, "depth3"):
            strategy.compute_loss(batch)

    def test_fully_rgb_masking_depth_strategy_replaces_masked_rgb_with_depth3(self):
        args = Namespace(**vars(self.args))
        args.rgb_masking_ratio = 0.75
        model = CaptureModel(in_chns=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = FullyRGBMaskingDepthV1Strategy(args, model, optimizer, self.device)

        rgb = torch.tensor(
            [[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]]
        )
        depth3 = torch.tensor(
            [[
                [[101.0, 102.0], [103.0, 104.0]],
                [[105.0, 106.0], [107.0, 108.0]],
                [[109.0, 110.0], [111.0, 112.0]],
            ]]
        )
        label = torch.zeros((1, 2, 2), dtype=torch.long)
        batch = {"image": rgb, "depth3": depth3, "label": label}

        expected_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        strategy._build_random_mask = lambda tensor: expected_mask.to(tensor.device, tensor.dtype)

        loss_dict = strategy.compute_loss(batch)
        expected_volume = rgb * expected_mask + depth3 * (1.0 - expected_mask)

        self.assertEqual(set(loss_dict.keys()), {"total", "ce", "dice"})
        self.assertTrue(torch.allclose(model.last_input, expected_volume))

    def test_fully_rgb_masking_depth_strategy_defaults_to_75_percent_masking(self):
        model = TinyModel(in_chns=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = FullyRGBMaskingDepthV1Strategy(self.args, model, optimizer, self.device)

        self.assertAlmostEqual(strategy.mask_ratio, 0.75)

    def test_fully_rgb_masking_depth_strategy_visualization_path_is_repo_relative(self):
        model = TinyModel(in_chns=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = FullyRGBMaskingDepthV1Strategy(self.args, model, optimizer, self.device)
        expected = Path(__file__).resolve().parents[1] / "outputs" / "fully_rgb_masking_depth_v1_complementary.png"

        self.assertTrue(str(strategy.vis_path).endswith("outputs/fully_rgb_masking_depth_v1_complementary.png"))
        self.assertEqual(strategy.vis_path, expected)

    def test_fully_depth_pretrain_strategy_visualization_root_is_repo_relative(self):
        model = TinyModel(in_chns=3, class_num=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = FullyDepthPretrainStrategy(self.args, model, optimizer, self.device)
        expected = Path(__file__).resolve().parents[1] / "outputs"

        self.assertTrue(str(strategy.vis_root).endswith("outputs"))
        self.assertEqual(strategy.vis_root, expected)

    def test_fully_rgb_masking_depth_strategy_saves_complementary_visualization(self):
        model = TinyModel(in_chns=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = FullyRGBMaskingDepthV1Strategy(self.args, model, optimizer, self.device)
        volume = torch.tensor(
            [[
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
                [[8.0, 9.0], [10.0, 11.0]],
            ]]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.vis_path = Path(tmpdir) / "complementary.png"
            strategy._save_complementary_visualization(volume)

            self.assertTrue(strategy.vis_path.exists())
            self.assertGreater(strategy.vis_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
