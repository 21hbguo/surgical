from argparse import Namespace
from pathlib import Path
import unittest

import torch

from strategies import STRATEGY_REGISTRY, create_strategy
from strategies.base_strategy import BaseTrainingStrategy


class DummyStrategy(BaseTrainingStrategy):
    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        raise NotImplementedError


class StrategyRegistryTest(unittest.TestCase):
    def test_registry_maps_names_directly_to_classes(self):
        strategy_cls = STRATEGY_REGISTRY["mt"]
        self.assertIn("fully_contrast_v1", STRATEGY_REGISTRY)
        self.assertIn("fully_contrast_v1_1", STRATEGY_REGISTRY)
        self.assertIn("fully_rgb_masking_depth_v1", STRATEGY_REGISTRY)
        self.assertIn("semi_mean_teacher_contrast_v1", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_teacher_v1", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_v1", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_v1_2", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_v4", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_proto_v1", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_proto_teacher_v2", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_proto_teacher_v3", STRATEGY_REGISTRY)
        self.assertIn("mt_depth_guider_proto_teacher_v1", STRATEGY_REGISTRY)
        self.assertNotIn("mt_depth_guider_proto_v2", STRATEGY_REGISTRY)
        self.assertTrue(isinstance(strategy_cls, type))
        self.assertTrue(issubclass(strategy_cls, BaseTrainingStrategy))

    def test_create_strategy_uses_static_registry_class(self):
        args = Namespace(
            consistency=0.1,
            grad_clip=0.0,
            strong="s",
            labeled_bs=2,
            num_classes=2,
            use_depth=None,
            ema_decay=0.99,
            consistency_rampup=150,
            consistency_rampup_div=200,
            way="fully",
        )
        model = torch.nn.Conv2d(1, 2, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        device = torch.device("cpu")

        strategy = create_strategy("fully", args, model, optimizer, device)
        self.assertIsInstance(strategy, BaseTrainingStrategy)

    def test_registry_module_has_no_dynamic_import_or_getattr_loader(self):
        source = Path(__file__).resolve().parents[1] / "strategies" / "__init__.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("importlib", text)
        self.assertNotIn("def __getattr__", text)


if __name__ == "__main__":
    unittest.main()
