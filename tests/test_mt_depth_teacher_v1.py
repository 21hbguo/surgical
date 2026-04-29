from argparse import Namespace
import unittest

import torch

from strategies import STRATEGY_REGISTRY
from strategies.semi_mt_depth_teacher_v1 import MTDepthTeacherV1Strategy


class DummyUNetBase(torch.nn.Module):
    def __init__(self, in_chns=1, class_num=2):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num}
        self.net = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)

    def forward(self, x):
        return self.net(x)


class MTDepthTeacherV1StrategyTest(unittest.TestCase):
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
            consistency_start_iters=0,
            way="mt_depth_teacher_v1",
            lr=0.1,
        )
        self.model = DummyUNetBase(in_chns=1, class_num=2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device("cpu")
        self.strategy = MTDepthTeacherV1Strategy(self.args, self.model, self.optimizer, self.device)

    def test_strategy_registered(self):
        self.assertIn("mt_depth_teacher_v1", STRATEGY_REGISTRY)

    def test_compute_loss_returns_depth_teacher_terms(self):
        batch = {
            "image": torch.randn(4, 1, 32, 32),
            "label": torch.randint(0, 2, (4, 32, 32)),
        }

        loss_dict = self.strategy.compute_loss(batch, iter_num=1)

        self.assertEqual(
            set(loss_dict.keys()),
            {
                "total",
                "ce",
                "dice",
                "consistency",
                "ema_consistency",
                "depth_teacher_consistency",
                "consistency_weight",
            },
        )
        self.assertIn("teacher_pred", batch)
        self.assertIn("depth_teacher_pred", batch)
        self.assertEqual(tuple(batch["teacher_pred"].shape), (2, 2, 32, 32))
        self.assertEqual(tuple(batch["depth_teacher_pred"].shape), (2, 2, 32, 32))

    def test_get_state_dict_includes_depth_teacher_model(self):
        state_dict = self.strategy.get_state_dict()
        self.assertIn("model", state_dict)
        self.assertIn("depth_teacher_model", state_dict)


if __name__ == "__main__":
    unittest.main()
