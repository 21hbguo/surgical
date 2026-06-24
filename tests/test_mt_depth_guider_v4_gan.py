from argparse import Namespace
import unittest

import torch

from strategies import STRATEGY_REGISTRY
from strategies.semi_mt_depth_guider_gan_v1 import MTDepthGuiderV4GANStrategy


class DummyGuiderModel(torch.nn.Module):
    def __init__(self, in_chns=2, class_num=2):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num}
        self.conv = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MTDepthGuiderV4GANStrategyTest(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            consistency=0.1,
            grad_clip=0.0,
            strong="s",
            labeled_bs=2,
            num_classes=2,
            use_depth=1,
            ema_decay=0.99,
            consistency_rampup=150,
            consistency_rampup_div=200,
            consistency_start_iters=0,
            way="mt_depth_guider_v4_gan",
            gan_loss_weight=0.01,
            gan_lr=1e-4,
        )
        self.model = DummyGuiderModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device("cpu")
        self.strategy = MTDepthGuiderV4GANStrategy(self.args, self.model, self.optimizer, self.device)

    def test_strategy_registered(self):
        self.assertIn("mt_depth_guider_v4_gan", STRATEGY_REGISTRY)

    def test_compute_loss_returns_mt_and_gan_terms(self):
        batch = {
            "image": torch.randn(4, 1, 32, 32),
            "depth1": torch.randn(4, 1, 32, 32),
            "label": torch.randint(0, 2, (4, 32, 32)),
        }
        loss_dict = self.strategy.compute_loss(batch, iter_num=1)
        self.assertEqual(
            set(loss_dict.keys()),
            {"total", "ce", "dice", "consistency", "consistency_weight", "gan_adv", "gan_weight"},
        )
        self.assertIn("teacher_pred", batch)
        self.assertEqual(tuple(batch["teacher_pred"].shape), (2, 2, 32, 32))

    def test_training_step_updates_and_saves_discriminator(self):
        batch = {
            "image": torch.randn(4, 1, 32, 32),
            "depth1": torch.randn(4, 1, 32, 32),
            "label": torch.randint(0, 2, (4, 32, 32)),
        }
        loss_dict = self.strategy.training_step(batch, iter_num=1)
        self.assertIn("gan_disc", loss_dict)
        self.assertIn("gan_real", loss_dict)
        self.assertIn("gan_fake", loss_dict)
        state_dict = self.strategy.get_state_dict()
        self.assertIn("model", state_dict)
        self.assertIn("discriminator", state_dict)


if __name__ == "__main__":
    unittest.main()
