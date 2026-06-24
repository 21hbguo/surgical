from argparse import Namespace
import unittest

import torch

from strategies import STRATEGY_REGISTRY
from strategies.fully_supervised_depthGAN import FullySupervisedDepthGANStrategy


class DummySegModel(torch.nn.Module):
    def __init__(self, in_chns=3, class_num=2):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num}
        self.conv = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)
        self.seen_shape = None

    def forward(self, x):
        self.seen_shape = tuple(x.shape)
        return self.conv(x)


class FullySupervisedDepthGANStrategyTest(unittest.TestCase):
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
            way="fully_supervised_depthgan",
            in_chns=3,
            gan_loss_weight=0.01,
            gan_lr=1e-4,
        )
        self.model = DummySegModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device("cpu")
        self.strategy = FullySupervisedDepthGANStrategy(self.args, self.model, self.optimizer, self.device)

    def test_registry_contains_strategy(self):
        self.assertIn("fully_supervised_depthgan", STRATEGY_REGISTRY)

    def test_compute_loss_keeps_seg_model_rgb_only_and_uses_depth_for_gan(self):
        batch = {
            "image": torch.randn(2, 3, 16, 16),
            "depth1": torch.randn(2, 1, 16, 16),
            "label": torch.randint(0, 2, (2, 16, 16)),
        }
        loss_dict = self.strategy.compute_loss(batch)
        self.assertEqual(self.model.seen_shape, (2, 3, 16, 16))
        self.assertEqual(set(loss_dict.keys()), {"total", "ce", "dice", "gan_adv", "gan_weight"})
        self.assertEqual(self.strategy.discriminator.net[0].in_channels, 2)

    def test_training_step_updates_and_saves_discriminator(self):
        batch = {
            "image": torch.randn(2, 3, 16, 16),
            "depth1": torch.randn(2, 1, 16, 16),
            "label": torch.randint(0, 2, (2, 16, 16)),
        }
        loss_dict = self.strategy.training_step(batch)
        self.assertIn("gan_disc", loss_dict)
        state_dict = self.strategy.get_state_dict()
        self.assertIn("model", state_dict)
        self.assertIn("discriminator", state_dict)


if __name__ == "__main__":
    unittest.main()
