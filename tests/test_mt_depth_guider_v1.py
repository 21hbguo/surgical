from argparse import Namespace
import unittest

import torch
import torch.nn.functional as F

from models.networks.resnet import DepthGuider as ResNetDepthGuider
from models.networks.unet import DepthGuider as UNetDepthGuider, DepthGuiderV4, UNet_DepthGuiderV1_2, UNet_DepthGuiderV4
from strategies import STRATEGY_REGISTRY
from strategies.semi_mt_depth_guider_v1 import MTDepthGuiderV1Strategy


class DummyEncoder(torch.nn.Module):
    def __init__(self, in_chns):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_chns, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = F.relu(self.conv1(x))
        e2 = F.relu(self.conv2(F.avg_pool2d(e1, 2, 2)))
        e3 = F.relu(self.conv3(F.avg_pool2d(e2, 2, 2)))
        e4 = F.relu(self.conv4(F.avg_pool2d(e3, 2, 2)))
        e5 = F.relu(self.conv5(F.avg_pool2d(e4, 2, 2)))
        return [e1, e2, e3, e4, e5]


class DummyDecoder(torch.nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.head = torch.nn.Conv2d(8, class_num, kernel_size=1)

    def forward(self, features):
        return self.head(features[0])


class DummyUNetBase(torch.nn.Module):
    def __init__(self, in_chns=2, class_num=2):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num}
        self.encoder = DummyEncoder(in_chns)
        self.decoder = DummyDecoder(class_num)

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)


class MTDepthGuiderV1StrategyTest(unittest.TestCase):
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
            way="mt_depth_guider_v1",
        )
        self.model = DummyUNetBase(in_chns=2, class_num=2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device("cpu")
        self.strategy = MTDepthGuiderV1Strategy(self.args, self.model, self.optimizer, self.device)

    def test_strategy_registered(self):
        self.assertIn("mt_depth_guider_v1", STRATEGY_REGISTRY)

    def test_compute_loss_returns_mt_terms_with_teacher_pred(self):
        batch = {
            "image": torch.randn(4, 1, 32, 32),
            "depth1": torch.randn(4, 1, 32, 32),
            "label": torch.randint(0, 2, (4, 32, 32)),
        }

        loss_dict = self.strategy.compute_loss(batch, iter_num=1)

        self.assertEqual(
            set(loss_dict.keys()),
            {"total", "ce", "dice", "consistency", "consistency_weight"},
        )
        self.assertIn("teacher_pred", batch)
        self.assertEqual(tuple(batch["teacher_pred"].shape), (2, 2, 32, 32))

    def test_requires_depth1c(self):
        args = Namespace(**{**vars(self.args), "use_depth": None})
        with self.assertRaises(ValueError):
            MTDepthGuiderV1Strategy(args, self.model, self.optimizer, self.device)


class DepthGuiderModuleTest(unittest.TestCase):
    def test_resnet_depth_guider_init_identity(self):
        guider = ResNetDepthGuider(8, depth_channels=1)
        rgb = torch.randn(2, 8, 16, 16)
        depth = torch.randn(2, 1, 16, 16)
        out = guider(rgb, depth)
        gamma, beta = guider.compute_gamma_beta(rgb, depth)
        self.assertEqual(tuple(gamma.shape), (2, 8, 16, 16))
        self.assertEqual(tuple(beta.shape), (2, 8, 16, 16))
        self.assertTrue(torch.allclose(out, rgb, atol=1e-6))

    def test_unet_depth_guider_init_identity(self):
        guider = UNetDepthGuider(8, depth_channels=1)
        rgb = torch.randn(2, 8, 16, 16)
        depth = torch.randn(2, 1, 16, 16)
        out = guider(rgb, depth)
        gamma, beta = guider.compute_gamma_beta(rgb, depth)
        self.assertEqual(tuple(gamma.shape), (2, 8, 16, 16))
        self.assertEqual(tuple(beta.shape), (2, 8, 16, 16))
        self.assertTrue(torch.allclose(out, rgb, atol=1e-6))

    def test_unet_depth_guider_v1_2_encoder_guided(self):
        model = UNet_DepthGuiderV1_2(in_chns=3, class_num=2)
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 2, 32, 32))
        self.assertTrue(hasattr(model.encoder, "depth_guiders"))

    def test_unet_depth_guider_v4_init_identity(self):
        guider = DepthGuiderV4(8, depth_channels=1, pool_size=4)
        rgb = torch.randn(2, 8, 16, 16)
        depth = torch.randn(2, 1, 16, 16)
        out = guider(rgb, depth)
        gamma, beta = guider.compute_gamma_beta(rgb, depth)
        self.assertEqual(tuple(gamma.shape), (2, 8, 16, 16))
        self.assertEqual(tuple(beta.shape), (2, 8, 16, 16))
        self.assertTrue(torch.allclose(out, rgb, atol=1e-6))

    def test_unet_depth_guider_v4_scale_router_is_spatial(self):
        guider = DepthGuiderV4(8, depth_channels=1, pool_size=4)
        rgb = torch.randn(2, 8, 16, 16)
        depth = torch.randn(2, 1, 16, 16)
        depth = guider._resize_depth(depth, 16, 16)
        rgb_ctx = guider.rgb_proj(rgb)
        depth_feat, geom_feat = guider._encode_depth(depth)
        scale_weight = guider._compute_scale_weight(rgb_ctx, depth_feat, geom_feat)
        self.assertEqual(tuple(scale_weight.shape), (2, 3, 16, 16))
        self.assertTrue(torch.allclose(scale_weight.sum(dim=1), torch.ones(2, 16, 16), atol=1e-6))

    def test_unet_depth_guider_v4_encoder_guided(self):
        model = UNet_DepthGuiderV4(in_chns=3, class_num=2)
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 2, 32, 32))
        self.assertTrue(hasattr(model.encoder, "depth_guiders"))


if __name__ == "__main__":
    unittest.main()
