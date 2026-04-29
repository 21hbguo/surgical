from argparse import Namespace
import unittest

import torch
import torch.nn.functional as F

from strategies.fully_contrast_v1 import FullyContrastV1Strategy


class DummyContrastModel(torch.nn.Module):
    def __init__(self, in_chns=1, class_num=2, feature_dim=8):
        super().__init__()
        self.params = {"in_chns": in_chns, "class_num": class_num, "feature_dim": feature_dim}
        self.seg_head = torch.nn.Conv2d(in_chns, class_num, kernel_size=1)
        self.proj = torch.nn.Conv2d(in_chns, feature_dim, kernel_size=1)

    def forward(self, x):
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.seg_head(x), self.proj(pooled)


class FullyContrastV1StrategyTest(unittest.TestCase):
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
            way="fully_contrast_v1",
            contrast_loss_weight=0.05,
            contrast_temperature=0.1,
            contrast_boundary_width=1,
            contrast_min_pixels=2,
            contrast_max_samples=16,
        )
        self.model = DummyContrastModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device("cpu")
        self.strategy = FullyContrastV1Strategy(self.args, self.model, self.optimizer, self.device)

    def test_build_class_assignment_matches_nearest_downsampled_labels(self):
        labels = torch.tensor(
            [
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ]
            ],
            dtype=torch.long,
        )

        assigned = self.strategy._build_class_assignment(labels, target_size=(2, 2))

        expected = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
        self.assertTrue(torch.equal(assigned, expected))

    def test_boundary_region_helper_produces_disjoint_non_empty_masks(self):
        label_assign = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=torch.long,
        )

        boundary_masks, interior_masks = self.strategy._build_boundary_region_masks(label_assign)

        self.assertGreater(int(boundary_masks[:, 1].sum().item()), 0)
        self.assertGreater(int(interior_masks[:, 1].sum().item()), 0)
        self.assertEqual(int((boundary_masks & interior_masks).sum().item()), 0)

    def test_compute_loss_returns_contrastive_terms_without_proto(self):
        pattern = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        batch = {
            "image": torch.randn(2, 1, 8, 8),
            "label": torch.stack([pattern, pattern], dim=0),
        }

        loss_dict = self.strategy.compute_loss(batch)

        self.assertEqual(set(loss_dict.keys()), {"total", "ce", "dice", "contrastive", "contrast_weight"})
        self.assertNotIn("proto", loss_dict)
        self.assertEqual(loss_dict["contrastive"].ndim, 0)


if __name__ == "__main__":
    unittest.main()
