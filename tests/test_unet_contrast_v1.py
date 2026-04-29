import unittest

import torch

from models.networks.unet import UNet_ContrastV1


class UNetContrastV1Test(unittest.TestCase):
    def test_forward_returns_segmentation_and_contrast_features(self):
        model = UNet_ContrastV1(in_chns=3, class_num=4, feature_dim=32, filter_num=16)
        inputs = torch.randn(2, 3, 64, 64)

        seg_logits, contrast_feat = model(inputs)

        self.assertEqual(tuple(seg_logits.shape), (2, 4, 64, 64))
        self.assertEqual(contrast_feat.shape[0], 2)
        self.assertEqual(contrast_feat.shape[1], 32)
        self.assertEqual(tuple(contrast_feat.shape[2:]), (4, 4))


if __name__ == "__main__":
    unittest.main()

