import unittest

import torch

from models.networks.resnet import ResNetUNet_ContrastV1


class ResNetContrastV1Test(unittest.TestCase):
    def test_forward_returns_segmentation_and_contrast_features(self):
        model = ResNetUNet_ContrastV1(
            in_chns=3,
            class_num=4,
            feature_dim=32,
            load_encoder_pretrained=False,
        )
        inputs = torch.randn(2, 3, 64, 64)

        seg_logits, contrast_feat = model(inputs)

        self.assertEqual(tuple(seg_logits.shape), (2, 4, 64, 64))
        self.assertEqual(contrast_feat.shape[0], 2)
        self.assertEqual(contrast_feat.shape[1], 32)
        self.assertEqual(tuple(contrast_feat.shape[2:]), (2, 2))


if __name__ == "__main__":
    unittest.main()
