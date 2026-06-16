import unittest

import numpy as np

from core.args import add_normalize_suffix, build_train_parser
from data.transforms import _normalize_array


class NormalizeModesTest(unittest.TestCase):
    def test_parser_accepts_255_normalize_mode(self):
        args = build_train_parser().parse_args(["--task", "1", "--normalize", "255"])
        self.assertEqual(args.normalize, "255")

    def test_add_normalize_suffix_supports_255(self):
        self.assertEqual(add_normalize_suffix("toy/MT", "255"), "toy_255/MT")

    def test_255_normalization_scales_uint8_range(self):
        arr = np.array([[0, 128, 255]], dtype=np.uint8)
        out = _normalize_array(arr, method="255")
        np.testing.assert_allclose(out, np.array([[0.0, 128.0 / 255.0, 1.0]], dtype=np.float32))

    def test_255_normalization_keeps_unit_range_input(self):
        arr = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
        out = _normalize_array(arr, method="255")
        np.testing.assert_allclose(out, arr)

    def test_255_normalization_scales_uint16_range(self):
        arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
        out = _normalize_array(arr, method="255")
        np.testing.assert_allclose(out, np.array([[0.0, 32768.0 / 65535.0, 1.0]], dtype=np.float32))

    def test_255_normalization_keeps_non_unit_float_range(self):
        arr = np.array([[0.0, 1000.0, 2000.0]], dtype=np.float32)
        out = _normalize_array(arr, method="255")
        np.testing.assert_allclose(out, arr)


if __name__ == "__main__":
    unittest.main()
