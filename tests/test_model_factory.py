import unittest
from argparse import Namespace
from unittest.mock import patch

from models.factory import MODEL_REGISTRY, ModelSpec, create_model, resolve_default_model_name


class ModelFactoryTest(unittest.TestCase):
    def test_default_model_resolution_lives_in_factory(self):
        self.assertEqual(resolve_default_model_name("fully", "none"), "unet")
        self.assertEqual(resolve_default_model_name("semi_mean_teacher_contrast_v1", "none"), "unet_contrast_v1")
        self.assertEqual(resolve_default_model_name("semi_mean_teacher_contrast_v1", "resnet"), "resnet_contrast_v1")
        self.assertEqual(resolve_default_model_name("semi_mean_teacher_contrast_v1", "dinov3"), "dinov3_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_contrast_v1_1", "none"), "unet_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_contrast_v1_1", "resnet"), "resnet_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_contrast_v1_1", "dinov3"), "dinov3_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_contrast_v1", "none"), "unet_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_contrast_v1", "resnet"), "resnet_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_contrast_v1", "dinov3"), "dinov3_contrast_v1")
        self.assertEqual(resolve_default_model_name("fully_rgb_masking_depth_v1", "none"), "unet")
        self.assertEqual(resolve_default_model_name("fully_rgb_masking_depth_v1", "resnet"), "resnet")
        self.assertEqual(resolve_default_model_name("fully_rgb_masking_depth_v1", "depth"), "depth")
        self.assertEqual(resolve_default_model_name("fully_rgb_masking_depth_v1", "dinov3"), "dinov3")
        self.assertEqual(resolve_default_model_name("proto", "resnet"), "resnet_proto_v1")
        self.assertEqual(resolve_default_model_name("proto", "depth"), "depth_proto_v1")
        self.assertEqual(resolve_default_model_name("fully", "depth"), "depth")
        self.assertEqual(resolve_default_model_name("proto", "dinov3"), "dinov3_proto_v1")
        self.assertEqual(resolve_default_model_name("proto_v1", "resnet"), "resnet_proto_v1")
        self.assertEqual(resolve_default_model_name("proto_v1", "depth"), "depth_proto_v1")
        self.assertEqual(resolve_default_model_name("proto_v1", "dinov3"), "dinov3_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_teacher_v1", "none"), "unet")
        self.assertEqual(resolve_default_model_name("mt_depth_teacher_v1", "resnet"), "resnet")
        self.assertEqual(resolve_default_model_name("mt_depth_teacher_v1", "depth"), "depth")
        self.assertEqual(resolve_default_model_name("mt_depth_teacher_v1", "dinov3"), "dinov3")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1", "none"), "unet_depth_guider_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1", "resnet"), "resnet_depth_guider_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1", "depth"), "depth_depth_guider_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1", "dinov3"), "dinov3_depth_guider_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1_2", "none"), "unet_depth_guider_v1_2")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1_2", "resnet"), "resnet_depth_guider_v1_2")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1_2", "depth"), "unet_depth_guider_v1_2")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v1_2", "dinov3"), "unet_depth_guider_v1_2")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v3", "none"), "unet_depth_guider_v3")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v3", "resnet"), "resnet_depth_guider_v3")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v4", "none"), "unet_depth_guider_v4")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v4", "resnet"), "resnet_depth_guider_v4")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v4", "depth"), "unet_depth_guider_v4")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_v4", "dinov3"), "unet_depth_guider_v4")
        self.assertEqual(resolve_default_model_name("georisk_spc", "resnet"), "resnet_georisk_spc")
        self.assertEqual(resolve_default_model_name("georisk_spc_dgv4", "none"), "unet_georisk_spc_dgv4")
        self.assertEqual(resolve_default_model_name("georisk_spc_dgv4", "resnet"), "resnet_georisk_spc_dgv4")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_v1", "none"), "unet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_v1", "resnet"), "resnet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_v1", "depth"), "depth_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_v1", "dinov3"), "dinov3_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v2", "none"), "unet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v2", "resnet"), "resnet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v2", "depth"), "depth_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v2", "dinov3"), "dinov3_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v3", "none"), "unet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v3", "resnet"), "resnet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v3", "depth"), "depth_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v3", "dinov3"), "dinov3_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v1", "none"), "unet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v1", "resnet"), "resnet_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v1", "depth"), "depth_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("mt_depth_guider_proto_teacher_v1", "dinov3"), "dinov3_depth_guider_proto_v1")
        self.assertEqual(resolve_default_model_name("dformerv2_fully", "resnet"), "dformerv2_small")

    def test_create_model_uses_registry_arg_mapping(self):
        self.assertIn("unet_contrast_v1", MODEL_REGISTRY)
        self.assertIn("resnet_contrast_v1", MODEL_REGISTRY)
        self.assertIn("unet_depth_guider_v3", MODEL_REGISTRY)
        self.assertIn("unet_depth_guider_v1_2", MODEL_REGISTRY)
        self.assertIn("unet_depth_guider_v4", MODEL_REGISTRY)
        self.assertIn("resnet_depth_guider_v1_2", MODEL_REGISTRY)
        self.assertIn("resnet_depth_guider_v4", MODEL_REGISTRY)
        self.assertIn("resnet_georisk_spc", MODEL_REGISTRY)
        self.assertIn("resnet_georisk_spc_dgv4", MODEL_REGISTRY)
        captured = {}

        def builder(**kwargs):
            captured.update(kwargs)
            return kwargs

        dummy_spec = ModelSpec(
            builder=builder,
            arg_map={
                "foo": "foo_attr",
                "bar": "bar_attr",
            },
            static_kwargs={"baz": 3},
        )
        args = Namespace(model="dummy", foo_attr=1, bar_attr=2)

        with patch.dict(MODEL_REGISTRY, {"dummy": dummy_spec}, clear=True):
            built = create_model(args)

        self.assertEqual(built, {"foo": 1, "bar": 2, "baz": 3})
        self.assertEqual(captured, {"foo": 1, "bar": 2, "baz": 3})


if __name__ == "__main__":
    unittest.main()
