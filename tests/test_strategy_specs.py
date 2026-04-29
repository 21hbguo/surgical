import json
import os
import tempfile
import unittest

from strategies.specs import (
    get_strategy_names,
    get_strategy_spec,
    resolve_strategy_default_model_name,
    resolve_strategy_input_settings,
)


def _write_task_json(root_path: str, task: int, input_channels: int = 3) -> None:
    with open(os.path.join(root_path, f"task{task}.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_classes": 2,
                "input_channels": input_channels,
                "n_folds": 4,
                "classes": [{"name": "class0", "label_id": 0}],
            },
            handle,
        )


class StrategySpecsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_path = self.tmpdir.name
        _write_task_json(self.root_path, task=1, input_channels=3)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_strategy_names_are_exposed_from_central_specs(self):
        names = get_strategy_names()
        self.assertIn("fully", names)
        self.assertIn("mt_depth_teacher_v1", names)
        self.assertIn("mt_depth_guider_proto_teacher_v2", names)
        self.assertIn("mt_depth_guider_proto_teacher_v3", names)
        self.assertNotIn("mt_depth_guider_proto_v2", names)

    def test_strategy_spec_carries_semisupervised_and_input_override_metadata(self):
        teacher_spec = get_strategy_spec("mt_depth_teacher_v1")
        self.assertTrue(teacher_spec.is_semi)
        self.assertEqual(teacher_spec.in_chns, "metadata")

        masking_spec = get_strategy_spec("fully_rgb_masking_depth_v1")
        self.assertFalse(masking_spec.is_semi)
        self.assertEqual(masking_spec.in_chns, "metadata")

    def test_resolve_strategy_input_settings_uses_metadata_override_when_requested(self):
        resolved = resolve_strategy_input_settings(
            way="mt_depth_teacher_v1",
            root_path=self.root_path,
            task=1,
            use_depth=1,
        )
        self.assertEqual(resolved["metadata_in_chns"], 3)
        self.assertEqual(resolved["use_depth"], 1)
        self.assertEqual(resolved["in_chns"], 3)

    def test_resolve_strategy_input_settings_keeps_depth13_semantics_for_default_strategies(self):
        resolved = resolve_strategy_input_settings(
            way="mt_depth_guider_proto_teacher_v2",
            root_path=self.root_path,
            task=1,
            use_depth=13,
        )
        self.assertEqual(resolved["use_depth"], 13)
        self.assertEqual(resolved["in_chns"], 4)

    def test_resolve_strategy_default_model_name_comes_from_spec(self):
        self.assertEqual(
            resolve_strategy_default_model_name("mt_depth_guider_proto_teacher_v2", "resnet"),
            "resnet_depth_guider_proto_v1",
        )
        self.assertEqual(
            resolve_strategy_default_model_name("mt_depth_guider_proto_teacher_v3", "resnet"),
            "resnet_depth_guider_proto_v1",
        )
        self.assertEqual(
            resolve_strategy_default_model_name("dformerv2_fully", "resnet"),
            "dformerv2_small",
        )


if __name__ == "__main__":
    unittest.main()
