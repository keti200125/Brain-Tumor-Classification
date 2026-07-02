from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from brain_tumor_classifier.config import (
    ConfigValidationError,
    load_experiment_config,
)


def valid_config() -> dict:
    return {
        "experiment_name": "custom_cnn_test",
        "trace_name": "custom_cnn_test_v1",
        "report_name": "Custom CNN test",
        "hypothesis": "A test hypothesis.",
        "description": "A small test experiment.",
        "tags": ["custom_cnn", "test"],
        "seed": 42,
        "device": "cpu",
        "data": {
            "root": "DATA",
            "validation_fraction": 0.5,
            "max_train_samples": 8,
            "max_val_samples": 4,
            "max_test_samples": 4,
        },
        "model": {"name": "custom_cnn", "pretrained": False},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "num_workers": 0,
            "image_size": 64,
            "monitor_metric": "f1_weighted",
            "monitor_mode": "max",
        },
        "outputs": {
            "root": "outputs",
            "save_eda_plots": False,
            "save_training_curves": False,
            "save_prediction_grid": False,
            "save_best_checkpoint": True,
            "save_last_checkpoint": True,
            "generate_report_after_run": True,
        },
    }


class ConfigLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_config(self, content: dict, name: str = "experiment.yaml") -> Path:
        path = self.project_root / name
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
        return path

    def test_loads_typed_config_and_resolves_project_paths(self) -> None:
        path = self.write_config(valid_config())
        config = load_experiment_config(path, project_root=self.project_root)

        self.assertEqual(config.model.name, "custom_cnn")
        self.assertEqual(config.training.epochs, 1)
        self.assertEqual(config.data.root, (self.project_root / "DATA").resolve())
        self.assertEqual(
            config.outputs.root,
            (self.project_root / "outputs").resolve(),
        )
        self.assertTrue(config.outputs.generate_report_after_run)
        self.assertEqual(config.config_filename, "experiment.yaml")
        self.assertEqual(len(config.config_hash), 64)

    def test_hash_is_stable_and_changes_with_content(self) -> None:
        content = valid_config()
        first_path = self.write_config(content, "first.yaml")
        second_path = self.write_config(content, "second.yaml")

        first = load_experiment_config(first_path, project_root=self.project_root)
        second = load_experiment_config(second_path, project_root=self.project_root)
        self.assertEqual(first.config_hash, second.config_hash)

        content["seed"] = 99
        self.write_config(content, "second.yaml")
        changed = load_experiment_config(
            second_path,
            project_root=self.project_root,
        )
        self.assertNotEqual(first.config_hash, changed.config_hash)

    def test_missing_and_unknown_keys_are_rejected(self) -> None:
        missing = valid_config()
        del missing["hypothesis"]
        missing_path = self.write_config(missing, "missing.yaml")
        with self.assertRaisesRegex(ConfigValidationError, "missing.*hypothesis"):
            load_experiment_config(missing_path, project_root=self.project_root)

        unknown = valid_config()
        unknown["surprise"] = True
        unknown_path = self.write_config(unknown, "unknown.yaml")
        with self.assertRaisesRegex(ConfigValidationError, "unknown.*surprise"):
            load_experiment_config(unknown_path, project_root=self.project_root)

    def test_invalid_model_and_training_values_are_rejected(self) -> None:
        invalid_model = valid_config()
        invalid_model["model"]["name"] = "not_a_model"
        model_path = self.write_config(invalid_model, "model.yaml")
        with self.assertRaisesRegex(ConfigValidationError, "model.name"):
            load_experiment_config(model_path, project_root=self.project_root)

        invalid_training = valid_config()
        invalid_training["training"]["batch_size"] = 0
        training_path = self.write_config(invalid_training, "training.yaml")
        with self.assertRaisesRegex(ConfigValidationError, "batch_size"):
            load_experiment_config(training_path, project_root=self.project_root)

    def test_all_repository_experiment_configs_are_valid(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        paths = sorted((project_root / "configs" / "experiments").glob("*.yaml"))
        configs = [
            load_experiment_config(path, project_root=project_root)
            for path in paths
        ]

        self.assertEqual(len(configs), 5)
        self.assertEqual(
            {config.model.name for config in configs},
            {
                "custom_cnn",
                "inception_v3",
                "resnet18",
                "resnet50_sspanet",
                "vgg11",
            },
        )


if __name__ == "__main__":
    unittest.main()
