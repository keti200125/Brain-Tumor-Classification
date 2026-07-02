from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from brain_tumor_classifier.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.experiments import (
    create_run_paths,
    generate_run_id,
    make_filesystem_safe,
)


CONFIG_HASH = "a13f91be" + "0" * 56
RUN_TIME = datetime(2026, 6, 30, 14, 15, 22)


class ExperimentNamingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.config_path = self.project_root / "source_config.yaml"
        self.config_path.write_text(
            "experiment_name: source_config\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_config(
        self,
        *,
        experiment_name: str = "transfer_resnet18_lr_1e3_e3",
        trace_name: str = "transfer_resnet18_lr_1e3_e3_v1",
    ) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_name=experiment_name,
            trace_name=trace_name,
            report_name="Test report",
            hypothesis="Test hypothesis",
            description="Test description",
            tags=["test"],
            seed=42,
            device="cpu",
            data=DataConfig(
                root=self.project_root / "DATA",
                validation_fraction=0.5,
                max_train_samples=None,
                max_val_samples=None,
                max_test_samples=None,
            ),
            model=ModelConfig(name="resnet18", pretrained=False),
            training=TrainingConfig(
                epochs=1,
                batch_size=2,
                learning_rate=0.001,
                optimizer="AdamW",
                num_workers=0,
                image_size=224,
                monitor_metric="f1_weighted",
                monitor_mode="max",
            ),
            outputs=OutputConfig(
                root=self.project_root / "outputs",
                save_eda_plots=True,
                save_training_curves=True,
                save_prediction_grid=True,
                save_best_checkpoint=True,
                save_last_checkpoint=True,
                generate_report_after_run=True,
            ),
            config_path=self.config_path,
            config_filename=self.config_path.name,
            config_hash=CONFIG_HASH,
        )

    def test_generate_run_id_matches_documented_format(self) -> None:
        run_id = generate_run_id(
            "transfer_resnet18_lr_1e3_e3_v1",
            CONFIG_HASH,
            timestamp=RUN_TIME,
        )
        self.assertEqual(
            run_id,
            "20260630-141522__transfer_resnet18_lr_1e3_e3_v1__a13f91be",
        )

    def test_names_are_safe_single_path_components(self) -> None:
        safe_name = make_filesystem_safe("../../ Unsafe: experiment/name ")
        self.assertEqual(safe_name, "Unsafe_experiment_name")
        self.assertNotIn("/", safe_name)
        self.assertEqual(make_filesystem_safe("..."), "run")

    def test_create_run_paths_builds_complete_layout_and_snapshot(self) -> None:
        config = self.make_config(experiment_name="Unsafe / Experiment")
        paths = create_run_paths(config, timestamp=RUN_TIME)

        expected_run_id = (
            "20260630-141522__transfer_resnet18_lr_1e3_e3_v1__a13f91be"
        )
        expected_run_dir = (
            config.outputs.root
            / "experiments"
            / "Unsafe_Experiment"
            / expected_run_id
        )
        self.assertEqual(paths.run_dir, expected_run_dir)
        for directory in (
            paths.run_dir,
            paths.plots_dir,
            paths.logs_dir,
            paths.checkpoints_dir,
            paths.report_path.parent,
        ):
            self.assertTrue(directory.is_dir())

        self.assertEqual(paths.metrics_json_path.name, "metrics.json")
        self.assertEqual(paths.epoch_history_path.name, "history.csv")
        self.assertEqual(
            paths.experiment_history_path,
            config.outputs.root / "experiment_history.csv",
        )
        self.assertEqual(paths.report_path.name, "model_report.xlsx")

        snapshot = paths.config_snapshot_path.read_text(encoding="utf-8")
        self.assertIn("# source_config: source_config.yaml", snapshot)
        self.assertIn(f"# config_hash: {CONFIG_HASH}", snapshot)
        self.assertIn("experiment_name: source_config", snapshot)

    def test_same_second_collision_gets_stable_suffix(self) -> None:
        config = self.make_config()
        first = create_run_paths(config, timestamp=RUN_TIME)
        second = create_run_paths(config, timestamp=RUN_TIME)

        self.assertNotEqual(first.run_dir, second.run_dir)
        self.assertTrue(second.run_dir.name.endswith("__02"))
        self.assertEqual(
            second.checkpoints_dir.name,
            second.run_dir.name,
        )

    def test_invalid_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "hexadecimal"):
            generate_run_id("trace", "not-a-hash", timestamp=RUN_TIME)


if __name__ == "__main__":
    unittest.main()
