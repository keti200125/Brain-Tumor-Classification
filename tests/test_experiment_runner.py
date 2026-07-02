from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from brain_tumor_classifier.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.experiments import ExperimentRunner


RUN_TIME = datetime(2026, 6, 30, 14, 15, 22, tzinfo=timezone.utc)


class ExperimentRunnerTest(unittest.TestCase):
    def test_runner_writes_phase_seven_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root / "DATA")
            source_config = root / "experiment.yaml"
            source_config.write_text("experiment_name: phase7\n", encoding="utf-8")

            config = ExperimentConfig(
                experiment_name="phase7",
                trace_name="custom_cnn_smoke",
                report_name="Phase 7 report",
                hypothesis="Runner should orchestrate one experiment end to end",
                description="Synthetic end-to-end test",
                tags=["phase7", "test"],
                seed=7,
                device="cpu",
                data=DataConfig(root / "DATA", 0.5, None, None, None),
                model=ModelConfig("custom_cnn", False),
                training=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    optimizer="AdamW",
                    num_workers=0,
                    image_size=32,
                    monitor_metric="f1_weighted",
                    monitor_mode="max",
                ),
                outputs=OutputConfig(
                    root / "outputs",
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ),
                config_path=source_config,
                config_filename=source_config.name,
                config_hash="a13f91be" + ("0" * 56),
            )

            runner = ExperimentRunner(
                config=config,
                config_path=config.config_path,
                show_progress=False,
                timestamp=RUN_TIME,
            )
            result = runner.run()

            run_dir = Path(result.run_dir)
            self.assertEqual(
                result.run_id,
                "20260630-141522__custom_cnn_smoke__a13f91be",
            )
            self.assertTrue(run_dir.is_dir())
            self.assertTrue((run_dir / "config_snapshot.yaml").is_file())
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertTrue((run_dir / "history.csv").is_file())
            self.assertTrue((run_dir / "logs" / "run.log").is_file())
            self.assertTrue((run_dir / "plots" / "eda_class_samples.png").is_file())
            self.assertTrue((run_dir / "plots" / "eda_class_distribution.png").is_file())
            self.assertTrue((run_dir / "plots" / "eda_image_sizes.png").is_file())
            self.assertTrue((run_dir / "plots" / "train_vs_val_loss.png").is_file())
            self.assertTrue((run_dir / "plots" / "train_vs_val_f1.png").is_file())
            self.assertTrue((run_dir / "plots" / "prediction_grid.png").is_file())
            self.assertTrue(
                (root / "outputs" / "checkpoints" / "phase7" / result.run_id / "best_model.pt").is_file()
            )
            self.assertTrue(
                (root / "outputs" / "checkpoints" / "phase7" / result.run_id / "last_model.pt").is_file()
            )

            metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["seed"], 7)
            self.assertEqual(metrics_payload["experiment_name"], "phase7")
            self.assertEqual(metrics_payload["trace_name"], "custom_cnn_smoke")
            self.assertEqual(metrics_payload["run_id"], result.run_id)
            self.assertEqual(metrics_payload["config_file"], str(source_config))
            self.assertEqual(metrics_payload["config_hash"], config.config_hash)
            self.assertEqual(metrics_payload["model"]["name"], "custom_cnn")
            self.assertEqual(metrics_payload["model"]["architecture"], "CustomTumorCNN")
            self.assertFalse(metrics_payload["model"]["pretrained_requested"])
            self.assertFalse(metrics_payload["model"]["pretrained_used"])
            self.assertEqual(metrics_payload["data"]["train_size"], 8)
            self.assertEqual(metrics_payload["data"]["val_size"], 4)
            self.assertEqual(metrics_payload["data"]["test_size"], 4)
            self.assertEqual(
                metrics_payload["data"]["class_names"],
                ["glioma", "meningioma"],
            )
            self.assertEqual(metrics_payload["training"]["epochs"], 1)
            self.assertEqual(metrics_payload["training"]["batch_size"], 2)
            self.assertEqual(metrics_payload["training"]["learning_rate"], 1e-3)
            self.assertEqual(metrics_payload["training"]["optimizer"], "AdamW")
            self.assertEqual(metrics_payload["training"]["image_size"], 32)
            self.assertEqual(metrics_payload["training"]["best_epoch"], 1)
            self.assertIn("reload_f1_delta", metrics_payload["metrics"])
            self.assertEqual(
                metrics_payload["artifacts"]["checkpoint_path"],
                str(
                    root
                    / "outputs"
                    / "checkpoints"
                    / "phase7"
                    / result.run_id
                    / "best_model.pt"
                ),
            )
            self.assertEqual(
                metrics_payload["artifacts"]["last_checkpoint_path"],
                str(
                    root
                    / "outputs"
                    / "checkpoints"
                    / "phase7"
                    / result.run_id
                    / "last_model.pt"
                ),
            )
            self.assertEqual(metrics_payload["artifacts"]["run_dir"], str(run_dir))
            self.assertEqual(metrics_payload["artifacts"]["plots_dir"], str(run_dir / "plots"))
            self.assertEqual(
                metrics_payload["artifacts"]["history_csv_path"],
                str(run_dir / "history.csv"),
            )
            self.assertEqual(
                metrics_payload["artifacts"]["plot_paths"]["eda_class_samples_path"],
                str(run_dir / "plots" / "eda_class_samples.png"),
            )
            self.assertEqual(
                metrics_payload["artifacts"]["plot_paths"]["train_vs_val_loss_path"],
                str(run_dir / "plots" / "train_vs_val_loss.png"),
            )
            self.assertEqual(
                metrics_payload["artifacts"]["plot_paths"]["prediction_grid_path"],
                str(run_dir / "plots" / "prediction_grid.png"),
            )
            log_text = (run_dir / "logs" / "run.log").read_text(encoding="utf-8")
            self.assertIn("Config path:", log_text)
            self.assertIn("Run ID:", log_text)
            self.assertIn("Device:", log_text)
            self.assertIn("Dataset sizes:", log_text)
            self.assertIn("Model: custom_cnn", log_text)
            self.assertIn("Final checkpoint path:", log_text)
            self.assertIn("Final metrics:", log_text)

            with (run_dir / "history.csv").open("r", encoding="utf-8", newline="") as handle:
                history_rows = list(csv.DictReader(handle))
            self.assertEqual(len(history_rows), 1)
            self.assertEqual(history_rows[0]["model_name"], "custom_cnn")
            self.assertEqual(history_rows[0]["epoch"], "1")

            with (root / "outputs" / "experiment_history.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                global_rows = list(csv.DictReader(handle))
            self.assertEqual(len(global_rows), 1)
            self.assertEqual(global_rows[0]["status"], "success")
            self.assertEqual(global_rows[0]["run_id"], result.run_id)
            self.assertEqual(global_rows[0]["history_csv_path"], str(run_dir / "history.csv"))
            self.assertNotEqual(global_rows[0]["best_val_precision_weighted"], "")
            self.assertNotEqual(global_rows[0]["best_val_recall_weighted"], "")
            self.assertEqual(
                global_rows[0]["eda_class_samples_path"],
                str(run_dir / "plots" / "eda_class_samples.png"),
            )
            self.assertEqual(
                global_rows[0]["train_vs_val_loss_path"],
                str(run_dir / "plots" / "train_vs_val_loss.png"),
            )
            self.assertEqual(
                global_rows[0]["prediction_grid_path"],
                str(run_dir / "plots" / "prediction_grid.png"),
            )

    def _build_dataset(self, data_root: Path) -> None:
        for split in ("Training", "Testing"):
            for class_name, colors in {
                "glioma": [(240, 30, 30), (220, 40, 40), (200, 50, 50), (180, 60, 60)],
                "meningioma": [(30, 30, 240), (40, 40, 220), (50, 50, 200), (60, 60, 180)],
            }.items():
                class_dir = data_root / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for idx, color in enumerate(colors):
                    image = Image.new("RGB", (32, 32), color=color)
                    image.save(class_dir / f"{split.lower()}_{idx}.png")


if __name__ == "__main__":
    unittest.main()
