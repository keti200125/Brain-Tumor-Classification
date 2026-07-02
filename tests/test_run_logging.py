from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from brain_tumor_classifier.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.experiments import ExperimentRunner


RUN_TIME = datetime(2026, 6, 30, 14, 15, 22, tzinfo=timezone.utc)


class RunLoggingTest(unittest.TestCase):
    def test_failed_run_logs_exception_before_failed_history_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "broken.yaml"
            config_path.write_text("experiment_name: broken\n", encoding="utf-8")
            config = ExperimentConfig(
                experiment_name="broken_exp",
                trace_name="broken_trace_v1",
                report_name="Broken run",
                hypothesis="Failure path should be logged",
                description="Intentionally missing dataset directories",
                tags=["logging", "failure"],
                seed=42,
                device="cpu",
                data=DataConfig(root / "MISSING_DATA", 0.5, None, None, None),
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
                config_path=config_path,
                config_filename=config_path.name,
                config_hash="a13f91be" + ("0" * 56),
            )

            runner = ExperimentRunner(
                config=config,
                config_path=config.config_path,
                show_progress=False,
                timestamp=RUN_TIME,
            )

            with self.assertRaises(FileNotFoundError):
                runner.run()

            run_dir = (
                root
                / "outputs"
                / "experiments"
                / "broken_exp"
                / "20260630-141522__broken_trace_v1__a13f91be"
            )
            log_path = run_dir / "logs" / "run.log"
            self.assertTrue(log_path.is_file())

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("Config path:", log_text)
            self.assertIn("Run ID:", log_text)
            self.assertIn("Device:", log_text)
            self.assertIn("Experiment failed:", log_text)
            self.assertIn("Expected Training and Testing directories under", log_text)

            metrics_payload = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics_payload["status"], "failed")
            self.assertEqual(metrics_payload["seed"], 42)
            self.assertEqual(metrics_payload["experiment_name"], "broken_exp")
            self.assertEqual(
                metrics_payload["artifacts"]["history_csv_path"],
                str(run_dir / "history.csv"),
            )
            self.assertEqual(metrics_payload["data"]["train_size"], 0)
            self.assertEqual(metrics_payload["data"]["class_names"], [])
            self.assertIn(
                "Expected Training and Testing directories under",
                metrics_payload["error_message"],
            )

            history_path = root / "outputs" / "experiment_history.csv"
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("Expected Training and Testing directories under", rows[0]["error_message"])


if __name__ == "__main__":
    unittest.main()
