from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from brain_tumor_classifier.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.experiments.history import (
    GLOBAL_HISTORY_COLUMNS,
    RUN_HISTORY_COLUMNS,
    append_experiment_history_row,
    experiment_result_to_row,
    failed_experiment_row,
    write_run_history_csv,
)
from brain_tumor_classifier.experiments.runner import ExperimentResult
from brain_tumor_classifier.training import EpochRecord, EpochResult, TrainingResult


class ExperimentHistoryTest(unittest.TestCase):
    def test_write_run_history_csv_uses_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._make_config(root)
            path = root / "outputs" / "experiments" / "phase8" / "run" / "history.csv"

            write_run_history_csv(
                path=path,
                config=config,
                run_id="run-123",
                model_name="custom_cnn",
                training_result=self._make_training_result(),
                created_at="2026-06-30T14:15:22+00:00",
            )

            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, RUN_HISTORY_COLUMNS)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["experiment_name"], "phase8")
            self.assertEqual(rows[0]["trace_name"], "trace_a")
            self.assertEqual(rows[0]["run_id"], "run-123")
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertEqual(rows[1]["epoch"], "2")

    def test_append_experiment_history_row_always_appends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "outputs" / "experiment_history.csv"

            append_experiment_history_row(
                path,
                {
                    "created_at": "2026-06-30T14:15:22+00:00",
                    "experiment_name": "phase8",
                    "trace_name": "trace_a",
                    "run_id": "run-1",
                    "status": "success",
                },
            )
            append_experiment_history_row(
                path,
                {
                    "created_at": "2026-06-30T14:16:22+00:00",
                    "experiment_name": "phase8",
                    "trace_name": "trace_b",
                    "run_id": "run-2",
                    "status": "failed",
                    "error_message": "boom",
                },
            )

            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, GLOBAL_HISTORY_COLUMNS)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "run-1")
            self.assertEqual(rows[1]["run_id"], "run-2")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[1]["status"], "failed")
            self.assertEqual(rows[1]["error_message"], "boom")

    def test_append_experiment_history_row_upgrades_older_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "outputs" / "experiment_history.csv"
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "created_at",
                        "experiment_name",
                        "trace_name",
                        "run_id",
                        "plots_dir",
                        "status",
                        "error_message",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "created_at": "2026-06-30T14:15:22+00:00",
                        "experiment_name": "old_exp",
                        "trace_name": "old_trace",
                        "run_id": "old_run",
                        "plots_dir": "/tmp/plots",
                        "status": "success",
                        "error_message": "",
                    }
                )

            append_experiment_history_row(
                path,
                {
                    "created_at": "2026-06-30T14:16:22+00:00",
                    "experiment_name": "new_exp",
                    "trace_name": "new_trace",
                    "run_id": "new_run",
                    "plots_dir": "/tmp/plots",
                    "eda_class_samples_path": "/tmp/plots/eda_class_samples.png",
                    "status": "success",
                },
            )

            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, GLOBAL_HISTORY_COLUMNS)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "old_run")
            self.assertEqual(rows[1]["run_id"], "new_run")
            self.assertEqual(
                rows[1]["eda_class_samples_path"],
                "/tmp/plots/eda_class_samples.png",
            )

    def test_row_builders_fill_traceability_and_status_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._make_config(root)
            result = ExperimentResult(
                experiment_name="phase8",
                trace_name="trace_a",
                run_id="run-123",
                config_file=str(config.config_path),
                config_hash=config.config_hash,
                model_name="custom_cnn",
                architecture="CustomTumorCNN",
                pretrained_requested=False,
                pretrained_used=False,
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                optimizer="AdamW",
                image_size=32,
                train_size=8,
                val_size=4,
                test_size=4,
                best_epoch=2,
                best_val_loss=0.2,
                best_val_accuracy=0.9,
                best_val_f1_weighted=0.88,
                test_loss=0.25,
                test_accuracy=0.87,
                test_f1_weighted=0.86,
                test_precision_weighted=0.85,
                test_recall_weighted=0.87,
                reloaded_test_accuracy=0.87,
                reloaded_test_f1_weighted=0.86,
                checkpoint_path="/tmp/best_model.pt",
                run_dir="/tmp/run-123",
                created_at="2026-06-30T14:15:22+00:00",
            )

            success_row = experiment_result_to_row(
                result=result,
                config=config,
                best_val_precision_weighted=0.89,
                best_val_recall_weighted=0.9,
                last_checkpoint_path="/tmp/last_model.pt",
                plots_dir="/tmp/plots",
                plot_paths={
                    "eda_class_samples_path": "/tmp/plots/eda_class_samples.png",
                    "eda_class_distribution_path": "/tmp/plots/eda_class_distribution.png",
                    "eda_image_sizes_path": "/tmp/plots/eda_image_sizes.png",
                    "train_vs_val_loss_path": "/tmp/plots/train_vs_val_loss.png",
                    "train_vs_val_f1_path": "/tmp/plots/train_vs_val_f1.png",
                    "prediction_grid_path": "/tmp/plots/prediction_grid.png",
                },
                metrics_json_path="/tmp/metrics.json",
                history_csv_path="/tmp/history.csv",
                status="success",
            )
            failure_row = failed_experiment_row(
                config=config,
                run_id="run-999",
                created_at="2026-06-30T14:17:22+00:00",
                run_dir="/tmp/run-999",
                plots_dir="/tmp/plots",
                plot_paths={
                    "eda_class_samples_path": "/tmp/plots/eda_class_samples.png",
                    "eda_class_distribution_path": "/tmp/plots/eda_class_distribution.png",
                    "eda_image_sizes_path": "/tmp/plots/eda_image_sizes.png",
                    "train_vs_val_loss_path": "/tmp/plots/train_vs_val_loss.png",
                    "train_vs_val_f1_path": "/tmp/plots/train_vs_val_f1.png",
                    "prediction_grid_path": "/tmp/plots/prediction_grid.png",
                },
                metrics_json_path="/tmp/metrics.json",
                history_csv_path="/tmp/history.csv",
                error_message="training failed",
            )

            self.assertEqual(list(success_row.keys()), GLOBAL_HISTORY_COLUMNS)
            self.assertEqual(success_row["config_file"], str(config.config_path))
            self.assertEqual(success_row["config_hash"], config.config_hash)
            self.assertEqual(success_row["trace_name"], "trace_a")
            self.assertEqual(success_row["status"], "success")
            self.assertEqual(success_row["error_message"], "")
            self.assertEqual(success_row["reload_f1_delta"], 0.0)
            self.assertEqual(
                success_row["train_vs_val_f1_path"],
                "/tmp/plots/train_vs_val_f1.png",
            )

            self.assertEqual(failure_row["experiment_name"], "phase8")
            self.assertEqual(failure_row["trace_name"], "trace_a")
            self.assertEqual(failure_row["run_id"], "run-999")
            self.assertEqual(failure_row["status"], "failed")
            self.assertEqual(failure_row["error_message"], "training failed")
            self.assertEqual(failure_row["architecture"], "CustomTumorCNN")
            self.assertEqual(
                failure_row["prediction_grid_path"],
                "/tmp/plots/prediction_grid.png",
            )

    def _make_config(self, root: Path) -> ExperimentConfig:
        config_path = root / "phase8.yaml"
        config_path.write_text("experiment_name: phase8\n", encoding="utf-8")
        return ExperimentConfig(
            experiment_name="phase8",
            trace_name="trace_a",
            report_name="Phase 8 Report",
            hypothesis="History should be traceable",
            description="CSV contract test",
            tags=["phase8", "history"],
            seed=42,
            device="cpu",
            data=DataConfig(root / "DATA", 0.5, None, 5, 6),
            model=ModelConfig("custom_cnn", False),
            training=TrainingConfig(
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                optimizer="AdamW",
                num_workers=0,
                image_size=32,
                monitor_metric="f1_weighted",
                monitor_mode="max",
            ),
            outputs=OutputConfig(
                root / "outputs",
                False,
                False,
                False,
                True,
                True,
                True,
            ),
            config_path=config_path,
            config_filename=config_path.name,
            config_hash="a13f91be" + ("0" * 56),
        )

    def _make_training_result(self) -> TrainingResult:
        epoch_one = EpochRecord(
            epoch=1,
            train=EpochResult(
                loss=0.7,
                metrics={
                    "accuracy": 0.5,
                    "f1_weighted": 0.48,
                    "precision_weighted": 0.47,
                    "recall_weighted": 0.5,
                },
                sample_count=8,
            ),
            validation=EpochResult(
                loss=0.6,
                metrics={
                    "accuracy": 0.75,
                    "f1_weighted": 0.74,
                    "precision_weighted": 0.76,
                    "recall_weighted": 0.75,
                },
                sample_count=4,
            ),
            learning_rate=1e-3,
            duration_seconds=1.25,
        )
        epoch_two = EpochRecord(
            epoch=2,
            train=EpochResult(
                loss=0.4,
                metrics={
                    "accuracy": 0.875,
                    "f1_weighted": 0.87,
                    "precision_weighted": 0.88,
                    "recall_weighted": 0.875,
                },
                sample_count=8,
            ),
            validation=EpochResult(
                loss=0.3,
                metrics={
                    "accuracy": 1.0,
                    "f1_weighted": 1.0,
                    "precision_weighted": 1.0,
                    "recall_weighted": 1.0,
                },
                sample_count=4,
            ),
            learning_rate=1e-3,
            duration_seconds=1.15,
        )
        return TrainingResult(
            epochs=[epoch_one, epoch_two],
            best_state_dict={},
            last_state_dict={},
            best_epoch=2,
            best_val_metric=1.0,
            monitor_metric="f1_weighted",
            monitor_mode="max",
        )


if __name__ == "__main__":
    unittest.main()
