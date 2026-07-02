"""CSV writers for per-run and global experiment history."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from brain_tumor_classifier.config import ExperimentConfig
from brain_tumor_classifier.models import get_architecture_name
from brain_tumor_classifier.training import TrainingResult


RUN_HISTORY_COLUMNS = [
    "experiment_name",
    "trace_name",
    "run_id",
    "config_file",
    "config_hash",
    "model_name",
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_f1_weighted",
    "train_precision_weighted",
    "train_recall_weighted",
    "val_loss",
    "val_accuracy",
    "val_f1_weighted",
    "val_precision_weighted",
    "val_recall_weighted",
    "learning_rate",
    "epoch_duration_seconds",
    "created_at",
]

GLOBAL_HISTORY_COLUMNS = [
    "created_at",
    "experiment_name",
    "trace_name",
    "run_id",
    "config_file",
    "config_hash",
    "report_name",
    "hypothesis",
    "description",
    "tags",
    "model_name",
    "architecture",
    "pretrained_requested",
    "pretrained_used",
    "seed",
    "device",
    "image_size",
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "num_workers",
    "validation_fraction",
    "max_train_samples",
    "max_val_samples",
    "max_test_samples",
    "train_size",
    "val_size",
    "test_size",
    "best_epoch",
    "best_val_loss",
    "best_val_accuracy",
    "best_val_f1_weighted",
    "best_val_precision_weighted",
    "best_val_recall_weighted",
    "test_loss",
    "test_accuracy",
    "test_f1_weighted",
    "test_precision_weighted",
    "test_recall_weighted",
    "reloaded_test_accuracy",
    "reloaded_test_f1_weighted",
    "reload_f1_delta",
    "checkpoint_path",
    "last_checkpoint_path",
    "run_dir",
    "plots_dir",
    "eda_class_samples_path",
    "eda_class_distribution_path",
    "eda_image_sizes_path",
    "train_vs_val_loss_path",
    "train_vs_val_f1_path",
    "prediction_grid_path",
    "metrics_json_path",
    "history_csv_path",
    "status",
    "error_message",
]


def write_run_history_csv(
    *,
    path: Path,
    config: ExperimentConfig,
    run_id: str,
    model_name: str,
    training_result: TrainingResult,
    created_at: str,
) -> None:
    rows: list[dict[str, object]] = []
    for record in training_result.epochs:
        rows.append(
            {
                "experiment_name": config.experiment_name,
                "trace_name": config.trace_name,
                "run_id": run_id,
                "config_file": str(config.config_path),
                "config_hash": config.config_hash,
                "model_name": model_name,
                "epoch": record.epoch,
                "train_loss": record.train.loss,
                "train_accuracy": record.train.metrics["accuracy"],
                "train_f1_weighted": record.train.metrics["f1_weighted"],
                "train_precision_weighted": record.train.metrics["precision_weighted"],
                "train_recall_weighted": record.train.metrics["recall_weighted"],
                "val_loss": record.validation.loss,
                "val_accuracy": record.validation.metrics["accuracy"],
                "val_f1_weighted": record.validation.metrics["f1_weighted"],
                "val_precision_weighted": record.validation.metrics["precision_weighted"],
                "val_recall_weighted": record.validation.metrics["recall_weighted"],
                "learning_rate": record.learning_rate,
                "epoch_duration_seconds": record.duration_seconds,
                "created_at": created_at,
            }
        )

    _write_rows_atomically(path, RUN_HISTORY_COLUMNS, rows)


def append_experiment_history_row(
    path: Path,
    row: dict[str, object],
) -> None:
    normalized_row = {column: row.get(column, "") for column in GLOBAL_HISTORY_COLUMNS}
    path.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_global_history_if_needed(path)

    with path.open("a+", encoding="utf-8", newline="") as handle:
        _lock_file(handle)
        handle.seek(0, 2)
        file_is_empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=GLOBAL_HISTORY_COLUMNS)
        if file_is_empty:
            writer.writeheader()
        writer.writerow(normalized_row)
        handle.flush()
        _unlock_file(handle)


def write_metrics_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def experiment_result_to_row(
    *,
    result: object,
    config: ExperimentConfig,
    best_val_precision_weighted: float,
    best_val_recall_weighted: float,
    last_checkpoint_path: str,
    plots_dir: str,
    plot_paths: dict[str, str],
    metrics_json_path: str,
    history_csv_path: str,
    status: str,
    error_message: str = "",
) -> dict[str, object]:
    payload = asdict(result)
    payload.update(
        {
            "report_name": config.report_name,
            "hypothesis": config.hypothesis,
            "description": config.description,
            "tags": ",".join(config.tags),
            "seed": config.seed,
            "device": config.device,
            "num_workers": config.training.num_workers,
            "validation_fraction": config.data.validation_fraction,
            "max_train_samples": config.data.max_train_samples,
            "max_val_samples": config.data.max_val_samples,
            "max_test_samples": config.data.max_test_samples,
            "best_val_precision_weighted": best_val_precision_weighted,
            "best_val_recall_weighted": best_val_recall_weighted,
            "reload_f1_delta": payload["reloaded_test_f1_weighted"] - payload["test_f1_weighted"],
            "last_checkpoint_path": last_checkpoint_path,
            "plots_dir": plots_dir,
            **plot_paths,
            "metrics_json_path": metrics_json_path,
            "history_csv_path": history_csv_path,
            "status": status,
            "error_message": error_message,
        }
    )
    return {column: payload.get(column, "") for column in GLOBAL_HISTORY_COLUMNS}


def failed_experiment_row(
    *,
    config: ExperimentConfig,
    run_id: str,
    created_at: str,
    run_dir: str,
    plots_dir: str,
    plot_paths: dict[str, str],
    metrics_json_path: str,
    history_csv_path: str,
    error_message: str,
) -> dict[str, object]:
    base_row = {column: "" for column in GLOBAL_HISTORY_COLUMNS}
    base_row.update(
        {
            "created_at": created_at,
            "experiment_name": config.experiment_name,
            "trace_name": config.trace_name,
            "run_id": run_id,
            "config_file": str(config.config_path),
            "config_hash": config.config_hash,
            "report_name": config.report_name,
            "hypothesis": config.hypothesis,
            "description": config.description,
            "tags": ",".join(config.tags),
            "model_name": config.model.name,
            "architecture": get_architecture_name(config.model.name),
            "pretrained_requested": config.model.pretrained,
            "seed": config.seed,
            "device": config.device,
            "image_size": config.training.image_size,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.optimizer,
            "num_workers": config.training.num_workers,
            "validation_fraction": config.data.validation_fraction,
            "max_train_samples": config.data.max_train_samples,
            "max_val_samples": config.data.max_val_samples,
            "max_test_samples": config.data.max_test_samples,
            "run_dir": run_dir,
            "plots_dir": plots_dir,
            **plot_paths,
            "metrics_json_path": metrics_json_path,
            "history_csv_path": history_csv_path,
            "status": "failed",
            "error_message": error_message,
        }
    )
    return base_row


def _write_rows_atomically(
    path: Path,
    columns: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _upgrade_global_history_if_needed(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current_columns = reader.fieldnames or []
        if current_columns == GLOBAL_HISTORY_COLUMNS:
            return
        existing_rows = list(reader)

    normalized_rows = [
        {column: row.get(column, "") for column in GLOBAL_HISTORY_COLUMNS}
        for row in existing_rows
    ]
    _write_rows_atomically(path, GLOBAL_HISTORY_COLUMNS, normalized_rows)


def _lock_file(handle: object) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except (ImportError, AttributeError, OSError):
        return


def _unlock_file(handle: object) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (ImportError, AttributeError, OSError):
        return
