"""Evaluate an existing experiment checkpoint without retraining."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brain_tumor_classifier.config import load_experiment_config
from brain_tumor_classifier.data import INCEPTION_MIN_IMAGE_SIZE, BrainTumorDataModule
from brain_tumor_classifier.experiments import ExperimentResult
from brain_tumor_classifier.experiments.history import (
    append_experiment_history_row,
    experiment_result_to_row,
    write_metrics_json,
)
from brain_tumor_classifier.experiments.naming import make_filesystem_safe
from brain_tumor_classifier.models import build_model, get_architecture_name
from brain_tumor_classifier.reporting import build_plot_paths, generate_excel_report
from brain_tumor_classifier.training import EpochResult, evaluate_model, load_checkpoint
from brain_tumor_classifier.utils import seed_everything


@dataclass(frozen=True)
class CheckpointEvaluation:
    checkpoint_path: Path
    model_name: str
    test_size: int
    result: EpochResult
    results_recorded: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the deterministic test split",
    )
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--checkpoint", required=True, help="Saved .pt checkpoint path")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Evaluation device; defaults to the config value",
    )
    parser.add_argument(
        "--record-results",
        action="store_true",
        help="Recover metrics.json and append one successful history row",
    )
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def evaluate_saved_checkpoint(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device_name: str | None = None,
    show_progress: bool = True,
    record_results: bool = False,
) -> CheckpointEvaluation:
    """Load and evaluate a checkpoint without updating model parameters."""
    config = load_experiment_config(config_path, project_root=PROJECT_ROOT)
    checkpoint_path = _project_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    seed_everything(config.seed)
    device = _resolve_device(device_name or config.device)
    image_size = config.training.image_size
    if config.model.name == "inception_v3":
        image_size = max(image_size, INCEPTION_MIN_IMAGE_SIZE)

    data_module = BrainTumorDataModule(
        data_root=config.data.root,
        batch_size=config.training.batch_size,
        image_size=image_size,
        validation_fraction=config.data.validation_fraction,
        num_workers=config.training.num_workers,
        seed=config.seed,
        max_train_samples=config.data.max_train_samples,
        max_val_samples=config.data.max_val_samples,
        max_test_samples=config.data.max_test_samples,
    )
    data_module.setup()

    model, _ = build_model(
        config.model.name,
        num_classes=len(data_module.class_names),
        pretrained=False,
    )
    model.to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name != config.model.name:
        raise ValueError(
            "Checkpoint model does not match config: "
            f"{checkpoint_model_name!r} != {config.model.name!r}"
        )
    if checkpoint.get("class_names") != data_module.class_names:
        raise ValueError("Checkpoint class names do not match the dataset")
    if checkpoint.get("image_size") != image_size:
        raise ValueError("Checkpoint image size does not match the config")

    result = evaluate_model(
        model=model,
        loader=data_module.test_dataloader(),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        model_name=f"{config.model.name}-checkpoint-test",
        use_inception_aux=config.model.name == "inception_v3",
        show_progress=show_progress,
    )
    results_recorded = False
    if record_results:
        validation_result = evaluate_model(
            model=model,
            loader=data_module.val_dataloader(),
            criterion=nn.CrossEntropyLoss(),
            device=device,
            model_name=f"{config.model.name}-checkpoint-validation",
            use_inception_aux=config.model.name == "inception_v3",
            show_progress=show_progress,
        )
        results_recorded = _record_recovered_results(
            config=config,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            data_module=data_module,
            image_size=image_size,
            validation_result=validation_result,
            test_result=result,
        )
    return CheckpointEvaluation(
        checkpoint_path=checkpoint_path,
        model_name=config.model.name,
        test_size=len(data_module.test_samples),
        result=result,
        results_recorded=results_recorded,
    )


def _record_recovered_results(
    *,
    config: Any,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    data_module: BrainTumorDataModule,
    image_size: int,
    validation_result: EpochResult,
    test_result: EpochResult,
) -> bool:
    run_id = str(checkpoint.get("run_id", checkpoint_path.parent.name))
    if run_id != checkpoint_path.parent.name:
        raise ValueError("Checkpoint run ID does not match its directory")

    experiment_name = str(checkpoint.get("experiment_name", config.experiment_name))
    run_dir = (
        config.outputs.root
        / "experiments"
        / make_filesystem_safe(experiment_name, fallback="experiment")
        / run_id
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Original run directory not found: {run_dir}")

    plots_dir = run_dir / "plots"
    plot_paths = {
        key: str(path) for key, path in build_plot_paths(plots_dir).items()
    }
    metrics_json_path = run_dir / "metrics.json"
    epoch_history_path = run_dir / "history.csv"
    history_csv_value = str(epoch_history_path) if epoch_history_path.is_file() else ""
    last_checkpoint_path = checkpoint_path.parent / "last_model.pt"
    last_checkpoint_value = (
        str(last_checkpoint_path) if last_checkpoint_path.is_file() else ""
    )
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    result = ExperimentResult(
        experiment_name=experiment_name,
        trace_name=str(checkpoint.get("trace_name", config.trace_name)),
        run_id=run_id,
        config_file=str(config.config_path),
        config_hash=str(checkpoint.get("config_hash", config.config_hash)),
        model_name=config.model.name,
        architecture=get_architecture_name(config.model.name),
        pretrained_requested=config.model.pretrained,
        pretrained_used=bool(checkpoint.get("pretrained_used", False)),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        optimizer=config.training.optimizer,
        image_size=image_size,
        train_size=len(data_module.train_samples),
        val_size=len(data_module.val_samples),
        test_size=len(data_module.test_samples),
        best_epoch=int(checkpoint.get("best_epoch", 0)),
        best_val_loss=validation_result.loss,
        best_val_accuracy=validation_result.metrics["accuracy"],
        best_val_f1_weighted=validation_result.metrics["f1_weighted"],
        test_loss=test_result.loss,
        test_accuracy=test_result.metrics["accuracy"],
        test_f1_weighted=test_result.metrics["f1_weighted"],
        test_precision_weighted=test_result.metrics["precision_weighted"],
        test_recall_weighted=test_result.metrics["recall_weighted"],
        reloaded_test_accuracy=test_result.metrics["accuracy"],
        reloaded_test_f1_weighted=test_result.metrics["f1_weighted"],
        checkpoint_path=str(checkpoint_path),
        run_dir=str(run_dir),
        created_at=created_at,
    )
    report_path = config.outputs.root / "reports" / "model_report.xlsx"
    experiment_history_path = config.outputs.root / "experiment_history.csv"
    metrics_payload = {
        "seed": config.seed,
        "experiment_name": result.experiment_name,
        "trace_name": result.trace_name,
        "run_id": run_id,
        "config_file": str(config.config_path),
        "config_hash": result.config_hash,
        "model": {
            "name": result.model_name,
            "architecture": result.architecture,
            "pretrained_requested": result.pretrained_requested,
            "pretrained_used": result.pretrained_used,
        },
        "data": {
            "train_size": result.train_size,
            "val_size": result.val_size,
            "test_size": result.test_size,
            "class_names": list(data_module.class_names),
        },
        "training": {
            "epochs": result.epochs,
            "batch_size": result.batch_size,
            "learning_rate": result.learning_rate,
            "optimizer": result.optimizer,
            "image_size": result.image_size,
            "best_epoch": result.best_epoch,
        },
        "metrics": {
            "best_val_loss": result.best_val_loss,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_f1_weighted": result.best_val_f1_weighted,
            "test_loss": result.test_loss,
            "test_accuracy": result.test_accuracy,
            "test_f1_weighted": result.test_f1_weighted,
            "test_precision_weighted": result.test_precision_weighted,
            "test_recall_weighted": result.test_recall_weighted,
            "reloaded_test_accuracy": result.reloaded_test_accuracy,
            "reloaded_test_f1_weighted": result.reloaded_test_f1_weighted,
            "reload_f1_delta": 0.0,
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "last_checkpoint_path": last_checkpoint_value,
            "run_dir": str(run_dir),
            "plots_dir": str(plots_dir),
            "history_csv_path": history_csv_value,
            "plot_paths": plot_paths,
            "config_snapshot_path": str(run_dir / "config_snapshot.yaml"),
            "metrics_json_path": str(metrics_json_path),
            "experiment_history_path": str(experiment_history_path),
            "report_path": str(report_path),
        },
        "created_at": created_at,
        "status": "success",
        "recovered_from_checkpoint": True,
    }
    write_metrics_json(metrics_json_path, metrics_payload)

    appended = False
    if not _has_successful_history_row(experiment_history_path, run_id):
        append_experiment_history_row(
            experiment_history_path,
            experiment_result_to_row(
                result=result,
                config=config,
                best_val_precision_weighted=validation_result.metrics[
                    "precision_weighted"
                ],
                best_val_recall_weighted=validation_result.metrics[
                    "recall_weighted"
                ],
                last_checkpoint_path=last_checkpoint_value,
                plots_dir=str(plots_dir),
                plot_paths=plot_paths,
                metrics_json_path=str(metrics_json_path),
                history_csv_path=history_csv_value,
                status="success",
            ),
        )
        appended = True

    if config.outputs.generate_report_after_run and experiment_history_path.is_file():
        generate_excel_report(experiment_history_path, report_path)
    return appended


def _has_successful_history_row(path: Path, run_id: str) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        return any(
            row.get("run_id") == run_id and row.get("status") == "success"
            for row in csv.DictReader(handle)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = evaluate_saved_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        record_results=args.record_results,
    )
    metrics = evaluation.result.metrics
    print(f"Checkpoint: {evaluation.checkpoint_path}")
    print(f"Model: {evaluation.model_name}")
    print(f"Test samples: {evaluation.test_size}")
    print(
        "Test metrics: "
        f"loss={evaluation.result.loss:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"f1_weighted={metrics['f1_weighted']:.4f} "
        f"precision_weighted={metrics['precision_weighted']:.4f} "
        f"recall_weighted={metrics['recall_weighted']:.4f}"
    )
    if args.record_results:
        print(
            "Experiment history: "
            + ("success row appended" if evaluation.results_recorded else "success row already present")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
