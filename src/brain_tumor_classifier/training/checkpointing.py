"""Checkpoint persistence, loading, and reload validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_tumor_classifier.training.engine import EpochResult
from brain_tumor_classifier.training.evaluator import evaluate_model


@dataclass(frozen=True)
class CheckpointValidationResult:
    evaluation: EpochResult
    metrics_match: bool
    max_metric_delta: float


def build_checkpoint(
    *,
    experiment_name: str,
    trace_name: str,
    run_id: str,
    seed: int,
    config_file: str,
    config_hash: str,
    model_name: str,
    state_dict: dict[str, Any],
    class_names: list[str],
    class_to_idx: dict[str, int],
    image_size: int,
    pretrained_used: bool,
    best_epoch: int,
    best_val_metric: float,
    final_test_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build the stable checkpoint payload specified by the project plan."""
    return {
        "experiment_name": experiment_name,
        "trace_name": trace_name,
        "run_id": run_id,
        "seed": seed,
        "config_file": config_file,
        "config_hash": config_hash,
        "model_name": model_name,
        "state_dict": state_dict,
        "class_names": list(class_names),
        "class_to_idx": dict(class_to_idx),
        "image_size": image_size,
        "pretrained_used": pretrained_used,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "final_test_metrics": dict(final_test_metrics),
    }


def save_best_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    _save_checkpoint(path, checkpoint)


def save_last_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    _save_checkpoint(path, checkpoint)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    map_location: torch.device | str,
    strict: bool = True,
) -> dict[str, Any]:
    """Load checkpoint metadata and restore a model's state dictionary."""
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {path}")
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return checkpoint


def validate_reloaded_model(
    model: nn.Module,
    checkpoint_path: Path,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    expected_metrics: dict[str, float],
    *,
    use_inception_aux: bool = False,
    tolerance: float = 1e-6,
    show_progress: bool = True,
    raise_on_mismatch: bool = True,
) -> CheckpointValidationResult:
    """Reload, evaluate, and compare a checkpoint to original test metrics."""
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    evaluation = evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        model_name=model_name,
        use_inception_aux=use_inception_aux,
        show_progress=show_progress,
    )
    missing_metrics = sorted(set(expected_metrics) - evaluation.metrics.keys())
    if missing_metrics:
        raise ValueError(
            f"Reloaded evaluation is missing metrics: {missing_metrics}"
        )
    deltas = [
        abs(evaluation.metrics[name] - expected_value)
        for name, expected_value in expected_metrics.items()
    ]
    max_delta = max(deltas, default=0.0)
    metrics_match = max_delta <= tolerance
    if raise_on_mismatch and not metrics_match:
        raise ValueError(
            "Reloaded checkpoint metrics differ from original metrics by "
            f"{max_delta:.8f} (tolerance={tolerance:.8f})"
        )
    return CheckpointValidationResult(
        evaluation=evaluation,
        metrics_match=metrics_match,
        max_metric_delta=max_delta,
    )


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
