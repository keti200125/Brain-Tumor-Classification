"""Epoch execution and multi-epoch model training."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_tumor_classifier.models import unpack_inception_outputs
from brain_tumor_classifier.training.metrics import (
    CLASSIFICATION_METRIC_NAMES,
    compute_classification_metrics,
)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: dict[str, float]
    sample_count: int


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train: EpochResult
    validation: EpochResult
    learning_rate: float = 0.0
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class TrainingResult:
    epochs: list[EpochRecord]
    best_state_dict: dict[str, Any]
    last_state_dict: dict[str, Any]
    best_epoch: int
    best_val_metric: float
    monitor_metric: str
    monitor_mode: str

    def plot_history(self) -> dict[str, list[float]]:
        """Return the existing plotting shape during the reporting transition."""
        return {
            "train_loss": [record.train.loss for record in self.epochs],
            "val_loss": [record.validation.loss for record in self.epochs],
            "train_f1": [
                record.train.metrics["f1_weighted"] for record in self.epochs
            ],
            "val_f1": [
                record.validation.metrics["f1_weighted"] for record in self.epochs
            ],
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    epoch_idx: int,
    epochs: int,
    optimizer: torch.optim.Optimizer | None = None,
    use_inception_aux: bool = False,
    show_progress: bool = True,
) -> EpochResult:
    """Run one training or evaluation epoch and return structured metrics."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    sample_count = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress_label = "train" if is_training else "eval"
    batches = tqdm(
        loader,
        desc=f"{model_name} [{epoch_idx + 1}/{epochs}] {progress_label}",
        leave=False,
        disable=not show_progress,
    )

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in batches:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            logits, aux_logits = unpack_inception_outputs(outputs)
            loss = criterion(logits, labels)
            if use_inception_aux and is_training and aux_logits is not None:
                loss = loss + 0.4 * criterion(aux_logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            predictions = torch.argmax(logits, dim=1)
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            batches.set_postfix(loss=f"{loss.item():.4f}")

    if sample_count == 0:
        raise ValueError("cannot run an epoch with an empty DataLoader")
    return EpochResult(
        loss=total_loss / sample_count,
        metrics=compute_classification_metrics(all_targets, all_predictions),
        sample_count=sample_count,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
    epochs: int,
    *,
    monitor_metric: str = "f1_weighted",
    monitor_mode: str = "max",
    use_inception_aux: bool = False,
    show_progress: bool = True,
) -> TrainingResult:
    """Train a model and retain both best and final parameter states."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if monitor_metric not in CLASSIFICATION_METRIC_NAMES:
        raise ValueError(f"Unsupported monitor metric: {monitor_metric}")
    if monitor_mode not in {"min", "max"}:
        raise ValueError("monitor_mode must be either 'min' or 'max'")

    best_value = float("-inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    records: list[EpochRecord] = []

    for epoch_idx in range(epochs):
        started_at = time.perf_counter()
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            model_name=model_name,
            epoch_idx=epoch_idx,
            epochs=epochs,
            optimizer=optimizer,
            use_inception_aux=use_inception_aux,
            show_progress=show_progress,
        )
        val_result = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            model_name=f"{model_name}-val",
            epoch_idx=epoch_idx,
            epochs=epochs,
            optimizer=None,
            use_inception_aux=use_inception_aux,
            show_progress=show_progress,
        )
        record = EpochRecord(
            epoch=epoch_idx + 1,
            train=train_result,
            validation=val_result,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            duration_seconds=time.perf_counter() - started_at,
        )
        records.append(record)

        monitored_value = val_result.metrics[monitor_metric]
        improved = (
            monitored_value > best_value
            if monitor_mode == "max"
            else monitored_value < best_value
        )
        if improved:
            best_value = monitored_value
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[{model_name}] epoch {epoch_idx + 1}/{epochs} | "
            f"train loss={train_result.loss:.4f} "
            f"f1={train_result.metrics['f1_weighted']:.4f} | "
            f"val loss={val_result.loss:.4f} "
            f"f1={val_result.metrics['f1_weighted']:.4f}"
        )

    return TrainingResult(
        epochs=records,
        best_state_dict=best_state,
        last_state_dict=copy.deepcopy(model.state_dict()),
        best_epoch=best_epoch,
        best_val_metric=best_value,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
    )
