"""Typed schema for experiment YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root: Path
    validation_fraction: float
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    num_workers: int
    image_size: int
    monitor_metric: str
    monitor_mode: str


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    save_eda_plots: bool
    save_training_curves: bool
    save_prediction_grid: bool
    save_best_checkpoint: bool
    save_last_checkpoint: bool
    generate_report_after_run: bool


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    trace_name: str
    report_name: str
    hypothesis: str
    description: str
    tags: list[str]
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    outputs: OutputConfig
    config_path: Path
    config_filename: str
    config_hash: str
