"""YAML loading, strict validation, and config hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from brain_tumor_classifier.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.models import available_models


class ConfigValidationError(ValueError):
    """Raised when an experiment config does not match the schema."""


TOP_LEVEL_KEYS = {
    "experiment_name",
    "trace_name",
    "report_name",
    "hypothesis",
    "description",
    "tags",
    "seed",
    "device",
    "data",
    "model",
    "training",
    "outputs",
}
DATA_KEYS = {
    "root",
    "validation_fraction",
    "max_train_samples",
    "max_val_samples",
    "max_test_samples",
}
MODEL_KEYS = {"name", "pretrained"}
TRAINING_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "num_workers",
    "image_size",
    "monitor_metric",
    "monitor_mode",
}
OUTPUT_KEYS = {
    "root",
    "save_eda_plots",
    "save_training_curves",
    "save_prediction_grid",
    "save_best_checkpoint",
    "save_last_checkpoint",
    "generate_report_after_run",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_experiment_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load and validate an experiment config relative to the project root."""
    root = Path(project_root).resolve() if project_root else get_project_root()
    source_path = Path(config_path)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {source_path}")

    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"Invalid YAML in {source_path}: {exc}"
        ) from exc

    raw = _mapping(loaded, "config")
    _require_exact_keys(raw, TOP_LEVEL_KEYS, "config")

    data_raw = _mapping(raw["data"], "data")
    model_raw = _mapping(raw["model"], "model")
    training_raw = _mapping(raw["training"], "training")
    outputs_raw = _mapping(raw["outputs"], "outputs")
    _require_exact_keys(data_raw, DATA_KEYS, "data")
    _require_exact_keys(model_raw, MODEL_KEYS, "model")
    _require_exact_keys(training_raw, TRAINING_KEYS, "training")
    _require_exact_keys(outputs_raw, OUTPUT_KEYS, "outputs")

    model_name = _string(model_raw, "name", "model")
    supported_models = available_models()
    if model_name not in supported_models:
        raise ConfigValidationError(
            f"model.name must be one of {list(supported_models)}; "
            f"got {model_name!r}"
        )

    validation_fraction = _number(
        data_raw,
        "validation_fraction",
        "data",
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigValidationError(
            "data.validation_fraction must be between 0 and 1"
        )

    epochs = _positive_int(training_raw, "epochs", "training")
    batch_size = _positive_int(training_raw, "batch_size", "training")
    learning_rate = _number(training_raw, "learning_rate", "training")
    if learning_rate <= 0:
        raise ConfigValidationError(
            "training.learning_rate must be greater than zero"
        )
    optimizer = _string(training_raw, "optimizer", "training")
    if optimizer != "AdamW":
        raise ConfigValidationError(
            "training.optimizer currently supports only 'AdamW'"
        )
    num_workers = _integer(training_raw, "num_workers", "training")
    if num_workers < 0:
        raise ConfigValidationError("training.num_workers cannot be negative")
    image_size = _positive_int(training_raw, "image_size", "training")
    monitor_metric = _string(training_raw, "monitor_metric", "training")
    if monitor_metric not in {
        "accuracy",
        "f1_weighted",
        "precision_weighted",
        "recall_weighted",
    }:
        raise ConfigValidationError(
            "training.monitor_metric is not a supported metric"
        )
    monitor_mode = _string(training_raw, "monitor_mode", "training")
    if monitor_mode not in {"min", "max"}:
        raise ConfigValidationError(
            "training.monitor_mode must be either 'min' or 'max'"
        )

    device = _string(raw, "device", "config")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigValidationError(
            "device must be one of ['auto', 'cpu', 'cuda']"
        )

    tags = raw["tags"]
    if not isinstance(tags, list) or not all(
        isinstance(tag, str) and tag.strip() for tag in tags
    ):
        raise ConfigValidationError("config.tags must be a list of strings")

    data_config = DataConfig(
        root=_resolve_project_path(
            _string(data_raw, "root", "data"),
            root,
        ),
        validation_fraction=validation_fraction,
        max_train_samples=_optional_nonnegative_int(
            data_raw,
            "max_train_samples",
            "data",
        ),
        max_val_samples=_optional_nonnegative_int(
            data_raw,
            "max_val_samples",
            "data",
        ),
        max_test_samples=_optional_nonnegative_int(
            data_raw,
            "max_test_samples",
            "data",
        ),
    )
    model_config = ModelConfig(
        name=model_name,
        pretrained=_boolean(model_raw, "pretrained", "model"),
    )
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        num_workers=num_workers,
        image_size=image_size,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
    )
    output_config = OutputConfig(
        root=_resolve_project_path(
            _string(outputs_raw, "root", "outputs"),
            root,
        ),
        save_eda_plots=_boolean(outputs_raw, "save_eda_plots", "outputs"),
        save_training_curves=_boolean(
            outputs_raw,
            "save_training_curves",
            "outputs",
        ),
        save_prediction_grid=_boolean(
            outputs_raw,
            "save_prediction_grid",
            "outputs",
        ),
        save_best_checkpoint=_boolean(
            outputs_raw,
            "save_best_checkpoint",
            "outputs",
        ),
        save_last_checkpoint=_boolean(
            outputs_raw,
            "save_last_checkpoint",
            "outputs",
        ),
        generate_report_after_run=_boolean(
            outputs_raw,
            "generate_report_after_run",
            "outputs",
        ),
    )

    config_hash = hashlib.sha256(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    return ExperimentConfig(
        experiment_name=_string(raw, "experiment_name", "config"),
        trace_name=_string(raw, "trace_name", "config"),
        report_name=_string(raw, "report_name", "config"),
        hypothesis=_string(raw, "hypothesis", "config"),
        description=_string(raw, "description", "config"),
        tags=list(tags),
        seed=_integer(raw, "seed", "config"),
        device=device,
        data=data_config,
        model=model_config,
        training=training_config,
        outputs=output_config,
        config_path=source_path,
        config_filename=source_path.name,
        config_hash=config_hash,
    )


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConfigValidationError(f"{context} must be a mapping")
    return value


def _require_exact_keys(
    mapping: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    missing = sorted(expected - mapping.keys())
    unknown = sorted(mapping.keys() - expected)
    if missing:
        raise ConfigValidationError(
            f"{context} is missing required keys: {missing}"
        )
    if unknown:
        raise ConfigValidationError(f"{context} has unknown keys: {unknown}")


def _string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _boolean(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{context}.{key} must be a boolean")
    return value


def _integer(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(f"{context}.{key} must be an integer")
    return value


def _positive_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = _integer(mapping, key, context)
    if value <= 0:
        raise ConfigValidationError(f"{context}.{key} must be greater than zero")
    return value


def _optional_nonnegative_int(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> int | None:
    value = mapping[key]
    if value is None:
        return None
    value = _integer(mapping, key, context)
    if value < 0:
        raise ConfigValidationError(f"{context}.{key} cannot be negative")
    return value


def _number(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(f"{context}.{key} must be a number")
    return float(value)


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
