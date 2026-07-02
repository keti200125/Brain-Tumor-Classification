"""Reusable training, evaluation, metrics, and checkpoint APIs."""

from brain_tumor_classifier.training.checkpointing import (
    CheckpointValidationResult,
    build_checkpoint,
    load_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
    validate_reloaded_model,
)
from brain_tumor_classifier.training.engine import (
    EpochRecord,
    EpochResult,
    TrainingResult,
    run_epoch,
    train_model,
)
from brain_tumor_classifier.training.evaluator import (
    evaluate_model,
    predict_single_image,
)
from brain_tumor_classifier.training.metrics import (
    CLASSIFICATION_METRIC_NAMES,
    compute_classification_metrics,
    compute_confusion_matrix,
)

__all__ = [
    "CLASSIFICATION_METRIC_NAMES",
    "CheckpointValidationResult",
    "EpochRecord",
    "EpochResult",
    "TrainingResult",
    "build_checkpoint",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "evaluate_model",
    "load_checkpoint",
    "predict_single_image",
    "run_epoch",
    "save_best_checkpoint",
    "save_last_checkpoint",
    "train_model",
    "validate_reloaded_model",
]

