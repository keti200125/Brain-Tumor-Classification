"""Classification metric calculation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASSIFICATION_METRIC_NAMES = (
    "accuracy",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
)


def compute_classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> dict[str, float]:
    """Compute the standard metrics used by every experiment."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        raise ValueError("cannot compute metrics for an empty target list")

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_weighted": float(
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def compute_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    *,
    labels: list[int] | None = None,
) -> np.ndarray:
    """Return a confusion matrix with an optional explicit label order."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    return confusion_matrix(targets, predictions, labels=labels)

