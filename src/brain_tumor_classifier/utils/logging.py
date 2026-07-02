"""Per-run logging helpers for experiment execution."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_run_logger(
    *,
    experiment_name: str,
    run_id: str,
    log_path: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to both console and the run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(
        f"brain_tumor_classifier.experiment.{experiment_name}.{run_id}"
    )
    logger.setLevel(level)
    logger.propagate = False

    # Replace existing handlers so repeated test runs stay deterministic.
    close_logger(logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and detach any handlers from a logger."""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)
