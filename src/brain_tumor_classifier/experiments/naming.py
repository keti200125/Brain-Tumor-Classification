"""Filesystem-safe run IDs and standardized artifact paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain_tumor_classifier.config.schema import ExperimentConfig


UNSAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
REPEATED_SEPARATOR_PATTERN = re.compile(r"[_-]{2,}")
MAX_NAME_LENGTH = 96
MAX_COLLISION_ATTEMPTS = 100


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    plots_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    config_snapshot_path: Path
    metrics_json_path: Path
    epoch_history_path: Path
    experiment_history_path: Path
    report_path: Path


def make_filesystem_safe(value: str, *, fallback: str = "run") -> str:
    """Normalize user-controlled names into one safe path component."""
    normalized = UNSAFE_NAME_PATTERN.sub("_", value.strip())
    normalized = REPEATED_SEPARATOR_PATTERN.sub("_", normalized)
    normalized = normalized.strip("._-")[:MAX_NAME_LENGTH].rstrip("._-")
    if not normalized or normalized in {".", ".."}:
        return fallback
    return normalized


def generate_run_id(
    trace_name: str,
    config_hash: str,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Generate ``timestamp__trace__hash8`` using local wall-clock time."""
    safe_trace_name = make_filesystem_safe(trace_name, fallback="experiment")
    short_hash = config_hash.strip().lower()[:8]
    if len(short_hash) != 8 or not re.fullmatch(r"[0-9a-f]{8}", short_hash):
        raise ValueError("config_hash must begin with at least 8 hexadecimal characters")
    run_timestamp = timestamp or datetime.now()
    return (
        f"{run_timestamp.strftime('%Y%m%d-%H%M%S')}"
        f"__{safe_trace_name}__{short_hash}"
    )


def create_run_paths(
    config: ExperimentConfig,
    *,
    timestamp: datetime | None = None,
) -> RunPaths:
    """Create all directories for one config-driven experiment run."""
    output_root = config.outputs.root
    experiment_name = make_filesystem_safe(
        config.experiment_name,
        fallback="experiment",
    )
    base_run_id = generate_run_id(
        config.trace_name,
        config.config_hash,
        timestamp=timestamp,
    )

    experiments_dir = output_root / "experiments" / experiment_name
    checkpoints_root = output_root / "checkpoints" / experiment_name
    reports_dir = output_root / "reports"

    run_dir, checkpoints_dir = _create_unique_run_directories(
        experiments_dir,
        checkpoints_root,
        base_run_id,
    )
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"
    plots_dir.mkdir()
    logs_dir.mkdir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        config_snapshot_path=run_dir / "config_snapshot.yaml",
        metrics_json_path=run_dir / "metrics.json",
        epoch_history_path=run_dir / "history.csv",
        experiment_history_path=output_root / "experiment_history.csv",
        report_path=reports_dir / "model_report.xlsx",
    )
    _write_config_snapshot(config, paths.config_snapshot_path)
    return paths


def _create_unique_run_directories(
    experiments_dir: Path,
    checkpoints_root: Path,
    base_run_id: str,
) -> tuple[Path, Path]:
    for attempt in range(1, MAX_COLLISION_ATTEMPTS + 1):
        suffix = "" if attempt == 1 else f"__{attempt:02d}"
        run_id = f"{base_run_id}{suffix}"
        run_dir = experiments_dir / run_id
        checkpoints_dir = checkpoints_root / run_id
        if run_dir.exists() or checkpoints_dir.exists():
            continue

        run_dir.mkdir(parents=True)
        try:
            checkpoints_dir.mkdir(parents=True)
        except Exception:
            # Leave the run directory visible for diagnosis instead of hiding a
            # partially created run through destructive cleanup.
            raise
        return run_dir, checkpoints_dir

    raise FileExistsError(
        f"Could not allocate a unique run directory for {base_run_id!r}"
    )


def _write_config_snapshot(config: ExperimentConfig, destination: Path) -> None:
    source_text = config.config_path.read_text(encoding="utf-8")
    metadata = (
        f"# source_config: {config.config_filename}\n"
        f"# config_hash: {config.config_hash}\n"
    )
    destination.write_text(metadata + source_text, encoding="utf-8")

