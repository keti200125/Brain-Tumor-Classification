"""Run every experiment config in a directory and regenerate the report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brain_tumor_classifier.config import load_experiment_config
from brain_tumor_classifier.experiments import ExperimentRunner
from brain_tumor_classifier.reporting import generate_excel_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all experiment configs in a directory",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Directory containing experiment YAML files",
    )
    return parser


def discover_config_paths(config_dir: str | Path) -> list[Path]:
    config_dir = Path(config_dir)
    return sorted(config_dir.glob("*.yaml"))


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir = _resolve_project_path(args.config_dir)
    config_paths = discover_config_paths(config_dir)
    if not config_paths:
        raise FileNotFoundError(f"No .yaml configs found in {config_dir}")

    success_count = 0
    failure_count = 0
    report_targets: set[tuple[Path, Path]] = set()

    for config_path in config_paths:
        display_name = (
            str(config_path.relative_to(PROJECT_ROOT))
            if config_path.is_relative_to(PROJECT_ROOT)
            else str(config_path)
        )
        print(f"Running config: {display_name}")
        try:
            config = load_experiment_config(
                config_path,
                project_root=PROJECT_ROOT,
            )
            report_targets.add(
                (
                    config.outputs.root / "experiment_history.csv",
                    config.outputs.root / "reports" / "model_report.xlsx",
                )
            )
            result = ExperimentRunner(
                config=config,
                config_path=config.config_path,
            ).run()
            success_count += 1
            print(f"Run directory: {result.run_dir}")
            print(f"Checkpoint path: {result.checkpoint_path or '(not saved)'}")
            print(
                "Test metrics: "
                f"accuracy={result.test_accuracy:.4f} "
                f"f1_weighted={result.test_f1_weighted:.4f}"
            )
        except Exception as exc:
            failure_count += 1
            print(f"Failed config: {display_name}")
            print(f"Error: {exc}")

    for history_path, report_path in sorted(
        report_targets,
        key=lambda item: (str(item[0]), str(item[1])),
    ):
        if history_path.is_file():
            output_path = generate_excel_report(history_path, report_path)
            print(f"Output path: {output_path}")

    print(
        f"Completed batch run: successes={success_count} failures={failure_count}"
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
