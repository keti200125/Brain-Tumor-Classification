"""Run exactly one config-driven experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brain_tumor_classifier.experiments import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one experiment from a YAML config",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to an experiment YAML file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = ExperimentRunner.from_config_path(
        args.config,
        project_root=PROJECT_ROOT,
    )
    result = runner.run()

    print(f"Run directory: {result.run_dir}")
    print(f"Checkpoint path: {result.checkpoint_path or '(not saved)'}")
    print(
        "Test metrics: "
        f"accuracy={result.test_accuracy:.4f} "
        f"f1_weighted={result.test_f1_weighted:.4f} "
        f"precision_weighted={result.test_precision_weighted:.4f} "
        f"recall_weighted={result.test_recall_weighted:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
