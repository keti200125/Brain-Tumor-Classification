"""Generate the Excel model report from experiment history CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brain_tumor_classifier.reporting import generate_excel_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Excel model report from experiment history CSV",
    )
    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Path to outputs/experiment_history.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the .xlsx report",
    )
    return parser


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = generate_excel_report(
        history_path=_resolve_project_path(args.history),
        output_path=_resolve_project_path(args.output),
    )
    print(f"Output path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
