from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from brain_tumor_classifier.experiments.history import (
    GLOBAL_HISTORY_COLUMNS,
    append_experiment_history_row,
)


class HistoryWriterTest(unittest.TestCase):
    def test_experiment_history_csv_is_created_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "outputs" / "experiment_history.csv"

            append_experiment_history_row(
                history_path,
                {
                    "created_at": "2026-06-30T14:15:22+00:00",
                    "experiment_name": "exp_a",
                    "trace_name": "trace_a",
                    "run_id": "run_a",
                    "status": "success",
                },
            )

            self.assertTrue(history_path.is_file())
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, GLOBAL_HISTORY_COLUMNS)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run_a")

    def test_rows_are_appended_instead_of_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "outputs" / "experiment_history.csv"

            append_experiment_history_row(
                history_path,
                {
                    "created_at": "2026-06-30T14:15:22+00:00",
                    "experiment_name": "exp_a",
                    "trace_name": "trace_a",
                    "run_id": "run_a",
                    "status": "success",
                },
            )
            append_experiment_history_row(
                history_path,
                {
                    "created_at": "2026-06-30T14:16:22+00:00",
                    "experiment_name": "exp_b",
                    "trace_name": "trace_b",
                    "run_id": "run_b",
                    "status": "success",
                },
            )

            with history_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual([row["run_id"] for row in rows], ["run_a", "run_b"])

    def test_failed_rows_include_status_and_error_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "outputs" / "experiment_history.csv"

            append_experiment_history_row(
                history_path,
                {
                    "created_at": "2026-06-30T14:17:22+00:00",
                    "experiment_name": "exp_fail",
                    "trace_name": "trace_fail",
                    "run_id": "run_fail",
                    "status": "failed",
                    "error_message": "training exploded",
                },
            )

            with history_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["error_message"], "training exploded")


if __name__ == "__main__":
    unittest.main()
