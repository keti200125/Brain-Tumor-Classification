from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain_tumor_classifier.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.experiments import ExperimentResult
from scripts import generate_report, run_all_configs, run_experiment


class ScriptEntryPointTest(unittest.TestCase):
    def test_run_experiment_main_prints_required_fields(self) -> None:
        result = self._make_result()
        fake_runner = Mock()
        fake_runner.run.return_value = result

        with patch.object(
            run_experiment.ExperimentRunner,
            "from_config_path",
            return_value=fake_runner,
        ) as from_config_path:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = run_experiment.main(
                    ["--config", "configs/experiments/resnet18.yaml"]
                )

        self.assertEqual(exit_code, 0)
        from_config_path.assert_called_once_with(
            "configs/experiments/resnet18.yaml",
            project_root=run_experiment.PROJECT_ROOT,
        )
        output = stream.getvalue()
        self.assertIn("Run directory: /tmp/run-dir", output)
        self.assertIn("Checkpoint path: /tmp/best_model.pt", output)
        self.assertIn("accuracy=0.9100", output)
        self.assertIn("f1_weighted=0.9000", output)
        self.assertIn("precision_weighted=0.9200", output)
        self.assertIn("recall_weighted=0.8900", output)

    def test_generate_report_main_prints_output_path(self) -> None:
        with patch.object(
            generate_report,
            "generate_excel_report",
            return_value=Path("/tmp/model_report.xlsx"),
        ) as generate_excel_report_mock:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = generate_report.main(
                    [
                        "--history",
                        "outputs/experiment_history.csv",
                        "--output",
                        "outputs/reports/model_report.xlsx",
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = generate_excel_report_mock.call_args
        self.assertTrue(
            str(kwargs["history_path"]).endswith("outputs/experiment_history.csv")
        )
        self.assertTrue(
            str(kwargs["output_path"]).endswith("outputs/reports/model_report.xlsx")
        )
        self.assertIn("Output path: /tmp/model_report.xlsx", stream.getvalue())

    def test_discover_config_paths_finds_sorted_yaml_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "b.yaml").write_text("", encoding="utf-8")
            (config_dir / "a.yaml").write_text("", encoding="utf-8")
            (config_dir / "ignore.txt").write_text("", encoding="utf-8")

            config_paths = run_all_configs.discover_config_paths(config_dir)

        self.assertEqual([path.name for path in config_paths], ["a.yaml", "b.yaml"])

    def test_run_all_configs_continues_after_failures_and_regenerates_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            first_config = config_dir / "a.yaml"
            second_config = config_dir / "b.yaml"
            first_config.write_text("", encoding="utf-8")
            second_config.write_text("", encoding="utf-8")

            output_root = root / "outputs"
            history_path = output_root / "experiment_history.csv"
            report_path = output_root / "reports" / "model_report.xlsx"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text("created_at\n", encoding="utf-8")

            config_one = self._make_config(first_config, output_root)
            config_two = self._make_config(second_config, output_root)

            runner_one = Mock()
            runner_one.run.return_value = self._make_result()
            runner_two = Mock()
            runner_two.run.side_effect = RuntimeError("boom")

            with patch.object(
                run_all_configs,
                "load_experiment_config",
                side_effect=[config_one, config_two],
            ) as load_config_mock, patch.object(
                run_all_configs,
                "ExperimentRunner",
                side_effect=[runner_one, runner_two],
            ) as experiment_runner_mock, patch.object(
                run_all_configs,
                "generate_excel_report",
                return_value=report_path,
            ) as generate_report_mock:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    exit_code = run_all_configs.main(
                        ["--config-dir", str(config_dir)]
                    )

        self.assertEqual(exit_code, 1)
        self.assertEqual(load_config_mock.call_count, 2)
        self.assertEqual(experiment_runner_mock.call_count, 2)
        generate_report_mock.assert_called_once_with(history_path, report_path)
        output = stream.getvalue()
        self.assertIn("Running config:", output)
        self.assertIn("Failed config:", output)
        self.assertIn("Output path:", output)
        self.assertIn("successes=1 failures=1", output)

    def _make_result(self) -> ExperimentResult:
        return ExperimentResult(
            experiment_name="exp",
            trace_name="trace",
            run_id="run-1",
            config_file="/tmp/config.yaml",
            config_hash="abc12345",
            model_name="resnet18",
            architecture="ResNet18",
            pretrained_requested=True,
            pretrained_used=True,
            epochs=3,
            batch_size=8,
            learning_rate=1e-3,
            optimizer="AdamW",
            image_size=224,
            train_size=10,
            val_size=4,
            test_size=4,
            best_epoch=3,
            best_val_loss=0.2,
            best_val_accuracy=0.9,
            best_val_f1_weighted=0.89,
            test_loss=0.19,
            test_accuracy=0.91,
            test_f1_weighted=0.9,
            test_precision_weighted=0.92,
            test_recall_weighted=0.89,
            reloaded_test_accuracy=0.91,
            reloaded_test_f1_weighted=0.9,
            checkpoint_path="/tmp/best_model.pt",
            run_dir="/tmp/run-dir",
            created_at="2026-06-30T14:15:22+00:00",
        )

    def _make_config(self, config_path: Path, output_root: Path) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_name="exp",
            trace_name=config_path.stem,
            report_name="Report",
            hypothesis="Hypothesis",
            description="Description",
            tags=["test"],
            seed=42,
            device="cpu",
            data=DataConfig(output_root / "DATA", 0.5, None, None, None),
            model=ModelConfig("resnet18", True),
            training=TrainingConfig(
                epochs=3,
                batch_size=8,
                learning_rate=1e-3,
                optimizer="AdamW",
                num_workers=0,
                image_size=224,
                monitor_metric="f1_weighted",
                monitor_mode="max",
            ),
            outputs=OutputConfig(
                output_root,
                True,
                True,
                True,
                True,
                True,
                True,
            ),
            config_path=config_path,
            config_filename=config_path.name,
            config_hash="abc12345",
        )


if __name__ == "__main__":
    unittest.main()
