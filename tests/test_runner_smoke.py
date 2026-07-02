from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from brain_tumor_classifier.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from brain_tumor_classifier.data import ImageSample
from brain_tumor_classifier.legacy import ExperimentRunner


class FakeDataModule:
    def __init__(self) -> None:
        self.class_names = ["negative", "positive"]
        self.class_to_idx = {"negative": 0, "positive": 1}
        self.image_size = 32
        self.batch_size = 2
        self.eval_transform = None
        self.train_samples = [
            ImageSample(Path("unused-0.png"), 0, "negative"),
            ImageSample(Path("unused-1.png"), 1, "positive"),
        ]
        self.val_samples = list(self.train_samples)
        self.test_samples = list(self.train_samples)
        torch.manual_seed(11)
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        self.loader = DataLoader(
            TensorDataset(images, labels),
            batch_size=2,
            shuffle=False,
        )

    def setup(self) -> None:
        return None

    def train_dataloader(self) -> DataLoader:
        return self.loader

    def val_dataloader(self) -> DataLoader:
        return self.loader

    def test_dataloader(self) -> DataLoader:
        return self.loader


class RunnerSmokeTest(unittest.TestCase):
    def test_config_run_writes_and_validates_best_and_last_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_config = root / "experiment.yaml"
            source_config.write_text("experiment_name: smoke\n", encoding="utf-8")
            config = ExperimentConfig(
                experiment_name="smoke",
                trace_name="smoke_v1",
                report_name="Smoke report",
                hypothesis="Smoke hypothesis",
                description="Runner smoke test",
                tags=["test"],
                seed=42,
                device="cpu",
                data=DataConfig(root, 0.5, None, None, None),
                model=ModelConfig("custom_cnn", False),
                training=TrainingConfig(
                    1,
                    2,
                    0.001,
                    "AdamW",
                    0,
                    32,
                    "f1_weighted",
                    "max",
                ),
                outputs=OutputConfig(
                    root,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ),
                config_path=source_config,
                config_filename=source_config.name,
                config_hash="a13f91be" + "0" * 56,
            )
            run_dir = root / "run"
            checkpoints_dir = root / "checkpoints"
            run_dir.mkdir()
            checkpoints_dir.mkdir()
            args = argparse.Namespace(
                epochs=1,
                learning_rate=0.001,
                no_pretrained=True,
                seed=42,
            )
            runner = ExperimentRunner(
                data_module=FakeDataModule(),
                args=args,
                device=torch.device("cpu"),
                output_root=run_dir,
                plots_dir=run_dir,
                checkpoints_dir=checkpoints_dir,
                model_names=["custom_cnn"],
                save_eda_plots=False,
                save_training_curves=False,
                save_prediction_grid=False,
                experiment_config=config,
                run_id="20260630-141522__smoke_v1__a13f91be",
                show_progress=False,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                runner.run()

            self.assertTrue((checkpoints_dir / "best_model.pt").is_file())
            self.assertTrue((checkpoints_dir / "last_model.pt").is_file())
            self.assertTrue((run_dir / "data-science-2-model-report.md").is_file())


if __name__ == "__main__":
    unittest.main()
