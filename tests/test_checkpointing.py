from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brain_tumor_classifier.training import (
    build_checkpoint,
    evaluate_model,
    load_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
    validate_reloaded_model,
)


CHECKPOINT_KEYS = {
    "experiment_name",
    "trace_name",
    "run_id",
    "seed",
    "config_file",
    "config_hash",
    "model_name",
    "state_dict",
    "class_names",
    "class_to_idx",
    "image_size",
    "pretrained_used",
    "best_epoch",
    "best_val_metric",
    "final_test_metrics",
}


class CheckpointingTest(unittest.TestCase):
    def setUp(self) -> None:
        features = torch.tensor(
            [[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        self.loader = DataLoader(
            TensorDataset(features, labels),
            batch_size=2,
            shuffle=False,
        )
        self.model = nn.Linear(2, 2)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
            self.model.bias.zero_()
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cpu")
        self.evaluation = evaluate_model(
            model=self.model,
            loader=self.loader,
            criterion=self.criterion,
            device=self.device,
            model_name="tiny-test",
            show_progress=False,
        )

    def make_checkpoint(self) -> dict:
        return build_checkpoint(
            experiment_name="tiny_experiment",
            trace_name="tiny_experiment_v1",
            run_id="20260630-141522__tiny_experiment_v1__a13f91be",
            seed=42,
            config_file="configs/experiments/tiny.yaml",
            config_hash="a13f91be" + "0" * 56,
            model_name="tiny",
            state_dict=self.model.state_dict(),
            class_names=["negative", "positive"],
            class_to_idx={"negative": 0, "positive": 1},
            image_size=64,
            pretrained_used=False,
            best_epoch=2,
            best_val_metric=1.0,
            final_test_metrics=self.evaluation.metrics,
        )

    def test_checkpoint_format_and_atomic_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            best_path = Path(temp_dir) / "best_model.pt"
            last_path = Path(temp_dir) / "last_model.pt"
            checkpoint = self.make_checkpoint()
            save_best_checkpoint(best_path, checkpoint)
            save_last_checkpoint(last_path, checkpoint)

            restored_model = nn.Linear(2, 2)
            loaded = load_checkpoint(
                best_path,
                restored_model,
                map_location=self.device,
            )

            self.assertEqual(set(loaded), CHECKPOINT_KEYS)
            self.assertEqual(loaded["seed"], 42)
            self.assertTrue(best_path.is_file())
            self.assertTrue(last_path.is_file())
            self.assertFalse(best_path.with_suffix(".pt.tmp").exists())
            for expected, restored in zip(
                self.model.parameters(),
                restored_model.parameters(),
            ):
                self.assertTrue(torch.equal(expected, restored))

    def test_reloaded_checkpoint_metrics_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best_model.pt"
            save_best_checkpoint(checkpoint_path, self.make_checkpoint())
            restored_model = nn.Linear(2, 2)

            validation = validate_reloaded_model(
                model=restored_model,
                checkpoint_path=checkpoint_path,
                loader=self.loader,
                criterion=self.criterion,
                device=self.device,
                model_name="tiny-reloaded-test",
                expected_metrics=self.evaluation.metrics,
                show_progress=False,
            )

            self.assertTrue(validation.metrics_match)
            self.assertEqual(validation.max_metric_delta, 0.0)
            self.assertEqual(
                validation.evaluation.metrics,
                self.evaluation.metrics,
            )


if __name__ == "__main__":
    unittest.main()
