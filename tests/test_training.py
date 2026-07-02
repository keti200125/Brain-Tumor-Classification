from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from brain_tumor_classifier.training import (
    CLASSIFICATION_METRIC_NAMES,
    compute_classification_metrics,
    compute_confusion_matrix,
    predict_single_image,
    run_epoch,
    train_model,
)


class AuxiliaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Linear(2, 2)
        self.aux = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            logits=self.main(inputs),
            aux_logits=self.aux(inputs),
        )


def make_loader() -> DataLoader:
    features = torch.tensor(
        [[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)


class MetricsTest(unittest.TestCase):
    def test_standard_metrics_and_confusion_matrix(self) -> None:
        targets = [0, 0, 1, 1]
        predictions = [0, 1, 1, 1]

        metrics = compute_classification_metrics(targets, predictions)
        matrix = compute_confusion_matrix(
            targets,
            predictions,
            labels=[0, 1],
        )

        self.assertEqual(tuple(metrics), CLASSIFICATION_METRIC_NAMES)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])

    def test_empty_or_mismatched_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            compute_classification_metrics([], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_classification_metrics([0], [])


class TrainingEngineTest(unittest.TestCase):
    def test_run_epoch_returns_structured_result(self) -> None:
        torch.manual_seed(7)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        result = run_epoch(
            model=model,
            loader=make_loader(),
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            model_name="tiny",
            epoch_idx=0,
            epochs=1,
            optimizer=optimizer,
            show_progress=False,
        )

        self.assertEqual(result.sample_count, 4)
        self.assertGreater(result.loss, 0)
        self.assertEqual(tuple(result.metrics), CLASSIFICATION_METRIC_NAMES)

    def test_train_model_tracks_best_and_last_states(self) -> None:
        torch.manual_seed(7)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_model(
                model=model,
                train_loader=make_loader(),
                val_loader=make_loader(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                device=torch.device("cpu"),
                model_name="tiny",
                epochs=2,
                show_progress=False,
            )

        self.assertEqual(len(result.epochs), 2)
        self.assertIn(result.best_epoch, {1, 2})
        self.assertEqual(result.monitor_metric, "f1_weighted")
        self.assertEqual(len(result.plot_history()["train_loss"]), 2)
        self.assertEqual(
            set(result.best_state_dict),
            set(result.last_state_dict),
        )

    def test_train_model_reports_each_epoch_to_callback(self) -> None:
        torch.manual_seed(7)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reported_epochs = []

        train_model(
            model=model,
            train_loader=make_loader(),
            val_loader=make_loader(),
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            model_name="tiny",
            epochs=2,
            show_progress=False,
            on_epoch_end=reported_epochs.append,
        )

        self.assertEqual([record.epoch for record in reported_epochs], [1, 2])
        self.assertTrue(all(record.train.loss > 0 for record in reported_epochs))
        self.assertTrue(
            all(record.validation.loss > 0 for record in reported_epochs)
        )

    def test_inception_auxiliary_head_contributes_to_training(self) -> None:
        torch.manual_seed(7)
        model = AuxiliaryModel()
        original_aux_weight = model.aux.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        run_epoch(
            model=model,
            loader=make_loader(),
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            model_name="inception-test",
            epoch_idx=0,
            epochs=1,
            optimizer=optimizer,
            use_inception_aux=True,
            show_progress=False,
        )

        self.assertFalse(torch.equal(original_aux_weight, model.aux.weight))


class PredictionTest(unittest.TestCase):
    def test_predict_single_image_handles_rgb_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            Image.new("RGB", (4, 4), color="white").save(image_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
            with torch.no_grad():
                model[1].weight.zero_()
                model[1].bias.copy_(torch.tensor([0.0, 2.0]))

            predicted, confidence = predict_single_image(
                model=model,
                image_path=image_path,
                transform=transforms.ToTensor(),
                class_names=["negative", "positive"],
                device=torch.device("cpu"),
            )

        self.assertEqual(predicted, "positive")
        self.assertGreater(confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
