from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from brain_tumor_classifier.data import (
    BrainTumorDataModule,
    ImageSample,
    class_distribution,
    format_split_balance_report,
    summarize_split_balance,
)


class BrainTumorDataModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        for split, count in (("Training", 3), ("Testing", 4)):
            for class_name in ("alpha", "beta"):
                class_dir = self.data_root / split / class_name
                class_dir.mkdir(parents=True)
                for index in range(count):
                    Image.new("RGB", (12, 8), color=(index, 0, 0)).save(
                        class_dir / f"{index}.png"
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_data_module(self) -> BrainTumorDataModule:
        return BrainTumorDataModule(
            self.data_root,
            batch_size=2,
            image_size=64,
            validation_fraction=0.5,
            num_workers=0,
            seed=42,
        )

    def test_setup_discovers_classes_and_builds_stratified_splits(self) -> None:
        data_module = self.make_data_module()
        data_module.setup()

        self.assertEqual(data_module.class_names, ["alpha", "beta"])
        self.assertEqual(data_module.class_to_idx, {"alpha": 0, "beta": 1})
        self.assertEqual(len(data_module.train_samples), 6)
        self.assertEqual(len(data_module.val_samples), 4)
        self.assertEqual(len(data_module.test_samples), 4)
        self.assertEqual(data_module.image_size, 64)
        self.assertEqual(
            class_distribution(data_module.val_samples, data_module.class_names),
            {"alpha": 2, "beta": 2},
        )

    def test_loaders_require_setup_and_return_expected_shape(self) -> None:
        data_module = self.make_data_module()
        with self.assertRaisesRegex(RuntimeError, "setup"):
            data_module.train_dataloader()

        data_module.setup()
        images, labels = next(iter(data_module.train_dataloader()))

        self.assertEqual(
            tuple(images.shape),
            (2, 3, 64, 64),
        )
        self.assertEqual(tuple(labels.shape), (2,))

    def test_setup_is_deterministic_for_the_same_seed(self) -> None:
        first = self.make_data_module()
        second = self.make_data_module()
        first.setup()
        second.setup()

        self.assertEqual(
            [sample.path for sample in first.val_samples],
            [sample.path for sample in second.val_samples],
        )
        self.assertEqual(
            [sample.path for sample in first.test_samples],
            [sample.path for sample in second.test_samples],
        )

    def test_split_balance_report_is_ok_when_each_class_matches_expected_ratio(self) -> None:
        class_names = ["alpha", "beta"]
        train_samples = self._make_samples("train", {"alpha": 81, "beta": 80})
        validation_samples = self._make_samples("val", {"alpha": 10, "beta": 10})
        test_samples = self._make_samples("test", {"alpha": 9, "beta": 10})

        report = summarize_split_balance(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            class_names=class_names,
        )
        rendered = format_split_balance_report(report)

        self.assertEqual(report.status, "OK")
        self.assertEqual(report.warnings, [])
        self.assertIn("Split check result: OK", rendered)
        self.assertIn("alpha", rendered)
        self.assertIn("beta", rendered)

    def test_split_balance_report_warns_when_class_is_badly_imbalanced(self) -> None:
        class_names = ["alpha"]
        train_samples = self._make_samples("train", {"alpha": 70})
        validation_samples = self._make_samples("val", {"alpha": 20})
        test_samples = self._make_samples("test", {"alpha": 10})

        report = summarize_split_balance(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            class_names=class_names,
        )
        rendered = format_split_balance_report(report)

        self.assertEqual(report.status, "WARNING")
        self.assertTrue(report.warnings)
        self.assertIn("Split check result: WARNING", rendered)
        self.assertIn("expected around", rendered)

    def _make_samples(
        self,
        split_name: str,
        counts_by_class: dict[str, int],
    ) -> list[ImageSample]:
        samples: list[ImageSample] = []
        for class_name, count in counts_by_class.items():
            for index in range(count):
                samples.append(
                    ImageSample(
                        path=Path(f"{split_name}-{class_name}-{index}.png"),
                        label=0,
                        class_name=class_name,
                    )
                )
        return samples


if __name__ == "__main__":
    unittest.main()
