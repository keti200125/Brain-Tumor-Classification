"""Data discovery and PyTorch DataLoader construction."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from brain_tumor_classifier.data.dataset import BrainTumorDataset, ImageSample
from brain_tumor_classifier.data.splits import (
    SplitBalanceReport,
    collect_samples_from_split,
    format_split_balance_report,
    maybe_limit_samples,
    split_testing_into_validation_and_test,
    summarize_split_balance,
)
from brain_tumor_classifier.data.transforms import (
    get_eval_transform,
    get_train_transform,
)


class BrainTumorDataModule:
    """Own dataset setup and expose loaders without producing artifacts."""

    def __init__(
        self,
        data_root: Path,
        *,
        batch_size: int,
        image_size: int,
        validation_fraction: float,
        num_workers: int,
        seed: int,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_fraction = validation_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        self.class_names: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.train_samples: list[ImageSample] = []
        self.val_samples: list[ImageSample] = []
        self.test_samples: list[ImageSample] = []

        self.train_transform = get_train_transform(self.image_size)
        self.eval_transform = get_eval_transform(self.image_size)
        self._is_setup = False

    def setup(self) -> None:
        """Discover classes and prepare deterministic train/val/test samples."""
        resolved_root = self._resolve_dataset_root()
        training_dir = resolved_root / "Training"
        testing_dir = resolved_root / "Testing"
        if not training_dir.is_dir() or not testing_dir.is_dir():
            raise FileNotFoundError(
                "Expected Training and Testing directories under "
                f"{self.data_root}"
            )

        self.class_names = sorted(
            path.name for path in training_dir.iterdir() if path.is_dir()
        )
        if not self.class_names:
            raise ValueError(f"No class directories found under {training_dir}")
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        train_samples = collect_samples_from_split(training_dir, self.class_to_idx)
        val_samples, test_samples = split_testing_into_validation_and_test(
            testing_dir=testing_dir,
            class_to_idx=self.class_to_idx,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )

        self.train_samples = maybe_limit_samples(
            train_samples,
            self.max_train_samples,
            self.seed,
        )
        self.val_samples = maybe_limit_samples(
            val_samples,
            self.max_val_samples,
            self.seed + 1,
        )
        self.test_samples = maybe_limit_samples(
            test_samples,
            self.max_test_samples,
            self.seed + 2,
        )
        self._is_setup = True

    def _resolve_dataset_root(self) -> Path:
        candidates = [self.data_root]
        if self.data_root.parent != self.data_root:
            candidates.append(self.data_root.parent)

        for candidate in candidates:
            if (candidate / "Training").is_dir() and (candidate / "Testing").is_dir():
                return candidate

        return self.data_root

    def train_dataloader(self) -> DataLoader:
        self._require_setup()
        dataset = BrainTumorDataset(self.train_samples, self.train_transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        self._require_setup()
        dataset = BrainTumorDataset(self.val_samples, self.eval_transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        self._require_setup()
        dataset = BrainTumorDataset(self.test_samples, self.eval_transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError("Call setup() before requesting a DataLoader")

    def split_balance_report(self) -> SplitBalanceReport:
        self._require_setup()
        return summarize_split_balance(
            train_samples=self.train_samples,
            validation_samples=self.val_samples,
            test_samples=self.test_samples,
            class_names=self.class_names,
        )

    def formatted_split_balance_report(self) -> str:
        return format_split_balance_report(self.split_balance_report())
