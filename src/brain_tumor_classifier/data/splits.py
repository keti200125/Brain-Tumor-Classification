"""Sample discovery, limiting, and validation/test splitting."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from brain_tumor_classifier.data.dataset import ImageSample


EXPECTED_TRAIN_PERCENT = 81.0
EXPECTED_VALIDATION_PERCENT = 9.5
EXPECTED_TEST_PERCENT = 9.5
SPLIT_IMBALANCE_TOLERANCE_PERCENTAGE_POINTS = 5.0


@dataclass(frozen=True)
class ClassSplitSummary:
    class_name: str
    train_count: int
    validation_count: int
    test_count: int
    total_count: int
    train_percent: float
    validation_percent: float
    test_percent: float


@dataclass(frozen=True)
class SplitBalanceReport:
    class_summaries: list[ClassSplitSummary]
    warnings: list[str]

    @property
    def status(self) -> str:
        return "OK" if not self.warnings else "WARNING"


def collect_samples_from_split(
    split_dir: Path,
    class_to_idx: dict[str, int],
) -> list[ImageSample]:
    """Collect files under one directory per configured class."""
    samples: list[ImageSample] = []
    for class_name, label in class_to_idx.items():
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file():
                samples.append(
                    ImageSample(path=image_path, label=label, class_name=class_name)
                )
    return samples


def split_testing_into_validation_and_test(
    testing_dir: Path,
    class_to_idx: dict[str, int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    """Create deterministic, class-stratified validation and test samples."""
    rng = random.Random(seed)
    validation_samples: list[ImageSample] = []
    test_samples: list[ImageSample] = []

    for class_name, label in class_to_idx.items():
        image_paths = [
            path
            for path in sorted((testing_dir / class_name).glob("*"))
            if path.is_file()
        ]
        rng.shuffle(image_paths)

        split_index = int(len(image_paths) * validation_fraction)
        split_index = max(1, min(split_index, len(image_paths) - 1))

        validation_samples.extend(
            ImageSample(path=path, label=label, class_name=class_name)
            for path in image_paths[:split_index]
        )
        test_samples.extend(
            ImageSample(path=path, label=label, class_name=class_name)
            for path in image_paths[split_index:]
        )

    rng.shuffle(validation_samples)
    rng.shuffle(test_samples)
    return validation_samples, test_samples


def maybe_limit_samples(
    samples: list[ImageSample],
    limit: int | None,
    seed: int,
) -> list[ImageSample]:
    """Return a deterministic random subset when a limit is configured."""
    if limit is None or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    sampled = samples[:]
    rng.shuffle(sampled)
    return sampled[:limit]


def class_distribution(
    samples: list[ImageSample],
    class_names: list[str],
) -> dict[str, int]:
    """Count samples by class while preserving the configured class order."""
    counts = Counter(sample.class_name for sample in samples)
    return {name: counts.get(name, 0) for name in class_names}


def summarize_split_balance(
    train_samples: list[ImageSample],
    validation_samples: list[ImageSample],
    test_samples: list[ImageSample],
    class_names: list[str],
) -> SplitBalanceReport:
    """Summarize per-class counts and percentages across train/val/test."""
    train_counts = class_distribution(train_samples, class_names)
    validation_counts = class_distribution(validation_samples, class_names)
    test_counts = class_distribution(test_samples, class_names)

    summaries: list[ClassSplitSummary] = []
    warnings: list[str] = []
    expected_percentages = {
        "train": EXPECTED_TRAIN_PERCENT,
        "validation": EXPECTED_VALIDATION_PERCENT,
        "test": EXPECTED_TEST_PERCENT,
    }

    for class_name in class_names:
        train_count = train_counts[class_name]
        validation_count = validation_counts[class_name]
        test_count = test_counts[class_name]
        total_count = train_count + validation_count + test_count

        if total_count == 0:
            train_percent = 0.0
            validation_percent = 0.0
            test_percent = 0.0
        else:
            train_percent = (train_count / total_count) * 100.0
            validation_percent = (validation_count / total_count) * 100.0
            test_percent = (test_count / total_count) * 100.0

        summaries.append(
            ClassSplitSummary(
                class_name=class_name,
                train_count=train_count,
                validation_count=validation_count,
                test_count=test_count,
                total_count=total_count,
                train_percent=train_percent,
                validation_percent=validation_percent,
                test_percent=test_percent,
            )
        )

        observed_percentages = {
            "train": train_percent,
            "validation": validation_percent,
            "test": test_percent,
        }
        for split_name, observed in observed_percentages.items():
            expected = expected_percentages[split_name]
            if abs(observed - expected) > SPLIT_IMBALANCE_TOLERANCE_PERCENTAGE_POINTS:
                warnings.append(
                    "class "
                    f"`{class_name}` has {observed:.1f}% in {split_name}, "
                    f"expected around {expected:.1f}%."
                )

    return SplitBalanceReport(class_summaries=summaries, warnings=warnings)


def format_split_balance_report(report: SplitBalanceReport) -> str:
    """Render a human-readable split-balance report for console and logs."""
    lines = [
        "Class distribution by split",
        "",
        f"{'Class':<16}{'Train':>8}{'Validation':>14}{'Test':>8}",
    ]
    for summary in report.class_summaries:
        lines.append(
            f"{summary.class_name:<16}"
            f"{summary.train_count:>8}"
            f"{summary.validation_count:>14}"
            f"{summary.test_count:>8}"
        )

    lines.extend(
        [
            "",
            f"{'Class':<16}{'Train %':>10}{'Validation %':>16}{'Test %':>10}",
        ]
    )
    for summary in report.class_summaries:
        lines.append(
            f"{summary.class_name:<16}"
            f"{summary.train_percent:>9.1f}%"
            f"{summary.validation_percent:>15.1f}%"
            f"{summary.test_percent:>9.1f}%"
        )

    lines.append("")
    lines.append(f"Split check result: {report.status}")
    if report.warnings:
        lines.append(f"Reason: {report.warnings[0]}")

    return "\n".join(lines)
