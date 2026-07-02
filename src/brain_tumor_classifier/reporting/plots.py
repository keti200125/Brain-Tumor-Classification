"""Exploratory data analysis plots."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from brain_tumor_classifier.data.dataset import ImageSample
from brain_tumor_classifier.data.splits import class_distribution
from brain_tumor_classifier.training import predict_single_image


PLOT_FILENAMES = {
    "eda_class_samples_path": "eda_class_samples.png",
    "eda_class_distribution_path": "eda_class_distribution.png",
    "eda_image_sizes_path": "eda_image_sizes.png",
    "train_vs_val_loss_path": "train_vs_val_loss.png",
    "train_vs_val_f1_path": "train_vs_val_f1.png",
    "prediction_grid_path": "prediction_grid.png",
}


def build_plot_paths(plots_dir: Path) -> dict[str, Path]:
    """Return the standardized artifact paths for all plots in one run."""
    return {
        key: plots_dir / filename
        for key, filename in PLOT_FILENAMES.items()
    }


def save_class_examples_plot(
    samples: list[ImageSample],
    class_names: list[str],
    output_path: Path,
    seed: int,
    per_class: int = 5,
) -> None:
    rng = random.Random(seed)
    class_to_samples: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        class_to_samples[sample.class_name].append(sample)

    fig, axes = plt.subplots(
        len(class_names),
        per_class,
        figsize=(2.4 * per_class, 2.4 * len(class_names)),
    )
    axes = np.asarray(axes, dtype=object).reshape(len(class_names), per_class)

    for row, class_name in enumerate(class_names):
        class_items = class_to_samples[class_name][:]
        rng.shuffle(class_items)
        selected = class_items[:per_class]

        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(selected):
                with Image.open(selected[col].path) as source_image:
                    image = source_image.convert("RGB")
                ax.imshow(image)
            if col == 0:
                ax.set_title(class_name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_distribution_plot(
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    test_samples: list[ImageSample],
    class_names: list[str],
    output_path: Path,
) -> None:
    train_counts = class_distribution(train_samples, class_names)
    val_counts = class_distribution(val_samples, class_names)
    test_counts = class_distribution(test_samples, class_names)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(
        x - width,
        [train_counts[class_name] for class_name in class_names],
        width=width,
        label="train",
    )
    ax.bar(
        x,
        [val_counts[class_name] for class_name in class_names],
        width=width,
        label="validation",
    )
    ax.bar(
        x + width,
        [test_counts[class_name] for class_name in class_names],
        width=width,
        label="test",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_ylabel("Number of images")
    ax.set_title("Class Distribution Across Splits")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_image_size_plot(
    samples: list[ImageSample],
    output_path: Path,
) -> None:
    widths: list[int] = []
    heights: list[int] = []

    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                width, height = image.size
            widths.append(width)
            heights.append(height)
        except (OSError, ValueError):
            continue

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(widths, heights, alpha=0.4, s=8)
    ax.set_xlabel("Image width")
    ax.set_ylabel("Image height")
    ax.set_title("Image Resolution Scatter Plot")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_prediction_grid(
    samples: list[ImageSample],
    loaded_models: dict[str, object],
    class_names: list[str],
    transform: object,
    device: object,
    output_path: Path,
) -> None:
    if not samples:
        return

    fig, axes = plt.subplots(1, len(samples), figsize=(4.2 * len(samples), 5.2))
    if len(samples) == 1:
        axes = [axes]

    for idx, sample in enumerate(samples):
        ax = axes[idx]
        with Image.open(sample.path) as source_image:
            image = source_image.convert("RGB")
        ax.imshow(image)
        ax.axis("off")

        lines = [f"true: {sample.class_name}"]
        for model_name, model in loaded_models.items():
            predicted, confidence = predict_single_image(
                model=model,
                image_path=sample.path,
                transform=transform,
                class_names=class_names,
                device=device,
            )
            lines.append(f"{model_name}: {predicted} ({confidence:.2f})")

        ax.set_title("\n".join(lines), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_training_curves(
    histories: dict[str, dict[str, list[float]]],
    output_loss_path: Path,
    output_metric_path: Path,
) -> None:
    model_names = list(histories.keys())
    if not model_names:
        return

    loss_fig, loss_axes = plt.subplots(
        len(model_names),
        1,
        figsize=(8, 3.2 * len(model_names)),
    )
    metric_fig, metric_axes = plt.subplots(
        len(model_names),
        1,
        figsize=(8, 3.2 * len(model_names)),
    )

    if len(model_names) == 1:
        loss_axes = [loss_axes]
        metric_axes = [metric_axes]

    for idx, model_name in enumerate(model_names):
        history = histories[model_name]
        epochs = range(1, len(history["train_loss"]) + 1)

        loss_ax = loss_axes[idx]
        loss_ax.plot(epochs, history["train_loss"], label="train loss")
        loss_ax.plot(epochs, history["val_loss"], label="validation loss")
        loss_ax.set_title(f"{model_name} - Train vs Validation Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True, alpha=0.25)
        loss_ax.legend()

        metric_ax = metric_axes[idx]
        metric_ax.plot(epochs, history["train_f1"], label="train F1 weighted")
        metric_ax.plot(epochs, history["val_f1"], label="validation F1 weighted")
        metric_ax.set_title(f"{model_name} - Train vs Validation F1 Weighted")
        metric_ax.set_xlabel("Epoch")
        metric_ax.set_ylabel("F1 weighted")
        metric_ax.grid(True, alpha=0.25)
        metric_ax.legend()

    loss_fig.tight_layout()
    loss_fig.savefig(output_loss_path, dpi=180)
    plt.close(loss_fig)

    metric_fig.tight_layout()
    metric_fig.savefig(output_metric_path, dpi=180)
    plt.close(metric_fig)
