"""Model evaluation and single-image prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from brain_tumor_classifier.models import unpack_inception_outputs
from brain_tumor_classifier.training.engine import EpochResult, run_epoch


ImageTransform = Callable[[Image.Image], torch.Tensor]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    use_inception_aux: bool = False,
    show_progress: bool = True,
) -> EpochResult:
    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        model_name=model_name,
        epoch_idx=0,
        epochs=1,
        optimizer=None,
        use_inception_aux=use_inception_aux,
        show_progress=show_progress,
    )


def predict_single_image(
    model: nn.Module,
    image_path: Path,
    transform: ImageTransform,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    model.eval()
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        logits, _ = unpack_inception_outputs(outputs)
        probabilities = torch.softmax(logits, dim=1)
        confidence, label_idx = torch.max(probabilities, dim=1)

    return class_names[int(label_idx.item())], float(confidence.item())

