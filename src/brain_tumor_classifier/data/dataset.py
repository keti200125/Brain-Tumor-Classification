"""PyTorch dataset primitives for brain MRI images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class ImageSample:
    """Path and label metadata for one image."""

    path: Path
    label: int
    class_name: str


class BrainTumorDataset(Dataset):
    """Load RGB images from a precomputed list of samples."""

    def __init__(self, samples: list[ImageSample], transform: ImageTransform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        with Image.open(sample.path) as source_image:
            image = source_image.convert("RGB")
        return self.transform(image), sample.label

