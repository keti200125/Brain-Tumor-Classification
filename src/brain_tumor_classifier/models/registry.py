"""Explicit registry for every supported model builder."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from brain_tumor_classifier.models.custom_cnn import build_custom_cnn
from brain_tumor_classifier.models.torchvision_models import (
    build_inception_v3,
    build_resnet18,
    build_resnet50_sspanet,
    build_vgg11,
)


ModelBuilder = Callable[[int, bool], tuple[nn.Module, bool]]


class ModelRegistry:
    """Map stable config names to model builders and display names."""

    def __init__(self) -> None:
        self._builders: dict[str, ModelBuilder] = {}
        self._architecture_names: dict[str, str] = {}

    def register(
        self,
        name: str,
        architecture_name: str,
        builder: ModelBuilder,
    ) -> None:
        if name in self._builders:
            raise ValueError(f"Model '{name}' is already registered")
        self._builders[name] = builder
        self._architecture_names[name] = architecture_name

    def build(
        self,
        name: str,
        num_classes: int,
        pretrained: bool,
    ) -> tuple[nn.Module, bool]:
        builder = self._builders.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(self.available_models())}"
            )
        return builder(num_classes, pretrained)

    def architecture_name(self, name: str) -> str:
        return self._architecture_names.get(name, name)

    def available_models(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


model_registry = ModelRegistry()
model_registry.register("custom_cnn", "CustomTumorCNN", build_custom_cnn)
model_registry.register("resnet18", "ResNet18", build_resnet18)
model_registry.register(
    "resnet50_sspanet",
    "ResNet50 + SSPANet",
    build_resnet50_sspanet,
)
model_registry.register("vgg11", "VGG11", build_vgg11)
model_registry.register("inception_v3", "InceptionV3", build_inception_v3)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, bool]:
    """Build a registered model by its stable public name."""
    return model_registry.build(model_name, num_classes, pretrained)


def get_architecture_name(model_name: str) -> str:
    return model_registry.architecture_name(model_name)


def available_models() -> tuple[str, ...]:
    return model_registry.available_models()
