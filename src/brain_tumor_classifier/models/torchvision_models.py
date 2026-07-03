"""Torchvision transfer-learning model builders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from brain_tumor_classifier.models.attention import SSPANet


def freeze_all_but(model: nn.Module, trainable_keys: tuple[str, ...]) -> None:
    """Freeze every parameter except names containing a configured key."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if any(key in name for key in trainable_keys):
            parameter.requires_grad = True


def build_vgg11(num_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
            got_pretrained = True
        except Exception:
            model = models.vgg11(weights=None)
    else:
        model = models.vgg11(weights=None)

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes,
    )
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("classifier.6",))
    return model, got_pretrained


def build_inception_v3(
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            model = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT,
                aux_logits=True,
            )
            got_pretrained = True
        except Exception:
            model = models.inception_v3(weights=None, aux_logits=True)
    else:
        model = models.inception_v3(weights=None, aux_logits=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features,
            num_classes,
        )
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("fc", "AuxLogits.fc"))
    return model, got_pretrained


def build_resnet18(
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            got_pretrained = True
        except Exception:
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("fc",))
    return model, got_pretrained


def build_resnet50(
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, bool]:
    """Build a plain torchvision ResNet50 classification baseline."""
    got_pretrained = False
    if pretrained:
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            got_pretrained = True
        except Exception:
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("fc",))
    return model, got_pretrained


class ResNet50SSPANet(nn.Module):
    """ResNet50 backbone with an SSPANet feature-refinement head."""

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.attention = SSPANet(channels=2048)
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_resnet50_sspanet(
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            got_pretrained = True
        except Exception:
            backbone = models.resnet50(weights=None)
    else:
        backbone = models.resnet50(weights=None)

    model = ResNet50SSPANet(backbone=backbone, num_classes=num_classes)
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("attention", "fc"))
    return model, got_pretrained


def unpack_inception_outputs(
    outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize Inception and ordinary model outputs into logits/aux logits."""
    if hasattr(outputs, "logits"):
        return outputs.logits, getattr(outputs, "aux_logits", None)
    if isinstance(outputs, tuple):
        if not outputs:
            raise ValueError("Inception output tuple is empty")
        logits = outputs[0]
        aux_logits = outputs[1] if len(outputs) > 1 else None
        return logits, aux_logits
    return outputs, None
