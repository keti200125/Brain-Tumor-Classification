from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from brain_tumor_classifier.models import (
    CustomTumorCNN,
    ModelRegistry,
    ResNet50SSPANet,
    SSPANet,
    available_models,
    build_model,
    build_resnet50_sspanet,
    freeze_all_but,
    get_architecture_name,
    unpack_inception_outputs,
)


class DefaultModelRegistryTest(unittest.TestCase):
    def test_all_required_models_are_registered(self) -> None:
        self.assertEqual(
            available_models(),
            (
                "custom_cnn",
                "inception_v3",
                "resnet18",
                "resnet50_sspanet",
                "vgg11",
            ),
        )
        self.assertEqual(get_architecture_name("custom_cnn"), "CustomTumorCNN")
        self.assertEqual(get_architecture_name("resnet18"), "ResNet18")
        self.assertEqual(
            get_architecture_name("resnet50_sspanet"),
            "ResNet50 + SSPANet",
        )

    def test_public_builder_constructs_custom_cnn(self) -> None:
        model, pretrained_used = build_model(
            "custom_cnn",
            num_classes=4,
            pretrained=True,
        )

        self.assertIsInstance(model, CustomTumorCNN)
        self.assertFalse(pretrained_used)
        self.assertEqual(tuple(model(torch.randn(2, 3, 64, 64)).shape), (2, 4))

    def test_public_builder_replaces_resnet_head(self) -> None:
        model, pretrained_used = build_model(
            "resnet18",
            num_classes=4,
            pretrained=False,
        )

        self.assertFalse(pretrained_used)
        self.assertEqual(model.fc.out_features, 4)

    def test_public_builder_constructs_resnet50_sspanet(self) -> None:
        model, pretrained_used = build_model(
            "resnet50_sspanet",
            num_classes=4,
            pretrained=False,
        )

        self.assertIsInstance(model, ResNet50SSPANet)
        self.assertFalse(pretrained_used)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 224, 224)).shape), (2, 4))

    def test_unknown_model_lists_available_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown model 'missing'.*custom_cnn.*resnet18",
        ):
            build_model("missing", num_classes=4, pretrained=False)


class ModelRegistryTest(unittest.TestCase):
    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ModelRegistry()

        def builder(num_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
            del pretrained
            return nn.Linear(1, num_classes), False

        registry.register("tiny", "Tiny", builder)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("tiny", "Tiny", builder)

    def test_freeze_all_but_keeps_only_matching_parameters_trainable(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        freeze_all_but(model, trainable_keys=("1",))

        trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(trainable_names, {"1.weight", "1.bias"})

    def test_unpack_inception_outputs_supports_all_output_shapes(self) -> None:
        logits = torch.randn(2, 4)
        aux_logits = torch.randn(2, 4)

        plain_logits, plain_aux = unpack_inception_outputs(logits)
        tuple_logits, tuple_aux = unpack_inception_outputs((logits, aux_logits))
        object_logits, object_aux = unpack_inception_outputs(
            SimpleNamespace(logits=logits, aux_logits=aux_logits)
        )

        self.assertIs(plain_logits, logits)
        self.assertIsNone(plain_aux)
        self.assertIs(tuple_logits, logits)
        self.assertIs(tuple_aux, aux_logits)
        self.assertIs(object_logits, logits)
        self.assertIs(object_aux, aux_logits)

    def test_sspanet_preserves_feature_map_shape(self) -> None:
        attention = SSPANet(channels=2048)
        features = torch.randn(2, 2048, 7, 7)

        refined = attention(features)

        self.assertEqual(tuple(refined.shape), (2, 2048, 7, 7))

    def test_pretrained_resnet50_sspanet_freezes_backbone_only(self) -> None:
        from unittest.mock import patch
        from torchvision import models as tv_models

        original_resnet50 = tv_models.resnet50
        with patch(
            "brain_tumor_classifier.models.torchvision_models.models.resnet50",
            side_effect=lambda weights: original_resnet50(weights=None),
        ):
            model, pretrained_used = build_resnet50_sspanet(
                num_classes=4,
                pretrained=True,
            )

        self.assertTrue(pretrained_used)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith(("attention.", "fc.")) for name in trainable_names)
        )
        self.assertIn("fc.weight", trainable_names)


if __name__ == "__main__":
    unittest.main()
