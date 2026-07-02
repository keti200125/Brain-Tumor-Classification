"""Model definitions and the public model-construction API."""

from brain_tumor_classifier.models.attention import SSPANet
from brain_tumor_classifier.models.custom_cnn import (
    CustomTumorCNN,
    build_custom_cnn,
)
from brain_tumor_classifier.models.registry import (
    ModelRegistry,
    available_models,
    build_model,
    get_architecture_name,
    model_registry,
)
from brain_tumor_classifier.models.torchvision_models import (
    ResNet50SSPANet,
    build_inception_v3,
    build_resnet18,
    build_resnet50_sspanet,
    build_vgg11,
    freeze_all_but,
    unpack_inception_outputs,
)

__all__ = [
    "CustomTumorCNN",
    "ModelRegistry",
    "ResNet50SSPANet",
    "SSPANet",
    "available_models",
    "build_custom_cnn",
    "build_inception_v3",
    "build_model",
    "build_resnet18",
    "build_resnet50_sspanet",
    "build_vgg11",
    "freeze_all_but",
    "get_architecture_name",
    "model_registry",
    "unpack_inception_outputs",
]
