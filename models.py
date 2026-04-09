"""
EfficientNet-B0 and ResNet-50 with 4-class head: GAP -> Dropout(0.5) -> Linear(4).
Pretrained on ImageNet; grayscale MRIs are replicated to 3ch in the data pipeline.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

NUM_CLASSES = 4
DROPOUT_P = 0.5


def _replace_classifier(module: nn.Module, in_features: int, num_classes: int = NUM_CLASSES) -> None:
    module.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )


def efficientnet_b0(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """EfficientNet-B0 with 4-class head. Model already has avgpool+flatten before classifier."""
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    # torchvision EfficientNet: forward does features -> avgpool -> flatten -> classifier; classifier is (Dropout, Linear).
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    return m


def resnet50(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with 4-class head (GAP -> Dropout -> Linear)."""
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    in_features = m.fc.in_features
    m.avgpool = nn.AdaptiveAvgPool2d(1)
    m.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    return m


def build_model(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    name = name.lower().strip()
    if name == "efficientnet_b0":
        return efficientnet_b0(pretrained=pretrained, num_classes=num_classes)
    if name == "resnet50":
        return resnet50(pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}. Use efficientnet_b0 or resnet50.")
