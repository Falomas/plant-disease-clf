"""MobileNetV2 и EfficientNet-B0. Голова всегда обучается."""

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)


def build_model(architecture, num_classes, unfreeze_last_blocks):
    if architecture == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    elif architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        raise ValueError(f"неизвестная архитектура: {architecture}")

    # У обеих сетей голова — Dropout + Linear.
    linear = model.classifier[1]
    model.classifier[1] = nn.Linear(linear.in_features, num_classes)
    _freeze_features(model, unfreeze_last_blocks)
    return model


def _freeze_features(model, unfreeze_last_blocks):
    for param in model.features.parameters():
        param.requires_grad = False
    if unfreeze_last_blocks:
        layers = list(model.features.children())
        for layer in layers[-unfreeze_last_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def trainable_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
