import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    mobilenet_v3_small,
    resnet18,
)


def build_model(model_name, num_classes):
    if model_name == "mobilenetv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenetv3":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Modelo {model_name} no soportado.")

    return model