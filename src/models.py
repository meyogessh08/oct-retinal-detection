"""Model definitions: a custom CNN and four ImageNet-pretrained backbones.

Transfer-learning modes
-----------------------
* "feature_extraction": freeze the backbone, train only the new final layer.
* "fine_tune": unfreeze the last part of the backbone and train it with a low LR.
"""
import torch
import torch.nn as nn
from torchvision import models

from src import config


# ------------------------------------------------------------------ custom CNN
class CustomCNN(nn.Module):
    """Small CNN built from scratch: 4 x (Conv -> BN -> ReLU -> MaxPool) + classifier."""

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = 0.5,
                 use_batchnorm: bool = True):
        super().__init__()

        def block(c_in, c_out):
            layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 32),     # 224 -> 112
            block(32, 64),    # 112 -> 56
            block(64, 128),   # 56  -> 28
            block(128, 256),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # makes the head independent of input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ------------------------------------------------------------------ helpers
def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


# ------------------------------------------------------------------ factory
def build_model(name: str, num_classes: int = config.NUM_CLASSES,
                mode: str = "fine_tune", pretrained: bool = True,
                dropout: float = 0.5) -> nn.Module:
    """Create a model by name.

    name : one of config.MODEL_NAMES
    mode : 'feature_extraction' or 'fine_tune' (ignored for custom_cnn)
    """
    if name == "custom_cnn":
        return CustomCNN(num_classes, dropout=dropout)

    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        head, last_block = model.fc, model.layer4

    elif name == "vgg16":
        model = models.vgg16(weights=weights)
        in_f = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_f, num_classes)
        head, last_block = model.classifier[6], model.features[24:]

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        head, last_block = model.classifier, model.features[14:]

    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes)
        )
        head, last_block = model.classifier, model.features.denseblock4

    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {config.MODEL_NAMES}")

    if mode == "feature_extraction":
        _freeze(model)
        _unfreeze(head)
    elif mode == "fine_tune":
        _freeze(model)
        _unfreeze(last_block)
        _unfreeze(head)
    elif mode != "full":
        raise ValueError("mode must be 'feature_extraction', 'fine_tune' or 'full'")
    return model


def get_gradcam_target_layer(model: nn.Module, name: str):
    """Last convolutional layer for each architecture (used by Grad-CAM)."""
    if name == "resnet18":
        return model.layer4[-1]
    if name == "custom_cnn":
        return model.features[-1][0]  # last Conv2d (before BN/ReLU/Pool)
    if name == "vgg16":
        return model.features[28]
    if name == "mobilenet_v2":
        return model.features[-1]
    if name == "densenet121":
        return model.features.denseblock4
    raise ValueError(name)


def save_checkpoint(model: nn.Module, name: str, path, class_names=None, extra=None) -> None:
    """Save weights + the info the app needs to rebuild the model."""
    torch.save({
        "model_name": name,
        "state_dict": model.state_dict(),
        "class_names": class_names or config.CLASS_NAMES,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a checkpoint saved by save_checkpoint()."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["class_names"]),
                        pretrained=False, mode="full")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["model_name"], ckpt["class_names"]
