"""Modified ResNet-18 models for 13-channel terrain raster patch classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18BinaryClassifier(nn.Module):
    """ResNet-18 binary classifier with 13-channel small-patch stem."""

    def __init__(
        self,
        in_channels: int = 14,
        dropout: float = 0.4,
        small_patch_stem: bool = True,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        if small_patch_stem:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512, 1))
        self._initialize_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(1)


class ResNet18Encoder(nn.Module):
    """ResNet-18 encoder that returns spatial feature maps before pooling."""

    def __init__(
        self,
        in_channels: int = 14,
        small_patch_stem: bool = True,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        if small_patch_stem:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self._initialize_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def create_resnet18_binary_classifier(
    in_channels: int = 14,
    dropout: float = 0.4,
    small_patch_stem: bool = True,
    pretrained: bool = False,
) -> nn.Module:
    """Create a from-scratch modified ResNet-18 binary classifier."""

    if pretrained:
        raise ValueError("This baseline does not support pretrained weights.")
    return ResNet18BinaryClassifier(
        in_channels=in_channels,
        dropout=dropout,
        small_patch_stem=small_patch_stem,
    )


def create_resnet18_encoder_for_ssl(
    in_channels: int = 14,
    small_patch_stem: bool = True,
    pretrained: bool = False,
    return_feature_map: bool = True,
) -> nn.Module:
    """Create a from-scratch ResNet-18 encoder for SSL reconstruction."""

    if pretrained:
        raise ValueError("SSL masked reconstruction pretraining starts from scratch.")
    if not return_feature_map:
        raise ValueError("Masked reconstruction requires spatial feature maps.")
    return ResNet18Encoder(in_channels=in_channels, small_patch_stem=small_patch_stem)


def create_resnet18_encoder_for_contrastive(
    in_channels: int = 14,
    small_patch_stem: bool = True,
    pretrained: bool = False,
    projection_dim: int = 128,
) -> nn.Module:
    """Create a from-scratch ResNet-18 encoder for contrastive SSL."""

    if pretrained:
        raise ValueError("Contrastive SSL pretraining starts from scratch.")
    _ = projection_dim
    return ResNet18Encoder(in_channels=in_channels, small_patch_stem=small_patch_stem)


def create_resnet18_encoder_for_jigsaw(
    in_channels: int = 14,
    small_patch_stem: bool = True,
    pretrained: bool = False,
) -> nn.Module:
    """Create a from-scratch ResNet-18 encoder for Jigsaw SSL."""

    if pretrained:
        raise ValueError("Jigsaw SSL pretraining starts from scratch.")
    return ResNet18Encoder(in_channels=in_channels, small_patch_stem=small_patch_stem)


def create_resnet18_encoder_for_rotation(
    in_channels: int = 14,
    small_patch_stem: bool = True,
    pretrained: bool = False,
) -> nn.Module:
    """Create a from-scratch ResNet-18 encoder for rotation SSL."""

    if pretrained:
        raise ValueError("Rotation SSL pretraining starts from scratch.")
    return ResNet18Encoder(in_channels=in_channels, small_patch_stem=small_patch_stem)


def create_resnet18_encoder_for_strip_jigsaw(
    in_channels: int = 14,
    small_patch_stem: bool = True,
    pretrained: bool = False,
) -> nn.Module:
    """Create a from-scratch ResNet-18 encoder for 1D strip jigsaw SSL."""

    if pretrained:
        raise ValueError("Strip jigsaw SSL pretraining starts from scratch.")
    return ResNet18Encoder(in_channels=in_channels, small_patch_stem=small_patch_stem)


def load_ssl_encoder_weights_into_classifier(
    classifier_model: nn.Module,
    encoder_checkpoint_path: str | Path,
    strict_encoder: bool = True,
) -> dict[str, object]:
    """Load SSL-pretrained encoder weights into a binary classifier backbone.

    Decoder and classifier-head weights are intentionally ignored. The classifier
    architecture must expose the same backbone keys as ``ResNet18Encoder``:
    conv1, bn1, layer1, layer2, layer3, and layer4.
    """

    checkpoint = torch.load(Path(encoder_checkpoint_path).resolve(), map_location="cpu")
    if "encoder_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {encoder_checkpoint_path} does not contain "
            "'encoder_state_dict'."
        )

    encoder_state = checkpoint["encoder_state_dict"]
    classifier_state = classifier_model.state_dict()
    loadable_state = {}
    skipped_keys = []
    for key, value in encoder_state.items():
        if key in classifier_state and classifier_state[key].shape == value.shape:
            loadable_state[key] = value
        else:
            skipped_keys.append(key)

    load_result = classifier_model.load_state_dict(loadable_state, strict=False)
    loaded_keys = sorted(loadable_state)
    missing_keys = sorted(load_result.missing_keys)
    unexpected_keys = sorted(load_result.unexpected_keys + skipped_keys)

    expected_head_missing = [key for key in missing_keys if key.startswith("fc.")]
    backbone_missing = [key for key in missing_keys if not key.startswith("fc.")]
    if strict_encoder and backbone_missing:
        raise RuntimeError(
            "SSL encoder loading failed: backbone keys are missing from the "
            f"classifier load result: {backbone_missing[:20]}"
        )
    if strict_encoder and len(loaded_keys) < max(1, int(0.9 * len(encoder_state))):
        raise RuntimeError(
            "SSL encoder loading failed: too few encoder keys loaded "
            f"({len(loaded_keys)} of {len(encoder_state)})."
        )

    print(f"Loaded SSL encoder keys: {len(loaded_keys)}")
    print(f"Missing keys after load: {len(missing_keys)}")
    print(f"Unexpected/skipped keys: {len(unexpected_keys)}")
    if expected_head_missing:
        print(f"Expected classifier-head missing keys: {expected_head_missing}")
    if unexpected_keys:
        print(f"Unexpected/skipped key examples: {unexpected_keys[:10]}")

    return {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "checkpoint_config": checkpoint.get("config", {}),
        "checkpoint_channel_means": checkpoint.get("channel_means"),
        "checkpoint_channel_stds": checkpoint.get("channel_stds"),
    }
