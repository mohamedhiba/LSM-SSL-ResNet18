"""Rotation prediction SSL utilities for 13-channel terrain raster patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn

from src.models_resnet18 import create_resnet18_encoder_for_rotation
from src.patch_dataset import (
    DEFAULT_NODATA_VALUE,
    apply_norm_and_append_mask,
    audit_raster_alignment,
    center_is_valid,
    list_raster_files,
    read_boundless_patch_from_sources,
    valid_context_mask,
)
from src.ssl_masked_recon import create_unlabeled_patch_index


ROTATION_CLASS_LABELS = ["0°", "90°", "180°", "270°"]


def apply_rotation(X: torch.Tensor, rotation_class: int) -> torch.Tensor:
    """Rotate a 14xHxW tensor by 0/90/180/270 degrees over spatial axes."""

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (channels, height, width), got {X.shape}.")
    if rotation_class not in (0, 1, 2, 3):
        raise ValueError(f"rotation_class must be 0, 1, 2, or 3, got {rotation_class}.")
    return torch.rot90(X, k=int(rotation_class), dims=(1, 2)).contiguous()


class RotationRasterPatchDataset(torch.utils.data.Dataset):
    """Lazy unlabeled raster patch dataset for rotation prediction SSL."""

    def __init__(
        self,
        patch_index_csv: str | Path,
        raster_dir: str | Path,
        patch_size: int = 32,
        nodata_value: float = DEFAULT_NODATA_VALUE,
        normalize: bool = True,
        channel_means=None,
        channel_stds=None,
        return_metadata: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.patch_index_csv = Path(patch_index_csv).resolve()
        self.raster_files = list_raster_files(raster_dir)
        audit_raster_alignment(self.raster_files, expected_nodata=nodata_value)
        self.patch_size = int(patch_size)
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
        self.random_seed = int(random_seed)
        self.index = pd.read_csv(self.patch_index_csv)
        self.index = self.index.loc[self.index["valid_patch"].astype(bool)].reset_index(drop=True)
        self._sources = None

        if self.normalize:
            if channel_means is None or channel_stds is None:
                raise ValueError("channel_means and channel_stds are required when normalize=True.")
            self.channel_means = np.asarray(channel_means, dtype="float32")[:, None, None]
            self.channel_stds = np.asarray(channel_stds, dtype="float32")[:, None, None]
        else:
            self.channel_means = None
            self.channel_stds = None

    def _get_sources(self):
        if self._sources is None:
            self._sources = [rasterio.open(path) for path in self.raster_files]
        return self._sources

    def close(self) -> None:
        if self._sources is not None:
            for src in self._sources:
                src.close()
            self._sources = None

    def __len__(self) -> int:
        return len(self.index)

    def read_normalized_patch(self, index: int) -> torch.Tensor:
        row = self.index.iloc[index]
        window = {
            "window_row_start": int(row["window_row_start"]),
            "window_row_stop": int(row["window_row_stop"]),
            "window_col_start": int(row["window_col_start"]),
            "window_col_stop": int(row["window_col_stop"]),
        }
        patch = read_boundless_patch_from_sources(self._get_sources(), window, self.nodata_value)
        expected_shape = (len(self.raster_files), self.patch_size, self.patch_size)
        if patch.shape != expected_shape:
            raise ValueError(f"Patch shape mismatch for patch_id={row['patch_id']}: {patch.shape} != {expected_shape}.")
        mask = valid_context_mask(patch, self.nodata_value)
        if not center_is_valid(mask):
            raise ValueError(f"Patch patch_id={row['patch_id']} has an invalid (NoData) center pixel.")
        patch = apply_norm_and_append_mask(
            patch,
            mask,
            self.channel_means if self.normalize else None,
            self.channel_stds if self.normalize else None,
        )
        return torch.as_tensor(patch, dtype=torch.float32)

    def __getitem__(self, index: int):
        X = self.read_normalized_patch(index)
        rotation_class = int(torch.randint(0, 4, (1,)).item())
        X_rot = apply_rotation(X, rotation_class)
        y_rot = torch.tensor(rotation_class, dtype=torch.long)
        if not torch.isfinite(X_rot).all():
            raise ValueError(f"Rotated patch index={index} contains NaN or inf.")
        if not self.return_metadata:
            return X_rot, y_rot
        row = self.index.iloc[index]
        metadata = {
            "patch_id": row["patch_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
            "rotation_class": rotation_class,
        }
        return X_rot, y_rot, metadata


class RotationHead(nn.Module):
    """Classification head for rotation prediction."""

    def __init__(
        self,
        feature_dim: int = 512,
        n_rotation_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, n_rotation_classes),
        )

    def forward(self, x):
        return self.net(x)


class RotationResNet18Model(nn.Module):
    """ResNet-18 encoder plus rotation classification head."""

    def __init__(
        self,
        in_channels: int = 14,
        n_rotation_classes: int = 4,
        small_patch_stem: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = create_resnet18_encoder_for_rotation(
            in_channels=in_channels,
            small_patch_stem=small_patch_stem,
            pretrained=False,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rotation_head = RotationHead(
            feature_dim=512,
            n_rotation_classes=n_rotation_classes,
            dropout=dropout,
        )

    def forward(self, x):
        feature_map = self.encoder(x)
        h = torch.flatten(self.avgpool(feature_map), 1)
        return self.rotation_head(h)


def rotation_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 rotation accuracy as a fraction."""

    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())
