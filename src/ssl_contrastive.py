"""SimCLR-style contrastive SSL utilities for 13-channel terrain raster patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch import nn

from src.models_resnet18 import create_resnet18_encoder_for_contrastive
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


DEFAULT_AUGMENTATION_CONFIG = {
    "use_hflip": True,
    "use_vflip": True,
    "use_rot90": True,
    "use_crop_resize": True,
    "crop_min_size": 28,
    "crop_max_size": 32,
    "use_gaussian_noise": True,
    "noise_prob": 0.5,
    "noise_std": 0.02,
    "channel_dropout_prob": 0.0,
    "max_channels_to_drop": 2,
}


def apply_geospatial_augmentations(
    X: torch.Tensor,
    patch_size: int = 32,
    config: dict | None = None,
    mask_channels: int = 0,
) -> torch.Tensor:
    """Apply conservative spatial/noise augmentations to one patch.

    The trailing ``mask_channels`` (the valid-context mask) follow the geometric
    transforms (flip/rotate/crop) so they stay aligned with the terrain, but are
    excluded from gaussian noise and channel dropout and are re-binarized after
    crop-resize so the mask stays a clean 0/1 indicator.
    """

    cfg = {**DEFAULT_AUGMENTATION_CONFIG, **(config or {})}
    view = X.clone()
    n_feat = view.shape[0] - int(mask_channels)

    if cfg["use_hflip"] and torch.rand(()) < 0.5:
        view = torch.flip(view, dims=[2])
    if cfg["use_vflip"] and torch.rand(()) < 0.5:
        view = torch.flip(view, dims=[1])
    if cfg["use_rot90"] and torch.rand(()) < 0.5:
        k = int(torch.randint(0, 4, (1,)).item())
        view = torch.rot90(view, k=k, dims=[1, 2])

    if cfg["use_crop_resize"]:
        crop_min = int(cfg["crop_min_size"])
        crop_max = int(cfg["crop_max_size"])
        crop_size = int(torch.randint(crop_min, crop_max + 1, (1,)).item())
        if crop_size < patch_size:
            max_offset = patch_size - crop_size
            top = int(torch.randint(0, max_offset + 1, (1,)).item())
            left = int(torch.randint(0, max_offset + 1, (1,)).item())
            cropped = view[:, top : top + crop_size, left : left + crop_size]
            view = F.interpolate(
                cropped.unsqueeze(0),
                size=(patch_size, patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if mask_channels > 0:
                view[n_feat:] = (view[n_feat:] >= 0.5).float()

    if cfg["use_gaussian_noise"] and torch.rand(()) < float(cfg["noise_prob"]):
        view[:n_feat] = view[:n_feat] + torch.randn_like(view[:n_feat]) * float(cfg["noise_std"])

    channel_dropout_prob = float(cfg.get("channel_dropout_prob", 0.0))
    if channel_dropout_prob > 0 and torch.rand(()) < channel_dropout_prob:
        max_drop = min(int(cfg.get("max_channels_to_drop", 2)), n_feat)
        if max_drop >= 1:
            n_drop = int(torch.randint(1, max_drop + 1, (1,)).item())
            channels = torch.randperm(n_feat)[:n_drop]
            view[channels] = 0.0

    return view.contiguous()


class ContrastiveRasterPatchDataset(torch.utils.data.Dataset):
    """Lazy unlabeled patch dataset that returns two augmented views."""

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
        augment: bool = True,
        augmentation_config: dict | None = None,
    ) -> None:
        self.patch_index_csv = Path(patch_index_csv).resolve()
        self.raster_files = list_raster_files(raster_dir)
        audit_raster_alignment(self.raster_files, expected_nodata=nodata_value)
        self.patch_size = int(patch_size)
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
        self.augment = bool(augment)
        self.augmentation_config = {**DEFAULT_AUGMENTATION_CONFIG, **(augmentation_config or {})}
        self.index = pd.read_csv(self.patch_index_csv)
        self.index = self.index.loc[self.index["valid_patch"].astype(bool)].reset_index(drop=True)
        self._sources = None

        if self.normalize:
            if channel_means is None or channel_stds is None:
                raise ValueError("channel_means and channel_stds are required when normalize=True.")
            self.channel_means = np.asarray(channel_means, dtype="float32")[:, None, None]
            self.channel_stds = np.asarray(channel_stds, dtype="float32")[:, None, None]
            if len(self.channel_means) != len(self.raster_files):
                raise ValueError("channel_means length must match number of raster channels.")
            if len(self.channel_stds) != len(self.raster_files):
                raise ValueError("channel_stds length must match number of raster channels.")
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

    def __getitem__(self, index: int):
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
        X = torch.as_tensor(patch, dtype=torch.float32)
        if self.augment:
            view1 = apply_geospatial_augmentations(X, self.patch_size, self.augmentation_config, mask_channels=1)
            view2 = apply_geospatial_augmentations(X, self.patch_size, self.augmentation_config, mask_channels=1)
        else:
            view1 = X.clone()
            view2 = X.clone()

        if not torch.isfinite(view1).all() or not torch.isfinite(view2).all():
            raise ValueError(f"Augmented views for patch_id={row['patch_id']} contain NaN or inf.")

        if not self.return_metadata:
            return view1, view2
        metadata = {
            "patch_id": row["patch_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
        }
        return view1, view2, metadata


class ProjectionHead(nn.Module):
    """SimCLR MLP projection head."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveResNet18Model(nn.Module):
    """ResNet-18 encoder plus SimCLR projection head."""

    def __init__(
        self,
        in_channels: int = 14,
        projection_dim: int = 128,
        small_patch_stem: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = create_resnet18_encoder_for_contrastive(
            in_channels=in_channels,
            small_patch_stem=small_patch_stem,
            pretrained=False,
            projection_dim=projection_dim,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_head = ProjectionHead(
            feature_dim=512,
            hidden_dim=512,
            projection_dim=projection_dim,
        )

    def forward(self, x):
        feature_map = self.encoder(x)
        h = torch.flatten(self.avgpool(feature_map), 1)
        z = F.normalize(self.projection_head(h), dim=1)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Compute the NT-Xent loss for paired normalized embeddings."""

    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}.")
    if z1.ndim != 2:
        raise ValueError(f"Expected z tensors with shape (batch, dim), got {z1.shape}.")

    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent loss requires batch_size >= 2.")

    z = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(z, z.T) / float(temperature)
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -torch.inf)
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)
