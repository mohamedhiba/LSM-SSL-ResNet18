"""Masked reconstruction SSL utilities for 13-channel terrain raster patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn

from src.models_resnet18 import create_resnet18_encoder_for_ssl
from src.patch_dataset import (
    DEFAULT_NODATA_VALUE,
    apply_norm_and_append_mask,
    audit_raster_alignment,
    center_is_valid,
    compute_nodata_ratio,
    compute_patch_window,
    is_window_inside_raster,
    list_raster_files,
    read_boundless_patch_from_sources,
    read_multichannel_patch_from_sources,
    valid_context_mask,
)


class MaskedReconstructionRasterDataset(torch.utils.data.Dataset):
    """Lazy unlabeled raster patch dataset for masked reconstruction."""

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
    ) -> None:
        self.patch_index_csv = Path(patch_index_csv).resolve()
        self.raster_files = list_raster_files(raster_dir)
        audit_raster_alignment(self.raster_files, expected_nodata=nodata_value)
        self.patch_size = int(patch_size)
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
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
        if not self.return_metadata:
            return X
        metadata = {
            "patch_id": row["patch_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
        }
        return X, metadata


def _xy_from_row_col(row: int, col: int, transform) -> tuple[float, float]:
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def create_unlabeled_patch_index(
    raster_dir: str | Path,
    output_csv: str | Path,
    patch_size: int = 32,
    n_patches: int = 20_000,
    nodata_value: float = DEFAULT_NODATA_VALUE,
    max_nodata_ratio: float = 0.0,
    random_seed: int = 42,
    max_attempts: int = 1_000_000,
    force_regenerate: bool = False,
    center_only: bool = False,
) -> pd.DataFrame:
    """Generate or load a valid unlabeled patch index.

    With ``center_only=False`` (default) patches must be fully inside the raster
    and fully valid (``nodata_ratio <= max_nodata_ratio``); the appended
    valid-context mask is therefore all-ones. With ``center_only=True`` only the
    center pixel must be valid — windows may overhang the raster edge or contain
    local NoData (boundless/zero-padded), so the valid-context mask varies, which
    matches the downstream labeled-patch distribution.
    """

    output_csv = Path(output_csv).resolve()
    if output_csv.exists() and not force_regenerate:
        existing = pd.read_csv(output_csv)
        if len(existing.loc[existing["valid_patch"].astype(bool)]) >= n_patches:
            return existing.iloc[:n_patches].copy()

    raster_files = list_raster_files(raster_dir)
    audit = audit_raster_alignment(raster_files, expected_nodata=nodata_value)
    rng = np.random.default_rng(random_seed)
    half = patch_size // 2
    records: list[dict[str, object]] = []
    attempts = 0

    sources = [rasterio.open(path) for path in raster_files]
    try:
        while len(records) < n_patches and attempts < max_attempts:
            attempts += 1
            if center_only:
                # Sample the center anywhere in the raster; pad overhang/NoData.
                row = int(rng.integers(0, audit.height))
                col = int(rng.integers(0, audit.width))
                window = compute_patch_window(row, col, patch_size)
                patch = read_boundless_patch_from_sources(sources, window, nodata_value)
                if patch.shape != (len(raster_files), patch_size, patch_size):
                    continue
                mask = valid_context_mask(patch, nodata_value)
                if not center_is_valid(mask):
                    continue
                nodata_ratio = compute_nodata_ratio(patch, nodata_value)
            else:
                row = int(rng.integers(half, audit.height - half))
                col = int(rng.integers(half, audit.width - half))
                window = compute_patch_window(row, col, patch_size)
                if not is_window_inside_raster(window, audit.width, audit.height):
                    continue
                patch = read_multichannel_patch_from_sources(sources, window)
                if patch.shape != (len(raster_files), patch_size, patch_size):
                    continue
                if not np.isfinite(patch).all():
                    continue
                nodata_ratio = compute_nodata_ratio(patch, nodata_value)
                if nodata_ratio > max_nodata_ratio:
                    continue
            x, y = _xy_from_row_col(row, col, audit.transform)
            records.append(
                {
                    "patch_id": f"U_SSL_{len(records) + 1:06d}",
                    "x": x,
                    "y": y,
                    "row": row,
                    "col": col,
                    "patch_size": patch_size,
                    "window_row_start": window["window_row_start"],
                    "window_row_stop": window["window_row_stop"],
                    "window_col_start": window["window_col_start"],
                    "window_col_stop": window["window_col_stop"],
                    "valid_patch": True,
                    "nodata_ratio": float(nodata_ratio),
                }
            )
            if len(records) % 1000 == 0:
                print(f"Collected {len(records)} / {n_patches} valid SSL patches after {attempts} attempts.", flush=True)
    finally:
        for src in sources:
            src.close()

    if len(records) < n_patches:
        raise RuntimeError(
            f"Only collected {len(records)} valid patches after max_attempts={max_attempts}."
        )
    index = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_csv, index=False)
    return index


def compute_ssl_channel_stats(
    dataset: MaskedReconstructionRasterDataset,
    sample_size: int = 5000,
    batch_size: int = 64,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute channel means/stds from a random subset of unlabeled patches."""

    from torch.utils.data import DataLoader, Subset

    rng = np.random.default_rng(random_seed)
    n = len(dataset)
    sample_size = min(sample_size, n)
    indices = rng.choice(np.arange(n), size=sample_size, replace=False)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    # X is (B, terrain + 1 mask): the last channel is the valid-context mask and
    # padded pixels are already zeroed in the terrain channels, so terrain stats
    # accumulate over valid pixels only (counted by the mask). Output length is
    # the terrain channel count (the mask channel is not normalized).
    channel_sum = None
    channel_sumsq = None
    pixel_count = 0.0
    for X in loader:
        X = X.float()
        terrain = X[:, :-1]
        ctx_mask = X[:, -1:]
        if channel_sum is None:
            channel_sum = terrain.sum(dim=(0, 2, 3))
            channel_sumsq = (terrain ** 2).sum(dim=(0, 2, 3))
        else:
            channel_sum += terrain.sum(dim=(0, 2, 3))
            channel_sumsq += (terrain ** 2).sum(dim=(0, 2, 3))
        pixel_count += float(ctx_mask.sum().item())
    means = (channel_sum / pixel_count).numpy()
    variances = (channel_sumsq / pixel_count).numpy() - means**2
    stds = np.sqrt(np.maximum(variances, 1e-12))
    return means.astype("float32"), stds.astype("float32")


class ResNet18ReconstructionDecoder(nn.Module):
    """Lightweight decoder from ResNet-18 feature maps to 13x32x32 patches."""

    def __init__(self, out_channels: int = 13) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


class MaskedReconstructionModel(nn.Module):
    """Full masked reconstruction model: ResNet-18 encoder plus decoder."""

    def __init__(self, in_channels: int = 14, out_channels: int = 13) -> None:
        super().__init__()
        self.encoder = create_resnet18_encoder_for_ssl(
            in_channels=in_channels,
            small_patch_stem=True,
            pretrained=False,
            return_feature_map=True,
        )
        self.decoder = ResNet18ReconstructionDecoder(out_channels=out_channels)

    def forward(self, x_masked):
        features = self.encoder(x_masked)
        return self.decoder(features)


def generate_block_mask(
    batch_size: int,
    height: int = 32,
    width: int = 32,
    mask_ratio: float = 0.5,
    block_size: int = 4,
    device=None,
) -> torch.Tensor:
    """Generate block-wise spatial masks with 1 for masked pixels."""

    device = device or torch.device("cpu")
    mask = torch.zeros((batch_size, 1, height, width), device=device)
    n_block_rows = height // block_size
    n_block_cols = width // block_size
    n_blocks = n_block_rows * n_block_cols
    n_mask_blocks = max(1, int(round(mask_ratio * n_blocks)))
    for batch_index in range(batch_size):
        block_ids = torch.randperm(n_blocks, device=device)[:n_mask_blocks]
        for block_id in block_ids:
            block_id = int(block_id.item())
            br = block_id // n_block_cols
            bc = block_id % n_block_cols
            r0 = br * block_size
            c0 = bc * block_size
            mask[batch_index, :, r0 : r0 + block_size, c0 : c0 + block_size] = 1.0
    return mask


def masked_reconstruction_loss(X_recon: torch.Tensor, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss over masked pixels only."""

    squared_error = ((X_recon - X) ** 2) * mask
    denom = mask.sum() * X.shape[1]
    return squared_error.sum() / denom.clamp_min(1.0)
