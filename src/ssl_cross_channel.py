"""Cross-channel masking SSL utilities for terrain raster patches.

The pretext task masks one entire terrain conditioning-factor channel (default:
TWI, alphabetical channel index 12 after landcover is dropped) by zeroing it in
the normalized input, then predicts that channel from the remaining 12 terrain
channels plus the valid-context mask. Because masking happens after per-channel
normalization, "zero" corresponds to the channel mean in raw units, the intended
neutral fill value. The encoder input is 14-channel (13 terrain + 1 valid-context
mask) so pretrained weights transfer directly to the downstream 14-channel
classifier. The valid-context mask channel (index 13) is never the masked target.
"""

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
    list_raster_files,
    read_boundless_patch_from_sources,
    valid_context_mask,
)
from src.ssl_masked_recon import (  # noqa: F401  (re-exported for scripts/notebooks)
    ResNet18ReconstructionDecoder,
    compute_ssl_channel_stats,
    create_unlabeled_patch_index,
)


DEFAULT_MASKED_CHANNEL_INDEX = 12  # twi_dinf_30m.tif, alphabetically last of 13 (landcover dropped)


class CrossChannelMaskRasterDataset(torch.utils.data.Dataset):
    """Lazy unlabeled raster patch dataset for cross-channel masking.

    The dataset returns the full normalized 14-channel patch (13 terrain + 1
    valid-context mask); channel masking is applied in the training loop so the
    masked channel index stays a single configuration knob (same pattern as
    block-mask generation for masked reconstruction).
    """

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
        cache_in_memory: bool = False,
        with_mask: bool = True,
    ) -> None:
        self.patch_index_csv = Path(patch_index_csv).resolve()
        self.raster_files = list_raster_files(raster_dir)
        audit_raster_alignment(self.raster_files, expected_nodata=nodata_value)
        self.patch_size = int(patch_size)
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
        self.cache_in_memory = bool(cache_in_memory)
        self.with_mask = bool(with_mask)
        self.index = pd.read_csv(self.patch_index_csv)
        self.index = self.index.loc[self.index["valid_patch"].astype(bool)].reset_index(drop=True)
        self._sources = None
        self._cache = None

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

    def _read_raw_patch(self, index: int, window: dict[str, int]) -> np.ndarray:
        if self.cache_in_memory:
            if self._cache is None:
                self._build_cache()
            return self._cache[index]
        return read_boundless_patch_from_sources(self._get_sources(), window, self.nodata_value)

    def _build_cache(self) -> None:
        # Cache holds the raw terrain channels (NoData sentinels in padded
        # pixels); the valid-context mask is derived per item in __getitem__.
        patches = np.empty(
            (len(self.index), len(self.raster_files), self.patch_size, self.patch_size),
            dtype="float32",
        )
        sources = self._get_sources()
        for position in range(len(self.index)):
            row = self.index.iloc[position]
            window = {
                "window_row_start": int(row["window_row_start"]),
                "window_row_stop": int(row["window_row_stop"]),
                "window_col_start": int(row["window_col_start"]),
                "window_col_stop": int(row["window_col_stop"]),
            }
            patches[position] = read_boundless_patch_from_sources(sources, window, self.nodata_value)
        self._cache = patches

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
        patch = self._read_raw_patch(index, window)
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
            append_mask=self.with_mask,
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


def apply_channel_mask(
    X: torch.Tensor,
    masked_channel_index: int = DEFAULT_MASKED_CHANNEL_INDEX,
    mask_channel_present: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero one entire terrain channel and return (X_masked, target).

    X is a normalized batch with shape (batch, channels, height, width) whose
    last channel is the valid-context mask. The masked terrain channel is set to
    0.0 (the per-channel mean in normalized space) and the target is that channel
    with shape (batch, 1, height, width). The valid-context mask channel is never
    masked and is excluded from the valid index range.
    """

    if X.ndim != 4:
        raise ValueError(f"Expected X with shape (batch, channels, height, width), got {X.shape}.")
    n_channels = X.shape[1]
    # The last channel is the valid-context mask unless it has been ablated.
    n_terrain = n_channels - 1 if mask_channel_present else n_channels
    if not 0 <= masked_channel_index < n_terrain:
        suffix = (
            f" (the last channel {n_channels - 1} is the valid-context mask)"
            if mask_channel_present
            else ""
        )
        raise ValueError(
            f"masked_channel_index must be a terrain channel in [0, {n_terrain - 1}]"
            f"{suffix}, got {masked_channel_index}."
        )
    X_masked = X.clone()
    X_masked[:, masked_channel_index, :, :] = 0.0
    target = X[:, masked_channel_index : masked_channel_index + 1, :, :]
    return X_masked, target


def cross_channel_loss(X_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on the masked channel only (the prediction is that channel)."""

    if X_pred.shape != target.shape:
        raise ValueError(f"Prediction shape {X_pred.shape} does not match target shape {target.shape}.")
    return nn.functional.mse_loss(X_pred, target)


class CrossChannelModel(nn.Module):
    """Cross-channel masking model: ResNet-18 encoder plus 1-channel decoder.

    The encoder consumes the full 14-channel input (13 terrain + valid-context
    mask) with one terrain channel zeroed; the decoder predicts only the masked
    terrain channel.
    """

    def __init__(self, in_channels: int = 14, out_channels: int = 1) -> None:
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
