"""1D horizontal-strip jigsaw SSL utilities for 13-channel terrain raster patches.

The pretext task splits each 32x32 patch into 3 horizontal strips, reorders
them by one of the 3! = 6 fixed permutations, and trains the model to predict
the permutation class. Because 32 is not divisible by 3, strips use
array-split sizes [11, 11, 10]; the permuted strips always reassemble to the
full 32 rows. The unequal last strip leaves a residual strip-size cue for the
pretext task; this is documented as an accepted limitation, consistent with
how the repo reports jigsaw/rotation pretext solvability rather than
engineering it away.
"""

from __future__ import annotations

from itertools import permutations as _iter_permutations
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn

from src.models_resnet18 import create_resnet18_encoder_for_strip_jigsaw
from src.patch_dataset import (
    DEFAULT_NODATA_VALUE,
    apply_norm_and_append_mask,
    audit_raster_alignment,
    center_is_valid,
    list_raster_files,
    read_boundless_patch_from_sources,
    valid_context_mask,
)
from src.ssl_jigsaw import topk_accuracy  # noqa: F401  (re-exported for train_ssl)
from src.ssl_masked_recon import create_unlabeled_patch_index  # noqa: F401


N_STRIPS = 3
STRIP_PERMUTATIONS = np.asarray(
    list(_iter_permutations(range(N_STRIPS))), dtype=np.int64
)  # shape (6, 3); row 0 is the identity (0, 1, 2)


def strip_sizes(height: int = 32, n_strips: int = N_STRIPS) -> list[int]:
    """Return array-split strip heights, e.g. 32 rows / 3 strips -> [11, 11, 10]."""

    if height <= 0 or n_strips <= 0:
        raise ValueError(f"height and n_strips must be positive, got {height}, {n_strips}.")
    if n_strips > height:
        raise ValueError(f"n_strips ({n_strips}) cannot exceed height ({height}).")
    base, remainder = divmod(height, n_strips)
    return [base + (1 if strip_index < remainder else 0) for strip_index in range(n_strips)]


def apply_strip_permutation(
    X: torch.Tensor,
    permutation: torch.Tensor | np.ndarray,
    n_strips: int = N_STRIPS,
) -> torch.Tensor:
    """Reorder horizontal strips of a 14xHxW tensor by one permutation."""

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (channels, height, width), got {X.shape}.")
    if isinstance(permutation, np.ndarray):
        permutation = torch.as_tensor(permutation, dtype=torch.long, device=X.device)
    else:
        permutation = permutation.to(device=X.device, dtype=torch.long)
    if len(permutation) != n_strips:
        raise ValueError("Permutation length must equal n_strips.")
    if sorted(int(value) for value in permutation.tolist()) != list(range(n_strips)):
        raise ValueError(f"Permutation must reorder strip indices 0..{n_strips - 1}.")

    sizes = strip_sizes(X.shape[1], n_strips)
    strips = list(torch.split(X, sizes, dim=1))
    return torch.cat([strips[int(strip_index.item())] for strip_index in permutation], dim=1).contiguous()


def save_strip_permutation_bank(permutations: np.ndarray, output_csv: str | Path) -> Path:
    """Save the strip permutation bank to CSV (provenance only)."""

    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for class_id, perm in enumerate(permutations):
        record = {"perm_class": int(class_id)}
        for position_index, strip_index in enumerate(perm):
            record[f"position_{position_index:02d}"] = int(strip_index)
        records.append(record)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return output_csv


def load_or_create_strip_permutation_bank(
    output_csv: str | Path,
    n_strips: int = N_STRIPS,
) -> np.ndarray:
    """Return the fixed 3! permutation bank, writing the provenance CSV if absent.

    Unlike the 100-class tile jigsaw bank, the in-code ``STRIP_PERMUTATIONS``
    enumeration is the source of truth; the CSV records it for reproducibility.
    """

    if n_strips != N_STRIPS:
        raise ValueError(f"Only n_strips={N_STRIPS} is supported, got {n_strips}.")
    output_csv = Path(output_csv).resolve()
    required_columns = ["perm_class", *[f"position_{index:02d}" for index in range(n_strips)]]
    if output_csv.exists():
        table = pd.read_csv(output_csv)
        if len(table) == len(STRIP_PERMUTATIONS) and all(column in table.columns for column in required_columns):
            permutations = table[[f"position_{index:02d}" for index in range(n_strips)]].to_numpy(dtype=np.int64)
            if np.array_equal(permutations, STRIP_PERMUTATIONS):
                return STRIP_PERMUTATIONS.copy()
        raise ValueError(
            f"Existing strip permutation bank {output_csv} does not match the "
            "fixed 3-strip permutation enumeration."
        )

    save_strip_permutation_bank(STRIP_PERMUTATIONS, output_csv)
    return STRIP_PERMUTATIONS.copy()


class StripJigsawRasterPatchDataset(torch.utils.data.Dataset):
    """Lazy unlabeled raster patch dataset for 1D strip jigsaw SSL."""

    def __init__(
        self,
        patch_index_csv: str | Path,
        raster_dir: str | Path,
        patch_size: int = 32,
        n_strips: int = N_STRIPS,
        nodata_value: float = DEFAULT_NODATA_VALUE,
        normalize: bool = True,
        channel_means=None,
        channel_stds=None,
        return_metadata: bool = False,
        random_seed: int = 42,
        cache_in_memory: bool = False,
    ) -> None:
        self.patch_index_csv = Path(patch_index_csv).resolve()
        self.raster_files = list_raster_files(raster_dir)
        audit_raster_alignment(self.raster_files, expected_nodata=nodata_value)
        self.patch_size = int(patch_size)
        self.n_strips = int(n_strips)
        self.permutations = STRIP_PERMUTATIONS.copy()
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
        self.random_seed = int(random_seed)
        self.cache_in_memory = bool(cache_in_memory)
        self.index = pd.read_csv(self.patch_index_csv)
        self.index = self.index.loc[self.index["valid_patch"].astype(bool)].reset_index(drop=True)
        self._sources = None
        self._cache = None

        if self.n_strips != N_STRIPS:
            raise ValueError(f"Only n_strips={N_STRIPS} is supported, got {self.n_strips}.")
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
        # Cache holds raw terrain channels (NoData sentinels in padded pixels);
        # the valid-context mask is derived per item in read_normalized_patch.
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

    def read_normalized_patch(self, index: int) -> torch.Tensor:
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
        )
        return torch.as_tensor(patch, dtype=torch.float32)

    def __getitem__(self, index: int):
        X = self.read_normalized_patch(index)
        perm_class = int(torch.randint(0, len(self.permutations), (1,)).item())
        permutation = self.permutations[perm_class]
        X_strip = apply_strip_permutation(X, permutation, n_strips=self.n_strips)
        y_perm = torch.tensor(perm_class, dtype=torch.long)
        if not torch.isfinite(X_strip).all():
            raise ValueError(f"Strip jigsaw patch index={index} contains NaN or inf.")
        if not self.return_metadata:
            return X_strip, y_perm
        row = self.index.iloc[index]
        metadata = {
            "patch_id": row["patch_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
            "perm_class": perm_class,
        }
        return X_strip, y_perm, metadata


class StripJigsawHead(nn.Module):
    """Classification head for strip permutation prediction."""

    def __init__(
        self,
        feature_dim: int = 512,
        n_permutation_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, n_permutation_classes),
        )

    def forward(self, x):
        return self.net(x)


class StripJigsawResNet18Model(nn.Module):
    """ResNet-18 encoder plus strip permutation classification head."""

    def __init__(
        self,
        in_channels: int = 14,
        n_permutation_classes: int = 6,
        small_patch_stem: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = create_resnet18_encoder_for_strip_jigsaw(
            in_channels=in_channels,
            small_patch_stem=small_patch_stem,
            pretrained=False,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.strip_jigsaw_head = StripJigsawHead(
            feature_dim=512,
            n_permutation_classes=n_permutation_classes,
            dropout=dropout,
        )

    def forward(self, x):
        feature_map = self.encoder(x)
        h = torch.flatten(self.avgpool(feature_map), 1)
        return self.strip_jigsaw_head(h)
