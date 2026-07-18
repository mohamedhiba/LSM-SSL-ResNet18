"""Jigsaw self-supervised learning utilities for 13-channel terrain raster patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn

from src.models_resnet18 import create_resnet18_encoder_for_jigsaw
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


def generate_jigsaw_permutation_bank(
    n_tiles: int = 16,
    n_permutations: int = 100,
    random_seed: int = 42,
    min_hamming_distance: int | None = None,
    n_candidates: int = 20_000,
) -> np.ndarray:
    """Generate a deterministic diverse bank of tile permutations."""

    rng = np.random.default_rng(random_seed)
    identity = np.arange(n_tiles)
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < n_candidates and attempts < n_candidates * 20:
        attempts += 1
        perm = rng.permutation(n_tiles)
        if np.array_equal(perm, identity):
            continue
        key = tuple(int(value) for value in perm)
        if key in seen:
            continue
        if min_hamming_distance is not None:
            if int((perm != identity).sum()) < min_hamming_distance:
                continue
        seen.add(key)
        candidates.append(perm)

    if len(candidates) < n_permutations:
        print(
            "WARNING: Greedy diverse permutation generation had too few "
            "candidates; using available unique random permutations."
        )
        return np.asarray(candidates[:n_permutations], dtype=np.int64)

    candidates_array = np.asarray(candidates, dtype=np.int64)
    selected_indices = [0]
    min_distances = (candidates_array != candidates_array[0]).sum(axis=1)
    min_distances[0] = -1

    while len(selected_indices) < n_permutations:
        best_index = int(np.argmax(min_distances))
        if min_hamming_distance is not None and min_distances[best_index] < min_hamming_distance:
            print(
                "WARNING: Could not maintain requested min_hamming_distance "
                f"{min_hamming_distance}; continuing greedy selection."
            )
        selected_indices.append(best_index)
        new_distances = (candidates_array != candidates_array[best_index]).sum(axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected_indices] = -1

    return candidates_array[selected_indices].astype(np.int64)


def save_jigsaw_permutation_bank(permutations: np.ndarray, output_csv: str | Path) -> Path:
    """Save permutation bank to CSV."""

    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for class_id, perm in enumerate(permutations):
        record = {"perm_class": int(class_id)}
        for position_index, tile_index in enumerate(perm):
            record[f"position_{position_index:02d}"] = int(tile_index)
        records.append(record)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return output_csv


def load_or_create_jigsaw_permutation_bank(
    output_csv: str | Path,
    n_tiles: int = 16,
    n_permutations: int = 100,
    random_seed: int = 42,
    min_hamming_distance: int | None = None,
) -> np.ndarray:
    """Load existing permutation bank or create a deterministic new one."""

    output_csv = Path(output_csv).resolve()
    required_columns = ["perm_class", *[f"position_{index:02d}" for index in range(n_tiles)]]
    if output_csv.exists():
        table = pd.read_csv(output_csv)
        if len(table) == n_permutations and all(column in table.columns for column in required_columns):
            permutations = table[[f"position_{index:02d}" for index in range(n_tiles)]].to_numpy(dtype=np.int64)
            valid_rows = sorted(table["perm_class"].astype(int).tolist()) == list(range(n_permutations))
            valid_values = all(sorted(row.tolist()) == list(range(n_tiles)) for row in permutations)
            if valid_rows and valid_values:
                return permutations

    permutations = generate_jigsaw_permutation_bank(
        n_tiles=n_tiles,
        n_permutations=n_permutations,
        random_seed=random_seed,
        min_hamming_distance=min_hamming_distance,
    )
    save_jigsaw_permutation_bank(permutations, output_csv)
    return permutations


def apply_jigsaw_permutation(
    X: torch.Tensor,
    permutation: torch.Tensor | np.ndarray,
    grid_size: int = 4,
) -> torch.Tensor:
    """Apply one tile permutation to a 14x32x32 tensor."""

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (channels, height, width), got {X.shape}.")
    channels, height, width = X.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError("Patch height and width must be divisible by grid_size.")
    tile_h = height // grid_size
    tile_w = width // grid_size
    if isinstance(permutation, np.ndarray):
        permutation = torch.as_tensor(permutation, dtype=torch.long, device=X.device)
    else:
        permutation = permutation.to(device=X.device, dtype=torch.long)
    if len(permutation) != grid_size * grid_size:
        raise ValueError("Permutation length must equal grid_size * grid_size.")

    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            tiles.append(X[:, row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w])

    output_rows = []
    for row in range(grid_size):
        row_tiles = []
        for col in range(grid_size):
            output_position = row * grid_size + col
            original_tile_index = int(permutation[output_position].item())
            row_tiles.append(tiles[original_tile_index])
        output_rows.append(torch.cat(row_tiles, dim=2))
    return torch.cat(output_rows, dim=1).contiguous()


class JigsawRasterPatchDataset(torch.utils.data.Dataset):
    """Lazy unlabeled raster patch dataset for Jigsaw SSL."""

    def __init__(
        self,
        patch_index_csv: str | Path,
        raster_dir: str | Path,
        permutation_bank_csv: str | Path,
        patch_size: int = 32,
        grid_size: int = 4,
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
        self.permutation_bank_csv = Path(permutation_bank_csv).resolve()
        self.permutation_table = pd.read_csv(self.permutation_bank_csv)
        self.permutation_columns = [column for column in self.permutation_table.columns if column.startswith("position_")]
        self.permutations = self.permutation_table[self.permutation_columns].to_numpy(dtype=np.int64)
        self.patch_size = int(patch_size)
        self.grid_size = int(grid_size)
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
        self.random_seed = int(random_seed)
        self.index = pd.read_csv(self.patch_index_csv)
        self.index = self.index.loc[self.index["valid_patch"].astype(bool)].reset_index(drop=True)
        self._sources = None

        if self.patch_size % self.grid_size != 0:
            raise ValueError("patch_size must be divisible by grid_size.")
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
        perm_class = int(torch.randint(0, len(self.permutations), (1,)).item())
        permutation = self.permutations[perm_class]
        X_jigsaw = apply_jigsaw_permutation(X, permutation, grid_size=self.grid_size)
        y_perm = torch.tensor(perm_class, dtype=torch.long)
        if not torch.isfinite(X_jigsaw).all():
            raise ValueError(f"Jigsaw patch index={index} contains NaN or inf.")
        if not self.return_metadata:
            return X_jigsaw, y_perm
        row = self.index.iloc[index]
        metadata = {
            "patch_id": row["patch_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
            "perm_class": perm_class,
        }
        return X_jigsaw, y_perm, metadata


class JigsawHead(nn.Module):
    """Classification head for Jigsaw permutation prediction."""

    def __init__(
        self,
        feature_dim: int = 512,
        n_permutation_classes: int = 100,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, n_permutation_classes),
        )

    def forward(self, x):
        return self.net(x)


class JigsawResNet18Model(nn.Module):
    """ResNet-18 encoder plus Jigsaw permutation classification head."""

    def __init__(
        self,
        in_channels: int = 14,
        n_permutation_classes: int = 100,
        small_patch_stem: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = create_resnet18_encoder_for_jigsaw(
            in_channels=in_channels,
            small_patch_stem=small_patch_stem,
            pretrained=False,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.jigsaw_head = JigsawHead(
            feature_dim=512,
            n_permutation_classes=n_permutation_classes,
            dropout=dropout,
        )

    def forward(self, x):
        feature_map = self.encoder(x)
        h = torch.flatten(self.avgpool(feature_map), 1)
        return self.jigsaw_head(h)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> dict[int, float]:
    """Compute top-k accuracies as fractions."""

    max_k = max(topk)
    max_k = min(max_k, logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    results = {}
    for k in topk:
        k = min(k, logits.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[int(k)] = float((correct_k / targets.numel()).item())
    return results
