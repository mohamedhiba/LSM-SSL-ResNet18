"""Patch index generation and lazy raster patch Dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

try:
    import torch as _torch

    _TorchDatasetBase = _torch.utils.data.Dataset
except ModuleNotFoundError:
    _torch = None
    _TorchDatasetBase = object


FACTOR_NAMES = [f"factor_{index:02d}" for index in range(1, 15)]
DEFAULT_NODATA_VALUE = -9999

# Landcover is dropped from the CNN channel stack: it is a categorical,
# human-modifiable variable, not a continuous terrain feature, and cannot be
# recovered from the other terrain derivatives. The file may stay on disk; it
# is filtered out of the channel glob by name so every dataset/model sees the
# 13 terrain channels. See list_raster_files / _is_excluded_raster.
EXCLUDED_RASTER_NAME_STEMS = ("landcover",)
EXPECTED_RASTER_COUNT = 13


@dataclass(frozen=True)
class RasterGridAudit:
    """Shared metadata for an aligned raster stack."""

    crs: object
    transform: object
    width: int
    height: int
    resolution: tuple[float, float]
    bounds: object
    nodata: float


def _is_excluded_raster(path: Path) -> bool:
    """Return True if a raster file is excluded from the CNN channel stack."""

    name = path.name.lower()
    return any(stem in name for stem in EXCLUDED_RASTER_NAME_STEMS)


def list_raster_files(raster_dir: str | Path) -> list[Path]:
    """Return the cleaned terrain raster files sorted alphabetically.

    Landcover is excluded from the channel stack by name (see
    EXCLUDED_RASTER_NAME_STEMS); the file may stay on disk for provenance. The
    result is the EXPECTED_RASTER_COUNT terrain channels used by every dataset
    and model.
    """

    raster_dir = Path(raster_dir).resolve()
    raster_files = [
        path
        for path in sorted(raster_dir.glob("*.tif"), key=lambda path: path.name.lower())
        if not _is_excluded_raster(path)
    ]
    if len(raster_files) != EXPECTED_RASTER_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_RASTER_COUNT} terrain .tif rasters in "
            f"{raster_dir} (landcover excluded), found {len(raster_files)}."
        )
    return raster_files


def _transform_equal(left: object, right: object, atol: float = 1e-9) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def _bounds_equal(left: object, right: object, atol: float = 1e-9) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def audit_raster_alignment(
    raster_files: Sequence[str | Path],
    expected_nodata: float = DEFAULT_NODATA_VALUE,
) -> RasterGridAudit:
    """Check cleaned rasters share grid metadata and nodata = -9999."""

    raster_files = [Path(path).resolve() for path in raster_files]
    if len(raster_files) != EXPECTED_RASTER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RASTER_COUNT} rasters for audit, found {len(raster_files)}."
        )

    with rasterio.open(raster_files[0]) as reference:
        ref_crs = reference.crs
        ref_transform = reference.transform
        ref_width = reference.width
        ref_height = reference.height
        ref_resolution = reference.res
        ref_bounds = reference.bounds
        ref_nodata = reference.nodata

    errors: list[str] = []
    if not np.isclose(ref_nodata, expected_nodata, rtol=0.0, atol=0.0):
        errors.append(
            f"{raster_files[0].name}: nodata is {ref_nodata}, expected {expected_nodata}."
        )

    for path in raster_files[1:]:
        with rasterio.open(path) as src:
            mismatches: list[str] = []
            if src.crs != ref_crs:
                mismatches.append("CRS")
            if not _transform_equal(src.transform, ref_transform):
                mismatches.append("transform")
            if src.width != ref_width:
                mismatches.append("width")
            if src.height != ref_height:
                mismatches.append("height")
            if not np.allclose(src.res, ref_resolution, rtol=0.0, atol=1e-9):
                mismatches.append("resolution")
            if not _bounds_equal(src.bounds, ref_bounds):
                mismatches.append("bounds")
            if not np.isclose(src.nodata, expected_nodata, rtol=0.0, atol=0.0):
                mismatches.append(f"nodata={src.nodata}")
            if mismatches:
                errors.append(f"{path.name}: {', '.join(mismatches)}")

    if errors:
        raise ValueError(
            "Cleaned raster audit failed. Mismatches: " + "; ".join(errors)
        )

    return RasterGridAudit(
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        resolution=ref_resolution,
        bounds=ref_bounds,
        nodata=float(ref_nodata),
    )


def xy_to_rowcol(x: float, y: float, transform: object) -> tuple[int, int]:
    """Convert x/y to raster row/column using the inverse affine transform."""

    col_float, row_float = (~transform) * (float(x), float(y))
    return int(np.floor(row_float)), int(np.floor(col_float))


def compute_patch_window(row: int, col: int, patch_size: int) -> dict[str, int]:
    """Compute centered even-sized patch window bounds."""

    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError(f"patch_size must be a positive even integer, got {patch_size}.")
    half = patch_size // 2
    return {
        "half_size_top": half,
        "half_size_bottom": half,
        "half_size_left": half,
        "half_size_right": half,
        "window_row_start": int(row - half),
        "window_row_stop": int(row + half),
        "window_col_start": int(col - half),
        "window_col_stop": int(col + half),
    }


def is_window_inside_raster(window: dict[str, int], width: int, height: int) -> bool:
    """Return True when the patch window is fully inside raster bounds."""

    return (
        window["window_row_start"] >= 0
        and window["window_col_start"] >= 0
        and window["window_row_stop"] <= height
        and window["window_col_stop"] <= width
    )


def _window_to_rasterio(window: dict[str, int]) -> Window:
    return Window(
        col_off=window["window_col_start"],
        row_off=window["window_row_start"],
        width=window["window_col_stop"] - window["window_col_start"],
        height=window["window_row_stop"] - window["window_row_start"],
    )


def read_multichannel_patch(
    raster_files: Sequence[str | Path],
    window: dict[str, int],
) -> np.ndarray:
    """Read a raster window from all terrain rasters as (channels, rows, cols)."""

    raster_window = _window_to_rasterio(window)
    patches = []
    for path in raster_files:
        with rasterio.open(path) as src:
            band = src.read(1, window=raster_window, boundless=False)
            patches.append(band)
    return np.stack(patches, axis=0)


def read_multichannel_patch_from_sources(
    sources: Sequence[object],
    window: dict[str, int],
) -> np.ndarray:
    """Read a raster window from already-open rasterio sources."""

    raster_window = _window_to_rasterio(window)
    return np.stack(
        [src.read(1, window=raster_window, boundless=False) for src in sources],
        axis=0,
    )


def compute_nodata_ratio(
    patch: np.ndarray,
    nodata_value: float = DEFAULT_NODATA_VALUE,
) -> float:
    """Compute fraction of values equal to nodata across all channels/pixels."""

    return float(np.isclose(patch, nodata_value, rtol=0.0, atol=0.0).sum() / patch.size)


# ---------------------------------------------------------------------------
# Boundary-aware padding + valid-context mask channel
#
# Patches now only require a valid center pixel. Boundary or local NoData pixels
# are read as the NoData sentinel (boundless read), zeroed in the terrain
# channels, and flagged by an appended binary valid-context mask channel
# (1 = real pixel, 0 = padded/NoData) so the model knows which pixels are real
# conditioning factors. Input channels become EXPECTED_RASTER_COUNT terrain + 1
# mask. The mask is always the LAST channel.
# ---------------------------------------------------------------------------


def read_boundless_patch_from_sources(
    sources: Sequence[object],
    window: dict[str, int],
    nodata_value: float = DEFAULT_NODATA_VALUE,
) -> np.ndarray:
    """Read a (possibly edge-overhanging) window, padding outside with nodata.

    Unlike ``read_multichannel_patch_from_sources`` (boundless=False), this
    always returns a full ``(channels, patch_size, patch_size)`` array; pixels
    outside the raster are filled with ``nodata_value`` and later masked.
    """

    raster_window = _window_to_rasterio(window)
    return np.stack(
        [
            src.read(1, window=raster_window, boundless=True, fill_value=nodata_value)
            for src in sources
        ],
        axis=0,
    ).astype("float32", copy=False)


def valid_context_mask(
    patch: np.ndarray,
    nodata_value: float = DEFAULT_NODATA_VALUE,
) -> np.ndarray:
    """Return a (rows, cols) float32 mask: 1 where every channel is a real pixel.

    A pixel is real when it is finite and not equal to ``nodata_value`` in all
    channels (the aligned terrain rasters share one NoData footprint).
    """

    finite = np.isfinite(patch)
    not_nodata = ~np.isclose(patch, nodata_value, rtol=0.0, atol=0.0)
    return (finite & not_nodata).all(axis=0).astype("float32")


def center_is_valid(mask: np.ndarray) -> bool:
    """Return True when the patch center pixel is a real (unmasked) pixel."""

    rows, cols = mask.shape
    return bool(mask[rows // 2, cols // 2] >= 0.5)


def apply_norm_and_append_mask(
    patch_raw: np.ndarray,
    mask: np.ndarray,
    channel_means: np.ndarray | None = None,
    channel_stds: np.ndarray | None = None,
    append_mask: bool = True,
) -> np.ndarray:
    """Normalize terrain, zero padded pixels, and append the mask channel.

    ``patch_raw`` is ``(C, ps, ps)`` with NoData sentinels in padded/invalid
    pixels. Terrain channels are normalized (when stats are given), then
    multiplied by ``mask`` so padded pixels become 0 (the per-channel mean in
    normalized space, the intended neutral fill). When ``append_mask`` is True the
    binary ``mask`` is concatenated as the final channel, giving ``(C + 1, ps,
    ps)``; when False the terrain-only ``(C, ps, ps)`` is returned (mask-channel
    ablation).
    """

    terrain = patch_raw
    if channel_means is not None and channel_stds is not None:
        terrain = (terrain - channel_means) / (channel_stds + 1e-6)
    terrain = terrain * mask[None, :, :]
    if not append_mask:
        return terrain.astype("float32", copy=False)
    out = np.concatenate([terrain, mask[None, :, :]], axis=0)
    return out.astype("float32", copy=False)


def _required_sample_columns(factor_names: Sequence[str]) -> list[str]:
    return ["sample_id", "x", "y", "label", "source", "cluster_id", *factor_names]


def create_patch_index(
    samples_df: pd.DataFrame,
    raster_files: Sequence[str | Path],
    patch_size: int,
    max_nodata_ratio: float = 0.0,
    factor_names: Sequence[str] | None = None,
    nodata_value: float = DEFAULT_NODATA_VALUE,
) -> pd.DataFrame:
    """Create a patch index using boundary-aware (center-only) validity.

    A patch is valid when its center pixel is real (in-bounds and not NoData).
    Boundary or local NoData pixels are kept and zero-padded at load time, with
    a valid-context mask channel appended by the dataset. ``nodata_ratio`` and
    ``edge_touching`` are recorded for information only and no longer gate
    validity; ``max_nodata_ratio`` is accepted for backward compatibility but is
    not used as the validity threshold.
    """

    _ = max_nodata_ratio  # retained for signature compatibility; center-only gating
    factor_names = list(factor_names or FACTOR_NAMES)
    missing = [
        column
        for column in _required_sample_columns(factor_names)
        if column not in samples_df.columns
    ]
    if missing:
        raise ValueError(f"Sample table is missing columns: {missing}")

    raster_files = [Path(path).resolve() for path in raster_files]
    audit = audit_raster_alignment(raster_files, expected_nodata=nodata_value)
    records: list[dict[str, object]] = []

    sources = [rasterio.open(path) for path in raster_files]
    try:
        for _, sample in samples_df.iterrows():
            row, col = xy_to_rowcol(sample["x"], sample["y"], audit.transform)
            inside_point = 0 <= row < audit.height and 0 <= col < audit.width
            window = compute_patch_window(row, col, patch_size)
            edge_touching = not is_window_inside_raster(window, audit.width, audit.height)

            nodata_ratio = 1.0
            valid_patch = False
            if inside_point:
                patch = read_boundless_patch_from_sources(sources, window, nodata_value)
                expected_shape = (len(raster_files), patch_size, patch_size)
                if patch.shape != expected_shape:
                    raise ValueError(
                        f"Patch shape mismatch for sample {sample['sample_id']}: "
                        f"got {patch.shape}, expected {expected_shape}."
                    )
                mask = valid_context_mask(patch, nodata_value)
                nodata_ratio = compute_nodata_ratio(patch, nodata_value)
                valid_patch = center_is_valid(mask)

            record = {
                "sample_id": sample["sample_id"],
                "x": sample["x"],
                "y": sample["y"],
                "label": int(sample["label"]),
                "source": sample["source"],
                "cluster_id": int(sample["cluster_id"]),
                "row": int(row),
                "col": int(col),
                "patch_size": int(patch_size),
                **window,
                "valid_patch": bool(valid_patch),
                "nodata_ratio": float(nodata_ratio),
                "edge_touching": bool(edge_touching),
            }
            for factor_name in factor_names:
                record[factor_name] = sample[factor_name]
            records.append(record)
    finally:
        for src in sources:
            src.close()

    columns = [
        "sample_id",
        "x",
        "y",
        "label",
        "source",
        "cluster_id",
        "row",
        "col",
        "patch_size",
        "half_size_top",
        "half_size_bottom",
        "half_size_left",
        "half_size_right",
        "window_row_start",
        "window_row_stop",
        "window_col_start",
        "window_col_stop",
        "valid_patch",
        "nodata_ratio",
        "edge_touching",
        *factor_names,
    ]
    return pd.DataFrame(records, columns=columns)


def save_patch_index(patch_index: pd.DataFrame, output_csv: str | Path) -> Path:
    """Save one patch index CSV."""

    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    patch_index.to_csv(output_csv, index=False)
    return output_csv


def load_patch_indices(paths_by_patch_size: dict[int, str | Path]) -> dict[int, pd.DataFrame]:
    """Load patch index CSV files keyed by patch size."""

    return {
        int(patch_size): pd.read_csv(Path(path).resolve())
        for patch_size, path in sorted(paths_by_patch_size.items())
    }


def valid_sample_ids_by_patch_size(
    patch_indices: dict[int, pd.DataFrame],
) -> dict[int, set[str]]:
    """Return valid sample IDs for each patch size."""

    valid_ids = {}
    for patch_size, patch_index in patch_indices.items():
        if "valid_patch" not in patch_index.columns:
            raise ValueError(f"Patch index ps{patch_size} is missing valid_patch.")
        valid = patch_index.loc[patch_index["valid_patch"].astype(bool)]
        valid_ids[patch_size] = set(valid["sample_id"].astype(str))
    return valid_ids


def common_valid_sample_ids(patch_indices: dict[int, pd.DataFrame]) -> set[str]:
    """Compute sample IDs valid in all provided patch index files."""

    valid_ids = valid_sample_ids_by_patch_size(patch_indices)
    if not valid_ids:
        raise ValueError("No patch indices supplied.")
    return set.intersection(*valid_ids.values())


def build_common_valid_dataframe(
    patch_indices: dict[int, pd.DataFrame],
    common_ids: set[str],
    reference_patch_size: int | None = None,
) -> pd.DataFrame:
    """Create common-valid sample metadata from one patch index file."""

    if reference_patch_size is None:
        reference_patch_size = sorted(patch_indices)[0]
    reference = patch_indices[reference_patch_size].copy()
    common = reference.loc[reference["sample_id"].astype(str).isin(common_ids)].copy()
    common = common.loc[common["valid_patch"].astype(bool)].copy()
    if common["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id found in common valid dataframe.")
    return common


def cluster_wise_balance_common_valid(
    common_valid_df: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select equal label 0/1 sample counts within each cluster."""

    selected_parts = []
    for cluster_id in sorted(common_valid_df["cluster_id"].unique()):
        cluster = common_valid_df.loc[common_valid_df["cluster_id"] == cluster_id]
        label0 = cluster.loc[cluster["label"] == 0]
        label1 = cluster.loc[cluster["label"] == 1]
        n_select = min(len(label0), len(label1))
        if n_select == 0:
            raise ValueError(
                f"Cluster {cluster_id} has no samples for one label in the common "
                "valid subset."
            )
        original_total = len(cluster)
        retained_total = 2 * n_select
        lost_fraction = 1.0 - retained_total / original_total
        if lost_fraction >= 0.25:
            print(
                f"WARNING: Cluster {cluster_id} loses {original_total - retained_total} "
                f"of {original_total} common-valid samples during label balancing."
            )

        selected_parts.append(
            label0.sample(
                n=n_select,
                replace=False,
                random_state=random_seed + int(cluster_id) * 2,
            )
        )
        selected_parts.append(
            label1.sample(
                n=n_select,
                replace=False,
                random_state=random_seed + int(cluster_id) * 2 + 1,
            )
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["cluster_id", "label", "sample_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return selected


def make_common_valid_balanced_sample_id_table(
    selected_common_valid: pd.DataFrame,
) -> pd.DataFrame:
    """Build the common selected sample ID table."""

    table = selected_common_valid[
        ["sample_id", "label", "source", "cluster_id"]
    ].copy()
    table["selected"] = True
    return table.sort_values(["cluster_id", "label", "sample_id"]).reset_index(drop=True)


def quality_check_common_balanced_outputs(
    selected_ids: set[str],
    output_patch_indices: dict[int, pd.DataFrame],
) -> None:
    """Validate common-balanced patch index outputs before saving."""

    reference_ids = None
    for patch_size, patch_index in output_patch_indices.items():
        ids = set(patch_index["sample_id"].astype(str))
        if ids != selected_ids:
            raise ValueError(
                f"Patch size {patch_size} selected IDs differ from selected table."
            )
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("Selected sample IDs are not identical across patch sizes.")
        if not patch_index["valid_patch"].astype(bool).all():
            raise ValueError(f"Patch size {patch_size} contains invalid patches.")
        # NOTE: nonzero nodata_ratio is allowed under boundary-aware padding;
        # validity is gated by the center pixel and padded pixels are flagged by
        # the appended valid-context mask channel, not rejected here.

        cluster_label_counts = pd.crosstab(patch_index["cluster_id"], patch_index["label"])
        for cluster_id, row in cluster_label_counts.iterrows():
            count0 = int(row.get(0, 0))
            count1 = int(row.get(1, 0))
            if count0 != count1:
                raise ValueError(
                    f"Patch size {patch_size}, cluster {cluster_id} is imbalanced: "
                    f"label0={count0}, label1={count1}."
                )

        label_counts = patch_index["label"].value_counts()
        if int(label_counts.get(0, 0)) != int(label_counts.get(1, 0)):
            raise ValueError(
                f"Patch size {patch_size} is not overall label balanced."
            )


def filter_patch_indices_to_selected_ids(
    patch_indices: dict[int, pd.DataFrame],
    selected_ids: set[str],
) -> dict[int, pd.DataFrame]:
    """Filter each patch index to identical selected sample IDs."""

    filtered = {}
    for patch_size, patch_index in patch_indices.items():
        part = patch_index.loc[patch_index["sample_id"].astype(str).isin(selected_ids)].copy()
        part = part.sort_values(["cluster_id", "label", "sample_id"]).reset_index(drop=True)
        filtered[patch_size] = part
    quality_check_common_balanced_outputs(selected_ids, filtered)
    return filtered


def save_common_valid_balanced_outputs(
    selected_table: pd.DataFrame,
    filtered_patch_indices: dict[int, pd.DataFrame],
    selected_table_path: str | Path,
    output_paths_by_patch_size: dict[int, str | Path],
) -> dict[str, Path]:
    """Save selected ID table and common-balanced patch index CSV files."""

    selected_ids = set(selected_table["sample_id"].astype(str))
    quality_check_common_balanced_outputs(selected_ids, filtered_patch_indices)

    selected_table_path = Path(selected_table_path).resolve()
    selected_table_path.parent.mkdir(parents=True, exist_ok=True)
    selected_table.to_csv(selected_table_path, index=False)

    saved: dict[str, Path] = {"selected_table": selected_table_path}
    for patch_size, patch_index in filtered_patch_indices.items():
        output_path = Path(output_paths_by_patch_size[patch_size]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        patch_index.to_csv(output_path, index=False)
        saved[f"ps{patch_size}"] = output_path
    return saved


class RasterPatchDataset(_TorchDatasetBase):
    """Lazy-loading terrain raster patch dataset (landcover excluded).

    The class imports PyTorch at initialization so patch index generation can run
    in geospatial-only environments. In a PyTorch environment, instances behave
    like ``torch.utils.data.Dataset`` objects and can be used by DataLoader.
    """

    def __init__(
        self,
        patch_index_csv: str | Path,
        raster_dir: str | Path,
        patch_size: int,
        factor_names: Sequence[str] | None = None,
        nodata_value: float = DEFAULT_NODATA_VALUE,
        normalize: bool = False,
        channel_means: Sequence[float] | None = None,
        channel_stds: Sequence[float] | None = None,
        return_metadata: bool = False,
        valid_only: bool = True,
        cache_in_memory: bool = False,
        with_mask: bool = True,
    ) -> None:
        if _torch is None:
            raise ModuleNotFoundError(
                "PyTorch is required to instantiate RasterPatchDataset. Install "
                "torch in the active environment before running Dataset smoke tests "
                "or model training."
            )

        super().__init__()
        self.torch = _torch
        self.patch_index_csv = Path(patch_index_csv).resolve()
        self.raster_files = list_raster_files(raster_dir)
        audit_raster_alignment(self.raster_files, expected_nodata=nodata_value)
        self.patch_size = int(patch_size)
        self.factor_names = list(factor_names or FACTOR_NAMES)
        self.nodata_value = float(nodata_value)
        self.normalize = bool(normalize)
        self.return_metadata = bool(return_metadata)
        self.cache_in_memory = bool(cache_in_memory)
        self.with_mask = bool(with_mask)

        self.index = pd.read_csv(self.patch_index_csv)
        if valid_only:
            self.index = self.index.loc[self.index["valid_patch"].astype(bool)].copy()
        self.index = self.index.reset_index(drop=True)
        self._sources = None
        self._cache = None

        if self.normalize:
            if channel_means is None or channel_stds is None:
                raise ValueError(
                    "channel_means and channel_stds must be provided when "
                    "normalize=True."
                )
            self.channel_means = np.asarray(channel_means, dtype="float32")[:, None, None]
            self.channel_stds = np.asarray(channel_stds, dtype="float32")[:, None, None]
            if len(self.channel_means) != len(self.raster_files):
                raise ValueError("channel_means length must match number of rasters.")
            if len(self.channel_stds) != len(self.raster_files):
                raise ValueError("channel_stds length must match number of rasters.")
        else:
            self.channel_means = None
            self.channel_stds = None

    def _get_sources(self):
        if self._sources is None:
            self._sources = [rasterio.open(path) for path in self.raster_files]
        return self._sources

    def _read_raw_patch(self, index: int, window: dict[str, int]) -> np.ndarray:
        """Return the raw (pre-normalization) float32 patch from cache or rasters."""

        if self.cache_in_memory:
            if self._cache is None:
                self._build_cache()
            return self._cache[index]
        return read_boundless_patch_from_sources(
            self._get_sources(), window, self.nodata_value
        )

    def _build_cache(self) -> None:
        """Read every raw terrain patch once into a (N, channels, ps, ps) array.

        The cache holds the raw (pre-normalization) terrain channels with NoData
        sentinels in padded pixels; the valid-context mask is derived per item
        in ``__getitem__`` so the cache stays terrain-deep and memory-equal.
        """

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
            patches[position] = read_boundless_patch_from_sources(
                sources, window, self.nodata_value
            )
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
        try:
            patch = self._read_raw_patch(index, window)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read patch for sample_id={row['sample_id']} from "
                f"{self.patch_index_csv}."
            ) from exc

        expected_shape = (len(self.raster_files), self.patch_size, self.patch_size)
        if patch.shape != expected_shape:
            raise ValueError(
                f"Patch shape mismatch for sample_id={row['sample_id']}: got "
                f"{patch.shape}, expected {expected_shape}."
            )

        mask = valid_context_mask(patch, self.nodata_value)
        if not center_is_valid(mask):
            raise ValueError(
                f"Patch for sample_id={row['sample_id']} has an invalid (NoData) "
                "center pixel."
            )
        patch = apply_norm_and_append_mask(
            patch,
            mask,
            self.channel_means if self.normalize else None,
            self.channel_stds if self.normalize else None,
            append_mask=self.with_mask,
        )

        X = self.torch.as_tensor(patch, dtype=self.torch.float32)
        y = self.torch.tensor(int(row["label"]), dtype=self.torch.long)
        if not self.return_metadata:
            return X, y

        metadata = {
            "sample_id": row["sample_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "source": row["source"],
            "cluster_id": int(row["cluster_id"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
        }
        return X, y, metadata
