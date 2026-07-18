"""Point-to-patch sampling for point-based landslide inventories (KGS Kentucky).

Implements the positional-uncertainty-aware sampling designed in
KENTUCKY_KGS_PLAN.md, for inventories whose labels are POINTS that may fall
anywhere within a slide (crown / scarp / middle / toe):

  * reproject lon/lat -> a metric CRS (UTM) so buffers/jitter are in meters,
  * spatial blocking via KMeans on coordinates (for buffered spatial CV only,
    NOT for negative selection),
  * jitter augmentation: K patches per point, each re-centered by a random offset
    within a buffer (label-preserving; any window in the buffer still contains the
    slide), and
  * patch extraction at a given center against an aligned raster stack (reuses the
    boundary-aware/valid-context-mask machinery from patch_dataset).

The sampling/blocking/jitter functions are runnable on the points alone (no raster
stack required); ``extract_patch_at`` activates once the Kentucky terrain stack
exists.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd

# Eastern Kentucky default projected CRS (UTM zone 17N, NAD83) - meters.
DEFAULT_UTM_CRS = "EPSG:26917"
WGS84 = "EPSG:4326"


def load_inventory_points(geojson_path: str | Path) -> pd.DataFrame:
    """Load a point inventory GeoJSON into a DataFrame with lon/lat + attributes."""

    fc = json.loads(Path(geojson_path).read_text())
    rows = []
    for i, feat in enumerate(fc["features"]):
        geom = feat.get("geometry")
        if not geom or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        props = feat.get("properties", {}) or {}
        rows.append({"point_id": f"KGS_{i:05d}", "lon": float(lon), "lat": float(lat), **props})
    return pd.DataFrame(rows)


def to_metric_crs(lon: np.ndarray, lat: np.ndarray, dst_crs: str = DEFAULT_UTM_CRS) -> tuple[np.ndarray, np.ndarray]:
    """Reproject lon/lat (WGS84) to a metric CRS using rasterio.warp (no pyproj)."""

    from rasterio.warp import transform

    xs, ys = transform(WGS84, dst_crs, list(map(float, lon)), list(map(float, lat)))
    return np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")


def assign_spatial_blocks(x: np.ndarray, y: np.ndarray, n_blocks: int = 5, seed: int = 42) -> np.ndarray:
    """KMeans spatial blocks on metric coordinates -> fold_id (for spatial CV only).

    NOTE: blocking only. Do NOT reuse these clusters to *select* negatives
    (Gu et al. 2024 show K-means negative selection overfits).
    """

    from sklearn.cluster import KMeans

    coords = np.column_stack([x, y])
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    return km.fit_predict(coords).astype(int)


def jitter_positive_centers(
    df_xy: pd.DataFrame,
    k: int = 6,
    max_offset_m: float = 175.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate K jittered patch centers per point within a metric buffer.

    ``df_xy`` must have columns point_id, x, y, fold_id. Each output row is one
    label-preserving positive patch center (offset drawn uniformly in a disk of
    radius ``max_offset_m``). Includes the un-jittered center as jitter_idx 0.
    """

    rng = np.random.default_rng(seed)
    out = []
    for _, r in df_xy.iterrows():
        for j in range(k):
            if j == 0:
                dx = dy = 0.0
            else:
                rad = max_offset_m * np.sqrt(rng.random())
                ang = 2.0 * np.pi * rng.random()
                dx, dy = rad * np.cos(ang), rad * np.sin(ang)
            out.append({
                "point_id": r["point_id"], "fold_id": int(r["fold_id"]),
                "jitter_idx": j, "x": float(r["x"] + dx), "y": float(r["y"] + dy),
                "label": 1,
            })
    return pd.DataFrame(out)


def fold_buffer_report(df_xy: pd.DataFrame, dead_zone_m: float = 640.0) -> pd.DataFrame:
    """Report, per fold, how many held-out points sit within ``dead_zone_m`` of a
    training point (these would need dropping/buffering to avoid patch leakage).
    """

    from scipy.spatial import cKDTree

    rows = []
    for fold in sorted(df_xy["fold_id"].unique()):
        test = df_xy[df_xy["fold_id"] == fold]
        train = df_xy[df_xy["fold_id"] != fold]
        tree = cKDTree(train[["x", "y"]].to_numpy())
        d, _ = tree.query(test[["x", "y"]].to_numpy(), k=1)
        rows.append({
            "fold": int(fold), "n_test": len(test), "n_train": len(train),
            "n_test_within_deadzone": int((d < dead_zone_m).sum()),
            "min_test_train_dist_m": round(float(d.min()), 1),
        })
    return pd.DataFrame(rows)


def extract_patch_at(
    center_x: float,
    center_y: float,
    sources,
    transform,
    patch_px: int = 64,
    nodata_value: float = -9999.0,
):
    """Extract a (C, patch_px, patch_px) boundary-aware patch + valid-context mask
    at a metric center, against an aligned raster stack. Ready for the Kentucky
    stack; reuses patch_dataset helpers. Returns (terrain_with_mask, center_valid).
    """

    from src.patch_dataset import (
        center_is_valid, compute_patch_window, read_boundless_patch_from_sources,
        valid_context_mask, xy_to_rowcol,
    )

    row, col = xy_to_rowcol(center_x, center_y, transform)
    window = compute_patch_window(row, col, patch_px)
    patch = read_boundless_patch_from_sources(sources, window, nodata_value)
    mask = valid_context_mask(patch, nodata_value)
    return patch, mask, center_is_valid(mask)


def build_patch_centers(
    geojson_path: str | Path,
    n_blocks: int = 5,
    k_jitter: int = 6,
    max_offset_m: float = 175.0,
    dst_crs: str = DEFAULT_UTM_CRS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end positive-patch-center construction from a point inventory.

    Returns (centers_df, fold_report_df). centers_df is one row per jittered
    positive patch center, grouped by point_id and fold_id (use GroupKFold by
    point_id so a point's K patches never split across folds).
    """

    pts = load_inventory_points(geojson_path)
    x, y = to_metric_crs(pts["lon"].to_numpy(), pts["lat"].to_numpy(), dst_crs)
    fold_id = assign_spatial_blocks(x, y, n_blocks=n_blocks, seed=seed)
    base = pd.DataFrame({"point_id": pts["point_id"], "x": x, "y": y, "fold_id": fold_id})
    centers = jitter_positive_centers(base, k=k_jitter, max_offset_m=max_offset_m, seed=seed)
    report = fold_buffer_report(base)
    return centers, report
