"""PU-Bagging reliable non-landslide sample selection workflow.

This module intentionally writes no intermediate files. The only workflow
function that writes to disk is ``save_reliable_nonlandslides``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


DEFAULT_RUN_MODE = "standard_small"

RUN_MODE_PARAMETERS: dict[str, dict[str, object]] = {
    "quick_test": {
        "n_unlabeled_candidates": 2_000,
        "n_iterations": 20,
        "n_estimators": 50,
    },
    "standard_small": {
        "n_unlabeled_candidates": 5_000,
        "n_iterations": 50,
        "n_estimators": 100,
    },
    "full": {
        "n_unlabeled_candidates": 50_000,
        "n_iterations": 100,
        "n_estimators": 300,
    },
}


def get_run_mode_parameters(run_mode: str = DEFAULT_RUN_MODE) -> dict[str, object]:
    """Return the PU-Bagging parameter overrides for a named run mode."""

    if run_mode not in RUN_MODE_PARAMETERS:
        available = ", ".join(sorted(RUN_MODE_PARAMETERS))
        raise ValueError(f"Unknown run_mode={run_mode!r}. Available modes: {available}.")
    return dict(RUN_MODE_PARAMETERS[run_mode])


@dataclass(frozen=True)
class PUBaggingConfig:
    """Configuration for PU-Bagging reliable negative selection."""

    project_root: Path = Path(".")
    run_mode: str = DEFAULT_RUN_MODE
    raster_dir: Path = Path("data/processed/rasters_cleaned")
    landslide_points_csv: Path = Path("data/raw/samples/landslide_points.csv")
    samples_dir: Path = Path("data/raw/samples")
    boundary_dir: Path = Path("data/raw/boundary")
    output_csv: Path = Path(
        "data/processed/pu_bagging/reliable_nonlandslide_points_all.csv"
    )
    n_unlabeled_candidates: int = 5_000
    n_iterations: int = 50
    temp_negative_ratio: float = 1.0
    tau: float = 0.5
    reliable_vote_ratio_threshold: float = 0.5
    landslide_point_buffer_m: float = 90.0
    random_seed: int = 42
    n_estimators: int = 100
    max_depth: int | None = None
    n_jobs: int = -1
    max_attempts: int = 1_000_000
    candidate_batch_size: int = 20_000

    @classmethod
    def for_run_mode(
        cls,
        run_mode: str = DEFAULT_RUN_MODE,
        **overrides: object,
    ) -> "PUBaggingConfig":
        """Build a config using one named workload mode plus optional overrides."""

        parameters = get_run_mode_parameters(run_mode)
        parameters.update(overrides)
        return cls(run_mode=run_mode, **parameters)

    def resolve(self) -> "PUBaggingConfig":
        """Return a copy with relative paths resolved against project_root."""

        root = Path(self.project_root).resolve()

        def _resolve_path(path: Path) -> Path:
            path = Path(path)
            return path if path.is_absolute() else root / path

        return PUBaggingConfig(
            project_root=root,
            run_mode=self.run_mode,
            raster_dir=_resolve_path(self.raster_dir),
            landslide_points_csv=_resolve_path(self.landslide_points_csv),
            samples_dir=_resolve_path(self.samples_dir),
            boundary_dir=_resolve_path(self.boundary_dir),
            output_csv=_resolve_path(self.output_csv),
            n_unlabeled_candidates=self.n_unlabeled_candidates,
            n_iterations=self.n_iterations,
            temp_negative_ratio=self.temp_negative_ratio,
            tau=self.tau,
            reliable_vote_ratio_threshold=self.reliable_vote_ratio_threshold,
            landslide_point_buffer_m=self.landslide_point_buffer_m,
            random_seed=self.random_seed,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            max_attempts=self.max_attempts,
            candidate_batch_size=self.candidate_batch_size,
        )

    @property
    def total_tree_workload(self) -> int:
        """Total number of Random Forest trees fit across all PU iterations."""

        return int(self.n_iterations) * int(self.n_estimators)


@dataclass(frozen=True)
class RasterFactorStack:
    """Validated raster factor metadata."""

    paths: tuple[Path, ...]
    factor_names: tuple[str, ...]
    crs: object
    transform: Affine
    width: int
    height: int
    resolution: tuple[float, float]
    nodata: object
    bounds: object


@dataclass(frozen=True)
class PUBaggingResult:
    """Per-candidate PU-Bagging vote and probability statistics."""

    votes: np.ndarray
    times: np.ndarray
    probability_sum: np.ndarray
    probability_squared_sum: np.ndarray
    negative_vote_ratio: np.ndarray
    mean_oob_landslide_probability: np.ndarray
    std_oob_landslide_probability: np.ndarray


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _nodata_equal(left: object, right: object) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        return bool(np.isclose(float(left), float(right), equal_nan=True))
    except (TypeError, ValueError):
        return left == right


def _transform_equal(left: Affine, right: Affine, atol: float = 1e-9) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def read_raster_factors(raster_dir: str | Path) -> RasterFactorStack:
    """Read and validate the 14 aligned conditioning factor rasters."""

    raster_dir = _as_path(raster_dir)
    paths = tuple(sorted(raster_dir.glob("*.tif"), key=lambda path: path.name.lower()))
    if len(paths) != 14:
        raise ValueError(
            f"Expected exactly 14 .tif rasters in {raster_dir}, found {len(paths)}."
        )

    with rasterio.open(paths[0]) as reference:
        ref_crs = reference.crs
        ref_transform = reference.transform
        ref_width = reference.width
        ref_height = reference.height
        ref_resolution = reference.res
        ref_nodata = reference.nodata
        ref_bounds = reference.bounds

    alignment_errors: list[str] = []
    for path in paths[1:]:
        with rasterio.open(path) as src:
            errors: list[str] = []
            if src.crs != ref_crs:
                errors.append("CRS")
            if not _transform_equal(src.transform, ref_transform):
                errors.append("transform")
            if src.width != ref_width:
                errors.append("width")
            if src.height != ref_height:
                errors.append("height")
            if not np.allclose(src.res, ref_resolution, rtol=0.0, atol=1e-9):
                errors.append("resolution")
            if not _nodata_equal(src.nodata, ref_nodata):
                errors.append("nodata")
            if errors:
                alignment_errors.append(f"{path.name}: {', '.join(errors)}")

    if alignment_errors:
        details = "; ".join(alignment_errors)
        raise ValueError(
            "Raster alignment check failed against the first alphabetic raster "
            f"({paths[0].name}). Mismatched fields: {details}. No resampling was "
            "performed."
        )

    factor_names = tuple(f"factor_{index:02d}" for index in range(1, 15))
    return RasterFactorStack(
        paths=paths,
        factor_names=factor_names,
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        resolution=ref_resolution,
        nodata=ref_nodata,
        bounds=ref_bounds,
    )


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {str(column).lower(): str(column) for column in columns}


def _first_existing_column(columns: dict[str, str], names: Sequence[str]) -> str | None:
    for name in names:
        found = columns.get(name.lower())
        if found is not None:
            return found
    return None


def _find_xy_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    columns = _column_lookup(df.columns)
    x_col = _first_existing_column(columns, ("x",))
    y_col = _first_existing_column(columns, ("y",))
    if x_col and y_col:
        return x_col, y_col, "projected"

    lon_col = _first_existing_column(columns, ("lon", "longitude", "point_x"))
    lat_col = _first_existing_column(columns, ("lat", "latitude", "point_y"))
    if lon_col and lat_col:
        return lon_col, lat_col, "geographic"

    raise ValueError(
        "Landslide point CSV must contain x/y columns in raster CRS, or lon/lat "
        "columns such as lon/lat, longitude/latitude, or POINT_X/POINT_Y."
    )


def load_landslide_points(
    landslide_points_csv: str | Path,
    raster_crs: object,
) -> gpd.GeoDataFrame:
    """Load landslide points and return point geometries in the raster CRS."""

    landslide_points_csv = _as_path(landslide_points_csv)
    if not landslide_points_csv.exists():
        raise FileNotFoundError(
            f"Landslide point CSV not found: {landslide_points_csv}. Expected "
            "data/raw/samples/landslide_points.csv unless a different path is "
            "provided."
        )

    df = pd.read_csv(landslide_points_csv)
    x_col, y_col, coordinate_kind = _find_xy_columns(df)
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid_xy = x.notna() & y.notna()
    if not valid_xy.all():
        dropped = int((~valid_xy).sum())
        warnings.warn(f"Dropping {dropped} landslide rows with invalid coordinates.")
        df = df.loc[valid_xy].copy()
        x = x.loc[valid_xy]
        y = y.loc[valid_xy]

    if "sample_id" not in df.columns:
        df["sample_id"] = [f"L_{index:06d}" for index in range(1, len(df) + 1)]
    else:
        missing_id = df["sample_id"].isna() | (df["sample_id"].astype(str).str.len() == 0)
        if missing_id.any():
            generated = [f"L_{index:06d}" for index in range(1, len(df) + 1)]
            df.loc[missing_id, "sample_id"] = np.asarray(generated, dtype=object)[
                missing_id.to_numpy()
            ]

    df["label"] = 1

    source_crs = "EPSG:4326" if coordinate_kind == "geographic" else raster_crs
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(x.to_numpy(), y.to_numpy()),
        crs=source_crs,
    )
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def _points_inside_raster_bounds(
    coords: np.ndarray,
    bounds: object,
) -> np.ndarray:
    return (
        (coords[:, 0] >= bounds.left)
        & (coords[:, 0] <= bounds.right)
        & (coords[:, 1] >= bounds.bottom)
        & (coords[:, 1] <= bounds.top)
    )


def extract_factor_values(
    points_gdf: gpd.GeoDataFrame,
    raster_stack: RasterFactorStack,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract raster factor values and return attributes plus valid row mask."""

    if points_gdf.empty:
        empty = pd.DataFrame(columns=raster_stack.factor_names)
        return empty, np.zeros(0, dtype=bool)

    coords = np.column_stack((points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()))
    values = np.full((len(points_gdf), len(raster_stack.paths)), np.nan, dtype="float64")
    valid_mask = _points_inside_raster_bounds(coords, raster_stack.bounds)

    for factor_index, path in enumerate(raster_stack.paths):
        with rasterio.open(path) as src:
            inverse_transform = ~src.transform
            cols_float, rows_float = inverse_transform * (coords[:, 0], coords[:, 1])
            rows = np.floor(rows_float).astype("int64")
            cols = np.floor(cols_float).astype("int64")
            inside_grid = (
                (rows >= 0)
                & (rows < src.height)
                & (cols >= 0)
                & (cols < src.width)
            )
            band_values = np.full(len(points_gdf), np.nan, dtype="float64")
            if inside_grid.any():
                raster_data = src.read(1, masked=False)
                sampled_values = raster_data[rows[inside_grid], cols[inside_grid]].astype(
                    "float64",
                    copy=False,
                )
                valid_values = np.isfinite(sampled_values)
                if src.nodata is not None:
                    valid_values &= ~np.isclose(
                        sampled_values,
                        float(src.nodata),
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                inside_indices = np.flatnonzero(inside_grid)
                band_values[inside_indices[valid_values]] = sampled_values[valid_values]
            values[:, factor_index] = band_values
            valid_mask &= inside_grid

    valid_mask &= np.isfinite(values).all(axis=1)
    factors = pd.DataFrame(values, columns=raster_stack.factor_names, index=points_gdf.index)
    return factors, valid_mask


def attach_valid_factor_values(
    points_gdf: gpd.GeoDataFrame,
    raster_stack: RasterFactorStack,
) -> gpd.GeoDataFrame:
    """Attach factor values and drop rows with invalid raster values."""

    factors, valid_mask = extract_factor_values(points_gdf, raster_stack)
    valid_points = points_gdf.loc[valid_mask].copy()
    for factor_name in raster_stack.factor_names:
        valid_points[factor_name] = factors.loc[valid_mask, factor_name].to_numpy()
    return valid_points


def read_study_boundary(
    boundary_dir: str | Path,
    raster_crs: object,
) -> gpd.GeoDataFrame:
    """Read the first available supported study area boundary vector file."""

    boundary_dir = _as_path(boundary_dir)
    supported_extensions = (".shp", ".gpkg", ".geojson")
    candidates = sorted(
        (
            path
            for path in boundary_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        ),
        key=lambda path: path.name.lower(),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported boundary file found in {boundary_dir}. Supported formats: "
            ".shp, .gpkg, .geojson."
        )

    boundary = gpd.read_file(candidates[0])
    if boundary.empty:
        raise ValueError(f"Study area boundary is empty: {candidates[0]}")
    if boundary.crs is None:
        raise ValueError(f"Study area boundary has no CRS: {candidates[0]}")
    if boundary.crs != raster_crs:
        boundary = boundary.to_crs(raster_crs)
    boundary = boundary.loc[~boundary.geometry.is_empty & boundary.geometry.notna()].copy()
    if boundary.empty:
        raise ValueError(f"Study area boundary has no valid geometries: {candidates[0]}")
    return boundary


def find_landslide_polygon_file(samples_dir: str | Path) -> Path | None:
    """Find an optional landslide_polygons vector file."""

    samples_dir = _as_path(samples_dir)
    supported_extensions = (".shp", ".gpkg", ".geojson")
    candidates = sorted(
        (
            path
            for path in samples_dir.glob("landslide_polygons.*")
            if path.suffix.lower() in supported_extensions
        ),
        key=lambda path: path.name.lower(),
    )
    return candidates[0] if candidates else None


def read_landslide_polygons(
    samples_dir: str | Path,
    raster_crs: object,
) -> gpd.GeoDataFrame | None:
    """Read optional mapped landslide polygons, if available."""

    polygon_file = find_landslide_polygon_file(samples_dir)
    if polygon_file is None:
        return None

    polygons = gpd.read_file(polygon_file)
    if polygons.empty:
        return None
    if polygons.crs is None:
        raise ValueError(f"Landslide polygon file has no CRS: {polygon_file}")
    if polygons.crs != raster_crs:
        polygons = polygons.to_crs(raster_crs)
    polygons = polygons.loc[~polygons.geometry.is_empty & polygons.geometry.notna()].copy()
    return polygons if not polygons.empty else None


def _geometry_union(geometries: gpd.GeoSeries):
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def make_exclusion_geometry(
    landslide_points: gpd.GeoDataFrame,
    raster_crs: object,
    samples_dir: str | Path,
    landslide_point_buffer_m: float = 90.0,
):
    """Return landslide polygon exclusion geometry or buffered point geometry."""

    polygons = read_landslide_polygons(samples_dir, raster_crs)
    if polygons is not None:
        return _geometry_union(polygons.geometry)

    if getattr(raster_crs, "is_geographic", False):
        raise ValueError(
            "Raster CRS is geographic, so a landslide point buffer in meters cannot "
            "be applied safely. Provide landslide polygons or use a projected CRS."
        )

    if landslide_points.empty:
        return None
    buffered = landslide_points.geometry.buffer(landslide_point_buffer_m)
    return _geometry_union(buffered)


def _random_points_in_bounds(
    bounds: tuple[float, float, float, float],
    size: int,
    rng: np.random.Generator,
) -> gpd.GeoSeries:
    minx, miny, maxx, maxy = bounds
    xs = rng.uniform(minx, maxx, size=size)
    ys = rng.uniform(miny, maxy, size=size)
    return gpd.GeoSeries(gpd.points_from_xy(xs, ys))


def generate_valid_unlabeled_candidates(
    boundary: gpd.GeoDataFrame,
    landslide_points: gpd.GeoDataFrame,
    raster_stack: RasterFactorStack,
    config: PUBaggingConfig,
) -> gpd.GeoDataFrame:
    """Generate valid unlabeled candidate points and attach factor values."""

    if boundary.crs != raster_stack.crs:
        boundary = boundary.to_crs(raster_stack.crs)

    config = config.resolve()
    boundary_union = _geometry_union(boundary.geometry)
    exclusion_geometry = make_exclusion_geometry(
        landslide_points=landslide_points,
        raster_crs=raster_stack.crs,
        samples_dir=config.samples_dir,
        landslide_point_buffer_m=config.landslide_point_buffer_m,
    )

    rng = np.random.default_rng(config.random_seed)
    accepted: list[gpd.GeoDataFrame] = []
    accepted_count = 0
    attempts = 0
    batch_size = max(1, int(config.candidate_batch_size))
    target = int(config.n_unlabeled_candidates)

    while accepted_count < target and attempts < config.max_attempts:
        remaining_attempts = config.max_attempts - attempts
        draw_size = min(batch_size, remaining_attempts)
        attempts += draw_size

        candidate_geometries = _random_points_in_bounds(
            boundary_union.bounds,
            size=draw_size,
            rng=rng,
        )
        candidate_geometries = gpd.GeoSeries(candidate_geometries, crs=raster_stack.crs)
        inside_boundary = candidate_geometries.within(boundary_union)
        if exclusion_geometry is not None:
            outside_landslides = ~candidate_geometries.within(exclusion_geometry)
            geometry_mask = inside_boundary & outside_landslides
        else:
            geometry_mask = inside_boundary

        if not geometry_mask.any():
            continue

        candidate_gdf = gpd.GeoDataFrame(
            geometry=candidate_geometries.loc[geometry_mask].to_numpy(),
            crs=raster_stack.crs,
        )
        candidate_gdf["x"] = candidate_gdf.geometry.x
        candidate_gdf["y"] = candidate_gdf.geometry.y
        candidate_gdf = attach_valid_factor_values(candidate_gdf, raster_stack)
        if candidate_gdf.empty:
            continue

        needed = target - accepted_count
        if len(candidate_gdf) > needed:
            candidate_gdf = candidate_gdf.iloc[:needed].copy()
        accepted.append(candidate_gdf)
        accepted_count += len(candidate_gdf)

    if not accepted:
        raise RuntimeError(
            "No valid unlabeled candidates could be generated. Check the study area, "
            "exclusion geometry, raster validity, and max_attempts."
        )

    candidates = pd.concat(accepted, ignore_index=True)
    if len(candidates) < target:
        warnings.warn(
            f"Generated {len(candidates)} valid unlabeled candidates before reaching "
            f"max_attempts={config.max_attempts}; requested {target}."
        )

    candidates["candidate_id"] = [
        f"U_{index:06d}" for index in range(1, len(candidates) + 1)
    ]
    ordered_columns = ["candidate_id", "x", "y", "geometry", *raster_stack.factor_names]
    return gpd.GeoDataFrame(candidates[ordered_columns], geometry="geometry", crs=raster_stack.crs)


def run_pu_bagging(
    positive_points: gpd.GeoDataFrame,
    unlabeled_candidates: gpd.GeoDataFrame,
    factor_names: Sequence[str],
    config: PUBaggingConfig,
) -> PUBaggingResult:
    """Run PU-Bagging and return per-candidate vote/probability statistics."""

    config = config.resolve()
    factor_names = list(factor_names)
    if positive_points.empty:
        raise ValueError("No valid landslide positive samples are available.")
    if unlabeled_candidates.empty:
        raise ValueError("No valid unlabeled candidate samples are available.")

    X_positive = positive_points[factor_names].to_numpy(dtype="float64")
    X_unlabeled = unlabeled_candidates[factor_names].to_numpy(dtype="float64")
    if not np.isfinite(X_positive).all():
        raise ValueError("Positive sample factors contain NaN or infinite values.")
    if not np.isfinite(X_unlabeled).all():
        raise ValueError("Unlabeled candidate factors contain NaN or infinite values.")

    n_positive = X_positive.shape[0]
    n_unlabeled = X_unlabeled.shape[0]
    temp_negative_size = int(config.temp_negative_ratio * n_positive)
    if temp_negative_size < 1:
        raise ValueError(
            "Temporary negative subset size is less than 1. Increase "
            "temp_negative_ratio or provide more positive samples."
        )
    if temp_negative_size >= n_unlabeled:
        raise ValueError(
            "Temporary negative subset size must be smaller than the number of "
            f"unlabeled candidates so out-of-bag evaluation is possible. Got "
            f"temp_negative_size={temp_negative_size}, n_unlabeled={n_unlabeled}."
        )

    votes = np.zeros(n_unlabeled, dtype="int64")
    times = np.zeros(n_unlabeled, dtype="int64")
    probability_sum = np.zeros(n_unlabeled, dtype="float64")
    probability_squared_sum = np.zeros(n_unlabeled, dtype="float64")
    all_indices = np.arange(n_unlabeled)

    for iteration in range(config.n_iterations):
        iteration_seed = config.random_seed + iteration
        rng = np.random.default_rng(iteration_seed)
        temp_negative_indices = rng.choice(
            all_indices,
            size=temp_negative_size,
            replace=False,
        )
        oob_mask = np.ones(n_unlabeled, dtype=bool)
        oob_mask[temp_negative_indices] = False
        oob_indices = all_indices[oob_mask]

        X_train = np.vstack((X_positive, X_unlabeled[temp_negative_indices]))
        y_train = np.concatenate(
            (
                np.ones(n_positive, dtype="int64"),
                np.zeros(temp_negative_size, dtype="int64"),
            )
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        classifier = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_jobs=config.n_jobs,
            random_state=iteration_seed,
        )
        classifier.fit(X_train_scaled, y_train)

        class_1_matches = np.where(classifier.classes_ == 1)[0]
        if len(class_1_matches) != 1:
            raise RuntimeError(
                f"Class 1 probability column could not be identified. Classes: "
                f"{classifier.classes_}"
            )
        class_1_index = class_1_matches[0]

        X_oob_scaled = scaler.transform(X_unlabeled[oob_indices])
        proba = classifier.predict_proba(X_oob_scaled)
        p_landslide = proba[:, class_1_index]

        times[oob_indices] += 1
        probability_sum[oob_indices] += p_landslide
        probability_squared_sum[oob_indices] += p_landslide**2
        votes[oob_indices] += p_landslide < config.tau

    evaluated = times > 0
    negative_vote_ratio = np.full(n_unlabeled, np.nan, dtype="float64")
    mean_probability = np.full(n_unlabeled, np.nan, dtype="float64")
    std_probability = np.full(n_unlabeled, np.nan, dtype="float64")

    negative_vote_ratio[evaluated] = votes[evaluated] / times[evaluated]
    mean_probability[evaluated] = probability_sum[evaluated] / times[evaluated]
    variance = (
        probability_squared_sum[evaluated] / times[evaluated]
        - mean_probability[evaluated] ** 2
    )
    std_probability[evaluated] = np.sqrt(np.maximum(variance, 0.0))

    return PUBaggingResult(
        votes=votes,
        times=times,
        probability_sum=probability_sum,
        probability_squared_sum=probability_squared_sum,
        negative_vote_ratio=negative_vote_ratio,
        mean_oob_landslide_probability=mean_probability,
        std_oob_landslide_probability=std_probability,
    )


def select_reliable_nonlandslides(
    unlabeled_candidates: gpd.GeoDataFrame,
    pu_result: PUBaggingResult,
    factor_names: Sequence[str],
    reliable_vote_ratio_threshold: float = 0.5,
) -> pd.DataFrame:
    """Select reliable non-landslide points from PU-Bagging statistics."""

    factor_names = list(factor_names)
    selected_mask = (
        (pu_result.times > 0)
        & (pu_result.negative_vote_ratio > reliable_vote_ratio_threshold)
    )

    selected = unlabeled_candidates.loc[selected_mask].copy()
    selected["label"] = 0
    selected["votes"] = pu_result.votes[selected_mask]
    selected["times"] = pu_result.times[selected_mask]
    selected["negative_vote_ratio"] = pu_result.negative_vote_ratio[selected_mask]
    selected["mean_oob_landslide_probability"] = (
        pu_result.mean_oob_landslide_probability[selected_mask]
    )
    selected["std_oob_landslide_probability"] = (
        pu_result.std_oob_landslide_probability[selected_mask]
    )

    output_columns = [
        "candidate_id",
        "x",
        "y",
        "label",
        "votes",
        "times",
        "negative_vote_ratio",
        "mean_oob_landslide_probability",
        "std_oob_landslide_probability",
        *factor_names,
    ]
    selected = pd.DataFrame(selected[output_columns]).reset_index(drop=True)
    quality_check_reliable_nonlandslides(selected, factor_names)
    return selected


def quality_check_reliable_nonlandslides(
    reliable_nonlandslides: pd.DataFrame,
    factor_names: Sequence[str],
) -> None:
    """Run final quality checks before saving reliable non-landslide points."""

    factor_names = list(factor_names)
    required_columns = [
        "candidate_id",
        "x",
        "y",
        "label",
        "votes",
        "times",
        "negative_vote_ratio",
        "mean_oob_landslide_probability",
        "std_oob_landslide_probability",
        *factor_names,
    ]
    missing = [column for column in required_columns if column not in reliable_nonlandslides]
    if missing:
        raise ValueError(f"Reliable non-landslide output is missing columns: {missing}")

    if reliable_nonlandslides["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate_id values found in reliable negatives.")

    factor_values = reliable_nonlandslides[factor_names].to_numpy(dtype="float64")
    if not np.isfinite(factor_values).all():
        raise ValueError("NaN or infinite values found in factor columns.")

    if not (reliable_nonlandslides["label"] == 0).all():
        raise ValueError("All selected reliable non-landslide labels must be 0.")

    if not (reliable_nonlandslides["times"] > 0).all():
        raise ValueError("All selected reliable non-landslide points must have times > 0.")

    ratio = reliable_nonlandslides["negative_vote_ratio"]
    if not ratio.between(0, 1, inclusive="both").all():
        raise ValueError("negative_vote_ratio values must be between 0 and 1.")

    mean_probability = reliable_nonlandslides["mean_oob_landslide_probability"]
    if not mean_probability.between(0, 1, inclusive="both").all():
        raise ValueError(
            "mean_oob_landslide_probability values must be between 0 and 1."
        )


def save_reliable_nonlandslides(
    reliable_nonlandslides: pd.DataFrame,
    output_csv: str | Path,
    factor_names: Sequence[str],
) -> Path:
    """Save the one allowed final CSV file."""

    output_csv = _as_path(output_csv)
    quality_check_reliable_nonlandslides(reliable_nonlandslides, factor_names)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reliable_nonlandslides.to_csv(output_csv, index=False)
    return output_csv


def run_reliable_negative_selection(config: PUBaggingConfig) -> dict[str, object]:
    """Run the full workflow and save only the final reliable negative CSV."""

    config = config.resolve()
    raster_stack = read_raster_factors(config.raster_dir)
    landslide_points = load_landslide_points(config.landslide_points_csv, raster_stack.crs)
    valid_landslide_points = attach_valid_factor_values(landslide_points, raster_stack)
    boundary = read_study_boundary(config.boundary_dir, raster_stack.crs)
    unlabeled_candidates = generate_valid_unlabeled_candidates(
        boundary=boundary,
        landslide_points=valid_landslide_points,
        raster_stack=raster_stack,
        config=config,
    )
    pu_result = run_pu_bagging(
        positive_points=valid_landslide_points,
        unlabeled_candidates=unlabeled_candidates,
        factor_names=raster_stack.factor_names,
        config=config,
    )
    reliable_nonlandslides = select_reliable_nonlandslides(
        unlabeled_candidates=unlabeled_candidates,
        pu_result=pu_result,
        factor_names=raster_stack.factor_names,
        reliable_vote_ratio_threshold=config.reliable_vote_ratio_threshold,
    )
    output_csv = save_reliable_nonlandslides(
        reliable_nonlandslides=reliable_nonlandslides,
        output_csv=config.output_csv,
        factor_names=raster_stack.factor_names,
    )

    reliable_percentage = (
        100.0 * len(reliable_nonlandslides) / len(unlabeled_candidates)
        if len(unlabeled_candidates)
        else 0.0
    )
    return {
        "raster_stack": raster_stack,
        "valid_landslide_points": valid_landslide_points,
        "unlabeled_candidates": unlabeled_candidates,
        "pu_result": pu_result,
        "reliable_nonlandslides": reliable_nonlandslides,
        "output_csv": output_csv,
        "summary": {
            "run_mode": config.run_mode,
            "n_valid_landslide_points": len(valid_landslide_points),
            "n_valid_unlabeled_candidates": len(unlabeled_candidates),
            "candidate_to_positive_ratio": (
                len(unlabeled_candidates) / len(valid_landslide_points)
                if len(valid_landslide_points)
                else float("nan")
            ),
            "n_iterations": config.n_iterations,
            "n_estimators": config.n_estimators,
            "total_tree_workload": config.total_tree_workload,
            "n_reliable_nonlandslide_points": len(reliable_nonlandslides),
            "reliable_nonlandslide_percentage": reliable_percentage,
            "output_csv": str(output_csv),
        },
    }
