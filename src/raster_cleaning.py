"""Raster NoData cleaning utilities for PU-Bagging preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata
from scipy import ndimage


COMMON_NODATA = -9999
CATEGORICAL_KEYWORDS = (
    "landcover",
    "land_cover",
    "lulc",
    "lithology",
    "geology",
    "class",
    "category",
    "categorical",
)


@dataclass(frozen=True)
class RasterAudit:
    """Common raster grid metadata."""

    crs: object
    transform: object
    width: int
    height: int
    resolution: tuple[float, float]
    bounds: object


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def find_raster_files(raster_dir: str | Path) -> list[Path]:
    """Find the 14 raw conditioning factor rasters in alphabetic order."""

    raster_dir = _as_path(raster_dir)
    raster_files = sorted(raster_dir.glob("*.tif"), key=lambda path: path.name.lower())
    if len(raster_files) != 14:
        raise ValueError(
            f"Expected exactly 14 .tif rasters in {raster_dir}, found "
            f"{len(raster_files)}."
        )
    return raster_files


def load_study_area_boundary(
    boundary_dir: str | Path,
    target_crs: object,
) -> gpd.GeoDataFrame:
    """Load the first supported boundary file and reproject to target_crs."""

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
    if boundary.crs != target_crs:
        boundary = boundary.to_crs(target_crs)
    boundary = boundary.loc[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError(f"Study area boundary has no valid geometries: {candidates[0]}")
    return boundary


def _transform_equal(left: object, right: object, atol: float = 1e-9) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def _bounds_equal(left: object, right: object, atol: float = 1e-9) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def audit_raster_alignment(raster_files: Iterable[str | Path]) -> RasterAudit:
    """Confirm all rasters share one grid. No resampling is performed."""

    raster_files = [Path(path) for path in raster_files]
    if not raster_files:
        raise ValueError("No rasters supplied for alignment audit.")

    with rasterio.open(raster_files[0]) as reference:
        ref_crs = reference.crs
        ref_transform = reference.transform
        ref_width = reference.width
        ref_height = reference.height
        ref_resolution = reference.res
        ref_bounds = reference.bounds

    errors: list[str] = []
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
            if mismatches:
                errors.append(f"{path.name}: {', '.join(mismatches)}")

    if errors:
        raise ValueError(
            "Raster alignment audit failed against the first alphabetic raster "
            f"({raster_files[0].name}). Mismatched fields: {'; '.join(errors)}. "
            "No resampling or realignment was performed."
        )

    return RasterAudit(
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        resolution=ref_resolution,
        bounds=ref_bounds,
    )


def create_study_area_mask(
    reference_raster: str | Path,
    study_area_boundary: gpd.GeoDataFrame,
) -> np.ndarray:
    """Create a boolean raster-grid mask where True means inside study area."""

    reference_raster = _as_path(reference_raster)
    with rasterio.open(reference_raster) as src:
        boundary = study_area_boundary
        if boundary.crs != src.crs:
            boundary = boundary.to_crs(src.crs)
        geometries = [geometry for geometry in boundary.geometry if geometry is not None]
        return geometry_mask(
            geometries=geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )


def detect_raster_type(filename: str | Path) -> str:
    """Detect raster variable type: continuous, categorical, or aspect."""

    name = Path(filename).name.lower()
    if "aspect" in name:
        return "aspect"
    if any(keyword in name for keyword in CATEGORICAL_KEYWORDS):
        return "categorical"
    return "continuous"


def _invalid_data_mask(data: np.ndarray, nodata: float | int | None) -> np.ndarray:
    invalid = ~np.isfinite(data)
    if nodata is not None:
        try:
            invalid |= np.isclose(data, nodata, rtol=0.0, atol=0.0, equal_nan=True)
        except TypeError:
            invalid |= data == nodata
    return invalid


def _base_profile(src: rasterio.io.DatasetReader, dtype: str, common_nodata: int):
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=common_nodata,
        compress="deflate",
        predictor=2 if dtype.startswith("float") else 1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _cleaning_counts(data: np.ndarray, nodata: float | int | None, study_area_mask: np.ndarray):
    invalid_before = _invalid_data_mask(data, nodata) & study_area_mask
    total_inside = int(study_area_mask.sum())
    return invalid_before, total_inside


def _warning_for_large_gap(filename: str, invalid_count: int, total_inside: int) -> None:
    if total_inside == 0:
        raise ValueError("Study area mask contains no raster cells.")
    percent = 100.0 * invalid_count / total_inside
    if percent >= 25.0:
        warnings.warn(
            f"{filename} has large internal NoData coverage before cleaning: "
            f"{invalid_count} pixels ({percent:.2f}% of study area)."
        )


def clean_continuous_raster(
    input_path: str | Path,
    output_path: str | Path,
    study_area_mask: np.ndarray,
    common_nodata: int = COMMON_NODATA,
) -> dict[str, object]:
    """Fill internal continuous NoData gaps with rasterio.fill.fillnodata."""

    input_path = _as_path(input_path)
    output_path = _as_path(output_path)
    with rasterio.open(input_path) as src:
        original_nodata = src.nodata
        data = src.read(1).astype("float32", copy=False)
        invalid_before, total_inside = _cleaning_counts(data, original_nodata, study_area_mask)
        _warning_for_large_gap(input_path.name, int(invalid_before.sum()), total_inside)

        output = data.copy()
        valid_source = (~_invalid_data_mask(data, original_nodata)) & study_area_mask
        fill_targets = invalid_before

        if fill_targets.any():
            fill_input = output.copy()
            fill_input[fill_targets] = common_nodata
            filled = fillnodata(
                fill_input,
                mask=valid_source.astype("uint8"),
                max_search_distance=100,
                smoothing_iterations=0,
            ).astype("float32", copy=False)
            output[fill_targets] = filled[fill_targets]

        output[~study_area_mask] = common_nodata
        invalid_after = _invalid_data_mask(output, common_nodata) & study_area_mask
        profile = _base_profile(src, "float32", common_nodata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output.astype("float32", copy=False), 1)

    return {
        "raster": input_path.name,
        "detected_type": "continuous",
        "original_nodata": original_nodata,
        "nodata_inside_before": int(invalid_before.sum()),
        "nodata_inside_percent_before": 100.0 * int(invalid_before.sum()) / total_inside,
        "nodata_inside_after": int(invalid_after.sum()),
        "output_path": str(output_path),
        "status": "pass" if int(invalid_after.sum()) == 0 else "fail",
    }


def _smallest_integer_dtype(values: np.ndarray, common_nodata: int) -> str:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "int16"
    min_value = min(float(finite_values.min()), float(common_nodata))
    max_value = max(float(finite_values.max()), float(common_nodata))
    if np.iinfo(np.int16).min <= min_value and max_value <= np.iinfo(np.int16).max:
        return "int16"
    return "int32"


def clean_categorical_raster_nearest(
    input_path: str | Path,
    output_path: str | Path,
    study_area_mask: np.ndarray,
    common_nodata: int = COMMON_NODATA,
) -> dict[str, object]:
    """Fill internal categorical NoData gaps with nearest valid class."""

    input_path = _as_path(input_path)
    output_path = _as_path(output_path)
    with rasterio.open(input_path) as src:
        original_nodata = src.nodata
        data = src.read(1).astype("float64", copy=False)
        invalid_before, total_inside = _cleaning_counts(data, original_nodata, study_area_mask)
        _warning_for_large_gap(input_path.name, int(invalid_before.sum()), total_inside)

        valid_source = (~_invalid_data_mask(data, original_nodata)) & study_area_mask
        if not valid_source.any():
            raise ValueError(f"No valid categorical source pixels inside study area: {input_path}")

        output = data.copy()
        if invalid_before.any():
            _, indices = ndimage.distance_transform_edt(
                ~valid_source,
                return_distances=True,
                return_indices=True,
            )
            output[invalid_before] = data[tuple(indices[:, invalid_before])]

        output[~study_area_mask] = common_nodata
        output = np.rint(output)
        invalid_after = _invalid_data_mask(output, common_nodata) & study_area_mask
        dtype = _smallest_integer_dtype(output, common_nodata)
        profile = _base_profile(src, dtype, common_nodata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output.astype(dtype, copy=False), 1)

    return {
        "raster": input_path.name,
        "detected_type": "categorical",
        "original_nodata": original_nodata,
        "nodata_inside_before": int(invalid_before.sum()),
        "nodata_inside_percent_before": 100.0 * int(invalid_before.sum()) / total_inside,
        "nodata_inside_after": int(invalid_after.sum()),
        "output_path": str(output_path),
        "status": "pass" if int(invalid_after.sum()) == 0 else "fail",
    }


def clean_aspect_raster(
    input_path: str | Path,
    output_path: str | Path,
    study_area_mask: np.ndarray,
    common_nodata: int = COMMON_NODATA,
) -> dict[str, object]:
    """Fill internal aspect NoData gaps with nearest valid aspect value."""

    input_path = _as_path(input_path)
    output_path = _as_path(output_path)
    with rasterio.open(input_path) as src:
        original_nodata = src.nodata
        data = src.read(1).astype("float32", copy=False)
        invalid_before, total_inside = _cleaning_counts(data, original_nodata, study_area_mask)
        _warning_for_large_gap(input_path.name, int(invalid_before.sum()), total_inside)

        valid_source = (~_invalid_data_mask(data, original_nodata)) & study_area_mask
        if not valid_source.any():
            raise ValueError(f"No valid aspect source pixels inside study area: {input_path}")

        output = data.copy()
        if invalid_before.any():
            _, indices = ndimage.distance_transform_edt(
                ~valid_source,
                return_distances=True,
                return_indices=True,
            )
            output[invalid_before] = data[tuple(indices[:, invalid_before])]

        output[~study_area_mask] = common_nodata
        invalid_after = _invalid_data_mask(output, common_nodata) & study_area_mask
        profile = _base_profile(src, "float32", common_nodata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output.astype("float32", copy=False), 1)

    return {
        "raster": input_path.name,
        "detected_type": "aspect",
        "original_nodata": original_nodata,
        "nodata_inside_before": int(invalid_before.sum()),
        "nodata_inside_percent_before": 100.0 * int(invalid_before.sum()) / total_inside,
        "nodata_inside_after": int(invalid_after.sum()),
        "output_path": str(output_path),
        "status": "pass" if int(invalid_after.sum()) == 0 else "fail",
    }


def clean_raster_by_type(
    input_path: str | Path,
    output_dir: str | Path,
    study_area_mask: np.ndarray,
    common_nodata: int = COMMON_NODATA,
) -> dict[str, object]:
    """Clean one raster using the detected variable-specific strategy."""

    input_path = _as_path(input_path)
    output_path = _as_path(output_dir) / input_path.name
    raster_type = detect_raster_type(input_path)
    if raster_type == "categorical":
        return clean_categorical_raster_nearest(
            input_path, output_path, study_area_mask, common_nodata
        )
    if raster_type == "aspect":
        return clean_aspect_raster(input_path, output_path, study_area_mask, common_nodata)
    return clean_continuous_raster(input_path, output_path, study_area_mask, common_nodata)


def clean_all_rasters(
    raster_dir: str | Path,
    boundary_dir: str | Path,
    cleaned_dir: str | Path,
    common_nodata: int = COMMON_NODATA,
) -> pd.DataFrame:
    """Clean all 14 rasters and return a printable summary table."""

    raster_files = find_raster_files(raster_dir)
    audit = audit_raster_alignment(raster_files)
    boundary = load_study_area_boundary(boundary_dir, audit.crs)
    study_area_mask = create_study_area_mask(raster_files[0], boundary)

    summaries = []
    for raster_file in raster_files:
        summaries.append(
            clean_raster_by_type(
                raster_file,
                cleaned_dir,
                study_area_mask,
                common_nodata=common_nodata,
            )
        )

    cleaned_audit = audit_cleaned_rasters(cleaned_dir, common_nodata=common_nodata)
    summary = pd.DataFrame(summaries)
    if not cleaned_audit["passes"]:
        summary["status"] = "fail"
        raise ValueError(f"Cleaned raster audit failed: {cleaned_audit['errors']}")
    return summary


def audit_cleaned_rasters(
    cleaned_dir: str | Path,
    common_nodata: int = COMMON_NODATA,
) -> dict[str, object]:
    """Audit cleaned rasters for shared alignment and common NoData metadata."""

    cleaned_files = find_raster_files(cleaned_dir)
    errors: list[str] = []
    try:
        audit = audit_raster_alignment(cleaned_files)
    except ValueError as exc:
        audit = None
        errors.append(str(exc))

    for path in cleaned_files:
        with rasterio.open(path) as src:
            if src.nodata != common_nodata:
                errors.append(
                    f"{path.name}: nodata metadata is {src.nodata}, expected "
                    f"{common_nodata}."
                )
            data = src.read(1, masked=False)
            invalid = _invalid_data_mask(data, src.nodata)
            if not np.isfinite(data[~invalid]).all():
                errors.append(f"{path.name}: finite-data audit failed.")

    return {
        "passes": len(errors) == 0,
        "errors": errors,
        "raster_files": cleaned_files,
        "audit": audit,
    }
