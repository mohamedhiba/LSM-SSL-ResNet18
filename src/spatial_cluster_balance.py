"""Spatial KMeans clustering and cluster-wise 1:1 sample balancing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.pu_bagging import (
    attach_valid_factor_values,
    load_landslide_points,
    read_raster_factors,
)


FACTOR_COLUMNS = [f"factor_{index:02d}" for index in range(1, 15)]


class OrdinaryKMeans:
    """Small ordinary KMeans implementation with KMeans++ initialization.

    This avoids platform-specific native-threading failures while preserving the
    KMeans behavior needed here: labels_, inertia_, cluster_centers_, and predict().
    It does not enforce balanced clusters.
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int,
        n_init: int = 20,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype="float64")
        first_index = int(rng.integers(0, n_samples))
        centers[0] = X[first_index]

        closest_distance_sq = self._squared_distances(X, centers[:1]).ravel()
        for center_index in range(1, self.n_clusters):
            total = float(closest_distance_sq.sum())
            if total <= 0:
                selected_index = int(rng.integers(0, n_samples))
            else:
                probabilities = closest_distance_sq / total
                selected_index = int(rng.choice(n_samples, p=probabilities))
            centers[center_index] = X[selected_index]
            new_distance_sq = self._squared_distances(
                X, centers[center_index : center_index + 1]
            ).ravel()
            closest_distance_sq = np.minimum(closest_distance_sq, new_distance_sq)
        return centers

    def fit(self, X: np.ndarray) -> "OrdinaryKMeans":
        X = np.asarray(X, dtype="float64")
        if X.ndim != 2:
            raise ValueError("KMeans input must be a 2D array.")
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} samples, found {X.shape[0]}."
            )

        rng = np.random.default_rng(self.random_state)
        best_centers = None
        best_labels = None
        best_inertia = float("inf")

        for _ in range(self.n_init):
            centers = self._init_centers(X, rng)
            labels = np.zeros(X.shape[0], dtype="int64")
            inertia = float("inf")

            for _iteration in range(self.max_iter):
                distances = self._squared_distances(X, centers)
                labels = distances.argmin(axis=1)
                inertia = float(distances[np.arange(X.shape[0]), labels].sum())

                new_centers = centers.copy()
                for cluster_id in range(self.n_clusters):
                    cluster_mask = labels == cluster_id
                    if cluster_mask.any():
                        new_centers[cluster_id] = X[cluster_mask].mean(axis=0)
                    else:
                        farthest_index = int(
                            np.argmax(distances[np.arange(X.shape[0]), labels])
                        )
                        new_centers[cluster_id] = X[farthest_index]

                center_shift = float(((centers - new_centers) ** 2).sum())
                centers = new_centers
                if center_shift <= self.tol:
                    break

            distances = self._squared_distances(X, centers)
            labels = distances.argmin(axis=1)
            inertia = float(distances[np.arange(X.shape[0]), labels].sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans model has not been fitted.")
        X = np.asarray(X, dtype="float64")
        return self._squared_distances(X, self.cluster_centers_).argmin(axis=1)


@dataclass(frozen=True)
class ClusterBalanceConfig:
    """Configuration for spatial clustering and cluster-wise balancing."""

    project_root: Path = Path(".")
    landslide_points_csv: Path = Path("data/raw/samples/landslide_points.csv")
    reliable_nonlandslide_csv: Path = Path(
        "data/processed/pu_bagging/reliable_nonlandslide_points_all.csv"
    )
    cleaned_raster_dir: Path = Path("data/processed/rasters_cleaned")
    output_csv: Path = Path("data/processed/samples/final_cluster_balanced_dataset.csv")
    n_clusters: int = 5
    random_seed: int = 42
    kmeans_seed_start: int = 0
    kmeans_seed_end: int = 100
    min_landslide_per_cluster: int = 30
    max_imbalance_ratio: float = 3.0

    def resolve(self) -> "ClusterBalanceConfig":
        root = Path(self.project_root).resolve()

        def _resolve_path(path: Path) -> Path:
            path = Path(path)
            return path if path.is_absolute() else root / path

        return ClusterBalanceConfig(
            project_root=root,
            landslide_points_csv=_resolve_path(self.landslide_points_csv),
            reliable_nonlandslide_csv=_resolve_path(self.reliable_nonlandslide_csv),
            cleaned_raster_dir=_resolve_path(self.cleaned_raster_dir),
            output_csv=_resolve_path(self.output_csv),
            n_clusters=self.n_clusters,
            random_seed=self.random_seed,
            kmeans_seed_start=self.kmeans_seed_start,
            kmeans_seed_end=self.kmeans_seed_end,
            min_landslide_per_cluster=self.min_landslide_per_cluster,
            max_imbalance_ratio=self.max_imbalance_ratio,
        )


@dataclass(frozen=True)
class KMeansSearchResult:
    """Selected KMeans model and diagnostics."""

    kmeans: OrdinaryKMeans
    selected_random_state: int
    inertia: float
    cluster_counts: dict[int, int]
    min_count: int
    max_count: int
    imbalance_ratio: float
    met_constraints: bool
    diagnostics: pd.DataFrame


def resolve_landslide_points_csv(samples_dir: str | Path) -> Path:
    """Prefer landslide_points.csv, otherwise use the single landslide*.csv file."""

    samples_dir = Path(samples_dir).resolve()
    preferred = samples_dir / "landslide_points.csv"
    if preferred.exists():
        return preferred

    fallback_csvs = sorted(samples_dir.glob("landslide*.csv"))
    if len(fallback_csvs) == 1:
        warnings.warn(
            f"{preferred} was not found. Using available landslide CSV instead: "
            f"{fallback_csvs[0]}"
        )
        return fallback_csvs[0]

    raise FileNotFoundError(
        f"Expected {preferred}. Found {len(fallback_csvs)} fallback landslide*.csv "
        "files, so no unambiguous fallback can be selected."
    )


def _has_all_factor_columns(df: pd.DataFrame, factor_columns: Sequence[str]) -> bool:
    return all(column in df.columns for column in factor_columns)


def _valid_factor_mask(df: pd.DataFrame, factor_columns: Sequence[str]) -> pd.Series:
    values = df[list(factor_columns)].to_numpy(dtype="float64")
    valid = np.isfinite(values).all(axis=1)
    valid &= ~(np.isclose(values, -9999.0, rtol=0.0, atol=0.0).any(axis=1))
    return pd.Series(valid, index=df.index)


def load_landslide_samples(
    landslide_points_csv: str | Path,
    cleaned_raster_dir: str | Path,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Load landslide samples, extracting missing factor values from cleaned rasters."""

    raster_stack = read_raster_factors(cleaned_raster_dir)
    landslide_gdf = load_landslide_points(landslide_points_csv, raster_stack.crs)
    if not _has_all_factor_columns(landslide_gdf, factor_columns):
        landslide_gdf = attach_valid_factor_values(landslide_gdf, raster_stack)

    for factor_column in factor_columns:
        landslide_gdf[factor_column] = pd.to_numeric(
            landslide_gdf[factor_column], errors="coerce"
        )

    valid_mask = _valid_factor_mask(landslide_gdf, factor_columns)
    landslide = pd.DataFrame(landslide_gdf.loc[valid_mask].copy())

    if "sample_id" not in landslide.columns:
        landslide["sample_id"] = [
            f"L_{index:06d}" for index in range(1, len(landslide) + 1)
        ]
    else:
        missing_id = (
            landslide["sample_id"].isna()
            | (landslide["sample_id"].astype(str).str.len() == 0)
        )
        if missing_id.any():
            generated = [f"L_{index:06d}" for index in range(1, len(landslide) + 1)]
            landslide.loc[missing_id, "sample_id"] = np.asarray(
                generated, dtype=object
            )[missing_id.to_numpy()]

    landslide["label"] = 1
    landslide["source"] = "landslide"
    required = ["sample_id", "x", "y", "label", "source", *factor_columns]
    return landslide[required].reset_index(drop=True)


def load_reliable_nonlandslide_samples(
    reliable_nonlandslide_csv: str | Path,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Load reliable non-landslide samples from the PU-Bagging final CSV."""

    reliable_nonlandslide_csv = Path(reliable_nonlandslide_csv).resolve()
    if not reliable_nonlandslide_csv.exists():
        raise FileNotFoundError(
            f"Reliable non-landslide CSV not found: {reliable_nonlandslide_csv}"
        )

    negatives = pd.read_csv(reliable_nonlandslide_csv)
    missing = [
        column
        for column in ["x", "y", *factor_columns]
        if column not in negatives.columns
    ]
    if missing:
        raise ValueError(f"Reliable non-landslide CSV is missing columns: {missing}")

    if "sample_id" not in negatives.columns:
        negatives["sample_id"] = [
            f"N_{index:06d}" for index in range(1, len(negatives) + 1)
        ]
    else:
        missing_id = (
            negatives["sample_id"].isna()
            | (negatives["sample_id"].astype(str).str.len() == 0)
        )
        if missing_id.any():
            generated = [f"N_{index:06d}" for index in range(1, len(negatives) + 1)]
            negatives.loc[missing_id, "sample_id"] = np.asarray(
                generated, dtype=object
            )[missing_id.to_numpy()]

    for column in ["x", "y", *factor_columns]:
        negatives[column] = pd.to_numeric(negatives[column], errors="coerce")

    valid_mask = negatives[["x", "y"]].notna().all(axis=1)
    valid_mask &= _valid_factor_mask(negatives, factor_columns)
    negatives = negatives.loc[valid_mask].copy()
    negatives["label"] = 0
    negatives["source"] = "reliable_nonlandslide"

    required = ["sample_id", "x", "y", "label", "source", *factor_columns]
    return negatives[required].reset_index(drop=True)


def _cluster_count_dict(labels: np.ndarray, n_clusters: int) -> dict[int, int]:
    counts = np.bincount(labels, minlength=n_clusters)
    return {cluster_id: int(counts[cluster_id]) for cluster_id in range(n_clusters)}


def search_kmeans_landslide_clusters(
    landslide_samples: pd.DataFrame,
    n_clusters: int = 5,
    kmeans_seed_start: int = 0,
    kmeans_seed_end: int = 100,
    min_landslide_per_cluster: int = 30,
    max_imbalance_ratio: float = 3.0,
) -> KMeansSearchResult:
    """Fit ordinary KMeans on landslide coordinates and choose lowest inertia.

    The selected KMeans labels are used only for diagnostics. Final landslide
    cluster labels are assigned later by the capacity-constrained assignment.
    """

    if len(landslide_samples) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} landslide samples, found "
            f"{len(landslide_samples)}."
        )

    xy = landslide_samples[["x", "y"]].to_numpy(dtype="float64")
    rows: list[dict[str, object]] = []
    models: dict[int, OrdinaryKMeans] = {}

    for random_state in range(kmeans_seed_start, kmeans_seed_end + 1):
        kmeans = OrdinaryKMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(xy)
        counts = _cluster_count_dict(labels, n_clusters)
        count_values = np.asarray(list(counts.values()), dtype="int64")
        min_count = int(count_values.min())
        max_count = int(count_values.max())
        imbalance_ratio = float(max_count / min_count) if min_count else float("inf")
        rows.append(
            {
                "random_state": random_state,
                "inertia": float(kmeans.inertia_),
                "cluster_counts": counts,
                "min_count": min_count,
                "max_count": max_count,
                "imbalance_ratio": imbalance_ratio,
                "met_constraints": (
                    min_count >= min_landslide_per_cluster
                    and imbalance_ratio <= max_imbalance_ratio
                ),
            }
        )
        models[random_state] = kmeans

    diagnostics = pd.DataFrame(rows)
    selected_row = diagnostics.sort_values("inertia", ascending=True).iloc[0]

    selected_random_state = int(selected_row["random_state"])
    return KMeansSearchResult(
        kmeans=models[selected_random_state],
        selected_random_state=selected_random_state,
        inertia=float(selected_row["inertia"]),
        cluster_counts=dict(selected_row["cluster_counts"]),
        min_count=int(selected_row["min_count"]),
        max_count=int(selected_row["max_count"]),
        imbalance_ratio=float(selected_row["imbalance_ratio"]),
        met_constraints=bool(selected_row["met_constraints"]),
        diagnostics=diagnostics,
    )


def build_balanced_capacities(n_samples: int, n_clusters: int) -> list[int]:
    """Build nearly equal cluster capacities that sum to n_samples."""

    if n_samples < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} samples to create nonempty capacities; "
            f"found {n_samples}."
        )
    base = n_samples // n_clusters
    remainder = n_samples % n_clusters
    return [
        base + 1 if cluster_id < remainder else base
        for cluster_id in range(n_clusters)
    ]


def capacity_constrained_landslide_assignment(
    landslide_samples: pd.DataFrame,
    initial_centroids: np.ndarray,
    capacities: Sequence[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assign landslides to fixed-capacity centroid slots by minimum distance."""

    xy = landslide_samples[["x", "y"]].to_numpy(dtype="float64")
    capacities = list(map(int, capacities))
    n_clusters = len(capacities)
    if sum(capacities) != len(landslide_samples):
        raise ValueError(
            f"Capacity sum {sum(capacities)} must equal number of landslides "
            f"{len(landslide_samples)}."
        )

    distances = np.sqrt(((xy[:, None, :] - initial_centroids[None, :, :]) ** 2).sum(axis=2))
    slot_cluster_ids = np.repeat(np.arange(n_clusters), capacities)
    slot_cost = distances[:, slot_cluster_ids]
    row_indices, col_indices = linear_sum_assignment(slot_cost)

    assigned_clusters = np.empty(len(landslide_samples), dtype="int64")
    assigned_clusters[row_indices] = slot_cluster_ids[col_indices]

    landslides = landslide_samples.copy()
    landslides["cluster_id"] = assigned_clusters

    balanced_centroids = np.empty((n_clusters, 2), dtype="float64")
    for cluster_id in range(n_clusters):
        cluster_xy = xy[assigned_clusters == cluster_id]
        if len(cluster_xy) != capacities[cluster_id]:
            raise RuntimeError(
                f"Cluster {cluster_id} assignment size {len(cluster_xy)} does not "
                f"match target capacity {capacities[cluster_id]}."
            )
        balanced_centroids[cluster_id] = cluster_xy.mean(axis=0)

    return landslides, balanced_centroids


def assign_negatives_to_balanced_centroids(
    reliable_negatives: pd.DataFrame,
    balanced_centroids: np.ndarray,
) -> pd.DataFrame:
    """Assign reliable negatives to nearest balanced landslide-cluster centroid."""

    negatives = reliable_negatives.copy()
    xy = negatives[["x", "y"]].to_numpy(dtype="float64")
    distances = ((xy[:, None, :] - balanced_centroids[None, :, :]) ** 2).sum(axis=2)
    negatives["cluster_id"] = distances.argmin(axis=1).astype("int64")
    return negatives


def assign_clusters(
    landslide_samples: pd.DataFrame,
    reliable_negatives: pd.DataFrame,
    kmeans: OrdinaryKMeans,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign landslides from fitted labels and negatives by nearest KMeans center."""

    landslides = landslide_samples.copy()
    negatives = reliable_negatives.copy()
    landslides["cluster_id"] = kmeans.labels_.astype("int64")
    negatives["cluster_id"] = kmeans.predict(
        negatives[["x", "y"]].to_numpy(dtype="float64")
    ).astype("int64")
    return landslides, negatives


def cluster_wise_balance(
    clustered_landslides: pd.DataFrame,
    clustered_negatives: pd.DataFrame,
    n_clusters: int = 5,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one reliable negative per landslide within each spatial cluster."""

    selected_negative_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for cluster_id in range(n_clusters):
        landslide_cluster = clustered_landslides.loc[
            clustered_landslides["cluster_id"] == cluster_id
        ]
        negative_cluster = clustered_negatives.loc[
            clustered_negatives["cluster_id"] == cluster_id
        ]
        n_landslide = len(landslide_cluster)
        n_available = len(negative_cluster)
        n_select = min(n_landslide, n_available)
        if n_available < n_landslide:
            warnings.warn(
                f"Cluster {cluster_id} has insufficient reliable negatives: "
                f"available={n_available}, landslides={n_landslide}. Selecting all "
                "available negatives."
            )

        if n_select > 0:
            selected = negative_cluster.sample(
                n=n_select,
                replace=False,
                random_state=random_seed + cluster_id,
            )
            selected_negative_parts.append(selected)

        ratio = n_landslide / n_select if n_select else float("inf")
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "n_landslide": n_landslide,
                "n_reliable_negative_available": n_available,
                "n_negative_selected": n_select,
                "n_total_final": n_landslide + n_select,
                "positive_negative_ratio": ratio,
            }
        )

    selected_negatives = (
        pd.concat(selected_negative_parts, ignore_index=True)
        if selected_negative_parts
        else clustered_negatives.iloc[0:0].copy()
    )
    balance_summary = pd.DataFrame(summary_rows)
    return selected_negatives, balance_summary


def quality_check_final_dataset(
    final_dataset: pd.DataFrame,
    n_clusters: int,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
) -> None:
    """Validate final cluster-balanced labeled sample dataset before saving."""

    required_columns = [
        "sample_id",
        "x",
        "y",
        "label",
        "source",
        "cluster_id",
        *factor_columns,
    ]
    missing = [column for column in required_columns if column not in final_dataset]
    if missing:
        raise ValueError(f"Final dataset is missing columns: {missing}")

    if final_dataset["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found in final dataset.")
    if not set(final_dataset["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")
    if not set(final_dataset["source"].unique()).issubset(
        {"landslide", "reliable_nonlandslide"}
    ):
        raise ValueError(
            "source values must contain only landslide and reliable_nonlandslide."
        )

    expected_clusters = set(range(n_clusters))
    cluster_values = set(final_dataset["cluster_id"].astype(int).unique())
    if not cluster_values.issubset(expected_clusters):
        raise ValueError(
            f"cluster_id values must be integers from 0 to {n_clusters - 1}."
        )
    if final_dataset[["x", "y"]].isna().any().any():
        raise ValueError("Missing x or y values found in final dataset.")

    factor_values = final_dataset[list(factor_columns)].to_numpy(dtype="float64")
    if not np.isfinite(factor_values).all():
        raise ValueError("NaN or infinite factor values found in final dataset.")
    if np.isclose(factor_values, -9999.0, rtol=0.0, atol=0.0).any():
        raise ValueError("-9999 factor values found in final dataset.")

    for cluster_id in range(n_clusters):
        cluster = final_dataset.loc[final_dataset["cluster_id"] == cluster_id]
        n_positive = int((cluster["label"] == 1).sum())
        n_negative = int((cluster["label"] == 0).sum())
        if n_positive < 1 or n_negative < 1:
            raise ValueError(
                f"Cluster {cluster_id} must have at least one landslide and one "
                f"selected reliable non-landslide sample. Found positives="
                f"{n_positive}, negatives={n_negative}."
            )
        if n_positive != n_negative:
            warnings.warn(
                f"Cluster {cluster_id} is not 1:1 balanced: positives="
                f"{n_positive}, negatives={n_negative}."
            )

    positive_counts = (
        final_dataset.loc[final_dataset["label"] == 1]
        .groupby("cluster_id")
        .size()
        .reindex(range(n_clusters), fill_value=0)
    )
    if int(positive_counts.max() - positive_counts.min()) > 1:
        raise ValueError(
            "Landslide cluster counts are not approximately equal. Counts: "
            f"{positive_counts.to_dict()}"
        )


def build_final_cluster_balanced_dataset(
    clustered_landslides: pd.DataFrame,
    selected_negatives: pd.DataFrame,
    n_clusters: int,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Combine, sort, and validate landslides plus selected reliable negatives."""

    output_columns = [
        "sample_id",
        "x",
        "y",
        "label",
        "source",
        "cluster_id",
        *factor_columns,
    ]
    final_dataset = pd.concat(
        [clustered_landslides[output_columns], selected_negatives[output_columns]],
        ignore_index=True,
    )
    final_dataset["cluster_id"] = final_dataset["cluster_id"].astype("int64")
    final_dataset = final_dataset.sort_values(
        ["cluster_id", "label", "sample_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    quality_check_final_dataset(final_dataset, n_clusters, factor_columns)

    total_positive = int((final_dataset["label"] == 1).sum())
    total_negative = int((final_dataset["label"] == 0).sum())
    if total_positive != total_negative:
        warnings.warn(
            "Final dataset is not perfectly 1:1 balanced overall: "
            f"positives={total_positive}, negatives={total_negative}."
        )
    return final_dataset


def save_final_dataset(
    final_dataset: pd.DataFrame,
    output_csv: str | Path,
    n_clusters: int,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
) -> Path:
    """Save the only final CSV artifact for this preparation step."""

    output_csv = Path(output_csv).resolve()
    quality_check_final_dataset(final_dataset, n_clusters, factor_columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(output_csv, index=False)
    return output_csv


def run_spatial_cluster_balancing(config: ClusterBalanceConfig) -> dict[str, object]:
    """Run capacity-constrained KMeans assignment and cluster-wise balancing."""

    config = config.resolve()
    landslides = load_landslide_samples(
        config.landslide_points_csv,
        config.cleaned_raster_dir,
        FACTOR_COLUMNS,
    )
    reliable_negatives = load_reliable_nonlandslide_samples(
        config.reliable_nonlandslide_csv,
        FACTOR_COLUMNS,
    )
    search_result = search_kmeans_landslide_clusters(
        landslide_samples=landslides,
        n_clusters=config.n_clusters,
        kmeans_seed_start=config.kmeans_seed_start,
        kmeans_seed_end=config.kmeans_seed_end,
        min_landslide_per_cluster=config.min_landslide_per_cluster,
        max_imbalance_ratio=config.max_imbalance_ratio,
    )
    capacities = build_balanced_capacities(len(landslides), config.n_clusters)
    clustered_landslides, balanced_centroids = capacity_constrained_landslide_assignment(
        landslides,
        search_result.kmeans.cluster_centers_,
        capacities,
    )
    clustered_negatives = assign_negatives_to_balanced_centroids(
        reliable_negatives,
        balanced_centroids,
    )
    selected_negatives, balance_summary = cluster_wise_balance(
        clustered_landslides,
        clustered_negatives,
        n_clusters=config.n_clusters,
        random_seed=config.random_seed,
    )
    final_dataset = build_final_cluster_balanced_dataset(
        clustered_landslides,
        selected_negatives,
        n_clusters=config.n_clusters,
        factor_columns=FACTOR_COLUMNS,
    )
    output_csv = save_final_dataset(
        final_dataset,
        config.output_csv,
        config.n_clusters,
        FACTOR_COLUMNS,
    )

    total_landslide = int((final_dataset["label"] == 1).sum())
    total_negative = int((final_dataset["label"] == 0).sum())
    final_landslide_counts = (
        clustered_landslides.groupby("cluster_id")
        .size()
        .reindex(range(config.n_clusters), fill_value=0)
        .astype(int)
        .to_dict()
    )
    final_count_values = np.asarray(list(final_landslide_counts.values()), dtype="int64")
    final_imbalance_ratio = float(final_count_values.max() / final_count_values.min())
    return {
        "clustering_method": "capacity-constrained KMeans assignment",
        "landslides": landslides,
        "reliable_negatives": reliable_negatives,
        "search_result": search_result,
        "target_capacities": capacities,
        "balanced_centroids": balanced_centroids,
        "clustered_landslides": clustered_landslides,
        "clustered_negatives": clustered_negatives,
        "selected_negatives": selected_negatives,
        "balance_summary": balance_summary,
        "final_dataset": final_dataset,
        "output_csv": output_csv,
        "summary": {
            "clustering_method": "capacity-constrained KMeans assignment",
            "n_landslide": len(landslides),
            "target_capacities": capacities,
            "final_landslide_counts_by_cluster": final_landslide_counts,
            "final_imbalance_ratio": final_imbalance_ratio,
            "total_landslide_samples": total_landslide,
            "total_selected_nonlandslide_samples": total_negative,
            "total_samples": len(final_dataset),
            "number_of_clusters": config.n_clusters,
            "output_path": str(output_csv),
        },
    }
