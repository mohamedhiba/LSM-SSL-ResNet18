#!/usr/bin/env python
"""Build positive patch centers + spatial folds from the KGS Kentucky inventory.

Runs the point_to_patch design (reproject -> spatial blocks -> jitter -> buffered
fold report) on the downloaded KGS July 2022 points. Produces a concrete artifact
(data/kentucky/kentucky_positive_centers.csv) and a summary, WITHOUT needing the
terrain stack yet. When Te Pei's Kentucky 10m stack lands, the same centers feed
point_to_patch.extract_patch_at.

Usage:  python scripts/build_kentucky_patch_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POINTS = PROJECT_ROOT / "data/kentucky/kgs_july2022_points.geojson"
OUT = PROJECT_ROOT / "data/kentucky/kentucky_positive_centers.csv"


def main() -> None:
    from src.point_to_patch import build_patch_centers

    if not POINTS.exists():
        raise FileNotFoundError(f"{POINTS} not found - run scripts/fetch_kgs_inventory.py first.")

    centers, report = build_patch_centers(
        POINTS, n_blocks=5, k_jitter=6, max_offset_m=175.0, seed=42,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    centers.to_csv(OUT, index=False)

    n_points = centers["point_id"].nunique()
    print(f"points: {n_points}  ->  positive patch centers: {len(centers)} "
          f"(K={len(centers)//max(n_points,1)} per point)", flush=True)
    print("\nper-fold spatial block sizes (unique points):", flush=True)
    print(centers.drop_duplicates("point_id").groupby("fold_id").size().to_string(), flush=True)
    print("\nbuffered-fold leakage report (dead_zone=640 m):", flush=True)
    print(report.to_string(index=False), flush=True)
    print(f"\nsaved positive centers -> {OUT}", flush=True)
    print("NOTE: negatives (PU-bagging, buffer-excluded, 1:2) + patch extraction "
          "activate once the Kentucky 10m terrain stack is assembled.", flush=True)


if __name__ == "__main__":
    main()
