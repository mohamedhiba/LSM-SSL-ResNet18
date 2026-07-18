#!/usr/bin/env python
"""Stage 30 (KY): corrected PU mean-OOB negatives -> balanced 1:1 KY labeled set.

Mirrors Qianyi's `prepare_pu_mean_oob_protocol` exactly, except the positive
count is KY's (his code hardcodes NYC's 281). Reuses his heavy machinery
verbatim via the shim: `generate_pu_candidates` (50k valid centers, 90 m
positive buffer), `run_pu_bagging_scores` (RF 50 iters x 100 trees, OOB-scored),
`_add_patch_metadata`, `create_flat_cv_split` (StratifiedKFold(5, shuffle,
random_state=42) on label). Negative rule: mean_oob_landslide_probability <= 0.5
ONLY (vote ratio diagnostic). CPU-only data prep; no training.

Inputs (from stage 10 = scripts/build_ky_dual_view_indices.py):
  data/kentucky_dual_view/dual_view_padded_patch_index_ky10m.csv
Outputs (data/kentucky_dual_view/):
  dual_view_padded_patch_index_ky10m_pu_mean_oob_balanced.csv   (1:1, n = 2 x n_pos)
  ky_flat_cv_split.csv + ky_flat_cv_split_qa.csv
  pu_candidates_scored_ky10m.csv                                 (all 50k, with mean-OOB)

Usage: .venv/bin/python scripts/build_ky_pu_labeled.py [--max-positives 0]
  --max-positives N > 0 subsamples positives first (seeded) to bound RF cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ky_dual_view_shim as shim  # noqa: E402  (must precede any src.* import)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positives", type=int, default=0,
                    help="0 = use all KY positives; N>0 subsamples (seed 42) to bound RF cost")
    args = ap.parse_args()

    from src.patch_dataset import FINAL_10M_FACTOR_NAMES, list_raster_files_10m
    import src.final_pu_mean_oob_protocol as pu

    out_dir = shim.KY_OUT_DIR
    labeled_index_csv = out_dir / "dual_view_padded_patch_index_ky10m.csv"
    if not labeled_index_csv.exists():
        sys.exit(f"Run scripts/build_ky_dual_view_indices.py first ({labeled_index_csv} missing).")

    config = pu.PUMeanOOBProtocolConfig(
        project_root=shim.REPO_ROOT,
        raster_dir=shim.KY_RASTER_DIR,
        positive_source_index=labeled_index_csv,
        output_root=out_dir / "pu_mean_oob",
        figure_root=out_dir / "pu_mean_oob" / "figures",
        checkpoint_root=out_dir / "pu_mean_oob" / "checkpoints",
        normalization_stats=out_dir / "normalization_stats_ky10m_13factors.json",
    ).resolve()
    raster_files = list_raster_files_10m(config.raster_dir)

    # --- positives: his _load_positive_samples minus the ==281 assert ------------
    index = pd.read_csv(config.positive_source_index)
    positives = index.loc[index["label"].astype(int).eq(1)].copy()
    if args.max_positives and len(positives) > args.max_positives:
        positives = positives.sample(n=args.max_positives, random_state=42)
    positives = positives.reset_index(drop=True)
    positives["sample_id"] = positives["sample_id"].astype(str)
    positives["legacy_scv_cluster_id"] = positives["cluster_id"].astype(int)
    positives["cluster_id"] = positives["legacy_scv_cluster_id"]
    n_pos = len(positives)
    print(f"KY positives: {n_pos}", flush=True)

    sources = [rasterio.open(p) for p in raster_files]
    try:
        vals = []
        for row in positives.itertuples(index=False):
            ok, v = pu._read_center_values(int(row.row_10m), int(row.col_10m), sources)
            if not ok:
                raise ValueError(f"Positive center invalid in KY rasters: {row.sample_id}")
            vals.append(v)
    finally:
        for s in sources:
            s.close()
    vals = np.asarray(vals, dtype="float64")
    for i, name in enumerate(FINAL_10M_FACTOR_NAMES):
        positives[name] = vals[:, i]

    # --- his machinery, unmodified ------------------------------------------------
    candidates = pu.generate_pu_candidates(config, positives, raster_files)
    scored = pu.run_pu_bagging_scores(config, positives, candidates)
    scored.to_csv(out_dir / "pu_candidates_scored_ky10m.csv", index=False)

    # --- balanced index: his build_mean_oob_labeled_index with 281 -> n_pos -------
    positive_centers = set(zip(positives["row_10m"].astype(int), positives["col_10m"].astype(int)))
    pool = scored.loc[scored["mean_oob_landslide_probability"].astype(float).le(config.mean_oob_threshold)]
    pool = pool.loc[pool["center_valid"].astype(bool)]
    pool = pool.loc[~pool[["row_10m", "col_10m"]].apply(tuple, axis=1).isin(positive_centers)]
    pool = pool.drop_duplicates(["row_10m", "col_10m"]).reset_index(drop=True)
    print(f"mean-OOB<= {config.mean_oob_threshold} negative pool: {len(pool)}", flush=True)
    if len(pool) < n_pos:
        raise RuntimeError(f"Negative pool {len(pool)} < n_pos {n_pos}; raise --n-candidates or threshold.")
    selected = pool.sample(n=n_pos, replace=False, random_state=config.random_seed)
    selected = selected.sort_values("candidate_id").reset_index(drop=True)

    pos_rows = positives.copy()
    pos_rows["source"] = "landslide"
    pos_rows["notes"] = "positive retained from KY center-valid landslide set"
    pos_rows["pu_candidate_id"] = ""
    pos_rows["mean_oob_landslide_probability"] = np.nan
    pos_rows["negative_vote_ratio"] = np.nan
    pos_rows["negative_vote_ratio_used_for_selection"] = False
    pos_rows["row"] = pos_rows["row_10m"].astype(int)
    pos_rows["col"] = pos_rows["col_10m"].astype(int)

    neg_rows = selected.copy()
    neg_rows["sample_id"] = [f"NKY_PU_MEANOOB_{i:06d}" for i in range(1, len(neg_rows) + 1)]
    neg_rows["label"] = 0
    neg_rows["cluster_id"] = -1
    neg_rows["legacy_scv_cluster_id"] = -1
    neg_rows["source"] = "pu_mean_oob_random_negative"
    neg_rows["notes"] = "random negative after mean_oob_landslide_probability <= 0.5; vote ratio not used"
    neg_rows["pu_candidate_id"] = neg_rows["candidate_id"].astype(str)
    neg_rows["row"] = neg_rows["row_10m"].astype(int)
    neg_rows["col"] = neg_rows["col_10m"].astype(int)

    keep = ["sample_id", "label", "cluster_id", "legacy_scv_cluster_id", "x", "y",
            "row_10m", "col_10m", "row", "col", "source", "notes", "pu_candidate_id",
            "mean_oob_landslide_probability", "negative_vote_ratio",
            "negative_vote_ratio_used_for_selection"]
    combined = pd.concat([pos_rows[keep], neg_rows[keep]], ignore_index=True)
    combined = pu._add_patch_metadata(config, combined)
    combined = combined.sort_values(["label", "sample_id"]).reset_index(drop=True)
    conflicts = combined.groupby(["row_10m", "col_10m"])["label"].nunique()
    if int((conflicts > 1).sum()) or combined.duplicated(["row_10m", "col_10m"]).any():
        raise RuntimeError("Duplicate or label-conflict centers in KY mean-OOB labeled index.")

    balanced_csv = out_dir / "dual_view_padded_patch_index_ky10m_pu_mean_oob_balanced.csv"
    combined.to_csv(balanced_csv, index=False)

    # --- group-safe flat CV -------------------------------------------------------
    # KY positives are KGS inventory records; ~30% share an original mapped feature
    # (polygon -> up to 3 centroid cells). Plain StratifiedKFold can put cells of the
    # SAME landslide in train and test. Group folds by original_feature_id so all
    # cells of one feature stay together; negatives are singleton groups.
    from scipy.spatial import cKDTree
    from sklearn.model_selection import StratifiedGroupKFold

    v4 = pd.read_csv(shim.KY_RASTER_DIR.parent / "samples" /
                     "kentucky_kgs_v4_six_county_modeling_candidates_deduplicated.csv",
                     low_memory=False)
    parents = v4[["row_10m", "col_10m", "x", "y", "original_feature_id"]].drop_duplicates(
        ["row_10m", "col_10m"])
    combined = combined.merge(parents[["row_10m", "col_10m", "original_feature_id"]],
                              on=["row_10m", "col_10m"], how="left")
    pos_mask = combined["label"].astype(int).eq(1)
    unmatched = pos_mask & combined["original_feature_id"].isna()
    if unmatched.any():
        tree = cKDTree(parents[["x", "y"]].to_numpy())
        dist, nearest = tree.query(combined.loc[unmatched, ["x", "y"]].to_numpy(),
                                   distance_upper_bound=50.0)
        feat = parents["original_feature_id"].to_numpy()
        vals = [feat[j] if np.isfinite(d) else None for d, j in zip(dist, nearest)]
        combined.loc[unmatched, "original_feature_id"] = vals
    combined["cv_group"] = np.where(
        pos_mask & combined["original_feature_id"].notna(),
        "FEAT_" + combined["original_feature_id"].astype(str),
        combined["sample_id"].astype(str))
    print(f"grouped positives: {int((pos_mask & combined['original_feature_id'].notna()).sum())}"
          f"/{n_pos} mapped to {combined.loc[pos_mask, 'cv_group'].nunique()} features", flush=True)

    split = combined[["sample_id", "label", "cluster_id", "legacy_scv_cluster_id",
                      "x", "y", "row_10m", "col_10m", "source", "cv_group"]].copy()
    split["fold_id"] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
    for fold_id, (_, eval_idx) in enumerate(
            sgkf.split(split, split["label"], groups=split["cv_group"])):
        split.loc[eval_idx, "fold_id"] = int(fold_id)

    qa_rows = []
    for fold_id in range(5):
        fold = split.loc[split["fold_id"].eq(fold_id)]
        labels = fold["label"].value_counts().to_dict()
        qa_rows.append({"fold_id": fold_id, "n_samples": int(len(fold)),
                        "n_pos": int(labels.get(1, 0)), "n_neg": int(labels.get(0, 0))})
    qa = pd.DataFrame(qa_rows)
    grp_folds = split.loc[split["label"].eq(1)].groupby("cv_group")["fold_id"].nunique()
    qa["group_leak_count"] = int((grp_folds > 1).sum())
    qa["overall_qa_passed"] = bool(
        len(split) == 2 * n_pos
        and split["label"].value_counts().sort_index().to_dict() == {0: n_pos, 1: n_pos}
        and int((grp_folds > 1).sum()) == 0
        and int(split["sample_id"].duplicated().sum()) == 0
    )
    split.to_csv(out_dir / "ky_flat_cv_split.csv", index=False)
    qa.to_csv(out_dir / "ky_flat_cv_split_qa.csv", index=False)
    print(f"balanced index: {balanced_csv}  n={len(combined)} ({n_pos}+{n_pos})", flush=True)
    print(f"flat-CV split QA passed: {bool(qa['overall_qa_passed'].iloc[0])}", flush=True)
    print("KY_PU_MEAN_OOB_OK", flush=True)


if __name__ == "__main__":
    main()
