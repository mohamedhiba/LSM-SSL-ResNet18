#!/usr/bin/env python
"""Stage 10 (KY): build the Kentucky dual-view indices + normalization stats.

Runs Qianyi's `generate_10m_patch_indices_and_normalization` against the KY
raster stack via the shim. Produces, under data/kentucky_dual_view/:
  - dual_view_padded_patch_index_ky10m.csv          (labeled, from kgs6c manifest)
  - unlabeled_dual_view_padded_index_ky10m_n20000.csv
  - normalization_stats_ky10m_13factors.{json,csv}  (block-wise whole-raster stats)
plus his QA tables. CPU-only data prep; no training.

Usage: .venv/bin/python scripts/build_ky_dual_view_indices.py [--unlabeled-n 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ky_dual_view_shim as shim  # noqa: E402  (must precede any src.* import)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled-n", type=int, default=20000,
                    help="SSL corpus size (Qianyi's protocol: 20000; 50k is his PU pool, not SSL)")
    args = ap.parse_args()

    from src.prepare_10m_patch_indices import generate_10m_patch_indices_and_normalization

    config = shim.ky_ten_m_patch_config(target_unlabeled_n=args.unlabeled_n)
    print(f"rasters:  {config.raster_dir}", flush=True)
    print(f"manifest: {config.source_labeled_index_csv}", flush=True)
    print(f"out dir:  {config.output_dir}", flush=True)

    result = generate_10m_patch_indices_and_normalization(config)

    print(json.dumps({k: result[k] for k in (
        "reference_crs", "reference_shape", "reference_resolution",
        "labeled_index_path", "unlabeled_index_path", "normalization_json",
        "qa_passed", "elapsed_sec")}, indent=2, default=str))
    if not result["qa_passed"]:
        sys.exit("QA FAILED — inspect the QA tables in the output dir.")
    print("KY_DUAL_VIEW_STAGE10_OK", flush=True)


if __name__ == "__main__":
    main()
