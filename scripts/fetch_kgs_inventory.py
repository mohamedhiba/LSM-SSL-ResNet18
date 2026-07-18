#!/usr/bin/env python
"""Fetch the KGS July 2022 storm-event landslide inventory (points + polygons).

Source: Kentucky Geological Survey ArcGIS REST service (public). The July 2022
event is MapServer Layer 1 (points) and Layer 2 (areas/polygons). Pulls all
features as GeoJSON with paging (the service caps maxRecordCount), and writes
them under data/kentucky/. For research use; data is KGS/UK copyright (attribute;
"unsuitable for site-specific investigations").

Usage:  python scripts/fetch_kgs_inventory.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data/kentucky"
BASE = "https://kgs.uky.edu/arcgis/rest/services/Hazards/KYLandslideStormInventory/MapServer"
LAYERS = {"points": 1, "areas": 2}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _count(layer: int) -> int:
    q = urllib.parse.urlencode({"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return int(_get(f"{BASE}/{layer}/query?{q}").get("count", -1))


def fetch_layer(name: str, layer: int) -> Path:
    total = _count(layer)
    print(f"[{name}] layer {layer}: server reports {total} features", flush=True)
    features: list[dict] = []
    offset, page = 0, 1000
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1", "outFields": "*", "outSR": "4326",
            "resultOffset": offset, "resultRecordCount": page, "f": "geojson",
        })
        data = _get(f"{BASE}/{layer}/query?{q}")
        batch = data.get("features", [])
        features.extend(batch)
        print(f"  fetched {len(features)} / {total}", flush=True)
        if len(batch) < page or (total > 0 and len(features) >= total):
            break
        offset += page
    fc = {"type": "FeatureCollection", "features": features}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"kgs_july2022_{name}.geojson"
    out.write_text(json.dumps(fc))
    print(f"[{name}] wrote {len(features)} features -> {out}", flush=True)
    return out


def main() -> None:
    print("Fetching KGS July 2022 landslide inventory ...", flush=True)
    written = {}
    for name, layer in LAYERS.items():
        try:
            written[name] = fetch_layer(name, layer)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}", file=sys.stderr, flush=True)
    # quick provenance / schema summary for the points layer
    pts = written.get("points")
    if pts and pts.exists():
        fc = json.loads(pts.read_text())
        feats = fc["features"]
        print(f"\nPOINTS: {len(feats)} features", flush=True)
        if feats:
            props = feats[0].get("properties", {})
            print(f"attribute fields ({len(props)}): {sorted(props)}", flush=True)
            # distributions of the two label-critical fields, if present
            from collections import Counter
            for field in ("Failure_Location", "Movement_Type", "Material", "County", "Confidence"):
                vals = [f["properties"].get(field) for f in feats if field in f.get("properties", {})]
                if vals:
                    top = Counter(v for v in vals if v not in (None, "")).most_common(8)
                    print(f"  {field}: {top}", flush=True)


if __name__ == "__main__":
    main()
