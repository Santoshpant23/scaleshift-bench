#!/usr/bin/env python
"""Build a chip index for the Phase 1 starter Nepal Terai dataset.

For each pilot district, this script:
  1. Samples N random chip centers inside the district bbox (uniform).
  2. For each center, queries GEE Sentinel-2 SR Harmonized for the cleanest
     scene inside the configured season window.
  3. Records the chip's (lon, lat, scene_id, date, cloud_pct) into a JSONL
     index. No imagery is downloaded by this script - the index is the
     handoff to pull_chip_imagery.py.

Output: data/index/terai_starter.jsonl

Splitting index-building from imagery-pulling lets us:
  - Inspect the index for sanity before burning GEE quota
  - Re-pull a specific chip later by reading its row
  - Resume after a network failure
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import ee
import yaml

from scaleshift.utils.logging import banner, get_logger


log = get_logger("index")

DEFAULT_CONFIG = Path("configs/terai_districts.yaml")
DEFAULT_OUT = Path("data/index/terai_starter.jsonl")


@dataclass
class ChipIndexRow:
    district: str
    lon: float
    lat: float
    chip_size_px: int
    gsd_m: float
    scene_id: str
    scene_date: str
    cloud_pct: float
    season_window: str


def sample_centers(bbox: list[float], n: int, rng: random.Random) -> list[tuple[float, float]]:
    w, s, e, n_lat = bbox
    return [(rng.uniform(w, e), rng.uniform(s, n_lat)) for _ in range(n)]


def least_cloud_scene_at(
    lon: float, lat: float, start: str, end: str, max_cloud: float
) -> dict | None:
    pt = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(pt)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    n = col.size().getInfo()
    if n == 0:
        return None
    img = ee.Image(col.first())
    info = img.getInfo()
    props = info.get("properties", {})
    return {
        "scene_id": info.get("id", ""),
        "scene_date": str(props.get("system:time_start", "")),
        "cloud_pct": float(props.get("CLOUDY_PIXEL_PERCENTAGE", -1.0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--project", default=os.getenv("EE_PROJECT"),
                   help="GCP project ID for EE (or EE_PROJECT env var)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel GEE info requests; keep modest to avoid rate limits")
    p.add_argument("--limit-per-district", type=int, default=None,
                   help="override chips_per_district from config (smoke testing)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project:
        log.error("Earth Engine requires --project or EE_PROJECT env var.")
        return 2

    cfg = yaml.safe_load(args.config.read_text())
    starter = cfg["starter_dataset"]
    season_key = starter["season_window"]
    season = cfg["season_windows"][season_key]
    start, end = season["range"]
    max_cloud = starter["max_cloud_pct"]
    chip_size_px = starter["chip_size_px"]
    chips_per_district = args.limit_per_district or starter["chips_per_district"]
    rng = random.Random(starter["random_seed"])

    log.info("Initializing Earth Engine (project=%s)", args.project)
    ee.Initialize(project=args.project)

    banner(f"Building chip index across {len(cfg['districts'])} districts")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[ChipIndexRow] = []

    for dist_name, dist_cfg in cfg["districts"].items():
        log.info("[%s] sampling %d centers", dist_name, chips_per_district)
        centers = sample_centers(dist_cfg["bbox"], chips_per_district, rng)

        def _process(c: tuple[float, float], dist=dist_name) -> ChipIndexRow | None:
            lon, lat = c
            scene = least_cloud_scene_at(lon, lat, start, end, max_cloud)
            if scene is None:
                return None
            return ChipIndexRow(
                district=dist,
                lon=lon,
                lat=lat,
                chip_size_px=chip_size_px,
                gsd_m=10.0,
                scene_id=scene["scene_id"],
                scene_date=scene["scene_date"],
                cloud_pct=scene["cloud_pct"],
                season_window=season_key,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process, c) for c in centers]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                if row is not None:
                    rows.append(row)
                if i % 25 == 0:
                    log.info("[%s] %d/%d", dist_name, i, len(centers))
        log.info("[%s] kept %d/%d (rejected: no cloud-free scene)",
                 dist_name, sum(1 for r in rows if r.district == dist_name), len(centers))

    log.info("Writing index with %d rows -> %s", len(rows), args.out)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    banner(f"Index complete: {len(rows)} chips ready to pull")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
