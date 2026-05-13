#!/usr/bin/env python
"""Pull one Sentinel-2 L2A chip over Nepal's Terai for smoke testing.

Default AOI: a 2.5 km square centered on Chitwan district (84.45 E, 27.55 N).
Date window: post-monsoon 2024, lowest-cloud scene picked from the collection.

Output: tests/fixtures/terai_sample.tif (10 bands, 10 m, reflectance / 10000).

Prereqs:
    earthengine authenticate --auth_mode=notebook   # one-time
    export EE_PROJECT=<your-gcp-project-id>         # or pass --project
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ee
from scaleshift.utils.logging import get_logger


log = get_logger("download")

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
DEFAULT_OUT = Path("tests/fixtures/terai_sample.tif")


def least_cloud_scene(geom: ee.Geometry, start: str, end: str) -> ee.Image | None:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    # ee.Image(col.first()) returns a masked Image when the collection is empty,
    # not None. Resolve size on the server before constructing the Image.
    n = col.size().getInfo()
    if n == 0:
        return None
    return ee.Image(col.first()).select(S2_BANDS)


def export(image: ee.Image, geom: ee.Geometry, out: Path) -> None:
    url = image.getDownloadURL({
        "scale": 10,
        "region": geom,
        "format": "GEO_TIFF",
        "crs": "EPSG:32645",
    })
    import requests
    log.info("Downloading from %s", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    log.info("Wrote %s (%.1f MB)", out, len(r.content) / 1e6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lon", type=float, default=84.45)
    p.add_argument("--lat", type=float, default=27.55)
    p.add_argument("--size-m", type=int, default=2_560, help="square side in meters")
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--project",
        default=os.getenv("EE_PROJECT"),
        help="GCP project ID for Earth Engine (or set EE_PROJECT env var)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project:
        log.error(
            "Earth Engine requires a project ID. Pass --project <id> or set EE_PROJECT. "
            "Find yours at https://console.cloud.google.com/earth-engine"
        )
        return 2
    ee.Initialize(project=args.project)
    half = args.size_m / 2.0
    pt = ee.Geometry.Point([args.lon, args.lat])
    geom = pt.buffer(half).bounds()
    log.info("AOI: lon=%.4f lat=%.4f size=%dm", args.lon, args.lat, args.size_m)
    img = least_cloud_scene(geom, args.start, args.end)
    if img is None:
        log.error("No scene found in window %s - %s with cloud < 10%%", args.start, args.end)
        return 1
    export(img, geom, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
