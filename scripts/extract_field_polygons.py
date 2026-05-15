#!/usr/bin/env python
"""Extract field-like polygons from per-chip WorldCover cropland masks.

For each chip, the cropland mask is:
  1. Morphologically opened (3x3) to drop noise pixels.
  2. Connected-component labeled (8-connectivity).
  3. Vectorized via rasterio.features.shapes.
  4. Reprojected to a local UTM zone for accurate area in m^2.
  5. Filtered: keep polygons with area in [100 m^2, 2 km^2].

Output: data/labels/polygons_terai_starter.parquet (geoparquet)

Important caveat documented for the paper:
    WorldCover 2021 at 10 m resolution does not resolve bunds between
    smallholder fields. The polygons here are "contiguous cropland
    regions", not individual fields. The resulting size distribution
    will skew larger than the true smallholder field-size distribution.
    For the final benchmark, replace WorldCover polygons with RicePAL
    or JECAM field-level boundaries (Phase 1.5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform_geom
from scipy import ndimage as ndi
from shapely.geometry import shape

from scaleshift.data.labels import (
    FieldSizeBin,
    POLYGON_COLUMNS,
    WORLDCOVER_CROPLAND_CODE,
    read_manifest,
)
from scaleshift.utils.logging import banner, get_logger


log = get_logger("polygons")

MIN_AREA_M2 = 100.0          # drop noise polygons
MAX_AREA_M2 = 2_000_000.0    # drop region-spanning blobs (2 km^2)


def utm_epsg_for_lon(lon: float) -> str:
    """Pick UTM 44N or 45N for Terai chip centers."""
    zone = 44 if lon < 84.0 else 45
    return f"EPSG:326{zone:02d}"


def extract_for_chip(
    worldcover_path: Path,
    chip_id: str,
    district: str,
    lon: float,
) -> list[dict]:
    """Read one WorldCover GeoTIFF and return a list of polygon records."""
    with rasterio.open(worldcover_path) as src:
        classes = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

    binary = (classes == WORLDCOVER_CROPLAND_CODE).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # 3x3 opening to drop isolated noise pixels.
    opened = ndi.binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)

    # Connected components (8-connectivity).
    labels, n_comp = ndi.label(opened, structure=np.ones((3, 3)))
    if n_comp == 0:
        return []

    target_crs = utm_epsg_for_lon(lon)

    records: list[dict] = []
    pid = 0
    for geom, val in features.shapes(labels, mask=opened.astype(bool), transform=src_transform):
        if val == 0:
            continue
        # Reproject from source raster CRS to local UTM for accurate area.
        utm_geom_dict = transform_geom(str(src_crs), target_crs, geom)
        utm_geom = shape(utm_geom_dict)
        area_m2 = float(utm_geom.area)
        if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
            continue
        # Centroid in WGS84 for joining back to chip metadata.
        wgs_geom_dict = transform_geom(str(src_crs), "EPSG:4326", geom)
        wgs_geom = shape(wgs_geom_dict)
        cx, cy = wgs_geom.centroid.x, wgs_geom.centroid.y
        records.append({
            "chip_id": chip_id,
            "district": district,
            "polygon_id": f"{chip_id}_{pid}",
            "area_m2": area_m2,
            "size_bin": FieldSizeBin.from_area_m2(area_m2).value,
            "centroid_lon": cx,
            "centroid_lat": cy,
            "geometry": wgs_geom,
        })
        pid += 1

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="path to a chip manifest jsonl produced by build_chip_manifest.py")
    p.add_argument("--out", type=Path, default=Path("data/labels/polygons_terai_starter.parquet"))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        log.error("Manifest not found at %s", args.manifest)
        return 2

    manifest = read_manifest(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]
    banner(f"Extracting polygons from {len(manifest)} chips")

    all_records: list[dict] = []
    skipped_no_wc = 0
    for i, entry in enumerate(manifest, 1):
        if entry.worldcover_path is None or not Path(entry.worldcover_path).exists():
            skipped_no_wc += 1
            continue
        recs = extract_for_chip(
            Path(entry.worldcover_path),
            entry.chip_id,
            entry.region.district,
            entry.lon,
        )
        all_records.extend(recs)
        if i % 50 == 0:
            log.info("Processed %d/%d chips; running polygon count = %d",
                     i, len(manifest), len(all_records))

    log.info("Polygons extracted: %d (chips skipped for missing WorldCover: %d)",
             len(all_records), skipped_no_wc)

    if not all_records:
        log.warning("No polygons extracted; check WorldCover paths in manifest.")
        return 1

    gdf = gpd.GeoDataFrame(all_records, geometry="geometry", crs="EPSG:4326")[POLYGON_COLUMNS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.out)
    log.info("Wrote %s (%d polygons)", args.out, len(gdf))

    # Summary by size bin
    bin_counts = gdf["size_bin"].value_counts().reindex([b.value for b in FieldSizeBin.ordered()]).fillna(0).astype(int)
    banner("Polygon counts by field-size bin")
    for bin_name, count in bin_counts.items():
        log.info("  %-12s %6d", bin_name, count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
