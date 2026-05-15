#!/usr/bin/env python
"""Extract per-example FM features from chip imagery.

For every example (cropland polygon + non-cropland negative sample), crop a
square patch around the example's centroid from the chip's S2 GeoTIFF, run
it through each of the 4 FMs, and cache the pooled feature vector.

Output:
    data/features/features_meta.parquet           # one row per example
    data/features/features_<fm>.npy               # [N, D_fm] per FM

The metadata and feature matrices share row ordering (the parquet's
``row_idx`` column matches the numpy index).

Why per-FM .npy instead of one big parquet:
    - Feature dims differ across FMs (Clay 1024, Prithvi 1024, TerraMind 768,
      AnySat 768). One big parquet would force list-of-floats columns or
      padding, both ugly.
    - numpy memmap loads in O(1) for downstream eval.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.warp import transform as warp_transform

from scaleshift.data.chip import Chip, S2_BAND_ORDER
from scaleshift.data.labels import FieldSizeBin, read_manifest
from scaleshift.model_zoo import get_model
from scaleshift.utils.logging import banner, get_logger


log = get_logger("features")

DEFAULT_MANIFEST = Path("data/chips/manifest_terai_starter.jsonl")
DEFAULT_POLYGONS = Path("data/labels/polygons_terai_starter.parquet")
DEFAULT_NEGATIVES = Path("data/labels/negatives_terai_starter.parquet")
DEFAULT_OUT_DIR = Path("data/features")

FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]
PATCH_SIZE_PX = 128


def lonlat_to_pixel(lon: float, lat: float, chip_path: Path) -> tuple[int, int]:
    with rasterio.open(chip_path) as src:
        xs, ys = warp_transform("EPSG:4326", str(src.crs), [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        return row, col


def crop_patch(chip_path: Path, row: int, col: int, size_px: int) -> tuple[np.ndarray, dict]:
    """Crop a [bands, size_px, size_px] window centered at (row, col).

    Edge handling: clip to image bounds, then pad to size_px with zeros so
    every patch has identical shape. Returns the patch + the chip's metadata
    (transform, crs, gsd) so a Chip can be constructed.
    """
    with rasterio.open(chip_path) as src:
        half = size_px // 2
        r0 = max(0, row - half)
        c0 = max(0, col - half)
        r1 = min(src.height, row + half)
        c1 = min(src.width, col + half)
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        arr = src.read(window=window).astype(np.float32) / 10_000.0
        crs = str(src.crs)
        gsd = abs(src.transform.a)

    bands, h, w = arr.shape
    if h != size_px or w != size_px:
        # Pad to size_px on the right/bottom (or both sides if needed).
        out = np.zeros((bands, size_px, size_px), dtype=np.float32)
        pad_top = max(0, half - row)
        pad_left = max(0, half - col)
        out[:, pad_top:pad_top + h, pad_left:pad_left + w] = arr
        arr = out
    return arr, {"crs": crs, "gsd_m": gsd}


def make_chip_from_patch(
    patch: np.ndarray, lat: float, lon: float, gsd_m: float, n_bands: int
) -> Chip:
    return Chip(
        s2=patch,
        s2_bands=list(S2_BAND_ORDER[:n_bands]),
        lat=lat,
        lon=lon,
        gsd_m=gsd_m,
        date=datetime(2024, 11, 1, tzinfo=timezone.utc),
    )


def build_examples(
    manifest: list,
    polygons: gpd.GeoDataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    """Return one DataFrame with all examples (positives + negatives)."""
    chip_lookup = {e.chip_id: e for e in manifest}

    pos_rows = []
    for _, p in polygons.iterrows():
        cid = p["chip_id"]
        if cid not in chip_lookup:
            continue
        pos_rows.append({
            "example_id": p["polygon_id"],
            "chip_id": cid,
            "district": p["district"],
            "label": 1,
            "size_bin": p["size_bin"],
            "center_lon": p["centroid_lon"],
            "center_lat": p["centroid_lat"],
            "area_m2": float(p["area_m2"]),
        })
    pos_df = pd.DataFrame(pos_rows)

    neg_rows = []
    for _, n in negatives.iterrows():
        cid = n["chip_id"]
        if cid not in chip_lookup:
            continue
        neg_rows.append({
            "example_id": n["negative_id"],
            "chip_id": cid,
            "district": n["district"],
            "label": 0,
            "size_bin": "non_crop",
            "center_lon": n["center_lon"],
            "center_lat": n["center_lat"],
            "area_m2": np.nan,
        })
    neg_df = pd.DataFrame(neg_rows)

    df = pd.concat([pos_df, neg_df], ignore_index=True)
    df["row_idx"] = range(len(df))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--polygons", type=Path, default=DEFAULT_POLYGONS)
    p.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fms", nargs="*", default=FM_NAMES,
                   help="subset of FMs to run (default: all 4)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N examples (smoke testing)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    manifest = read_manifest(args.manifest)
    polygons = gpd.read_parquet(args.polygons)
    negatives = pd.read_parquet(args.negatives)

    examples = build_examples(manifest, polygons, negatives)
    if args.limit:
        examples = examples.head(args.limit).reset_index(drop=True)
        examples["row_idx"] = range(len(examples))
    banner(f"Extracting features for {len(examples)} examples across {len(args.fms)} FMs")
    log.info("  positives=%d  negatives=%d", int((examples.label == 1).sum()), int((examples.label == 0).sum()))

    chip_lookup = {e.chip_id: e for e in manifest}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(args.out_dir / "features_meta.parquet")

    # Group examples by chip to avoid re-loading chip rasters
    by_chip: dict[str, list[int]] = defaultdict(list)
    for idx, row in examples.iterrows():
        by_chip[row["chip_id"]].append(int(idx))

    for fm_name in args.fms:
        log.info("Loading FM: %s", fm_name)
        fm = get_model(fm_name, device=args.device)
        fm.load()
        n_bands = 12  # all FMs select their own subset via Chip.select_s2_bands

        # Run a single example to learn the feature dim
        sample = examples.iloc[0]
        entry = chip_lookup[sample["chip_id"]]
        r, c = lonlat_to_pixel(sample["center_lon"], sample["center_lat"], entry.s2_path)
        patch, meta = crop_patch(entry.s2_path, r, c, args.patch_size_px)
        chip = make_chip_from_patch(patch, sample["center_lat"], sample["center_lon"],
                                    meta["gsd_m"], n_bands)
        with torch.no_grad():
            out = fm.predict(chip)
        feat_dim = int(out.features.shape[-1])
        log.info("  feature dim = %d", feat_dim)

        feats = np.zeros((len(examples), feat_dim), dtype=np.float32)
        feats[0] = out.features.cpu().numpy().reshape(-1)

        done = 1
        for chip_id, idxs in by_chip.items():
            if entry := chip_lookup.get(chip_id):
                pass
            else:
                continue
            for idx in idxs:
                if idx == 0:
                    continue
                row = examples.iloc[idx]
                r, c = lonlat_to_pixel(row["center_lon"], row["center_lat"], entry.s2_path)
                patch, meta = crop_patch(entry.s2_path, r, c, args.patch_size_px)
                chip = make_chip_from_patch(patch, row["center_lat"], row["center_lon"],
                                            meta["gsd_m"], n_bands)
                try:
                    with torch.no_grad():
                        out = fm.predict(chip)
                    feats[idx] = out.features.cpu().numpy().reshape(-1)
                except Exception as e:
                    log.warning("[%s] %s failed: %s", fm_name, row["example_id"], e)
                    # leave feats[idx] as zeros; eval will filter or NaN them
                done += 1
                if done % 200 == 0:
                    log.info("  [%s] %d/%d", fm_name, done, len(examples))

        out_path = args.out_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"
        np.save(out_path, feats)
        log.info("Saved %s shape=%s", out_path, feats.shape)
        # free GPU
        del fm
        torch.cuda.empty_cache()

    banner("Feature extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
