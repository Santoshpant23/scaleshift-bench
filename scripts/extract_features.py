#!/usr/bin/env python
"""Extract per-example FM features from chip imagery.

For every example (cropland polygon + non-cropland negative sample), crop a
square patch around the example's centroid from the chip's S2 GeoTIFF, run
it through each of the 4 FMs, and cache the pooled feature vector.

Output:
    data/features/features_meta.parquet           # one row per example
    data/features/features_<fm>.npy               # [N, D_fm] per FM

Performance notes:
    - Each chip is opened ONCE per FM (not once per example), to avoid
      thrashing rasterio + GDAL on the same file.
    - GDAL emits a benign "Photometric/ExtraSamples mismatch" warning for
      every GEE-exported TIFF; we silence it at startup. The pixel data
      reads correctly.
    - AnySat's forward pass is ~1.5 s/chip (vs ~50 ms for the others) due
      to its internal token expansion. Skip it via --fms ... to defer.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# Silence the GDAL "Photometric/ExtraSamples mismatch" stream for GEE TIFFs.
# Must happen before rasterio is imported in any worker.
os.environ.setdefault("CPL_LOG_ERRORS", "OFF")
os.environ.setdefault("CPL_DEBUG", "OFF")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Photometric.*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

from scaleshift.data.chip import Chip, S2_BAND_ORDER  # noqa: E402
from scaleshift.data.labels import read_manifest  # noqa: E402
from scaleshift.model_zoo import get_model  # noqa: E402
from scaleshift.utils.logging import banner, get_logger  # noqa: E402


log = get_logger("features")

DEFAULT_MANIFEST = Path("data/chips/manifest_terai_starter.jsonl")
DEFAULT_POLYGONS = Path("data/labels/polygons_terai_starter.parquet")
DEFAULT_NEGATIVES = Path("data/labels/negatives_terai_starter.parquet")
DEFAULT_OUT_DIR = Path("data/features")

FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]
PATCH_SIZE_PX = 128


def crop_patch_from_src(
    src: rasterio.io.DatasetReader, row: int, col: int, size_px: int
) -> np.ndarray:
    """Crop a [bands, size_px, size_px] window from an already-open raster.

    Edge handling: clip to image bounds, then pad to size_px with zeros.
    """
    half = size_px // 2
    r0 = max(0, row - half)
    c0 = max(0, col - half)
    r1 = min(src.height, row + half)
    c1 = min(src.width, col + half)
    window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
    arr = src.read(window=window).astype(np.float32) / 10_000.0
    bands, h, w = arr.shape
    if h == size_px and w == size_px:
        return arr
    out = np.zeros((bands, size_px, size_px), dtype=np.float32)
    pad_top = max(0, half - row)
    pad_left = max(0, half - col)
    out[:, pad_top:pad_top + h, pad_left:pad_left + w] = arr
    return out


def lonlat_to_pixel_from_src(
    src: rasterio.io.DatasetReader, lon: float, lat: float
) -> tuple[int, int]:
    xs, ys = warp_transform("EPSG:4326", str(src.crs), [lon], [lat])
    row, col = src.index(xs[0], ys[0])
    return int(row), int(col)


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


def build_examples(manifest, polygons, negatives) -> pd.DataFrame:
    chip_lookup = {e.chip_id: e for e in manifest}

    pos_rows = []
    for _, p in polygons.iterrows():
        if p["chip_id"] not in chip_lookup:
            continue
        pos_rows.append({
            "example_id": p["polygon_id"], "chip_id": p["chip_id"],
            "district": p["district"], "label": 1, "size_bin": p["size_bin"],
            "center_lon": p["centroid_lon"], "center_lat": p["centroid_lat"],
            "area_m2": float(p["area_m2"]),
        })
    neg_rows = []
    for _, n in negatives.iterrows():
        if n["chip_id"] not in chip_lookup:
            continue
        neg_rows.append({
            "example_id": n["negative_id"], "chip_id": n["chip_id"],
            "district": n["district"], "label": 0, "size_bin": "non_crop",
            "center_lon": n["center_lon"], "center_lat": n["center_lat"],
            "area_m2": np.nan,
        })
    df = pd.concat([pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)], ignore_index=True)
    df["row_idx"] = range(len(df))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--polygons", type=Path, default=DEFAULT_POLYGONS)
    p.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fms", nargs="*", default=FM_NAMES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    manifest = read_manifest(args.manifest)

    import geopandas as gpd
    polygons = gpd.read_parquet(args.polygons)
    negatives = pd.read_parquet(args.negatives)

    examples = build_examples(manifest, polygons, negatives)
    if args.limit:
        examples = examples.head(args.limit).reset_index(drop=True)
        examples["row_idx"] = range(len(examples))
    banner(f"Extracting features for {len(examples)} examples across {len(args.fms)} FMs")
    log.info("  positives=%d  negatives=%d",
             int((examples.label == 1).sum()), int((examples.label == 0).sum()))

    chip_lookup = {e.chip_id: e for e in manifest}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(args.out_dir / "features_meta.parquet")

    # Group examples by chip so each chip is opened once per FM.
    by_chip: dict[str, list[int]] = defaultdict(list)
    for idx, row in examples.iterrows():
        by_chip[row["chip_id"]].append(int(idx))

    for fm_name in args.fms:
        log.info("Loading FM: %s", fm_name)
        t_load = time.perf_counter()
        fm = get_model(fm_name, device=args.device)
        fm.load()
        log.info("  loaded in %.1fs", time.perf_counter() - t_load)
        n_bands = 12

        # Discover feature dim with the first example.
        first_row = examples.iloc[0]
        first_entry = chip_lookup[first_row["chip_id"]]
        with rasterio.open(first_entry.s2_path) as src:
            r, c = lonlat_to_pixel_from_src(src, first_row["center_lon"], first_row["center_lat"])
            patch = crop_patch_from_src(src, r, c, args.patch_size_px)
            gsd_m = abs(src.transform.a)
        chip = make_chip_from_patch(patch, first_row["center_lat"],
                                    first_row["center_lon"], gsd_m, n_bands)
        with torch.no_grad():
            out = fm.predict(chip)
        feat_dim = int(out.features.shape[-1])
        log.info("  feature dim = %d", feat_dim)

        feats = np.zeros((len(examples), feat_dim), dtype=np.float32)
        feats[first_row["row_idx"]] = out.features.cpu().numpy().reshape(-1)

        # Per-FM inference loop: open each chip ONCE, iterate its examples.
        done = 1
        failures = 0
        t_fm = time.perf_counter()
        for chip_id, idxs in by_chip.items():
            entry = chip_lookup.get(chip_id)
            if entry is None:
                continue
            with rasterio.open(entry.s2_path) as src:
                gsd_m = abs(src.transform.a)
                for idx in idxs:
                    if idx == first_row["row_idx"]:
                        continue
                    row = examples.iloc[idx]
                    try:
                        r, c = lonlat_to_pixel_from_src(src, row["center_lon"], row["center_lat"])
                        patch = crop_patch_from_src(src, r, c, args.patch_size_px)
                        chip = make_chip_from_patch(patch, row["center_lat"],
                                                    row["center_lon"], gsd_m, n_bands)
                        with torch.no_grad():
                            out = fm.predict(chip)
                        feats[idx] = out.features.cpu().numpy().reshape(-1)
                    except Exception as e:
                        log.warning("[%s] %s failed: %s", fm_name, row["example_id"], e)
                        failures += 1
                    done += 1
                    if done % 200 == 0:
                        rate = done / max(time.perf_counter() - t_fm, 1e-6)
                        eta = (len(examples) - done) / max(rate, 1e-6)
                        log.info("  [%s] %d/%d (%.1f ex/s, eta %.0fs)",
                                 fm_name, done, len(examples), rate, eta)

        out_path = args.out_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"
        np.save(out_path, feats)
        log.info("[%s] saved %s shape=%s failures=%d elapsed=%.0fs",
                 fm_name, out_path, feats.shape, failures, time.perf_counter() - t_fm)
        del fm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    banner("Feature extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
