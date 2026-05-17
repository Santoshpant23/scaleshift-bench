#!/usr/bin/env python
"""Spectral-context purity hypothesis test.

The size-controlled experiment (center-pool) showed that for Prithvi and
TerraMind, the size effect persists even when every polygon is pooled
from a single token. The mechanism we proposed: at fixed pool size, the
spectral content WITHIN that token varies systematically with field
size -- larger fields have purer cropland signatures at their centroid,
smaller fields have more mixed/edge content.

This script tests that hypothesis directly. For each polygon and each
FM:
  - locate the FM's centroid-token pixel window in the chip
  - compute NDVI (B08/red ratio) and EVI mean + standard deviation
  - join with per-polygon classification error (from predictions.parquet)
  - report:
      * does spectral SD decrease with field size? (expected yes)
      * does high spectral SD predict classification error?
        (expected yes for Prithvi/TerraMind, weaker for Clay)

Output:
    data/results/spectral_purity_per_polygon.parquet
    data/results/spectral_purity_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CPL_LOG_ERRORS", "OFF")
logging.getLogger("rasterio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Photometric.*")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform_geom  # noqa: E402
from shapely.geometry import shape as shapely_shape  # noqa: E402

from scaleshift.data.labels import FieldSizeBin, read_manifest  # noqa: E402
from scaleshift.utils.logging import banner, get_logger  # noqa: E402


log = get_logger("spectral")

DEFAULT_MANIFEST = Path("data/chips/manifest_terai_starter.jsonl")
DEFAULT_POLYGONS = Path("data/labels/polygons_terai_starter.parquet")
DEFAULT_PREDS = Path("data/features_per_polygon/predictions.parquet")
DEFAULT_OUT_PARQUET = Path("data/results/spectral_purity_per_polygon.parquet")
DEFAULT_OUT_JSON = Path("data/results/spectral_purity_analysis.json")

# In our 12-band S2_BAND_ORDER (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12):
IDX_BLUE = 1   # B02
IDX_RED = 3    # B04
IDX_NIR = 7    # B08

# Window size (in chip pixels) per FM. Approximates the FM's centroid-token
# spatial coverage when the chip is fed in at 256 px and resized.
# Clay: 8 px tokens on 256 input from 256 chip -> 8 chip pixels per token.
# Prithvi/TerraMind: 16 px tokens on 224 input from 256 chip -> ~18 chip pixels.
WINDOW_PX = {
    "clay-v1": 8,
    "prithvi-eo-2.0-300m": 18,
    "terramind-v1-base": 18,
    # AnySat is tile-only; spectral context isn't well-defined per polygon
}


def polygon_centroid_chip_px(poly_wgs84, chip_path: Path) -> tuple[int, int, int]:
    with rasterio.open(chip_path) as src:
        poly_proj = transform_geom("EPSG:4326", str(src.crs),
                                    poly_wgs84.__geo_interface__)
        c = shapely_shape(poly_proj).centroid
        row, col = src.index(c.x, c.y)
        return int(row), int(col), max(src.height, src.width)


def window_metrics(arr: np.ndarray, r: int, c: int, half: int) -> dict:
    """Compute mean + sd of NDVI and EVI within a (2*half+1) x (2*half+1) window
    around (r, c). arr is [12, H, W] reflectance (0..1)."""
    h, w = arr.shape[-2:]
    r0 = max(0, r - half)
    c0 = max(0, c - half)
    r1 = min(h, r + half + 1)
    c1 = min(w, c + half + 1)
    red = arr[IDX_RED, r0:r1, c0:c1]
    nir = arr[IDX_NIR, r0:r1, c0:c1]
    blue = arr[IDX_BLUE, r0:r1, c0:c1]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    return {
        "ndvi_mean": float(ndvi.mean()),
        "ndvi_sd": float(ndvi.std()),
        "evi_mean": float(evi.mean()),
        "evi_sd": float(evi.std()),
        "n_pixels": int(ndvi.size),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--polygons", type=Path, default=DEFAULT_POLYGONS)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    p.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--classifier", default="lr",
                   help="classifier predictions to join against")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    chip_lookup = {e.chip_id: e for e in manifest}
    polygons = gpd.read_parquet(args.polygons)
    preds = pd.read_parquet(args.predictions)
    preds = preds[preds.classifier == args.classifier].copy()
    preds["correct"] = (preds.y_pred == preds.label).astype(int)

    banner(f"Computing spectral purity for {len(polygons)} polygons across {len(WINDOW_PX)} FMs")

    # Group by chip so each chip is opened once.
    by_chip: dict[str, list[int]] = defaultdict(list)
    for idx, row in polygons.iterrows():
        by_chip[row["chip_id"]].append(int(idx))

    records: list[dict] = []
    skipped = 0
    for ci, (chip_id, idxs) in enumerate(by_chip.items(), 1):
        entry = chip_lookup.get(chip_id)
        if entry is None:
            skipped += len(idxs)
            continue
        with rasterio.open(entry.s2_path) as src:
            arr = src.read().astype(np.float32) / 10_000.0
        for idx in idxs:
            poly = polygons.iloc[idx]
            r, c, _ = polygon_centroid_chip_px(poly.geometry, entry.s2_path)
            for fm_name, win in WINDOW_PX.items():
                half = win // 2
                m = window_metrics(arr, r, c, half)
                records.append({
                    "polygon_id": poly["polygon_id"],
                    "chip_id": chip_id,
                    "district": poly["district"],
                    "size_bin": poly["size_bin"],
                    "area_m2": float(poly["area_m2"]),
                    "fm_name": fm_name,
                    "window_px": int(win),
                    **m,
                })
        if ci % 50 == 0:
            log.info("Processed %d/%d chips (running records=%d)", ci, len(by_chip), len(records))

    if not records:
        log.error("no records produced (skipped=%d)", skipped)
        return 1
    spec = pd.DataFrame(records)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    spec.to_parquet(args.out_parquet)
    log.info("Wrote %s (%d rows)", args.out_parquet, len(spec))

    # --- ANALYSIS ---

    bins_ordered = [b.value for b in FieldSizeBin.ordered()]

    # Q1: does spectral SD decrease with field size?
    q1_rows = []
    for fm in WINDOW_PX:
        sub = spec[spec.fm_name == fm]
        for bin_name in bins_ordered:
            ss = sub[sub.size_bin == bin_name]
            if len(ss) == 0:
                continue
            q1_rows.append({
                "fm": fm,
                "size_bin": bin_name,
                "n": int(len(ss)),
                "ndvi_sd_mean": float(ss.ndvi_sd.mean()),
                "ndvi_sd_median": float(ss.ndvi_sd.median()),
                "evi_sd_mean": float(ss.evi_sd.mean()),
                "ndvi_mean_mean": float(ss.ndvi_mean.mean()),
            })
    q1 = pd.DataFrame(q1_rows)

    # Q2: does spectral SD predict per-polygon error?
    # Merge spec with preds (matching polygon_id and fm_name where the
    # polygon was a positive example in the test set).
    pos_preds = preds[preds.label == 1].copy()
    merged = spec.merge(
        pos_preds[["example_id", "fm_name", "correct"]],
        left_on=["polygon_id", "fm_name"], right_on=["example_id", "fm_name"],
        how="inner",
    )

    q2_rows = []
    for fm in WINDOW_PX:
        sub = merged[merged.fm_name == fm]
        if len(sub) < 50:
            continue
        # Quartile of ndvi_sd
        sub = sub.copy()
        sub["sd_quartile"] = pd.qcut(sub.ndvi_sd, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            ss = sub[sub.sd_quartile == q]
            if len(ss) == 0:
                continue
            q2_rows.append({
                "fm": fm,
                "ndvi_sd_quartile": q,
                "n": int(len(ss)),
                "recall": float(ss.correct.mean()),
                "ndvi_sd_min": float(ss.ndvi_sd.min()),
                "ndvi_sd_max": float(ss.ndvi_sd.max()),
            })
        # Point-biserial correlation between ndvi_sd and correctness.
        if len(sub) >= 50:
            from scipy.stats import pointbiserialr
            r, p = pointbiserialr(sub.correct.to_numpy(), sub.ndvi_sd.to_numpy())
            q2_rows.append({
                "fm": fm,
                "ndvi_sd_quartile": "_correlation",
                "n": int(len(sub)),
                "pointbiserial_r": float(r),
                "pointbiserial_p": float(p),
            })
    q2 = pd.DataFrame(q2_rows)

    out = {
        "classifier": args.classifier,
        "fms": list(WINDOW_PX.keys()),
        "Q1_sd_by_size_bin": q1.to_dict(orient="records"),
        "Q2_recall_by_sd_quartile": q2.to_dict(orient="records"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    banner("Q1: spectral SD vs field size (expected: SD decreases as size increases)")
    if not q1.empty:
        piv = q1.pivot_table(index="size_bin", columns="fm", values="ndvi_sd_mean")
        piv = piv.reindex(bins_ordered)
        print(piv.to_string(float_format="%.4f"))

    banner("Q2: recall vs NDVI-SD quartile (expected: Q4 < Q1 for Prithvi/TerraMind)")
    if not q2.empty:
        q2_table = q2[q2.ndvi_sd_quartile.isin(["Q1", "Q2", "Q3", "Q4"])]
        piv2 = q2_table.pivot_table(index="ndvi_sd_quartile", columns="fm", values="recall")
        print(piv2.to_string(float_format="%.3f"))
        print()
        corr_rows = q2[q2.ndvi_sd_quartile == "_correlation"]
        for _, r in corr_rows.iterrows():
            print(f"  {r.fm:20s}  pointbiserial(NDVI_SD, correct) = {r['pointbiserial_r']:+.4f}  p={r['pointbiserial_p']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
