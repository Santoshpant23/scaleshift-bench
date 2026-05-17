#!/usr/bin/env python
"""Extract per-polygon features by pooling each FM's spatial token grid.

For each chip:
    1. Read the whole 256x256 S2 GeoTIFF once.
    2. Run each FM once on the full chip, get back a token grid [H_tok, W_tok, D].
    3. For every polygon in the chip, compute its bbox in chip pixel coords,
       map into the FM's token grid, and mean-pool just those tokens.
    4. For every negative point, do the same with a 16x16-px window around
       the point so positives and negatives are at comparable granularity.

For AnySat (tile-only output), every example in a chip receives the same
chip-level tile feature -- a per-FM limitation that downstream eval reads
from the wrapper's pooling_method attribute.

Output:
    data/features_per_polygon/features_meta.parquet
    data/features_per_polygon/features_<fm>.npy
    data/features_per_polygon/n_tokens_per_example.parquet
        (diagnostic: how many tokens were pooled per example per FM)
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


os.environ.setdefault("CPL_LOG_ERRORS", "OFF")
os.environ.setdefault("CPL_DEBUG", "OFF")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Photometric.*")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

from scaleshift.data.chip import Chip, S2_BAND_ORDER  # noqa: E402
from scaleshift.data.labels import read_manifest  # noqa: E402
from scaleshift.data.token_pool import (  # noqa: E402
    point_bbox_in_chip_px,
    polygon_bbox_in_chip_px,
    pool_tokens_for_bbox,
    reshape_to_grid,
)
from scaleshift.model_zoo import get_model  # noqa: E402
from scaleshift.utils.logging import banner, get_logger  # noqa: E402


log = get_logger("features-pp")

DEFAULT_MANIFEST = Path("data/chips/manifest_terai_starter.jsonl")
DEFAULT_POLYGONS = Path("data/labels/polygons_terai_starter.parquet")
DEFAULT_NEGATIVES = Path("data/labels/negatives_terai_starter.parquet")
DEFAULT_OUT_DIR = Path("data/features_per_polygon")

FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]


def chip_from_geotiff_array(s2_path: Path) -> tuple[Chip, int]:
    """Load full chip; return Chip + max(height, width) for token-scale math."""
    with rasterio.open(s2_path) as src:
        arr = src.read().astype(np.float32) / 10_000.0
        h, w = src.height, src.width
        gsd = abs(src.transform.a)
        # Lat/lon of chip center for FM encoders that consume location.
        cy, cx = h // 2, w // 2
        x, y = src.xy(cy, cx)
        from rasterio.warp import transform as warp_transform
        lons, lats = warp_transform(src.crs, "EPSG:4326", [x], [y])
        chip = Chip(
            s2=arr,
            s2_bands=list(S2_BAND_ORDER[: arr.shape[0]]),
            lat=lats[0], lon=lons[0], gsd_m=gsd,
            date=datetime(2024, 11, 1, tzinfo=timezone.utc),
        )
    return chip, max(h, w)


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
    p.add_argument("--polygons", type=Path, default=DEFAULT_POLYGONS,
                   help="positive polygons parquet")
    p.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES,
                   help="negative samples parquet")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fms", nargs="*", default=FM_NAMES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--negative-half-px", type=int, default=8,
                   help="half-size of the negative-point window in chip pixels (default 8 = 160 m)")
    p.add_argument("--pool-strategy", choices=["mean", "max", "center"], default="mean",
                   help="token aggregation: 'mean' (default), 'max', or 'center' "
                        "(single centroid token, for the size-controlled experiment)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    manifest = read_manifest(args.manifest)
    polygons = gpd.read_parquet(args.polygons)
    negatives = pd.read_parquet(args.negatives)
    poly_by_id = {p["polygon_id"]: p for _, p in polygons.iterrows()}

    examples = build_examples(manifest, polygons, negatives)
    banner(f"Per-polygon features for {len(examples)} examples across {len(args.fms)} FMs")
    log.info("  positives=%d  negatives=%d",
             int((examples.label == 1).sum()), int((examples.label == 0).sum()))

    chip_lookup = {e.chip_id: e for e in manifest}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(args.out_dir / "features_meta.parquet")

    by_chip: dict[str, list[int]] = defaultdict(list)
    for idx, row in examples.iterrows():
        by_chip[row["chip_id"]].append(int(idx))

    # Pre-compute bboxes per example (one chip-open per chip total).
    log.info("Pre-computing per-example bboxes...")
    bboxes: dict[int, tuple[int, int, int, int, int]] = {}
    for chip_id, idxs in by_chip.items():
        entry = chip_lookup[chip_id]
        for idx in idxs:
            row = examples.iloc[idx]
            if row["label"] == 1:
                poly = poly_by_id[row["example_id"]]
                r0, c0, r1, c1, h, w = polygon_bbox_in_chip_px(poly["geometry"], entry.s2_path)
            else:
                r0, c0, r1, c1, h, w = point_bbox_in_chip_px(
                    row["center_lon"], row["center_lat"], entry.s2_path,
                    half_px=args.negative_half_px,
                )
            bboxes[idx] = (r0, c0, r1, c1, max(h, w))
    log.info("  done.")

    # Diagnostic: n_tokens_used per example per FM
    n_tokens_table: dict[str, np.ndarray] = {}

    for fm_name in args.fms:
        log.info("Loading FM: %s", fm_name)
        t_load = time.perf_counter()
        fm = get_model(fm_name, device=args.device)
        fm.load()
        log.info("  loaded in %.1fs", time.perf_counter() - t_load)

        is_tile_only = (fm_name == "anysat")
        feat_dim: int | None = None
        feats: np.ndarray | None = None
        n_tokens_used = np.zeros(len(examples), dtype=np.int32)

        t_fm = time.perf_counter()
        n_chips_done = 0
        for chip_id, idxs in by_chip.items():
            entry = chip_lookup[chip_id]
            chip, chip_size_px = chip_from_geotiff_array(entry.s2_path)
            try:
                with torch.no_grad():
                    out = fm.predict(chip, return_tokens=not is_tile_only)
            except Exception as e:
                log.warning("[%s] chip %s forward failed: %s; skipping all its examples",
                            fm_name, chip_id, e)
                continue

            if feat_dim is None:
                if is_tile_only or out.tokens is None:
                    feat_dim = int(out.features.shape[-1])
                else:
                    feat_dim = int(out.tokens.shape[-1])
                feats = np.zeros((len(examples), feat_dim), dtype=np.float32)
                log.info("  feature dim = %d  is_tile_only=%s", feat_dim, is_tile_only)

            if is_tile_only or out.tokens is None:
                feat_vec = out.features.cpu().numpy().reshape(-1)
                for idx in idxs:
                    feats[idx] = feat_vec
                    n_tokens_used[idx] = 1  # tile = "1 super-token"
            else:
                tokens_1d = out.tokens[0].cpu().numpy()
                tokens_2d = reshape_to_grid(tokens_1d)
                if tokens_2d is None:
                    log.warning("[%s] token count %d is not a perfect square; falling back to tile",
                                fm_name, tokens_1d.shape[0])
                    feat_vec = out.features.cpu().numpy().reshape(-1)
                    for idx in idxs:
                        feats[idx] = feat_vec
                        n_tokens_used[idx] = 1
                else:
                    for idx in idxs:
                        bbox = bboxes[idx]
                        chip_size = bbox[4]
                        pooled, n_used = pool_tokens_for_bbox(
                            tokens_2d,
                            bbox[:4],
                            chip_size_px=chip_size,
                            fm_input_size_px=fm.default_input_size_px,
                            fm_patch_size_px=fm.patch_size_px or 1,
                            strategy=args.pool_strategy,
                        )
                        feats[idx] = pooled
                        n_tokens_used[idx] = n_used

            n_chips_done += 1
            if n_chips_done % 10 == 0:
                rate = n_chips_done / max(time.perf_counter() - t_fm, 1e-6)
                eta = (len(by_chip) - n_chips_done) / max(rate, 1e-6)
                log.info("  [%s] %d/%d chips  (%.2f chips/s  eta %.0fs)",
                         fm_name, n_chips_done, len(by_chip), rate, eta)

        out_path = args.out_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"
        if feats is None:
            log.error("[%s] no chips processed", fm_name)
        else:
            np.save(out_path, feats)
            log.info("[%s] saved %s shape=%s  elapsed=%.0fs",
                     fm_name, out_path, feats.shape, time.perf_counter() - t_fm)
        n_tokens_table[fm_name] = n_tokens_used

        del fm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Merge with existing diagnostic so a partial re-run (e.g. --fms anysat
    # only) does not wipe the columns of FMs from a previous run.
    new_df = examples[["row_idx", "example_id", "label", "size_bin", "district"]].copy()
    for fm_name, arr in n_tokens_table.items():
        new_df[f"n_tokens_{fm_name}"] = arr

    diag_path = args.out_dir / "n_tokens_per_example.parquet"
    if diag_path.exists():
        existing = pd.read_parquet(diag_path)
        new_cols = set(new_df.columns)
        extra_cols = [c for c in existing.columns
                      if c.startswith("n_tokens_") and c not in new_cols]
        if extra_cols:
            new_df = new_df.merge(
                existing[["example_id"] + extra_cols], on="example_id", how="left"
            )
            log.info("Preserved %d existing FM column(s) in the diagnostic: %s",
                     len(extra_cols), extra_cols)
    new_df.to_parquet(diag_path)
    log.info("Wrote per-example token-count diagnostics to %s", diag_path)

    banner("Per-polygon feature extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
