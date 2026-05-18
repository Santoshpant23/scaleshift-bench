#!/usr/bin/env python
"""Layer-wise probe analysis for Clay v1.5 (Sprint task #39, Phase C).

For each transformer block in Clay's encoder, pool the per-block output
tokens over each polygon's bbox (same per-polygon token-pool methodology
as Phase 1), train a linear probe, and report:
  - Overall F1 / AUROC at that layer
  - Per-FieldSizeBin recall and span at that layer

Hypothesis: the size-recall span is largest at EARLY layers (where the
input patch tokenization is intact) and decreases at later layers (where
self-attention mixes context across tokens). If true, this localizes the
patch-tokenization bottleneck to specific blocks and tells us *where* in
the architecture the fix should live.

The script is Clay-only for now (Clay's patch-tokenization story is the
cleanest). Extending to Prithvi/TerraMind requires similar wrapper hooks.

Output: data/results/layerwise_clay.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CPL_LOG_ERRORS", "OFF")
logging.getLogger("rasterio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Photometric.*")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scaleshift.data.chip import Chip, S2_BAND_ORDER  # noqa: E402
from scaleshift.data.labels import FieldSizeBin, read_manifest  # noqa: E402
from scaleshift.data.token_pool import (  # noqa: E402
    point_bbox_in_chip_px,
    polygon_bbox_in_chip_px,
    pool_tokens_for_bbox,
    reshape_to_grid,
)
from scaleshift.model_zoo.clay import ClayFoundationModel  # noqa: E402
from scaleshift.utils.logging import banner, get_logger  # noqa: E402


log = get_logger("layerwise")


def chip_from_geotiff_array(s2_path: Path):
    with rasterio.open(s2_path) as src:
        arr = src.read().astype(np.float32) / 10_000.0
        h, w = src.height, src.width
        gsd = abs(src.transform.a)
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
    pos_rows, neg_rows = [], []
    for _, p in polygons.iterrows():
        if p["chip_id"] not in chip_lookup:
            continue
        pos_rows.append({
            "example_id": p["polygon_id"], "chip_id": p["chip_id"],
            "district": p["district"], "label": 1, "size_bin": p["size_bin"],
            "center_lon": p["centroid_lon"], "center_lat": p["centroid_lat"],
            "area_m2": float(p["area_m2"]),
        })
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


def build_stratum(row):
    if row["label"] == 0:
        return f"neg_{row['district']}"
    return f"pos_{row['district']}_{row['size_bin']}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("data/chips/manifest_terai_starter.jsonl"))
    p.add_argument("--polygons", type=Path, default=Path("data/labels/polygons_terai_starter.parquet"))
    p.add_argument("--negatives", type=Path, default=Path("data/labels/negatives_terai_starter.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/results/layerwise_clay.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-chips", type=int, default=None,
                   help="cap number of chips processed (smoke testing)")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--test-size", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()

    manifest = read_manifest(args.manifest)
    polygons = gpd.read_parquet(args.polygons)
    negatives = pd.read_parquet(args.negatives)
    poly_by_id = {p["polygon_id"]: p for _, p in polygons.iterrows()}

    examples = build_examples(manifest, polygons, negatives)
    banner(f"Layer-wise Clay probe on {len(examples)} examples")

    chip_lookup = {e.chip_id: e for e in manifest}
    by_chip: dict[str, list[int]] = defaultdict(list)
    for idx, row in examples.iterrows():
        by_chip[row["chip_id"]].append(int(idx))
    if args.limit_chips:
        chip_ids = list(by_chip.keys())[: args.limit_chips]
        by_chip = {k: by_chip[k] for k in chip_ids}
        keep = set(b for c in chip_ids for b in by_chip[c])
        examples = examples.iloc[sorted(keep)].reset_index(drop=True)
        examples["row_idx"] = range(len(examples))
        log.info("Limited to %d chips / %d examples", len(by_chip), len(examples))

    # Pre-compute bboxes
    log.info("Pre-computing bboxes...")
    bboxes = {}
    for chip_id, idxs in by_chip.items():
        entry = chip_lookup[chip_id]
        for idx in idxs:
            row = examples.iloc[idx]
            if row["label"] == 1:
                poly = poly_by_id[row["example_id"]]
                r0, c0, r1, c1, h, w = polygon_bbox_in_chip_px(poly["geometry"], entry.s2_path)
            else:
                r0, c0, r1, c1, h, w = point_bbox_in_chip_px(
                    row["center_lon"], row["center_lat"], entry.s2_path, half_px=8,
                )
            bboxes[idx] = (r0, c0, r1, c1, max(h, w))

    log.info("Loading Clay...")
    fm = ClayFoundationModel(device=args.device)
    fm.load()

    # Probe layer count via a dummy forward
    log.info("Probing layer count...")
    first_chip_id = next(iter(by_chip.keys()))
    first_chip, _ = chip_from_geotiff_array(chip_lookup[first_chip_id].s2_path)
    first_batch = fm.preprocess(first_chip)
    per_layer_probe = fm.encode_per_layer(first_batch)
    n_layers = len(per_layer_probe)
    log.info("Clay has %d encoder blocks", n_layers)

    # Pre-allocate one feature matrix per layer
    feat_dim = int(per_layer_probe[0].shape[-1])
    log.info("Per-block feature dim = %d", feat_dim)
    layer_feats: list[np.ndarray] = [
        np.zeros((len(examples), feat_dim), dtype=np.float32) for _ in range(n_layers)
    ]

    log.info("Extracting per-layer features for each chip...")
    n_chips_done = 0
    for chip_id, idxs in by_chip.items():
        entry = chip_lookup[chip_id]
        chip, chip_size_px = chip_from_geotiff_array(entry.s2_path)
        batch = fm.preprocess(chip)
        per_layer = fm.encode_per_layer(batch)
        for layer_idx, tokens in enumerate(per_layer):
            tokens_1d = tokens[0].cpu().numpy()
            # Clay prepends CLS at index 0; skip it for patch-tokens analysis.
            tokens_2d = reshape_to_grid(tokens_1d[1:])
            if tokens_2d is None:
                continue
            for idx in idxs:
                bbox = bboxes[idx]
                pooled, _ = pool_tokens_for_bbox(
                    tokens_2d, bbox[:4],
                    chip_size_px=bbox[4],
                    fm_input_size_px=fm.default_input_size_px,
                    fm_patch_size_px=fm.patch_size_px or 1,
                    strategy="mean",
                )
                layer_feats[layer_idx][idx] = pooled
        n_chips_done += 1
        if n_chips_done % 25 == 0:
            log.info("  processed %d/%d chips", n_chips_done, len(by_chip))

    # Train one linear probe per layer
    log.info("Training per-layer probes...")
    y_all = examples["label"].to_numpy()
    bins_all = examples["size_bin"].to_numpy()
    strata = examples.apply(build_stratum, axis=1)
    counts = strata.value_counts()
    keep_mask = strata.isin(counts[counts >= 2].index).to_numpy()
    indices = np.arange(len(examples))

    idx_tr, idx_te, y_tr, y_te, bins_tr, bins_te = train_test_split(
        indices[keep_mask], y_all[keep_mask], bins_all[keep_mask],
        test_size=args.test_size, stratify=strata[keep_mask].to_numpy(), random_state=args.seed,
    )

    bins_ordered = [b.value for b in FieldSizeBin.ordered()]
    results = {
        "fm": "clay-v1",
        "n_layers": n_layers,
        "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "n_pos_train": int(y_tr.sum()), "n_pos_test": int(y_te.sum()),
        "layers": [],
    }
    for layer_idx in range(n_layers):
        feats = layer_feats[layer_idx]
        # drop rows that are still zero (failed inference for that layer)
        nonzero_tr = ~np.all(feats[idx_tr] == 0, axis=1)
        nonzero_te = ~np.all(feats[idx_te] == 0, axis=1)
        idx_tr_keep = idx_tr[nonzero_tr]
        idx_te_keep = idx_te[nonzero_te]
        if len(idx_te_keep) < 100:
            results["layers"].append({"layer": layer_idx, "error": "insufficient nonzero test feats"})
            continue
        X_tr_s = StandardScaler().fit(feats[idx_tr_keep])
        Xt = X_tr_s.transform(feats[idx_tr_keep])
        Xe = X_tr_s.transform(feats[idx_te_keep])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        clf.fit(Xt, y_all[idx_tr_keep])
        y_pred = clf.predict(Xe)
        y_score = clf.predict_proba(Xe)[:, 1]
        y_te_keep = y_all[idx_te_keep]
        bins_te_keep = bins_all[idx_te_keep]
        per_bin = {}
        for bin_name in bins_ordered:
            mask = (y_te_keep == 1) & (bins_te_keep == bin_name)
            if mask.sum() == 0:
                per_bin[bin_name] = {"n": 0, "recall": None}
                continue
            per_bin[bin_name] = {
                "n": int(mask.sum()),
                "recall": float(recall_score(y_te_keep[mask], y_pred[mask], zero_division=0)),
            }
        recalls = [per_bin[b]["recall"] for b in bins_ordered if per_bin[b]["recall"] is not None]
        span = max(recalls) - min(recalls) if recalls else None
        results["layers"].append({
            "layer": layer_idx,
            "f1": float(f1_score(y_te_keep, y_pred)),
            "accuracy": float(accuracy_score(y_te_keep, y_pred)),
            "auroc": float(roc_auc_score(y_te_keep, y_score)) if len(set(y_te_keep)) == 2 else None,
            "span": float(span) if span is not None else None,
            "per_bin_recall_positive": per_bin,
        })
        ov = results["layers"][-1]
        log.info("  layer %2d: F1=%.3f  AUROC=%s  span=%s",
                 layer_idx, ov["f1"],
                 f"{ov['auroc']:.3f}" if ov['auroc'] is not None else "n/a",
                 f"{ov['span']:.3f}" if ov['span'] is not None else "n/a")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)

    banner("Per-layer F1 and span (Clay, Nepal cropland)")
    print(f"{'Layer':<7} {'F1':<8} {'AUROC':<8} {'Span':<8}")
    for entry in results["layers"]:
        if "error" in entry:
            print(f"{entry['layer']:<7} ERROR: {entry['error']}")
            continue
        print(f"{entry['layer']:<7} {entry['f1']:.3f}    "
              f"{entry['auroc']:.3f}    {entry['span']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
