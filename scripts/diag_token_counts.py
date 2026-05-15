#!/usr/bin/env python
"""Print the median and mean number of tokens pooled per polygon, per FM,
stratified by field-size bin.

Reads:  data/features_per_polygon/n_tokens_per_example.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scaleshift.data.labels import FieldSizeBin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--path",
        type=Path,
        default=Path("data/features_per_polygon/n_tokens_per_example.parquet"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.path.exists():
        print(f"ERROR: {args.path} not found. Run extract_features_per_polygon.py first.")
        return 2

    df = pd.read_parquet(args.path)
    pos = df[df.label == 1]
    fm_cols = [c for c in df.columns if c.startswith("n_tokens_")]
    ordered_bins = [b.value for b in FieldSizeBin.ordered()]

    print(f"Total positives: {len(pos)}")
    print(f"Total negatives: {(df.label == 0).sum()}\n")

    median_tbl = (
        pos.groupby("size_bin")[fm_cols]
        .median()
        .reindex(ordered_bins)
        .rename(columns=lambda c: c.replace("n_tokens_", ""))
    )
    mean_tbl = (
        pos.groupby("size_bin")[fm_cols]
        .mean()
        .reindex(ordered_bins)
        .rename(columns=lambda c: c.replace("n_tokens_", ""))
    )
    count_tbl = pos.groupby("size_bin").size().reindex(ordered_bins).rename("n_polygons")

    print("Per-FM median tokens pooled per polygon, by size bin:")
    print(median_tbl.to_string(float_format="%.1f"))
    print()
    print("Per-FM mean tokens pooled per polygon, by size bin:")
    print(mean_tbl.to_string(float_format="%.1f"))
    print()
    print("Polygon count per bin:")
    print(count_tbl.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
