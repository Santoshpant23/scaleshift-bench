#!/usr/bin/env python
"""Patch-boundary analysis: does recall depend on token-grid alignment?

Joins per-example predictions (from eval_zeroshot.py --save-predictions)
with the per-example token-count diagnostic from extract_features_per_polygon.py.

Two questions answered:

  1. For each FM, does recall increase with the number of tokens pooled?
     (Predicted yes -- polygons pooled from 1 token are at the FM's
     spatial resolution limit; polygons pooled from many tokens have
     more spatial detail in the feature.)

  2. Controlling for n_tokens, does recall still depend on field-size bin?
     (If no, then size effect IS the n_tokens effect, and the
     patch-tokenization story is the right explanation. If yes, then
     something else is going on -- label quality, surrounding context,
     edge effects, etc.)

Inputs:
    data/features_per_polygon/n_tokens_per_example.parquet
    data/features_per_polygon/predictions.parquet   (eval_zeroshot.py --save-predictions ...)

Outputs:
    data/results/boundary_recall_by_ntokens.json
    data/results/boundary_recall_by_ntokens.png
    data/results/boundary_recall_by_size_given_ntokens.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaleshift.data.labels import FieldSizeBin
from scaleshift.utils.logging import banner, get_logger


log = get_logger("boundary")

DEFAULT_TOKENS = Path("data/features_per_polygon/n_tokens_per_example.parquet")
DEFAULT_PREDS = Path("data/features_per_polygon/predictions.parquet")
DEFAULT_OUT_JSON = Path("data/results/boundary_recall_by_ntokens.json")
DEFAULT_OUT_PNG = Path("data/results/boundary_recall_by_ntokens.png")
DEFAULT_OUT_CSV = Path("data/results/boundary_recall_by_size_given_ntokens.csv")


def bucketize_tokens(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 2:
        return "2"
    if n <= 3:
        return "3"
    if n <= 7:
        return "4-7"
    if n <= 15:
        return "8-15"
    return "16+"


N_BUCKET_ORDER = ["1", "2", "3", "4-7", "8-15", "16+"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--classifier", default="lr",
                   help="filter predictions to this classifier kind")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.tokens.exists():
        log.error("missing %s", args.tokens)
        return 2
    if not args.predictions.exists():
        log.error("missing %s -- run eval_zeroshot.py with --save-predictions first", args.predictions)
        return 2

    tokens = pd.read_parquet(args.tokens)
    preds = pd.read_parquet(args.predictions)
    preds = preds[preds["classifier"] == args.classifier].copy()
    if len(preds) == 0:
        log.error("no predictions for classifier=%r", args.classifier)
        return 2

    fms = sorted(preds["fm_name"].unique())
    banner(f"Boundary analysis on {len(preds)} predictions across {len(fms)} FMs")

    # --- Question 1: recall vs n_tokens ---
    rows1: list[dict] = []
    for fm in fms:
        col = f"n_tokens_{fm}"
        if col not in tokens.columns:
            log.warning("n_tokens column missing for %s; skipping", fm)
            continue
        merged = preds[preds.fm_name == fm].merge(
            tokens[["example_id", col]], on="example_id", how="left"
        )
        merged["n_bucket"] = merged[col].fillna(1).astype(int).map(bucketize_tokens)
        # Recall on positives only
        pos = merged[merged.label == 1]
        for bucket in N_BUCKET_ORDER:
            sub = pos[pos.n_bucket == bucket]
            if len(sub) == 0:
                rows1.append({"fm": fm, "n_bucket": bucket, "n": 0, "recall": None})
                continue
            rec = float(sub.y_pred.mean())
            rows1.append({"fm": fm, "n_bucket": bucket, "n": int(len(sub)), "recall": rec})

    by_n = pd.DataFrame(rows1)

    # --- Question 2: recall by (n_bucket, size_bin) ---
    rows2: list[dict] = []
    for fm in fms:
        col = f"n_tokens_{fm}"
        if col not in tokens.columns:
            continue
        merged = preds[preds.fm_name == fm].merge(
            tokens[["example_id", col]], on="example_id", how="left"
        )
        merged["n_bucket"] = merged[col].fillna(1).astype(int).map(bucketize_tokens)
        pos = merged[merged.label == 1]
        for n_bucket in N_BUCKET_ORDER:
            for size_bin in [b.value for b in FieldSizeBin.ordered()]:
                sub = pos[(pos.n_bucket == n_bucket) & (pos.size_bin == size_bin)]
                if len(sub) < 5:  # too few to estimate
                    continue
                rows2.append({
                    "fm": fm,
                    "n_bucket": n_bucket,
                    "size_bin": size_bin,
                    "n": int(len(sub)),
                    "recall": float(sub.y_pred.mean()),
                })

    by_n_size = pd.DataFrame(rows2)

    # --- JSON output for question 1 ---
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "classifier": args.classifier,
        "fms": fms,
        "by_n_tokens": by_n.to_dict(orient="records"),
    }, indent=2))

    # --- CSV for question 2 ---
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_n_size.to_csv(args.out_csv, index=False)
    log.info("Wrote %s (%d cells)", args.out_csv, len(by_n_size))

    # --- Plot for question 1 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for fm in fms:
        sub = by_n[by_n.fm == fm]
        ys = [sub[sub.n_bucket == b].recall.iloc[0] if not sub[sub.n_bucket == b].empty else np.nan
              for b in N_BUCKET_ORDER]
        ns = [int(sub[sub.n_bucket == b].n.iloc[0]) if not sub[sub.n_bucket == b].empty else 0
              for b in N_BUCKET_ORDER]
        ax.plot(N_BUCKET_ORDER, ys, marker="o", linewidth=2, label=fm)
        for i, (y, n) in enumerate(zip(ys, ns)):
            if y is not None and not np.isnan(y) and n > 0:
                ax.annotate(f"n={n}", (i, y), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, ha="center", alpha=0.6)
    ax.set_xlabel("Tokens pooled per polygon")
    ax.set_ylabel("Positive-class recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Recall vs tokens pooled (classifier={args.classifier})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    log.info("Wrote %s and %s", args.out_json, args.out_png)

    # --- Pretty-print summary ---
    banner("Recall by n_tokens (positive class)")
    pivot = by_n.pivot_table(index="n_bucket", columns="fm", values="recall")
    pivot = pivot.reindex(N_BUCKET_ORDER)
    print(pivot.to_string(float_format="%.3f"))
    print()

    banner("Recall by (n_tokens x size_bin), showing whether size effect persists at fixed n_tokens")
    if not by_n_size.empty:
        for fm in fms:
            sub = by_n_size[by_n_size.fm == fm]
            if sub.empty:
                continue
            print(f"\n[{fm}]")
            ptab = sub.pivot_table(index="n_bucket", columns="size_bin", values="recall")
            ptab = ptab.reindex(N_BUCKET_ORDER)
            print(ptab.to_string(float_format="%.3f"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
