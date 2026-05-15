#!/usr/bin/env python
"""Plot recall-vs-field-size-bin for each FM.

The headline figure: one line per FM, x = field-size bin (ordered),
y = positive-class recall on the test set. If a clear monotone trend
appears (higher recall on larger fields), the size-effect claim has
empirical support on the starter dataset.

Output: data/results/recall_vs_field_size.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scaleshift.data.labels import FieldSizeBin
from scaleshift.utils.logging import get_logger


log = get_logger("plot")

DEFAULT_RESULTS = Path("data/results/eval_terai_starter.json")
DEFAULT_OUT = Path("data/results/recall_vs_field_size.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--title", default="Foundation-model recall vs field size (Nepal Terai, WorldCover labels)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results.exists():
        log.error("Results not found at %s", args.results)
        return 2

    data = json.loads(args.results.read_text())
    bins = [b.value for b in FieldSizeBin.ordered()]

    fig, ax = plt.subplots(figsize=(8, 5))
    for fm_name, res in data["fms"].items():
        per_bin = res["per_bin_recall_positive"]
        ys = [per_bin[b]["recall"] if per_bin[b]["recall"] is not None else np.nan for b in bins]
        ns = [per_bin[b]["n"] for b in bins]
        ax.plot(bins, ys, marker="o", linewidth=2, label=fm_name)
        for i, (y, n) in enumerate(zip(ys, ns)):
            if not np.isnan(y) and n > 0:
                ax.annotate(f"n={n}", (i, y), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, ha="center", alpha=0.6)

    ax.set_xlabel("Field size bin")
    ax.set_ylabel("Positive-class recall (test set)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    caveat = (
        "Labels: ESA WorldCover 2021 cropland; polygons from connected components, "
        "tend to merge across smallholder bunds. RicePAL/JECAM swap will sharpen the "
        "<0.5 ha bins."
    )
    fig.text(0.5, -0.02, caveat, ha="center", fontsize=8, style="italic", wrap=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
