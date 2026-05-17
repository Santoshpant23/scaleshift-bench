#!/usr/bin/env python
"""Cross-FM failure correlation analysis.

Reads predictions.parquet (from eval_zeroshot.py --save-predictions) and
computes pairwise correlations between FMs on which examples they get
wrong.

If two FMs miss the same examples (high correlation), the failures are
label-driven or data-driven (intrinsic to the inputs). If they miss
different examples (low correlation), the failures are model-specific
(architectures see the data differently).

Outputs:
    data/results/cross_fm_failure_correlation.json
    data/results/cross_fm_failure_correlation.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from scaleshift.utils.logging import banner, get_logger


log = get_logger("cross-fm")

DEFAULT_PREDS = Path("data/features_per_polygon/predictions.parquet")
DEFAULT_OUT_JSON = Path("data/results/cross_fm_failure_correlation.json")
DEFAULT_OUT_CSV = Path("data/results/cross_fm_failure_correlation.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    p.add_argument("--classifier", default="lr")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        log.error("missing %s", args.predictions)
        return 2

    preds = pd.read_parquet(args.predictions)
    preds = preds[preds.classifier == args.classifier].copy()
    if preds.empty:
        log.error("no predictions for classifier=%r", args.classifier)
        return 2

    fms = sorted(preds.fm_name.unique())
    banner(f"Cross-FM failure correlation, classifier={args.classifier}, FMs={fms}")

    # Pivot: rows = example_id, cols = FM, values = (y_pred == label) i.e. correct/incorrect
    preds["correct"] = (preds.y_pred == preds.label).astype(int)
    wide = preds.pivot_table(index="example_id", columns="fm_name", values="correct", aggfunc="first")
    wide = wide.dropna(how="any")
    log.info("Examples with predictions for all %d FMs: %d", len(fms), len(wide))

    # Wrong = NOT correct. Cohen's kappa on the WRONG indicator (so high
    # kappa means FMs agree on which examples they fail).
    wrong = 1 - wide

    pairs = list(itertools.combinations(fms, 2))
    rows = []
    for a, b in pairs:
        kappa = cohen_kappa_score(wrong[a], wrong[b])
        mcc = matthews_corrcoef(wrong[a], wrong[b])
        n_both_wrong = int(((wrong[a] == 1) & (wrong[b] == 1)).sum())
        n_a_only_wrong = int(((wrong[a] == 1) & (wrong[b] == 0)).sum())
        n_b_only_wrong = int(((wrong[a] == 0) & (wrong[b] == 1)).sum())
        n_neither_wrong = int(((wrong[a] == 0) & (wrong[b] == 0)).sum())
        rows.append({
            "fm_a": a,
            "fm_b": b,
            "kappa": float(kappa),
            "mcc": float(mcc),
            "n_both_wrong": n_both_wrong,
            "n_a_only_wrong": n_a_only_wrong,
            "n_b_only_wrong": n_b_only_wrong,
            "n_neither_wrong": n_neither_wrong,
        })

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = {
        "classifier": args.classifier,
        "fms": fms,
        "n_examples": int(len(wide)),
        "pairs": rows,
        "all_four_wrong": int((wrong[fms].sum(axis=1) == 4).sum()),
        "exactly_one_wrong": int((wrong[fms].sum(axis=1) == 1).sum()),
        "exactly_two_wrong": int((wrong[fms].sum(axis=1) == 2).sum()),
        "exactly_three_wrong": int((wrong[fms].sum(axis=1) == 3).sum()),
        "none_wrong": int((wrong[fms].sum(axis=1) == 0).sum()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    banner("Pairwise Cohen's kappa (FM agreement on errors)")
    print(df[["fm_a", "fm_b", "kappa", "mcc", "n_both_wrong"]].to_string(index=False))
    print()

    banner("Distribution: how many of the 4 FMs got each example wrong")
    print(f"  none wrong    : {summary['none_wrong']:6d}")
    print(f"  exactly 1     : {summary['exactly_one_wrong']:6d}")
    print(f"  exactly 2     : {summary['exactly_two_wrong']:6d}")
    print(f"  exactly 3     : {summary['exactly_three_wrong']:6d}")
    print(f"  all 4 wrong   : {summary['all_four_wrong']:6d}")
    print()
    print("Interpretation:")
    print("  - Many 'all 4 wrong' examples => label noise or intrinsically-hard data")
    print("  - Many 'exactly 1/2 wrong'    => model-specific failure modes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
