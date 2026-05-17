#!/usr/bin/env python
"""Cross-region distribution-shift eval.

Train a per-FM linear probe on one region's features, evaluate on
another region's test set. Reports overall F1/accuracy/AUROC and
per-FieldSizeBin positive-class recall, mirroring eval_zeroshot.py
but with the train/test split crossing geographies.

Answers the reviewer objection: "are the FM features region-invariant,
or does cropland-vs-non-cropland classification require region-specific
classifiers?"

Three modes:
    --train-features-dir A --test-features-dir B
       => train on ALL of A, test on ALL of B.

For comparison, the within-region baseline is whatever
eval_zeroshot.py with --features-dir A produced.

Usage:
    python scripts/eval_cross_region.py \
        --train-features-dir data/features_per_polygon \
        --test-features-dir  data/features_per_polygon_india \
        --train-name nepal --test-name india \
        --out data/results/cross_region_nepal_to_india.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scaleshift.data.labels import FieldSizeBin
from scaleshift.utils.logging import banner, get_logger


log = get_logger("cross")

FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]


def fm_features_path(features_dir: Path, fm_name: str) -> Path:
    return features_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"


def load_split(features_dir: Path, fm_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta_path = features_dir / "features_meta.parquet"
    feats_path = fm_features_path(features_dir, fm_name)
    if not meta_path.exists() or not feats_path.exists():
        return None, None, None
    meta = pd.read_parquet(meta_path)
    feats = np.load(feats_path)
    labels = meta["label"].to_numpy()
    bins = meta["size_bin"].to_numpy()
    nonzero = ~np.all(feats == 0, axis=1)
    return feats[nonzero], labels[nonzero], bins[nonzero]


def eval_one_pair(
    fm_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, bins_test: np.ndarray,
    seed: int,
) -> dict:
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xe = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(Xt, y_train)
    y_pred = clf.predict(Xe)
    y_score = clf.predict_proba(Xe)[:, 1]

    overall = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_pos_train": int(y_train.sum()),
        "n_pos_test": int(y_test.sum()),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_score)) if len(set(y_test)) == 2 else None,
    }
    per_bin = {}
    for bin_name in [b.value for b in FieldSizeBin.ordered()]:
        mask = (y_test == 1) & (bins_test == bin_name)
        if mask.sum() == 0:
            per_bin[bin_name] = {"n": 0, "recall": None}
            continue
        rec = recall_score(y_test[mask], y_pred[mask], zero_division=0)
        per_bin[bin_name] = {"n": int(mask.sum()), "recall": float(rec)}
    return {"overall": overall, "per_bin_recall_positive": per_bin}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-features-dir", type=Path, required=True)
    p.add_argument("--test-features-dir", type=Path, required=True)
    p.add_argument("--train-name", default="train")
    p.add_argument("--test-name", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fms", nargs="*", default=FM_NAMES)
    p.add_argument("--seed", type=int, default=20260517)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    banner(f"Cross-region: train={args.train_name}  test={args.test_name}")

    results: dict = {
        "train_name": args.train_name,
        "test_name": args.test_name,
        "train_features_dir": str(args.train_features_dir),
        "test_features_dir": str(args.test_features_dir),
        "seed": args.seed,
        "fms": {},
    }

    for fm_name in args.fms:
        Xtr, ytr, _ = load_split(args.train_features_dir, fm_name)
        Xte, yte, bins_te = load_split(args.test_features_dir, fm_name)
        if Xtr is None:
            log.warning("Skipping %s (missing train features)", fm_name)
            continue
        if Xte is None:
            log.warning("Skipping %s (missing test features)", fm_name)
            continue
        if Xtr.shape[1] != Xte.shape[1]:
            log.warning("Skipping %s (feature-dim mismatch: train=%d test=%d)",
                        fm_name, Xtr.shape[1], Xte.shape[1])
            continue
        res = eval_one_pair(fm_name, Xtr, ytr, Xte, yte, bins_te, args.seed)
        results["fms"][fm_name] = res
        ov = res["overall"]
        log.info("[%s] F1=%.3f  acc=%.3f  AUROC=%s",
                 fm_name, ov["f1"], ov["accuracy"],
                 f"{ov['auroc']:.3f}" if ov["auroc"] is not None else "n/a")
        for bin_name, rec in res["per_bin_recall_positive"].items():
            r = rec["recall"]
            r_str = f"{r:.3f}" if r is not None else "n/a"
            log.info("    %-12s n=%4d  recall=%s", bin_name, rec["n"], r_str)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
