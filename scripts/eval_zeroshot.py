#!/usr/bin/env python
"""Linear-probe evaluation of cached FM features on the starter dataset.

This is the closing artifact of Phase 1's public-data baseline:

  For each FM, train a logistic regression on the FM's pooled features
  to classify cropland (label=1, from polygons) vs non-cropland (label=0,
  from sampled negatives). Compute:
    - overall F1 / accuracy / AUROC on the test split
    - positive-class recall stratified by field-size bin

The per-bin recall is the headline number for the size-effect claim:
"Do FMs detect cropland more reliably on larger fields than smaller ones?"

Splits are stratified by (district, label, size_bin_for_positives) so each
size bin x district combination is represented in both train and test.

Output: data/results/eval_terai_starter.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from scaleshift.data.labels import FieldSizeBin
from scaleshift.utils.logging import banner, get_logger


def build_classifier(kind: str, seed: int):
    if kind == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    raise ValueError(f"unknown classifier kind: {kind!r}")


log = get_logger("eval")

DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_OUT = Path("data/results/eval_terai_starter.json")
FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]


def fm_features_path(features_dir: Path, fm_name: str) -> Path:
    return features_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"


def build_stratum_key(row: pd.Series) -> str:
    """Bucket each example so that stratify-aware split is balanced.

    Negatives stratify by district only; positives by district x size_bin.
    """
    if row["label"] == 0:
        return f"neg_{row['district']}"
    return f"pos_{row['district']}_{row['size_bin']}"


def eval_one_fm(
    fm_name: str,
    features: np.ndarray,
    meta: pd.DataFrame,
    test_size: float,
    seed: int,
    classifier_kind: str = "lr",
    save_predictions_path: Path | None = None,
) -> dict:
    strata = meta.apply(build_stratum_key, axis=1)
    # Drop tiny strata that scikit cannot split.
    counts = strata.value_counts()
    keep_mask = strata.isin(counts[counts >= 2].index)
    if not keep_mask.all():
        n_drop = (~keep_mask).sum()
        log.warning("[%s] dropping %d examples from tiny strata", fm_name, int(n_drop))

    X = features[keep_mask.to_numpy()]
    y = meta.loc[keep_mask, "label"].to_numpy()
    bins = meta.loc[keep_mask, "size_bin"].to_numpy()
    strata_v = strata[keep_mask].to_numpy()
    kept_meta = meta.loc[keep_mask].reset_index(drop=True)

    # Drop rows whose features are all-zero (failed inference).
    nonzero = ~np.all(X == 0, axis=1)
    if not nonzero.all():
        log.warning("[%s] %d examples have all-zero features (failed inference) — dropping",
                    fm_name, int((~nonzero).sum()))
        X, y, bins, strata_v = X[nonzero], y[nonzero], bins[nonzero], strata_v[nonzero]
        kept_meta = kept_meta.loc[nonzero].reset_index(drop=True)

    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, bins_train, bins_test, idx_train, idx_test = train_test_split(
        X, y, bins, indices, test_size=test_size, stratify=strata_v, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = build_classifier(classifier_kind, seed)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_score = clf.predict_proba(X_test_s)[:, 1]

    if save_predictions_path is not None:
        test_meta = kept_meta.iloc[idx_test].reset_index(drop=True)
        preds_df = pd.DataFrame({
            "example_id": test_meta["example_id"].to_numpy(),
            "chip_id": test_meta["chip_id"].to_numpy(),
            "district": test_meta["district"].to_numpy(),
            "label": y_test,
            "size_bin": bins_test,
            "y_pred": y_pred.astype(np.int8),
            "y_score": y_score.astype(np.float32),
            "fm_name": fm_name,
            "classifier": classifier_kind,
        })
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        # Append-or-create: keep predictions for all FMs together
        if save_predictions_path.exists():
            existing = pd.read_parquet(save_predictions_path)
            existing = existing[~((existing.fm_name == fm_name) & (existing.classifier == classifier_kind))]
            preds_df = pd.concat([existing, preds_df], ignore_index=True)
        preds_df.to_parquet(save_predictions_path)

    overall = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_pos_train": int(y_train.sum()),
        "n_pos_test": int(y_test.sum()),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_score)) if len(set(y_test)) == 2 else None,
    }

    # Positive-class recall per size_bin (positives only; bins are blank for negatives).
    per_bin: dict[str, dict] = {}
    for bin_name in [b.value for b in FieldSizeBin.ordered()]:
        mask = (y_test == 1) & (bins_test == bin_name)
        if mask.sum() == 0:
            per_bin[bin_name] = {"n": 0, "recall": None}
            continue
        rec = recall_score(y_test[mask], y_pred[mask], zero_division=0)
        per_bin[bin_name] = {"n": int(mask.sum()), "recall": float(rec)}

    return {
        "overall": overall,
        "per_bin_recall_positive": per_bin,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fms", nargs="*", default=FM_NAMES)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--classifier", choices=["lr", "mlp"], default="lr",
                   help="classification head: lr (logistic regression) or mlp (2-layer)")
    p.add_argument("--save-predictions", type=Path, default=None,
                   help="optional path to a parquet file collecting per-example predictions")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    meta_path = args.features_dir / "features_meta.parquet"
    if not meta_path.exists():
        log.error("Features metadata not found at %s", meta_path)
        return 2

    meta = pd.read_parquet(meta_path)
    banner(f"Evaluating {len(args.fms)} FMs on {len(meta)} examples")
    log.info("  positives=%d  negatives=%d  districts=%s",
             int((meta.label == 1).sum()),
             int((meta.label == 0).sum()),
             sorted(meta.district.unique()))

    results: dict = {
        "dataset": {
            "n_total": int(len(meta)),
            "n_positive": int((meta.label == 1).sum()),
            "n_negative": int((meta.label == 0).sum()),
            "districts": sorted(meta.district.unique()),
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "classifier": args.classifier,
        "fms": {},
    }

    for fm_name in args.fms:
        path = fm_features_path(args.features_dir, fm_name)
        if not path.exists():
            log.warning("Skipping %s (features missing at %s)", fm_name, path)
            continue
        feats = np.load(path)
        log.info("[%s] features shape=%s", fm_name, feats.shape)
        results["fms"][fm_name] = eval_one_fm(
            fm_name, feats, meta, args.test_size, args.seed,
            classifier_kind=args.classifier,
            save_predictions_path=args.save_predictions,
        )
        ov = results["fms"][fm_name]["overall"]
        log.info("[%s] F1=%.3f  acc=%.3f  AUROC=%s",
                 fm_name, ov["f1"], ov["accuracy"],
                 f"{ov['auroc']:.3f}" if ov["auroc"] is not None else "n/a")
        for bin_name, rec in results["fms"][fm_name]["per_bin_recall_positive"].items():
            r = rec["recall"]
            r_str = f"{r:.3f}" if r is not None else "n/a"
            log.info("    %-12s n=%4d  recall=%s", bin_name, rec["n"], r_str)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
