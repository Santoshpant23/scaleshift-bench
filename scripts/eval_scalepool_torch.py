#!/usr/bin/env python
"""Phase 5b — Learnable scale-conditioned adapter for ScalePool features.

The Phase 5a finding: an sklearn MLP on ScalePool features improves overall
F1 but exposes the n_tokens-distribution artifact (positives have variable
n_tokens, negatives have ~constant n_tokens; non-linear classifiers exploit
that as a shortcut).

The fix: explicitly condition the classifier on n_tokens by passing it as a
side input. Then the model cannot use n_tokens as a sneaky discrimination
signal -- the per-bin recall pattern should reflect actual feature-quality
differences rather than the methodology artifact.

Concretely:
    features  --> 3*D-dim ScalePool vector (mean pool at k=0/1/3 dilations)
    n_tokens  --> integer in [1, 19], bucketed and embedded as a 16-d vector
    --> concat --> 2-layer MLP --> binary cropland-vs-not

If the resulting per-bin recall is monotone-ish (no artifact collapse at >1
ha for Clay) AND F1 is at least as good as LR ScalePool, this is the
working Phase 5 method.

Usage:
    python scripts/eval_scalepool_torch.py \
        --features-dir data/features_per_polygon_scalepool \
        --tokens data/features_per_polygon_scalepool/n_tokens_per_example.parquet \
        --out data/results/eval_nepal_scalepool_torch.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scaleshift.data.labels import FieldSizeBin
from scaleshift.utils.logging import banner, get_logger


log = get_logger("scalepool-torch")

FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]
N_TOKENS_MAX_BIN = 19  # clamp; covers up to "16+" bucket


class ScalePoolAdapter(nn.Module):
    def __init__(self, in_dim: int, n_tok_bins: int = 20, n_tok_emb: int = 16,
                 hidden1: int = 256, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.n_tok_emb = nn.Embedding(n_tok_bins, n_tok_emb)
        self.head = nn.Sequential(
            nn.Linear(in_dim + n_tok_emb, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor, n_tokens: torch.Tensor) -> torch.Tensor:
        nt = self.n_tok_emb(n_tokens.clamp(max=N_TOKENS_MAX_BIN))
        return self.head(torch.cat([x, nt], dim=-1)).squeeze(-1)


def build_stratum(row: pd.Series) -> str:
    if row["label"] == 0:
        return f"neg_{row['district']}"
    return f"pos_{row['district']}_{row['size_bin']}"


def fm_features_path(features_dir: Path, fm_name: str) -> Path:
    return features_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"


def eval_one_fm(
    fm_name: str,
    features_dir: Path,
    tokens_df: pd.DataFrame,
    test_size: float,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict:
    feats_path = fm_features_path(features_dir, fm_name)
    meta = pd.read_parquet(features_dir / "features_meta.parquet")
    feats = np.load(feats_path)
    if feats.shape[0] != len(meta):
        raise ValueError(f"feature/meta length mismatch for {fm_name}")

    tok_col = f"n_tokens_{fm_name}"
    if tok_col not in tokens_df.columns:
        log.warning("Missing %s column in tokens parquet -- using 1 for all", tok_col)
        n_tok_arr = np.ones(len(meta), dtype=np.int64)
    else:
        n_map = dict(zip(tokens_df["example_id"], tokens_df[tok_col]))
        n_tok_arr = np.array(
            [int(n_map.get(e, 1)) for e in meta["example_id"]],
            dtype=np.int64,
        )

    strata = meta.apply(build_stratum, axis=1)
    counts = strata.value_counts()
    keep_mask = strata.isin(counts[counts >= 2].index)
    if not keep_mask.all():
        log.warning("[%s] dropping %d examples from tiny strata",
                    fm_name, int((~keep_mask).sum()))

    X = feats[keep_mask.to_numpy()]
    y = meta.loc[keep_mask, "label"].to_numpy()
    bins = meta.loc[keep_mask, "size_bin"].to_numpy()
    nt = n_tok_arr[keep_mask.to_numpy()]
    strata_v = strata[keep_mask].to_numpy()

    nonzero = ~np.all(X == 0, axis=1)
    X, y, bins, nt, strata_v = X[nonzero], y[nonzero], bins[nonzero], nt[nonzero], strata_v[nonzero]

    X_tr, X_te, y_tr, y_te, bins_tr, bins_te, nt_tr, nt_te = train_test_split(
        X, y, bins, nt, test_size=test_size, stratify=strata_v, random_state=seed
    )
    # Carve val off of train (15% of original) for early stopping
    X_tr, X_va, y_tr, y_va, nt_tr, nt_va = train_test_split(
        X_tr, y_tr, nt_tr, test_size=0.15, stratify=y_tr, random_state=seed
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    model = ScalePoolAdapter(in_dim=X_tr_s.shape[1]).to(device)
    pos_weight = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr_s), torch.from_numpy(nt_tr.astype(np.int64)),
        torch.from_numpy(y_tr.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_va_t = torch.from_numpy(X_va_s).to(device)
    nt_va_t = torch.from_numpy(nt_va.astype(np.int64)).to(device)
    y_va_arr = y_va

    best_val_f1 = -1.0
    best_state = None
    patience_left = patience
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, ntb, yb in train_loader:
            xb = xb.to(device); ntb = ntb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb, ntb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            logits_va = model(X_va_t, nt_va_t).cpu().numpy()
        val_pred = (logits_va > 0).astype(np.int64)
        val_f1 = f1_score(y_va_arr, val_pred)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        log.info("[%s] epoch %2d  train_loss=%.4f  val_F1=%.3f  (best=%.3f)",
                 fm_name, epoch, ep_loss, val_f1, best_val_f1)
        if patience_left <= 0:
            log.info("[%s] early stopping at epoch %d", fm_name, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    X_te_t = torch.from_numpy(X_te_s).to(device)
    nt_te_t = torch.from_numpy(nt_te.astype(np.int64)).to(device)
    with torch.no_grad():
        logits_te = model(X_te_t, nt_te_t).cpu().numpy()
    y_pred = (logits_te > 0).astype(np.int64)
    y_score = torch.sigmoid(torch.from_numpy(logits_te)).numpy()

    overall = {
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "n_pos_train": int(y_tr.sum()), "n_pos_test": int(y_te.sum()),
        "f1": float(f1_score(y_te, y_pred)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "auroc": float(roc_auc_score(y_te, y_score)) if len(set(y_te)) == 2 else None,
        "best_val_f1": float(best_val_f1),
    }
    per_bin = {}
    for bin_name in [b.value for b in FieldSizeBin.ordered()]:
        mask = (y_te == 1) & (bins_te == bin_name)
        if mask.sum() == 0:
            per_bin[bin_name] = {"n": 0, "recall": None}
            continue
        rec = recall_score(y_te[mask], y_pred[mask], zero_division=0)
        per_bin[bin_name] = {"n": int(mask.sum()), "recall": float(rec)}

    return {"overall": overall, "per_bin_recall_positive": per_bin}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--tokens", type=Path, default=None,
                   help="path to n_tokens_per_example.parquet "
                        "(defaults to <features-dir>/n_tokens_per_example.parquet)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fms", nargs="*", default=FM_NAMES)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tokens is None:
        args.tokens = args.features_dir / "n_tokens_per_example.parquet"
    if not args.tokens.exists():
        log.error("Missing n_tokens parquet at %s", args.tokens)
        return 2

    tokens_df = pd.read_parquet(args.tokens)
    meta_path = args.features_dir / "features_meta.parquet"
    meta = pd.read_parquet(meta_path)

    banner(f"ScalePool adapter (torch) on {len(meta)} examples, device={args.device}")

    results = {
        "dataset": {
            "n_total": int(len(meta)),
            "n_positive": int((meta.label == 1).sum()),
            "n_negative": int((meta.label == 0).sum()),
            "districts": sorted(meta.district.unique()),
            "test_size": args.test_size,
            "seed": args.seed,
            "device": args.device,
        },
        "classifier": "torch_mlp_ntokens_conditioned",
        "fms": {},
    }

    for fm_name in args.fms:
        log.info("==== %s ====", fm_name)
        try:
            res = eval_one_fm(
                fm_name, args.features_dir, tokens_df,
                test_size=args.test_size, seed=args.seed, device=args.device,
                epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            )
        except FileNotFoundError as e:
            log.warning("[%s] skipping (%s)", fm_name, e)
            continue
        results["fms"][fm_name] = res
        ov = res["overall"]
        log.info("[%s] F1=%.3f  acc=%.3f  AUROC=%s",
                 fm_name, ov["f1"], ov["accuracy"],
                 f"{ov['auroc']:.3f}" if ov["auroc"] is not None else "n/a")
        for bin_name, rec in res["per_bin_recall_positive"].items():
            r_str = f"{rec['recall']:.3f}" if rec["recall"] is not None else "n/a"
            log.info("    %-12s n=%4d  recall=%s", bin_name, rec["n"], r_str)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
