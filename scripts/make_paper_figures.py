#!/usr/bin/env python
"""Build paper-quality figures by reading the committed result JSONs.

CPU-only, no FM forwards. Produces (in data/results/figures/):

  fig_per_bin_recall_by_region.png   -- 1x3 panels (Nepal/India/Mozambique)
                                        showing per-bin recall for each FM
  fig_classifier_artifact.png         -- LR vs MLP per-bin recall on Nepal,
                                        per FM, showing MLP artifact pattern
  fig_scalepool_summary.png           -- Span delta (ScalePool vs mean baseline)
                                        per FM per region
  fig_clay_layerwise.png              -- F1 / AUROC / Span vs Clay layer depth
  fig_cross_region_heatmap.png        -- AUROC heatmap for cross-region transfer

  paper_results_unified.csv           -- one row per (FM, region, methodology),
                                        columns: F1, AUROC, span, per-bin recalls
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


RESULTS_DIR = Path("data/results")
FIG_DIR = RESULTS_DIR / "figures"

FMS = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base", "anysat"]
FM_LABELS = {
    "clay-v1": "Clay v1.5 (8px)",
    "prithvi-eo-2.0-300m": "Prithvi-EO-2.0 (16px)",
    "terramind-v1-base": "TerraMind (16px)",
    "anysat": "AnySat (tile)",
}
BINS = ["<0.1ha", "0.1-0.3ha", "0.3-0.5ha", "0.5-1ha", ">1ha"]
BIN_SHORT = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-1", ">1"]

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def load_eval(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def recall_vector(eval_d, fm):
    pb = eval_d["fms"][fm]["per_bin_recall_positive"]
    return [pb[b]["recall"] for b in BINS]


def overall(eval_d, fm):
    return eval_d["fms"][fm]["overall"]


def fig_per_bin_recall_by_region():
    regions = [
        ("Nepal Terai", load_eval("eval_per_polygon_600_lr.json")),
        ("India IGP", load_eval("eval_india_lr.json")),
        ("Mozambique", load_eval("eval_mozambique_lr.json")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(FMS)))
    for ax, (name, d) in zip(axes, regions):
        if d is None:
            ax.set_title(f"{name} (missing)")
            continue
        for fm, c in zip(FMS, colors):
            r = recall_vector(d, fm)
            ax.plot(BIN_SHORT, r, marker="o", color=c, linewidth=1.6,
                    label=FM_LABELS[fm])
        ax.set_title(name)
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Field-size bin (ha)")
    axes[0].set_ylabel("Positive-class recall (LR linear probe)")
    axes[-1].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Per-bin recall across regions (per-polygon token-pool, mean aggregation)", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_per_bin_recall_by_region.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_classifier_artifact():
    lr = load_eval("eval_per_polygon_600_lr.json")
    mlp = load_eval("eval_per_polygon_600_mlp.json")
    if lr is None or mlp is None:
        print("  skipping classifier artifact fig (missing eval)")
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
    for ax, fm in zip(axes, FMS):
        r_lr = recall_vector(lr, fm)
        r_mlp = recall_vector(mlp, fm)
        ax.plot(BIN_SHORT, r_lr, marker="o", color="#1f77b4", linewidth=1.8, label="LR")
        ax.plot(BIN_SHORT, r_mlp, marker="s", color="#d62728", linewidth=1.8, label="MLP")
        ax.set_title(FM_LABELS[fm])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Field-size bin")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Positive-class recall")
    axes[0].legend(loc="lower right")
    fig.suptitle("LR vs MLP head on baseline features: MLP overfits n_tokens "
                 "asymmetry (Clay collapses on >1 ha)", y=1.04)
    fig.tight_layout()
    out = FIG_DIR / "fig_classifier_artifact.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_scalepool_summary():
    pairs = [
        ("Nepal", load_eval("eval_per_polygon_600_lr.json"),
                  load_eval("eval_nepal_scalepool_lr.json")),
        ("India", load_eval("eval_india_lr.json"),
                   load_eval("eval_india_scalepool_lr.json")),
        ("Mozambique", load_eval("eval_mozambique_lr.json"),
                       load_eval("eval_mozambique_scalepool_lr.json")),
    ]
    rows = []
    for region, base, sp in pairs:
        if base is None or sp is None:
            continue
        for fm in FMS:
            b_recalls = recall_vector(base, fm)
            s_recalls = recall_vector(sp, fm)
            b_span = max(b_recalls) - min(b_recalls)
            s_span = max(s_recalls) - min(s_recalls)
            b_f1 = overall(base, fm)["f1"]
            s_f1 = overall(sp, fm)["f1"]
            rows.append({"region": region, "fm": FM_LABELS[fm],
                         "delta_span_pct": (s_span - b_span) / max(b_span, 1e-6) * 100,
                         "delta_f1_pt": (s_f1 - b_f1) * 100})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    pivot_span = df.pivot(index="fm", columns="region", values="delta_span_pct")
    pivot_f1 = df.pivot(index="fm", columns="region", values="delta_f1_pt")
    pivot_span = pivot_span.reindex(list(FM_LABELS.values()))
    pivot_f1 = pivot_f1.reindex(list(FM_LABELS.values()))
    pivot_span.plot(kind="bar", ax=ax1, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_title("ScalePool effect on per-bin recall span")
    ax1.set_ylabel("Δ span vs mean-pool baseline (%)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)
    pivot_f1.plot(kind="bar", ax=ax2, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_title("ScalePool effect on overall F1")
    ax2.set_ylabel("Δ F1 vs mean-pool baseline (pp)")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_scalepool_summary.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_clay_layerwise():
    p = RESULTS_DIR / "layerwise_clay.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    layers = [e["layer"] for e in d["layers"] if "f1" in e]
    f1 = [e["f1"] for e in d["layers"] if "f1" in e]
    auroc = [e["auroc"] for e in d["layers"] if "f1" in e]
    spans = [e["span"] for e in d["layers"] if "f1" in e]
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    ax1.plot(layers, f1, marker="o", color="#1f77b4", label="F1")
    ax1.plot(layers, auroc, marker="s", color="#2ca02c", label="AUROC")
    ax1.set_ylim(0.55, 0.75)
    ax1.set_xlabel("Clay v1.5 transformer block (0 = post patch-embed)")
    ax1.set_ylabel("F1 / AUROC", color="#1f77b4")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(layers, spans, color="#d62728", alpha=0.25, label="Span")
    ax2.set_ylabel("Per-bin recall span", color="#d62728")
    ax2.set_ylim(0.0, 0.18)
    ax2.legend(loc="upper right")
    ax1.set_title("Clay v1.5 layer-wise linear probe: F1 plateaus mid-network; "
                  "size gradient is largest at Layer 0 + final layers")
    fig.tight_layout()
    out = FIG_DIR / "fig_clay_layerwise.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_cross_region_heatmap():
    pairs = ["nepal", "india", "mozambique"]
    aurocs = {fm: np.full((3, 3), np.nan) for fm in FMS}
    # Within-region values from the within-region eval files
    within = {
        "nepal": load_eval("eval_per_polygon_600_lr.json"),
        "india": load_eval("eval_india_lr.json"),
        "mozambique": load_eval("eval_mozambique_lr.json"),
    }
    for i, reg in enumerate(pairs):
        if within[reg] is None:
            continue
        for fm in FMS:
            aurocs[fm][i, i] = overall(within[reg], fm)["auroc"]
    for a in pairs:
        for b in pairs:
            if a == b:
                continue
            d = load_eval(f"cross_region_{a}_to_{b}.json")
            if d is None:
                continue
            for fm in FMS:
                if fm not in d["fms"]:
                    continue
                v = d["fms"][fm]["overall"]["auroc"]
                if v is None:
                    continue
                aurocs[fm][pairs.index(a), pairs.index(b)] = v
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3), sharey=True)
    for ax, fm in zip(axes, FMS):
        m = aurocs[fm]
        im = ax.imshow(m, vmin=0.4, vmax=0.85, cmap="RdYlGn")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([p.capitalize() for p in pairs])
        ax.set_yticklabels([p.capitalize() for p in pairs])
        ax.set_xlabel("Test region")
        ax.set_title(FM_LABELS[fm])
        for i in range(3):
            for j in range(3):
                v = m[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color=("black" if 0.55 < v < 0.75 else "white"),
                        fontsize=8)
    axes[0].set_ylabel("Train region")
    fig.suptitle("Cross-region linear-probe AUROC (LR head, baseline mean pool)", y=1.02)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("AUROC")
    out = FIG_DIR / "fig_cross_region_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def make_unified_csv():
    rows = []
    region_files = {
        "Nepal_baseline_mean": "eval_per_polygon_600_lr.json",
        "Nepal_scalepool": "eval_nepal_scalepool_lr.json",
        "Nepal_baseline_MLP": "eval_per_polygon_600_mlp.json",
        "Nepal_scalepool_MLP": "eval_nepal_scalepool_mlp.json",
        "Nepal_scalepool_torch": "eval_nepal_scalepool_torch.json",
        "Nepal_centerpool": "eval_per_polygon_600_center.json",
        "India_baseline_mean": "eval_india_lr.json",
        "India_scalepool": "eval_india_scalepool_lr.json",
        "Mozambique_baseline_mean": "eval_mozambique_lr.json",
        "Mozambique_scalepool": "eval_mozambique_scalepool_lr.json",
        "Trees_Nepal_baseline_mean": "eval_trees_lr.json",
    }
    for label, fname in region_files.items():
        d = load_eval(fname)
        if d is None:
            continue
        for fm in FMS:
            if fm not in d["fms"]:
                continue
            ov = d["fms"][fm]["overall"]
            pb = d["fms"][fm]["per_bin_recall_positive"]
            recalls = [pb[b]["recall"] for b in BINS]
            recalls_n = [pb[b]["n"] for b in BINS]
            valid = [r for r in recalls if r is not None]
            span = max(valid) - min(valid) if valid else None
            rows.append({
                "experiment": label,
                "fm": fm,
                "f1": ov["f1"],
                "accuracy": ov["accuracy"],
                "auroc": ov["auroc"],
                "n_test": ov["n_test"],
                "span": span,
                **{f"recall_{b}": r for b, r in zip(BINS, recalls)},
                **{f"n_{b}": n for b, n in zip(BINS, recalls_n)},
            })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "paper_results_unified.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} rows)")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Building figures...")
    fig_per_bin_recall_by_region()
    fig_classifier_artifact()
    fig_scalepool_summary()
    fig_clay_layerwise()
    fig_cross_region_heatmap()
    print("Building unified CSV...")
    make_unified_csv()
    print("Done.")


if __name__ == "__main__":
    main()
