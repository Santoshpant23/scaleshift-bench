# Sprint 1.5 — Mechanistic Deep-Dive: What Actually Drives the Size Effect?

**Date:** 2026-05-16
**Status:** Complete. Findings significantly sharpen the paper's mechanistic claim.

---

## TL;DR (revised after the size-controlled experiment)

The size-recall gradient is **NOT** explained by a single mechanism. It
decomposes:

- **For Clay (8 px patches)**: size effect IS the n_tokens (patch-
  tokenization) effect. Forcing every polygon to be pooled from a single
  centroid token reduces the per-bin recall span from 0.116 to 0.064.
  Patch tokenization is the dominant mechanism.

- **For Prithvi and TerraMind (16 px patches)**: size effect PERSISTS even
  when n_tokens is forced to 1. Center-pool spans are 0.094 (Prithvi) and
  0.092 (TerraMind), both LARGER than mean-pool spans (0.063 and 0.055).
  Patch tokenization is NOT the main driver here -- something else
  varies with field size. Most likely: spectral content purity at the
  centroid pixel (smaller fields have more mixed/edge centroids).

- **For AnySat**: tile-only by design; the experiment doesn't apply.

The honest paper claim: *"The size effect on geospatial FM recall
decomposes by architecture. Small-patch ViTs (Clay 8 px) are bottlenecked
by patch tokenization. Larger-patch ViTs (Prithvi, TerraMind, both 16 px)
are bottlenecked by something else, plausibly spectral context purity at
the centroid pixel. The single 'FMs fail on small fields' headline hides
two distinct mechanisms."*

This is more nuanced than the earlier "patch tokenization explains it"
claim, and more defensible because we have a controlled experiment that
falsifies the uniform-mechanism story.

---

## Experiments

| Experiment | Question | Method |
|---|---|---|
| LR + MLP heads | Is the size effect a linear-probe artifact? | Two classifiers, compare per-bin recall |
| n_tokens stratification | Does recall depend on tokens-pooled-per-polygon, controlling for size? | `analyze_boundary.py` joins predictions with the n_tokens diagnostic |
| Center-pool re-extraction | Force n_tokens=1 for every polygon; does the size gradient persist? | New extractor flag `--pool-strategy center` |
| Cross-FM failure correlation | Are failures label-driven or model-specific? | Cohen's kappa pairwise on per-example error indicator |

---

## Finding 1 -- LR vs MLP head, mean pool

Overall F1 (LR / MLP):

| FM | LR | MLP |
|---|---|---|
| Clay | 0.646 | 0.722 |
| Prithvi | 0.747 | 0.763 |
| TerraMind | 0.768 | 0.769 |
| AnySat | 0.685 | 0.681 |

Per-bin recall, smallest -> largest:

| FM | LR | MLP |
|---|---|---|
| Clay | 0.65 -> 0.76 | **0.99 -> 0.06** |
| Prithvi | 0.77 -> 0.83 | 0.86 -> 0.66 |
| TerraMind | 0.82 -> 0.83 | 0.86 -> 0.63 |
| AnySat | 0.74 -> 0.79 | 0.72 -> 0.77 |

The MLP head fits a strong "n_tokens => non-cropland" rule, completely
inverting Clay's recall trend. This is a methodology artifact: negatives
are all sampled from a fixed 16 x 16 px window (~1-4 tokens) while
positives have variable n_tokens (1 to ~100). The MLP has enough
capacity to exploit this distributional asymmetry. The LR cannot,
which is why we report LR as the canonical classifier.

---

## Finding 2 -- n_tokens drives the LR size gradient (mean pool, by FM)

Recall as a function of how many tokens were pooled for that polygon,
LR head:

| Tokens pooled | Clay | Prithvi | TerraMind | AnySat |
|---|---|---|---|---|
| 1 | 0.61 | 0.78 | 0.82 | 0.75 |
| 2 | 0.67 | 0.80 | 0.82 | n/a |
| 3 | 0.69 | 0.84 | 0.84 | n/a |
| 4-7 | 0.67 | 0.82 | 0.82 | n/a |
| 8-15 | 0.77 | 0.82 | 0.83 | n/a |
| 16+ | 0.78 | 0.68 | 0.69 | n/a |

For Clay this is monotone -- more tokens => higher recall. For Prithvi/
TerraMind it is flat then drops sharply at 16+ (the WorldCover-merged-
blob artifact: >1 ha "polygons" that combine multiple fields end up
spectrally heterogeneous).

In the (n_tokens x size_bin) cross-tab (LR head, Clay), **at fixed
n_tokens the recall is roughly constant across size bins**, but at fixed
size_bin recall increases with n_tokens. Clay's size effect is the
n_tokens effect.

---

## Finding 3 -- Center-pool kills Clay's gradient, NOT Prithvi/TerraMind's

Per-bin recall, LR head, center pool (n_tokens forced to 1 for every
polygon):

| Bin | Clay | Prithvi | TerraMind | AnySat |
|---|---|---|---|---|
| <0.1 ha | 0.692 | 0.749 | 0.768 | 0.737 |
| 0.1-0.3 | 0.663 | 0.743 | 0.785 | 0.723 |
| 0.3-0.5 | 0.716 | 0.767 | 0.818 | 0.762 |
| 0.5-1 | 0.665 | 0.799 | 0.836 | 0.742 |
| >1 ha | 0.727 | **0.837** | **0.860** | 0.791 |

Span (best minus worst):

| FM | Mean pool | Center pool |
|---|---|---|
| Clay | 0.116 | **0.064 (↓)** |
| Prithvi | 0.063 | **0.094 (↑)** |
| TerraMind | 0.055 | **0.092 (↑)** |
| AnySat | 0.068 | 0.068 (unchanged) |

This is the cleanest mechanistic result of the sprint:

- Clay's span SHRINKS by ~45 percent. Patch tokenization (8 px) was the
  dominant cause.
- Prithvi and TerraMind's spans GROW by ~50 percent. Patch tokenization
  (16 px) was MASKING a different mechanism. Forcing a single-token
  pool reveals it.

What is that other mechanism? The data is consistent with **spectral
context purity** scaling with field size. A single 16 x 16 px patch on a
0.05 ha field includes more non-cropland edge content than the same
patch on a 5 ha field. Mean-pooling many tokens hides this because the
average smooths out individual-patch noise. Center-pool surfaces it.

This is a real, named mechanism distinct from patch-tokenization that
the paper now needs to discuss.

---

## Finding 4 -- Cross-FM failure correlation (LR, mean pool)

Pairwise Cohen's kappa on per-example error indicator:

| FM pair | kappa | MCC | n both wrong |
|---|---|---|---|
| AnySat / Clay | **0.614** | 0.616 | 1804 |
| Prithvi / TerraMind | **0.537** | 0.537 | 1120 |
| AnySat / Prithvi | 0.325 | 0.330 | 1067 |
| AnySat / TerraMind | 0.310 | 0.319 | 991 |
| Clay / Prithvi | 0.286 | 0.295 | 1080 |
| Clay / TerraMind | 0.277 | 0.289 | 1008 |

Distribution: 46 percent of examples are correctly classified by all 4
FMs; 9 percent are mis-classified by all 4 (likely label noise or
intrinsic hard cases); 36 percent are mis-classified by exactly 1 or 2
FMs (model-specific failures).

Two natural clusters emerge:
- **AnySat + Clay**: high agreement (kappa 0.61). Both use minimal
  spatial pooling for their feature (AnySat tile, Clay CLS).
- **Prithvi + TerraMind**: high agreement (kappa 0.54). Both are
  terratorch-loaded ViTs with 16 px patches.

Cross-cluster agreement is ~0.28-0.33. Modest. Failures are partly
shared (data-driven) and partly model-specific (architecture-driven).

The 9 percent "all-4-wrong" cohort is the upper bound on label-noise
contribution. Real model failures account for the rest.

---

## What this gives the paper

| Reviewer objection | Status | Evidence file |
|---|---|---|
| "Just linear-probe weakness?" | Answered | eval_per_polygon_600_{lr,mlp}.json |
| "What's the mechanism precisely?" | Sharper than before | boundary_recall_by_ntokens.json + boundary_recall_center.json |
| "Methodology artifact?" | Identified + flagged for fix | SPRINT_1_5_FINDINGS section above |
| "Labels driving the gap?" | Bounded at <=9 percent | cross_fm_failure_correlation.json |
| "Generalize beyond Nepal?" | Open (Phase 3) | -- |
| "Generalize beyond crops?" | Open (Phase 2) | -- |
| "What's the fix?" | Open (Phase 4) | -- |

Four of seven reviewer objections substantively addressed on Phase 1
data alone. The remaining three are tractable with the planned Phase
2-4 work.

---

## Open follow-ups before submission

1. **Per-pixel evaluation** (~1 sprint). Replaces the polygon-level
   token-pool with per-pixel classification on the chip's token grid.
   Eliminates the n_tokens-asymmetry methodology artifact entirely.
2. **n_tokens-matched negatives**. For each positive polygon, sample
   one negative with a window-size yielding the same n_tokens. Cleaner
   linear-probe.
3. **Spectral-context test for Prithvi/TerraMind**. Compute spectral
   purity (e.g. NDVI variance, EVI variance) within a centroid-window
   of each polygon. Correlate with per-polygon error. If purity
   predicts error, the "context purity" mechanism is confirmed.
4. **Aggregation operator ablation**. Mean vs max at the same n_tokens.
   Already in the codebase (`--pool-strategy max`); run when GPU is
   free and commit a max-pool variant for completeness.
5. **Layer-wise probe**. Train probes on each transformer layer's
   output. Find the layer where the size-recall gradient is largest.
   Locates the failure mechanism within the architecture.

---

## Artifacts

| Path | What |
|---|---|
| `data/results/eval_per_polygon_600_lr.json` | LR head, mean pool, headline |
| `data/results/eval_per_polygon_600_mlp.json` | MLP head, mean pool (ablation) |
| `data/results/eval_per_polygon_600_center.json` | LR head, center pool (size-controlled) |
| `data/results/boundary_recall_by_ntokens.json` | Recall x n_tokens stratification, LR |
| `data/results/boundary_recall_by_ntokens_mlp.json` | Same, MLP |
| `data/results/boundary_recall_center.json` | Center-pool stratification |
| `data/results/cross_fm_failure_correlation.json` | Pairwise FM agreement on errors |
| `data/results/boundary_recall_by_size_given_ntokens.csv` | LR cross-tab table |
| `data/results/boundary_recall_by_size_given_ntokens_mlp.csv` | MLP cross-tab table |
| `data/results/boundary_size_given_n_center.csv` | Center-pool cross-tab |
| `data/results/boundary_recall_by_ntokens.png` | LR figure |
| `data/results/boundary_recall_by_ntokens_mlp.png` | MLP figure |
| `data/results/boundary_recall_center.png` | Center-pool figure |
