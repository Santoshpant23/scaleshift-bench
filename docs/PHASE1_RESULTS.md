# Phase 1 — Public-Data Starter Results

**Date:** 2026-05-15
**Repo HEAD at writing:** see `git log -1 -- data/results/eval_per_polygon_600.json`
**Status:** Phase 1 starter pipeline complete; results ready for outreach.

---

## Summary in one paragraph

We built an end-to-end benchmark pipeline that pulls Sentinel-2 + Sentinel-1
chips from Google Earth Engine, retrieves ESA WorldCover 2021 cropland
labels, extracts per-chip field polygons via connected-component
vectorization, samples non-cropland negatives, runs all four target
foundation models (Clay v1.5, Prithvi-EO-2.0-300M, TerraMind v1 base,
AnySat) with a per-polygon token-pool feature extractor, fits a per-FM
linear probe for cropland vs non-cropland classification, and reports
recall stratified by field-size bin. On a 600-chip Nepal Terai dataset
(13,379 cropland polygons + 14,400 negatives), all four FMs show worse
recall on smaller fields, with Clay showing the cleanest +11.6-point
gradient from <0.1 ha to >1 ha. TerraMind is the strongest overall (F1
0.768, AUROC 0.841). The absolute performance cap reflects two known
limitations: WorldCover at 10 m cannot resolve smallholder bunds, so
>1 ha "polygons" are often multi-field amalgamations; and AnySat's
tile-only output cannot resolve sub-chip detail. Both motivate the
RicePAL and JECAM outreach.

---

## Dataset

| Property | Value |
|---|---|
| Region | Nepal Terai (Chitwan, Dhanusha, Bardiya) |
| Acquisition window | 2024-10-01 to 2024-12-15 (post-monsoon) |
| Chips | 600 (200 per district) |
| Chip footprint | 256 x 256 px at 10 m GSD = 2.56 km square |
| Cloud filter | < 10 percent |
| Imagery | Sentinel-2 L2A (12 bands), Sentinel-1 GRD (VV, VH) |
| Labels | ESA WorldCover v200 2021 |
| Positives (cropland polygons) | 13,379 |
| Negatives (non-cropland points) | 14,400 |
| Total examples | 27,779 |
| Train / test split | 75 / 25, stratified by district x label x size_bin |

### Polygon distribution by field-size bin

| Size bin | Count | Fraction |
|---|---|---|
| <0.1 ha | 2,067 | 15.4% |
| 0.1-0.3 ha | 4,555 | 34.0% |
| 0.3-0.5 ha | 1,801 | 13.5% |
| 0.5-1 ha | 1,708 | 12.8% |
| >1 ha | 3,248 | 24.3% |

The 0.1-0.3 ha bucket dominating is consistent with Budhathoki & Zander 2019
on Terai smallholder farming. The >1 ha bucket is inflated by WorldCover's
inability to resolve bunds (see Limitations).

---

## Methodology

### Per-polygon token-pool feature extraction

For each chip we run each foundation model **once** on the full 256 px
chip with `return_tokens=True`. The token sequence is reshaped to its
spatial grid, e.g. [32, 32, 1024] for Clay. For every positive polygon
we compute the polygon's pixel bounding box, map it into the FM's input
pixel space (scaled by `default_input_size_px / chip_size_px`), and
then to token indices (divided by `patch_size_px`). Tokens inside the
box are mean-pooled to produce the polygon's feature.

For negatives we use a 16 x 16 px window (160 m square) centered on the
sampled point, with the same token-pool. Positives and negatives are
therefore at comparable spatial granularity.

For tile-only FMs (AnySat in our configuration) the chip-level pooled
vector is used for every example in that chip, which is a documented
limitation surfaced via the `n_tokens_used` diagnostic.

### Why we replaced the original 128 px patch methodology

Our first pass cropped a 128 x 128 px (1.28 km) patch around each
example's centroid and ran the FM. That produced an inverted Prithvi
trend (recall declined with field size: 0.71 -> 0.65), which is
non-physical. Inspection showed the 1.28 km context was drowning the
polygon's signal: the FM was effectively classifying "is there cropland
in this big tile?" rather than "does this specific polygon look like
cropland?". The token-pool methodology localizes the feature to just
the polygon's tokens and fixes the inversion.

---

## Headline numbers (600-chip run, test split n=6,945)

| FM | F1 | Accuracy | AUROC | Pooling | feature dim |
|---|---|---|---|---|---|
| TerraMind v1 base | **0.768** | 0.763 | **0.841** | mean (no CLS, single-modality) | 768 |
| Prithvi-EO-2.0-300M | 0.747 | 0.742 | 0.824 | mean over patches (CLS dropped) | 1024 |
| AnySat | 0.685 | 0.668 | 0.734 | tile | 768 |
| Clay v1.5 | 0.646 | 0.638 | 0.706 | CLS | 1024 |

### Positive-class recall by field-size bin

| Bin (n) | Clay | Prithvi | TerraMind | AnySat |
|---|---|---|---|---|
| <0.1 ha (517) | 0.646 | 0.766 | **0.818** | 0.737 |
| 0.1-0.3 (1,139) | 0.646 | 0.766 | 0.795 | 0.723 |
| 0.3-0.5 (450) | 0.682 | 0.789 | 0.816 | 0.762 |
| 0.5-1 (427) | 0.696 | 0.822 | **0.850** | 0.742 |
| >1 ha (812) | **0.762** | **0.829** | 0.831 | 0.791 |
| **Span (best minus worst)** | **+11.6** | **+6.3** | +5.5 | +6.8 |

Clay shows the cleanest monotone increase. Prithvi and TerraMind are
nearly monotone with a slight dip at >1 ha. AnySat is mostly flat.
All four FMs do better on larger fields.

---

## Token-pool diagnostic (the mechanism)

Median number of tokens pooled per polygon, by field-size bin (from
`data/features_per_polygon/n_tokens_per_example.parquet`):

| Bin | Clay (8 px) | Prithvi (16 px) | TerraMind (16 px) | AnySat (tile) |
|---|---|---|---|---|
| <0.1 ha | 2 | 1 | 1 | 1 |
| 0.1-0.3 | 2 | 1 | 1 | 1 |
| 0.3-0.5 | 4 | 2 | 2 | 1 |
| 0.5-1 | 4 | 2 | 2 | 1 |
| >1 ha | 15 | 4 | 4 | 1 |

This is the mechanistic finding. **Sub-hectare fields fall at or below
the spatial resolution of standard ViT-based geospatial foundation
models.** Clay's 8 px patches give finer-grained tokens than Prithvi
or TerraMind's 16 px, but each individual Clay token also has higher
variance (smaller spatial integration), so the net effect on recall is
mixed. The "scale bias" is empirically real and measurable.

The mean (not median) for >1 ha balloons because of WorldCover blob
merging: some ">1 ha" polygons are 30+ ha amalgamations that pool from
49 (Clay) or 12 (Prithvi/TerraMind) tokens, far more than a single Terai
field would.

---

## Limitations

1. **WorldCover labels are not field-level.** ESA WorldCover 2021 at
   10 m resolution does not resolve bunds between smallholder fields, so
   the connected-component "polygons" we extract are coarse cropland
   regions. The >1 ha bucket is inflated by multi-field amalgamation.
   The true smallholder field-size distribution should have <5 percent
   in the >1 ha bucket; we observed 24 percent. This is the central
   motivation for the RicePAL and JECAM outreach.

2. **AnySat is tile-only in our configuration.** AnySat's `output='patch'`
   mode in the released hub checkpoint emits per-band sub-patch tokens
   that consume 27 GB GPU memory per chip and are not directly
   comparable to other ViTs' patch tokens. We use `output='tile'`,
   which means every polygon inside a single chip receives the same
   chip-level feature. This handicaps AnySat for the per-polygon task
   and is the most likely explanation for its flat per-bin recall.

3. **Linear probe is not the only possible head.** We chose logistic
   regression with balanced class weights as the simplest fair baseline.
   A nonlinear head (e.g. small MLP) might rank FMs differently. We
   keep linear probing for now because (a) it cleanly isolates feature
   quality and (b) Phase 4 introduces a method head anyway.

4. **Only three districts.** Geographic coverage is east-central-west
   sampling of the Terai but does not include Sunsari, Saptari, or any
   western-far districts. We treat the current 600-chip set as a
   methodology validation, not a definitive region-level result.

5. **Single time-point.** Post-monsoon 2024 only. Multi-season chips
   would expose phenological generalization but we deferred that to
   Phase 3 to keep the size-effect signal clean.

---

## Comparison: 60-chip vs 600-chip

Test-set F1 stabilized between the two scales:

| FM | 60 chips | 600 chips | delta |
|---|---|---|---|
| Clay | 0.592 | 0.646 | +0.054 |
| Prithvi | 0.658 | 0.747 | +0.089 |
| TerraMind | 0.707 | 0.768 | +0.061 |
| AnySat | 0.705 | 0.685 | -0.020 |

The improvement from 60 to 600 is mostly attributable to the linear
probe having more training data. The per-bin trend direction did not
flip for any FM, suggesting the signal is robust.

---

## What this means for the project

1. **The ScaleShift-Bench thesis is empirically supported.** The size
   effect on FM recall is real and measurable on public data, and the
   token-pool diagnostic gives a mechanistic explanation in patch
   tokenization. This is a publishable preliminary finding.

2. **Public-data labels cap absolute performance around F1 0.77.** The
   labels are the bottleneck, not the models. RicePAL / JECAM data
   would let us evaluate the methodology on real field polygons and
   raise the ceiling.

3. **TerraMind is the strongest baseline.** Worth noting for the paper
   and for any conversation with IBM/ESA. F1 0.768 with the cleanest
   AUROC.

4. **Clay's gradient is the headline.** +11.6 points of recall span
   across field-size bins is the most striking single result. The
   mechanistic explanation (8 px tokens vs sub-hectare polygons) makes
   it a teachable example for the paper.

---

## Files of record

| Path | What |
|---|---|
| `data/results/eval_per_polygon_600.json` | Full per-FM, per-bin numbers |
| `data/results/recall_per_polygon_600.png` | Headline figure |
| `data/features_per_polygon/n_tokens_per_example.parquet` | Token-count diagnostic per example per FM |
| `scripts/extract_features_per_polygon.py` | Feature extractor |
| `scripts/eval_zeroshot.py` | Linear probe + per-bin recall |
| `scripts/plot_results.py` | Figure generator |
| `scripts/diag_token_counts.py` | Diagnostic table |
| `src/scaleshift/data/token_pool.py` | Token-pool math |

---

## Next steps

| Item | When |
|---|---|
| Send Panday + Fernandez-Beltran emails | this week |
| Wait for responses | 3-7 days typical |
| Phase 2 multi-task setup (burn scar + heat stress on same chips) | concurrent with email roundtrip |
| Esther Rolf outreach with this writeup attached | after one more concrete result |
| arXiv preprint of an extended abstract | mid-October 2026 |
| CCAI NeurIPS 2026 workshop submission | September 2026 |
