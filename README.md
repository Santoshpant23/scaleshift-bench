# scaleshift-bench

**A benchmark and mechanism analysis of the field-size effect in geospatial
foundation models.** Per-polygon evaluation of Clay v1.5, Prithvi-EO-2.0,
TerraMind v1, and AnySat on Sentinel-2 cropland and tree-cover tasks across
Nepal, India, and Mozambique.

Target venue: NeurIPS Datasets & Benchmarks 2027.

---

## Headline findings

1. **Documented size-recall gradient.** Linear probes over per-polygon
   token-pooled foundation-model features show per-bin recall increasing
   with field size: e.g. Clay v1.5 recall climbs from 0.65 (<0.1 ha) to
   0.76 (>1 ha) on Nepal Terai cropland (span 0.116).
2. **Mechanism localized.** A layer-wise probe of Clay shows the size
   gradient is largest at Layer 0 (post patch-embed, pre attention,
   span 0.125), shrinks 33% by Layer 3 (span 0.083) via self-attention
   mixing, and re-emerges by Layer 23 (span 0.127). Mid-network features
   are the sweet spot for downstream cropland classification — the
   conventional final-layer probe is suboptimal.
3. **Region-conditional.** The effect appears in Nepal and India IGP
   (within-region AUROC 0.71-0.84) but vanishes in Mozambique smallholder
   regions (AUROC 0.58-0.61). When the FM lacks classification headroom,
   the size effect is masked by primary task noise.
4. **Cross-region transfer is poor.** Training on Nepal and testing on
   India: AUROC 0.55-0.68. Training on either and testing on Mozambique:
   AUROC 0.43-0.57 (near random). FM features for cropland are NOT
   region-invariant — operational deployment requires region-specific
   classifier training.
5. **Methodology limit identified.** Non-linear classifiers (MLP head)
   exploit the systematic n_tokens difference between positive polygons
   (variable size) and negative point-windows (constant ~4 tokens), so
   "higher F1" via MLP is partly methodology artifact rather than a
   real improvement. The proper fix is per-pixel evaluation; left for
   future work.
6. **Method (ScalePool v1).** A drop-in multi-scale token aggregation
   (concat of pools at k=0/1/3 dilations) reduces the recall span where
   the baseline has a real gradient (Nepal Clay −12%, Prithvi −19%,
   India TerraMind −35%) at a small (1-2 pp) F1 cost.

10 of 10 substantive reviewer objections substantively addressed.

---

## Repo layout

```
scaleshift-bench/
├── configs/
│   ├── terai_districts.yaml         # Nepal AOIs
│   ├── india_igp_districts.yaml     # India AOIs
│   └── mozambique_districts.yaml    # Mozambique AOIs
├── src/scaleshift/
│   ├── data/
│   │   ├── chip.py                  # Chip dataclass + S2/S1 band order
│   │   ├── labels.py                # FieldSizeBin enum, WorldCover codes
│   │   ├── worldcover.py            # WorldCover loader
│   │   └── token_pool.py            # Per-polygon token pool (mean/max/center/multiscale)
│   ├── model_zoo/
│   │   ├── base.py                  # FoundationModel ABC
│   │   ├── clay.py                  # Clay v1.5 wrapper (incl. encode_per_layer)
│   │   ├── prithvi.py               # Prithvi-EO-2.0
│   │   ├── terramind.py             # TerraMind v1 base
│   │   ├── anysat.py                # AnySat (tile mode)
│   │   ├── presto.py                # Presto (stub, Phase 1 deferred)
│   │   └── __init__.py              # Registry: get_model(name)
│   └── utils/
├── scripts/                         # End-to-end pipeline
│   ├── build_terai_chip_index.py    # GEE index builder
│   ├── pull_chip_imagery.py
│   ├── pull_worldcover_labels.py
│   ├── build_chip_manifest.py
│   ├── extract_field_polygons.py    # Cropland / tree polygons via connected components
│   ├── sample_negatives.py
│   ├── extract_features_per_polygon.py   # Per-polygon FM features (pool strategy: mean/max/center/multiscale)
│   ├── eval_zeroshot.py             # LR or MLP linear probe
│   ├── eval_cross_region.py         # Train-on-region-A, test-on-region-B
│   ├── eval_scalepool_torch.py      # Phase 5b torch adapter
│   ├── analyze_boundary.py          # n_tokens stratification
│   ├── analyze_cross_fm_failures.py # Pairwise Cohen's kappa
│   ├── analyze_spectral_purity.py   # NDVI SD per centroid window
│   ├── analyze_layerwise_clay.py    # Phase C
│   ├── plot_results.py
│   └── make_paper_figures.py        # Generate all 5 paper figures
├── data/
│   ├── chips/                       # (gitignored) S2/S1/WorldCover GeoTIFFs
│   ├── features_per_polygon/        # (gitignored) cached FM features
│   ├── labels/                      # (gitignored) extracted polygons / negatives
│   └── results/                     # COMMITTED: eval JSONs, figures, unified CSV
│       ├── figures/                 # 5 publication-ready PNGs
│       └── paper_results_unified.csv  # one row per (experiment, FM)
├── docs/                            # Phase-by-phase writeups
│   ├── PHASE1_RESULTS.md
│   ├── SPRINT_1_5_FINDINGS.md
│   ├── PHASE2_TREES_RESULTS.md
│   ├── PHASE3_INDIA_RESULTS.md
│   ├── PHASE3_MULTIREGION_RESULTS.md
│   ├── PHASE4_SCALEPOOL_RESULTS.md
│   ├── PHASE_C_LAYERWISE.md
│   └── outreach/                    # Email drafts
└── tests/                           # CPU-only pytest suite
```

---

## Reproducing the benchmark

### Setup (one-time, on a GPU host)

```bash
git clone https://github.com/Santoshpant23/scaleshift-bench.git
cd scaleshift-bench
bash scripts/setup_lambdavector2.sh      # uv venv, torch CUDA, FM extras
source .venv/bin/activate
earthengine authenticate --auth_mode=notebook
export EE_PROJECT=<your-gcp-project>
```

### One-region pipeline (Nepal Terai shown; same flow for India / Mozambique with their configs)

```bash
python scripts/build_terai_chip_index.py        # samples 600 chip centers via GEE
python scripts/pull_chip_imagery.py             # downloads S2+S1 GeoTIFFs
python scripts/pull_worldcover_labels.py        # downloads ESA WorldCover 2021
python scripts/build_chip_manifest.py --require-worldcover
python scripts/extract_field_polygons.py \
    --manifest data/chips/manifest_terai_starter.jsonl
python scripts/sample_negatives.py
python scripts/extract_features_per_polygon.py  # runs all 4 FMs, ~12 min on RTX A6000
python scripts/eval_zeroshot.py \
    --features-dir data/features_per_polygon \
    --out data/results/eval_per_polygon_600_lr.json \
    --classifier lr --save-predictions data/features_per_polygon/predictions.parquet
python scripts/analyze_boundary.py --classifier lr
python scripts/plot_results.py
```

### Reproducing ScalePool

```bash
python scripts/extract_features_per_polygon.py --pool-strategy multiscale \
    --out-dir data/features_per_polygon_scalepool
python scripts/eval_zeroshot.py --features-dir data/features_per_polygon_scalepool \
    --out data/results/eval_nepal_scalepool_lr.json --classifier lr
```

### Reproducing cross-region transfer

```bash
python scripts/eval_cross_region.py \
    --train-features-dir data/features_per_polygon \
    --test-features-dir data/features_per_polygon_india \
    --train-name nepal --test-name india \
    --out data/results/cross_region_nepal_to_india.json
```

### Reproducing layer-wise probe (Phase C)

```bash
python scripts/analyze_layerwise_clay.py     # ~20 min on RTX A6000
```

### Regenerating all paper figures from committed JSONs

```bash
python scripts/make_paper_figures.py    # CPU-only, ~5 seconds
```

---

## Key result tables

All numbers are mean over 600 chips per region, LR linear probe, mean
token-pool aggregation unless otherwise noted.

### Within-region per-bin recall span (≤ smaller is more uniform)

| FM | Nepal | India | Mozambique |
|---|---|---|---|
| Clay v1.5 (8 px) | 0.116 | 0.065 | 0.030 |
| Prithvi-EO-2.0 (16 px) | 0.063 | 0.118 | 0.076 |
| TerraMind (16 px) | 0.055 | 0.107 | 0.055 |
| AnySat (tile) | 0.068 | 0.096 | 0.029 |

### Within-region AUROC

| FM | Nepal | India | Mozambique |
|---|---|---|---|
| Clay v1.5 | 0.706 | 0.726 | 0.595 |
| Prithvi-EO-2.0 | 0.824 | 0.793 | 0.597 |
| TerraMind | 0.841 | 0.804 | 0.611 |
| AnySat | 0.734 | 0.769 | 0.579 |

### ScalePool effect (Δ span %, Δ F1 pp)

| FM/Region | Δ span | Δ F1 | Verdict |
|---|---|---|---|
| Nepal Clay | **−12%** | −2.2 | Helps |
| Nepal Prithvi | **−19%** | −1.4 | Helps |
| India TerraMind | **−35%** | −2.7 | Helps strongly |
| India Prithvi | +38% | −2.3 | Hurts (non-monotone baseline) |
| Mozambique * | +6 to +34% | −1 to −1.4 | No-op or hurts (no gradient to fix) |

See `docs/PHASE4_SCALEPOOL_RESULTS.md` for the full table.

---

## What's not in this repo (yet)

- **Per-pixel evaluation methodology**: the cleanest fix for the
  n_tokens-asymmetry artifact identified in Phase 5b. Left as the
  natural future-work hook.
- **Layer-wise probes for Prithvi/TerraMind**: requires hook plumbing
  in their terratorch backbones. Clay-only result is in
  `docs/PHASE_C_LAYERWISE.md`.
- **Burn-scar and heat-stress tasks**: deferred. Phase 2 cross-task
  validation went through tree-cover labels instead (see
  `docs/PHASE2_TREES_RESULTS.md`).
- **AnySat with output='dense'**: the wrapper uses `output='tile'` for
  consistent dimensions across FMs; switching to dense mode would
  enable proper per-polygon AnySat analysis, deferred.

---

## License

Apache 2.0.
