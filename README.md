# scaleshift-bench

**ScaleShift-Bench:** scale-aware benchmarking and adaptation of geospatial foundation models for smallholder agriculture.

Target venue: NeurIPS Datasets & Benchmarks 2027 (primary).

## What this repo will be

1. A unified wrapper around five geospatial foundation models: Clay, Prithvi-EO-2.0, Presto, AnySat, TerraMind.
2. A field-size-stratified benchmark for crop classification in Nepal's Terai + India + Mozambique + Vietnam.
3. A mechanistic analysis of why patch-based FMs fail on smallholder fields.
4. `ScalePool`: a drop-in scale-aware probing head that closes the gap.
5. Public HuggingFace releases + Gradio demo.

Phases are tracked in `docs/PLAN.md` (TODO) and on the task list.

## Repo layout

```
scaleshift-bench/
├── pyproject.toml
├── src/scaleshift/
│   ├── data/chip.py          # Chip dataclass
│   ├── model_zoo/
│   │   ├── base.py           # FoundationModel ABC + ModelOutput
│   │   ├── clay.py
│   │   ├── prithvi.py
│   │   ├── presto.py
│   │   ├── anysat.py
│   │   ├── terramind.py
│   │   └── __init__.py       # registry: get_model(name)
│   └── utils/logging.py
├── scripts/
│   ├── setup_lambdavector2.sh
│   ├── verify_install.py     # GPU smoke test for all 5 FMs
│   └── download_sample_chip.py
└── tests/
    ├── test_chip.py
    ├── test_registry.py
    └── test_models_smoke.py
```

## Phase 0 setup on lambdavector2

```bash
# 1. SSH in
ssh lambdavector2

# 2. Clone (assumes you've pushed this repo to GitHub)
git clone <repo-url> ~/scaleshift-bench
cd ~/scaleshift-bench

# 3. Run setup (installs uv, venv, torch, all FM extras)
bash scripts/setup_lambdavector2.sh

# 4. Authenticate Earth Engine (interactive)
.venv/bin/earthengine authenticate

# 5. (Optional) pull a real Terai chip
.venv/bin/python scripts/download_sample_chip.py

# 6. Run the full GPU verification across all 5 FMs
.venv/bin/python scripts/verify_install.py --device cuda \
    --chip tests/fixtures/terai_sample.tif
```

`verify_install.py` writes `outputs/verify_install.json` with load time, forward
time, peak GPU memory, and feature shape per model. **Send that file back** —
it's the input to Verification Agent A.

## What "Phase 0 done" means

- `pytest -m "not gpu"` passes locally
- `scripts/verify_install.py --device cuda` reports `ok=true` for at least 4 of 5 models
- Verification Agent A signs off on the report

## Things that are deliberately placeholders right now

- Normalization constants in `clay.py` and `prithvi.py` are illustrative. Replace with the canonical numbers from the upstream model cards before Phase 2.
- The `encode()` forward path of each wrapper depends on the upstream API. The current code is best-guess from each model's docs and **must** be reconciled against the actual installed package version when `verify_install.py` runs.
- `patch_size_px` per model is declared in each wrapper class. **Triple-check these** during Verification Agent A — every downstream mechanistic claim depends on them being correct.

## License

Apache 2.0.
