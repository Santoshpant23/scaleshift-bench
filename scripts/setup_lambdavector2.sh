#!/usr/bin/env bash
# Phase 0 setup for the Knox lambdavector2 server.
#
# Idempotent: safe to re-run after a failed step. Reuses an existing .venv if
# present (so torch + CUDA libs are not re-downloaded). Auto-detects the repo
# location from the script's own path, so it works from any clone location
# including paths with spaces.
#
# No root required. Everything installs into:
#   - $REPO_DIR/.venv/   (Python packages)
#   - $HOME/.local/bin/  (uv binary)
#   - $HOME/.cache/uv/   (uv build cache)
#   - $HOME/.cache/huggingface/  (model weights, default HF cache)
#   - $HOME/.config/earthengine/ (GEE credentials)
#
# To share a HuggingFace model cache with other users on the box, set
# HF_HOME=/shared/path/hf_cache before running.

set -euo pipefail

# Resolve script and repo directories, surviving symlinks and spaces.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu121}"            # set to cu118 if CUDA 11.8 driver

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup-warn]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[setup-error]\033[0m %s\n" "$*" >&2; exit 1; }

# ---- 1. Prerequisites ------------------------------------------------------
log "Checking prerequisites"
command -v git  >/dev/null || die "git not installed"
command -v curl >/dev/null || die "curl not installed"

if ! command -v nvidia-smi >/dev/null; then
    die "nvidia-smi not found - is this lambdavector2? Are GPU drivers loaded?"
fi
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"

# ---- 2. uv (env manager) ---------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    log "Installing uv to \$HOME/.local/bin (no root needed)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ---- 3. Repo layout --------------------------------------------------------
[ -d "$REPO_DIR" ] || die "Repo directory does not exist: $REPO_DIR"
[ -f "$REPO_DIR/pyproject.toml" ] || die "No pyproject.toml at $REPO_DIR - wrong directory?"
cd "$REPO_DIR"
log "Working in $REPO_DIR"

# ---- 4. Python env ---------------------------------------------------------
if [ ! -d ".venv" ]; then
    log "Creating venv with Python $PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" --seed
else
    log "Reusing existing .venv (delete it manually if you want a clean rebuild)"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing PyTorch ($CUDA_TAG wheels)"
uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "torch>=2.2,<2.6" "torchvision>=0.17"

log "Installing base package (editable)"
uv pip install -e .

log "Installing dev extras"
uv pip install -e ".[dev]"

# ---- 5. Model extras (each guarded so a single failure doesn't abort) -----
# AnySat is loaded at runtime via torch.hub.load (no pip install needed).
# Presto pins an old earthengine-api and is deferred to a Phase 1 sub-env.
declare -a EXTRA_RESULTS=()
for extra in clay prithvi terramind; do
    log "Installing model extra: $extra"
    if uv pip install -e ".[${extra}]"; then
        EXTRA_RESULTS+=("$extra: ok")
    else
        warn "Failed to install '$extra' extras. The wrapper will be skipped during verify_install.py."
        EXTRA_RESULTS+=("$extra: FAILED")
    fi
done
EXTRA_RESULTS+=("anysat: runtime via torch.hub (no pip install)")
EXTRA_RESULTS+=("presto: deferred to Phase 1 sub-env")

# ---- 6. Sanity checks ------------------------------------------------------
log "Smoke test: import package"
python -c "import scaleshift; print('scaleshift', scaleshift.__version__)"

log "Smoke test: list models in registry"
python -c "from scaleshift.model_zoo import list_models; print(list_models())"

log "Running CPU-only pytest"
if ! pytest -m "not gpu" --maxfail=5 -q; then
    warn "CPU pytest had failures - inspect output above. Continuing anyway."
fi

# ---- 7. GEE auth (interactive — user must run separately) -----------------
# Note: default 'earthengine authenticate' requires gcloud CLI (system tool).
# Use --auth_mode=notebook on a headless server without root access.
if [ -f "$HOME/.config/earthengine/credentials" ]; then
    log "Earth Engine credentials present"
else
    warn "Earth Engine credentials not found. After this script finishes:"
    warn "    source .venv/bin/activate"
    warn "    earthengine authenticate --auth_mode=notebook"
    warn "    export EE_PROJECT=<your-gcp-project-id>   # add to ~/.bashrc"
fi

# ---- 8. W&B login (optional, interactive on first run) --------------------
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    log "W&B not configured (optional). To enable: source .venv/bin/activate && wandb login"
fi

# ---- 9. Summary ------------------------------------------------------------
log "Model extra install results:"
for r in "${EXTRA_RESULTS[@]}"; do
    log "  $r"
done

log ""
log "Setup complete."
log "Next:"
log "  source .venv/bin/activate"
log "  earthengine authenticate --auth_mode=notebook   # one-time, headless-friendly"
log "  export EE_PROJECT=<your-gcp-project-id>          # add to ~/.bashrc"
log "  python scripts/download_sample_chip.py           # pull a Terai chip via GEE"
log "  python scripts/verify_install.py --device cuda \\"
log "      --chip tests/fixtures/terai_sample.tif"
