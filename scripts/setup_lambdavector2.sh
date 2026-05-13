#!/usr/bin/env bash
# Phase 0 setup script for the Knox lambdavector2 server.
# Idempotent: safe to re-run. Logs every step. Fails fast on errors.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/scaleshift-bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
WANDB_PROJECT="${WANDB_PROJECT:-scaleshift-bench}"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[setup-error]\033[0m %s\n" "$*" >&2; exit 1; }

# ---- 1. Prerequisites ------------------------------------------------------
log "Checking prerequisites"
command -v git  >/dev/null || die "git not installed"
command -v curl >/dev/null || die "curl not installed"

if ! command -v nvidia-smi >/dev/null; then
    die "nvidia-smi not found - is this lambdavector2? Are GPU drivers loaded?"
fi
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"

# ---- 2. uv (env manager) ---------------------------------------------------
if ! command -v uv >/dev/null; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ---- 3. Repo layout --------------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
    die "Repo not found at $REPO_DIR. Clone it first: git clone <url> $REPO_DIR"
fi
cd "$REPO_DIR"
log "Working in $REPO_DIR"

# ---- 4. Python env ---------------------------------------------------------
log "Creating venv with Python $PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" --seed
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing PyTorch (CUDA $CUDA_VERSION wheels)"
uv pip install --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}" \
    "torch>=2.2,<2.6" "torchvision>=0.17"

log "Installing base package"
uv pip install -e .

log "Installing dev extras"
uv pip install -e ".[dev]"

# ---- 5. Model extras (each in a guard so a single failure doesn't abort) ---
for extra in clay prithvi terramind anysat presto; do
    log "Installing $extra extras"
    if ! uv pip install -e ".[${extra}]"; then
        printf "\033[1;33m[setup-warn]\033[0m Failed to install %s extras. \
Continuing - the wrapper will be flagged as skipped during verify_install.py.\n" "$extra"
    fi
done

# ---- 6. Sanity checks ------------------------------------------------------
log "Smoke test: import package"
python -c "import scaleshift; print('scaleshift', scaleshift.__version__)"

log "Smoke test: import registry"
python -c "from scaleshift.model_zoo import list_models; print(list_models())"

log "Running CPU-only pytest"
pytest -m "not gpu" --maxfail=5 -q || die "pytest failed - inspect output above"

# ---- 7. GEE auth (interactive: prompts you to paste a token) ---------------
if [ ! -f "$HOME/.config/earthengine/credentials" ]; then
    log "Earth Engine credentials not found. Run 'earthengine authenticate' interactively."
else
    log "Earth Engine credentials present"
fi

# ---- 8. W&B login (interactive on first run) -------------------------------
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    log "W&B not configured. Run 'wandb login' if you want experiment tracking."
fi

log "Setup complete. Next: python scripts/verify_install.py --device cuda"
