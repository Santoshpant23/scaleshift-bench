#!/usr/bin/env python
"""Phase 0 verification script.

Run on lambdavector2 after setup. For each FM in the registry it:
  1. Instantiates the wrapper
  2. Loads weights onto CUDA
  3. Runs encode() on a synthetic chip
  4. Reports embedding shape, GPU memory, and wall time

Exit code 0 iff every requested model passes. Skip a model with --skip clay,presto.

Example:
    python scripts/verify_install.py --device cuda
    python scripts/verify_install.py --device cuda --skip presto,anysat
    python scripts/verify_install.py --device cuda --chip tests/fixtures/terai_sample.tif
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from scaleshift.data.chip import Chip
from scaleshift.model_zoo import (
    FoundationModelNotInstalledError,
    get_model,
    list_models,
)
from scaleshift.utils.logging import banner, get_logger


log = get_logger("verify")


def gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def verify_one(name: str, chip: Chip, device: str) -> dict[str, object]:
    record: dict[str, object] = {"name": name, "ok": False, "skipped": False}
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        fm = get_model(name, device=device)
        record["wrapper_info"] = fm.describe()
        log.info("[%s] loading weights...", name)
        t0 = time.perf_counter()
        fm.load()
        record["load_s"] = round(time.perf_counter() - t0, 2)
        log.info("[%s] running inference on synthetic chip...", name)
        t1 = time.perf_counter()
        out = fm.predict(chip)
        record["forward_s"] = round(time.perf_counter() - t1, 2)
        record["features_shape"] = list(out.features.shape)
        record["tokens_shape"] = list(out.tokens.shape) if out.tokens is not None else None
        record["features_dtype"] = str(out.features.dtype)
        record["peak_gpu_mb"] = round(gpu_mb(), 1)
        record["ok"] = True
        log.info(
            "[%s] OK  load=%.1fs  fwd=%.2fs  feat=%s  gpu=%.0fMB",
            name,
            record["load_s"], record["forward_s"], record["features_shape"], record["peak_gpu_mb"],
        )
    except FoundationModelNotInstalledError as e:
        record["skipped"] = True
        record["reason"] = str(e)
        log.warning("[%s] SKIPPED - %s", name, e)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        log.error("[%s] FAILED - %s", name, e)
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # Presto's preprocess() is a Phase 1 deliverable (see presto.py docstring).
    # Skip it by default until the dataops adapter is implemented.
    p.add_argument(
        "--skip",
        default="presto",
        help="comma-separated model names to skip (default: 'presto')",
    )
    p.add_argument(
        "--chip", type=Path, default=None,
        help="path to a real GeoTIFF; falls back to synthetic if omitted",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/verify_install.json"))
    p.add_argument("--size-px", type=int, default=224)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA requested but not available.")
        return 2

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    names = [n for n in list_models() if n not in skip]

    if args.chip and args.chip.exists():
        log.info("Loading chip from %s", args.chip)
        chip = Chip.from_geotiff(args.chip)
    else:
        log.info("Using synthetic chip (size_px=%d)", args.size_px)
        chip = Chip.synthetic(size_px=args.size_px)

    banner(f"Phase 0 verification on {args.device}")
    results = [verify_one(n, chip, args.device) for n in names]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)

    passed = sum(1 for r in results if r["ok"])
    skipped = sum(1 for r in results if r["skipped"])
    failed = len(results) - passed - skipped
    banner(f"Summary: {passed} passed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
