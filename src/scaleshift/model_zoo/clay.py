"""Clay v1.5 wrapper (Radiant Earth).

Reference:
    https://clay-foundation.github.io/model/getting-started/basic_use.html
    https://github.com/Clay-foundation/model
    Canonical metadata: configs/metadata.yaml in the Clay repo.

Clay v1.5 expects:
    - Sentinel-2 L2A bands at 10 m in raw (×10000) reflectance scale:
      B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    - A timestamps tensor of shape [B, 4] encoded as
      [sin(week*2π/52), cos(week*2π/52), sin(hour*2π/24), cos(hour*2π/24)]
    - A latlon tensor of shape [B, 4] encoded as
      [sin(lat_rad), cos(lat_rad), sin(lon_rad), cos(lon_rad)]
    - A wavelengths tensor [C] in nanometers (Sentinel-2 central wavelengths).

Patch tokenization: 8 px on a 256 px input (ViT-B-like backbone, dim 768).
"""

from __future__ import annotations

import contextlib
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from scaleshift.data.chip import Chip
from scaleshift.model_zoo.base import (
    FoundationModel,
    FoundationModelNotInstalledError,
    ModelOutput,
)


CLAY_METADATA_URL = (
    "https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml"
)
CLAY_CACHE_DIR = Path.home() / ".cache" / "scaleshift" / "clay"


def _ensure_clay_metadata_cwd() -> Path:
    """Cache Clay's configs/metadata.yaml and return the dir to chdir into.

    The Clay v1.5 checkpoint stores ``metadata_path='configs/metadata.yaml'``
    as a relative path in its saved hparams, so load_from_checkpoint fails to
    resolve the file when CWD is anything other than the Clay source tree.
    We cache the YAML under ~/.cache/scaleshift/clay/configs/ and chdir to
    its parent during load.
    """
    configs_dir = CLAY_CACHE_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = configs_dir / "metadata.yaml"
    if not metadata_file.exists():
        urllib.request.urlretrieve(CLAY_METADATA_URL, metadata_file)
    return CLAY_CACHE_DIR


@contextlib.contextmanager
def _chdir(path: Path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


# Canonical from Clay-foundation/model:configs/metadata.yaml (raw L2A scale).
# Band order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.
CLAY_S2_MEANS = np.array(
    [1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0],
    dtype=np.float32,
)
CLAY_S2_STDS = np.array(
    [1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0],
    dtype=np.float32,
)

# Sentinel-2 central wavelengths (nm) for B02..B12 in Clay's order.
CLAY_S2_WAVELENGTHS_NM = np.array(
    [493.0, 560.0, 665.0, 704.0, 740.0, 783.0, 833.0, 865.0, 1610.0, 2190.0],
    dtype=np.float32,
)


def _clay_timestamps(date: datetime) -> torch.Tensor:
    """[1, 4] = [sin(week), cos(week), sin(hour), cos(hour)]."""
    week = date.isocalendar().week
    hour = date.hour
    w = 2 * np.pi * (week / 52.0)
    h = 2 * np.pi * (hour / 24.0)
    return torch.tensor([[np.sin(w), np.cos(w), np.sin(h), np.cos(h)]], dtype=torch.float32)


def _clay_latlon(lat: float, lon: float) -> torch.Tensor:
    """[1, 4] = [sin(lat_rad), cos(lat_rad), sin(lon_rad), cos(lon_rad)]."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    return torch.tensor(
        [[np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)]],
        dtype=torch.float32,
    )


class ClayFoundationModel(FoundationModel):
    name: ClassVar[str] = "clay-v1"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 256
    patch_size_px: ClassVar[int | None] = 8
    pretrained_id: ClassVar[str | None] = "made-with-clay/Clay"
    checkpoint_filename: ClassVar[str] = "v1.5/clay-v1.5.ckpt"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from claymodel.module import ClayMAEModule
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "Clay deps not installed. Run: pip install -e '.[clay]'"
            ) from e

        ckpt_path = hf_hub_download(
            repo_id=self.pretrained_id, filename=self.checkpoint_filename
        )
        cache_dir = _ensure_clay_metadata_cwd()
        with _chdir(cache_dir):
            self._module = ClayMAEModule.load_from_checkpoint(ckpt_path)
        self._module = self._module.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        wanted = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        s2 = chip.select_s2_bands(wanted).astype(np.float32)
        # Chip stores values divided by 10000; Clay's normalization is in raw L2A.
        s2 = s2 * 10_000.0
        s2 = (s2 - CLAY_S2_MEANS[:, None, None]) / CLAY_S2_STDS[:, None, None]
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        t = F.interpolate(
            t,
            size=(self.default_input_size_px, self.default_input_size_px),
            mode="bilinear",
            align_corners=False,
        )
        date = chip.date or datetime(2024, 6, 15, tzinfo=timezone.utc)
        timestamps = _clay_timestamps(date).to(self.device).to(self.dtype)
        latlon = _clay_latlon(chip.lat, chip.lon).to(self.device).to(self.dtype)
        wavelengths = (
            torch.from_numpy(CLAY_S2_WAVELENGTHS_NM).to(self.device).to(self.dtype)
        )
        return {
            "pixels": t,
            "timestamps": timestamps,
            "latlon": latlon,
            "wavelengths": wavelengths,
            "gsd": torch.tensor([chip.gsd_m], device=self.device, dtype=self.dtype),
        }

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        encoder = self._module.model.encoder
        with torch.no_grad():
            # Clay v1.5 encoder signature (from basic_use.html): positional
            # (chips, timestamps, wavelengths). Some checkpoints additionally
            # accept latlon and gsd; we pass them via kwargs and let the
            # encoder ignore unknown ones. If your installed claymodel raises
            # TypeError on unknown kwargs, drop them.
            try:
                out = encoder(
                    batch["pixels"],
                    batch["timestamps"],
                    batch["wavelengths"],
                    latlon=batch["latlon"],
                    gsd=batch["gsd"],
                )
            except TypeError:
                out = encoder(
                    batch["pixels"], batch["timestamps"], batch["wavelengths"]
                )
        # Encoder returns either tokens [B, N, D] (CLS already prepended) or
        # a tuple (tokens, cls). Handle both.
        if isinstance(out, tuple):
            tokens, cls_tok = out
        elif isinstance(out, dict):
            tokens = out.get("patches") or out.get("tokens")
            cls_tok = out.get("cls")
        else:
            tokens = out
            cls_tok = None
        features = cls_tok if cls_tok is not None else tokens.mean(dim=1)
        return ModelOutput(
            tokens=tokens if return_tokens else None,
            features=features,
            attention=None,
        )
