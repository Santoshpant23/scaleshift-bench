"""AnySat wrapper (CVPR 2025, Astruc et al.).

Reference (source of truth):
    https://raw.githubusercontent.com/gastruc/AnySat/main/hubconf.py
    https://raw.githubusercontent.com/gastruc/AnySat/main/README.md
    Loaded via ``torch.hub.load('gastruc/anysat', 'anysat', pretrained=True)``.

AnySat enforces 5D inputs per modality: ``[B, T, C, H, W]``. For each
time-series modality it also requires a ``{modality}_dates`` tensor of shape
``[B, T]`` containing day-of-year integers (01/01 = 0, 31/12 = 364).

``patch_size`` is specified at call time in meters and must be a multiple of 10.
README examples use ``patch_size=10`` and ``patch_size=20``. ``40`` is NOT
documented as a supported value. We default to **20 m** so the token tile is
2x2 pixels at S2 10 m GSD. For Phase 3 mechanistic analysis, this means AnySat
has more tokens per chip than Clay/Prithvi/TerraMind, which is fine - the
patch-boundary effect still applies, just at a finer granularity.
"""

from __future__ import annotations

from datetime import datetime, timezone
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


class AnySatFoundationModel(FoundationModel):
    name: ClassVar[str] = "anysat"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 240
    # patch_size_px = patch_size_m / gsd_m = 20 / 10 = 2 at S2 native.
    patch_size_px: ClassVar[int | None] = 2
    pretrained_id: ClassVar[str | None] = "gastruc/anysat"
    patch_size_m: ClassVar[int] = 20  # 20 m is the largest documented value in the README

    def load(self) -> None:
        if self._loaded:
            return
        try:
            self._model = torch.hub.load(
                "gastruc/anysat",
                "anysat",
                pretrained=True,
                flash_attn=False,
                trust_repo=True,
            )
        except Exception as e:
            raise FoundationModelNotInstalledError(
                f"Failed to load AnySat via torch.hub: {e}. Confirm internet access "
                "and that the gastruc/anysat repo is reachable from lambdavector2."
            ) from e
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    # AnySat's S2 projector was trained on 10 bands (the 10/20 m bands; B01 and
    # B09 are 60 m bands that AnySat does not consume). Selecting the canonical
    # 10-band subset prevents a matmul shape mismatch against the [10, 96] weight.
    S2_BANDS: ClassVar[list[str]] = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    ]

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        if chip.s2 is None:
            raise ValueError("AnySat preprocessing requires S2 data.")
        s2 = chip.select_s2_bands(self.S2_BANDS).astype(np.float32)
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        if t.dim() == 4:
            t = F.interpolate(
                t,
                size=(self.default_input_size_px, self.default_input_size_px),
                mode="bilinear",
                align_corners=False,
            )
        # AnySat enforces 5D inputs per modality: [B, T, C, H, W]. T=1 for single-time.
        t = t.unsqueeze(1)
        # s2_dates: day-of-year integers, shape [B, T] (per README convention).
        date = chip.date or datetime(2024, 6, 15, tzinfo=timezone.utc)
        doy = date.timetuple().tm_yday - 1  # 0..364
        s2_dates = torch.tensor([[doy]], dtype=torch.long, device=self.device)
        return {"s2": t, "s2_dates": s2_dates}

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        with torch.no_grad():
            tokens = self._model(
                batch,
                patch_size=self.patch_size_m,
                output="patch" if return_tokens else "tile",
            )
        if tokens.dim() == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        features = tokens.mean(dim=1) if tokens.dim() == 3 else tokens
        return ModelOutput(
            tokens=tokens if return_tokens and tokens.dim() == 3 else None,
            features=features,
            attention=None,
        )
