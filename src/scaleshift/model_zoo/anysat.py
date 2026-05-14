"""AnySat wrapper (CVPR 2025, Astruc et al.).

Reference:
    https://github.com/gastruc/AnySat
    https://arxiv.org/abs/2412.14123
    Loaded via ``torch.hub.load('gastruc/anysat', 'anysat', pretrained=True)``.

AnySat is a JEPA-based, multi-resolution, multi-modal FM. ``patch_size`` is
specified at call time **in meters**, not pixels. At S2 native 10 m GSD:
    patch_size=10  -> 1 px per token  (degenerate, do not use for patch-boundary analysis)
    patch_size=20  -> 2 px per token  (matches Sentinel-2 10 m / 20 m hybrid)
    patch_size=40  -> 4 px per token  (default useful granularity)

We expose ``patch_size_m`` (the model's native arg) and derive ``patch_size_px``
from the chip's GSD at preprocess time. For Phase 3 mechanistic analysis we use
``patch_size_m=40`` so the token tile is 4×4 pixels — comparable to other FMs.
"""

from __future__ import annotations

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
    default_input_size_px: ClassVar[int] = 240  # AnySat prefers sizes divisible by 40m / 10m = 4
    # patch_size_px below = patch_size_m / gsd_m. We pick 40m → 4 px at S2 10m GSD,
    # consistent with the patch-boundary analysis in Phase 3.
    patch_size_px: ClassVar[int | None] = 4
    pretrained_id: ClassVar[str | None] = "gastruc/anysat"
    patch_size_m: ClassVar[int] = 40

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch.hub  # noqa: F401  (torch.hub is part of torch, but guard anyway)
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "torch.hub unavailable — should not happen with PyTorch."
            ) from e
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

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        if chip.s2 is None:
            raise ValueError("AnySat preprocessing requires S2 data.")
        s2 = chip.s2.astype(np.float32)
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        if t.dim() == 4:
            t = F.interpolate(
                t,
                size=(self.default_input_size_px, self.default_input_size_px),
                mode="bilinear",
                align_corners=False,
            )
        # AnySat enforces 5D inputs per modality: [B, T, C, H, W]. For
        # single-time chips we insert T=1.
        t = t.unsqueeze(1)
        return {"s2": t}

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
