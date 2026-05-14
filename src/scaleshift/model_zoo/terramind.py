"""TerraMind wrapper (IBM / ESA, 2025).

Reference:
    https://ibm.github.io/terramind/
    HF: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base  (lowercase 'base')
    Loaded via terratorch's BACKBONE_REGISTRY.

TerraMind is an any-to-any generative FM across 9 modalities. For this project
we use its S2L2A encoder for representation extraction. The model is invoked
directly (``__call__``) with a dict of modality tensors; ``forward_features``
is not exposed on the terratorch backbone.

Patch tokenization: 16 px on a 224 px input (ViT-B-style 14×14 = 196 tokens).
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


# TerraMind S2L2A patch-embed expects 12 channels per token. The proj weight
# shape is 3072 x 768 (3072 = 16*16*12). Supplying 10 bands fails with a matmul
# shape mismatch. Canonical ESA L2A order (excludes B10 which is L1C-only).
TERRAMIND_S2_BANDS: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]


class TerraMindFoundationModel(FoundationModel):
    name: ClassVar[str] = "terramind-v1-base"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 224
    patch_size_px: ClassVar[int | None] = 16
    pretrained_id: ClassVar[str | None] = "ibm-esa-geospatial/TerraMind-1.0-base"
    backbone_name: ClassVar[str] = "terramind_v1_base"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "TerraMind deps not installed. Run: pip install -e '.[terramind]'"
            ) from e

        self._model = BACKBONE_REGISTRY.build(
            self.backbone_name,
            pretrained=True,
            modalities=["S2L2A"],
        )
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        s2 = chip.select_s2_bands(TERRAMIND_S2_BANDS).astype(np.float32)
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        t = F.interpolate(
            t,
            size=(self.default_input_size_px, self.default_input_size_px),
            mode="bilinear",
            align_corners=False,
        )
        return {"S2L2A": t}

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        with torch.no_grad():
            tokens = self._model(batch)
        if tokens.dim() == 4:
            tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, D]
        features = tokens.mean(dim=1)
        return ModelOutput(
            tokens=tokens if return_tokens else None,
            features=features,
            attention=None,
        )
