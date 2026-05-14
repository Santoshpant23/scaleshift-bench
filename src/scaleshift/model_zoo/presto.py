"""Presto wrapper (NASA Harvest).

Reference:
    https://github.com/nasaharvest/presto
    https://arxiv.org/abs/2304.14065

Presto is a **pixel-time-series** model, not an image patch model. The natural
unit is a single pixel's monthly time series across S1+S2+ERA5+SRTM. Encoder
signature:

    encoder.forward(
        x: [B, T, len(NORMED_BANDS)],
        dynamic_world: [B, T],        # required positional
        latlons: [B, 2],
        mask: [B, T, len(NORMED_BANDS)] | None,
        month: int,
        eval_task: bool = True,
    )

Presto's band layout is dictated by ``presto.dataops.BANDS`` /
``NORMED_BANDS`` / ``BANDS_GROUPS_IDX``, which concatenates S1, S2, ERA5, and
SRTM channels per timestep. A correct adapter requires:

  1. Building the canonical 17-channel layout (or whatever the installed
     version exposes), with masking for missing modalities.
  2. Supplying ``dynamic_world`` as a per-timestep integer label tensor
     (``DYNAMIC_WORLD_NULL_CLASS`` = unknown) of shape ``[B, T]``.
  3. Choosing the correct ``month`` index per acquisition.

That logic is non-trivial and depends on the installed presto version. We
defer implementation to Phase 1 and ``verify_install.py`` skips this model by
default for Phase 0. The wrapper below is structurally complete (load() works,
the ABC contract is honored) but ``preprocess()`` raises NotImplementedError
until Phase 1.

``patch_size_px`` is ``None`` here. For ScalePool comparisons, Presto is treated
as a pixel-based baseline outside the patch-aware family.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from scaleshift.data.chip import Chip
from scaleshift.model_zoo.base import (
    FoundationModel,
    FoundationModelNotInstalledError,
    ModelOutput,
)


class PrestoFoundationModel(FoundationModel):
    name: ClassVar[str] = "presto"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 1  # per-pixel
    patch_size_px: ClassVar[int | None] = None
    pretrained_id: ClassVar[str | None] = "nasaharvest/presto"
    pooling_method: ClassVar[str] = "pixel"
    has_cls_token: ClassVar[bool] = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from presto import Presto
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "Presto deps not installed. Run: pip install -e '.[presto]'"
            ) from e

        self._model = Presto.load_pretrained()
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        # TODO(Phase 1): implement the canonical chip → Presto dataops layout.
        # The adapter must:
        #   - map Chip bands to NORMED_BANDS order
        #   - construct dynamic_world placeholder ([B, T], int, DYNAMIC_WORLD_NULL_CLASS)
        #   - select month index from chip.date
        # See presto/dataops.py for the exact contract.
        raise NotImplementedError(
            "Presto preprocess() is a Phase 1 deliverable. "
            "Skip this model in verify_install.py for now via --skip presto."
        )

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        # The correct call is:
        #   encoder(x, dynamic_world, latlons=latlons, mask=mask, month=month)
        # Implemented in Phase 1 alongside preprocess().
        raise NotImplementedError("Presto encode() is a Phase 1 deliverable.")
