"""Placeholder for the Phase 2/3 LeWM world-model filter."""

from __future__ import annotations

import pandas as pd

from lewm_harness.filters.base import RegimeFilter
from lewm_harness.types import Bar, FilterState


class LeWMFilter(RegimeFilter):
    """Interface stub for a JEPA-style market world model.

    Phase 2/3 can drop a PyTorch checkpoint into this class without
    touching the engine. The causal contract stays the same as the
    classical filters:

    * ``fit`` may read a training frame and store parameters or a
      loaded checkpoint. It must not keep the frame for later smoothing.
    * ``reset`` clears recurrent latent state.
    * ``update`` consumes only the current :class:`Bar` plus that latent
      state and returns a :class:`FilterState`. Map the model's one-step
      surprise onto ``surprise`` and its high-vol probability onto
      ``regime_prob``.

    Until that checkpoint path exists, ``update`` raises
    ``NotImplementedError``. ``checkpoint_path`` is reserved for the
    weights file Phase 2 will load.
    """

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        if "ret" not in train_df.columns:
            raise ValueError("LeWMFilter.fit expected a 'ret' column")
        # Record that fit was called, but do not retain the frame.
        self._fitted = True

    def reset(self) -> None:
        return None

    def update(self, obs: Bar) -> FilterState:
        del obs
        raise NotImplementedError(
            "LeWMFilter is a Phase 1 stub. Implement the causal PyTorch "
            "checkpoint update before calling update "
            f"(checkpoint_path={self.checkpoint_path!r}, fitted={self._fitted})."
        )
