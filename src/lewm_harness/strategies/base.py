"""Strategy interface and the shared filter overlay."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lewm_harness.types import Bar, FilterState


class Strategy(ABC):
    """Map the current bar and filter state to a target position in [-1, 1].

    Subclasses compute a raw signal from past and present observations,
    then pass it through :meth:`apply_overlay`. The overlay is the shared
    regime rule:

    * if ``regime_cutoff`` is set and ``state.regime_prob`` exceeds it,
      the position is paused at 0;
    * otherwise the raw signal is scaled by
      ``1 / (1 + surprise_scale * max(0, surprise - surprise_threshold))``.

    The backtester clips again, so a strategy that forgets to clip cannot
    publish a position outside ``[-1, 1]``.
    """

    def __init__(
        self,
        *,
        surprise_scale: float = 1.0,
        surprise_threshold: float = 0.0,
        regime_cutoff: float | None = None,
    ) -> None:
        if surprise_scale < 0.0:
            raise ValueError("surprise_scale must be non-negative")
        if regime_cutoff is not None and not 0.0 <= regime_cutoff <= 1.0:
            raise ValueError("regime_cutoff must be in [0, 1] when set")
        self.surprise_scale = float(surprise_scale)
        self.surprise_threshold = float(surprise_threshold)
        self.regime_cutoff = None if regime_cutoff is None else float(regime_cutoff)

    @abstractmethod
    def reset(self) -> None:
        """Forget rolling signal state. Overlay settings stay in place."""

    @abstractmethod
    def step(self, obs: Bar, state: FilterState) -> float:
        """Return the target position for the current bar."""

    def apply_overlay(self, raw_position: float, state: FilterState) -> float:
        """Scale or pause ``raw_position`` using the current filter state."""

        if self.regime_cutoff is not None and state.regime_prob > self.regime_cutoff:
            return 0.0
        excess = max(0.0, state.surprise - self.surprise_threshold)
        scale = 1.0 / (1.0 + self.surprise_scale * excess)
        return float(np.clip(raw_position * scale, -1.0, 1.0))
