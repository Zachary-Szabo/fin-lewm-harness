"""Strategy interface and the shared filter overlay."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lewm_harness.types import Bar, FilterState


class Strategy(ABC):
    """Map the current bar and filter state to a target position in [-1, 1].

    Subclasses compute a raw signal from past and present observations,
    then pass it through :meth:`apply_overlay`. The overlay is shared, but
    the high-vol pause is strategy-specific via ``regime_cutoff``:

    * if ``regime_cutoff`` is set and ``state.regime_prob`` exceeds it,
      the position is paused at 0 (mean reversion and pairs do this;
      momentum leaves the cutoff unset);
    * otherwise the raw signal is scaled by
      ``1 / (1 + surprise_scale * max(0, surprise - surprise_threshold))``,
      where ``surprise`` is the filter's training-set z-score;
    * if a vol target is active, that result is multiplied by
      ``min(1, target_vol / max(pred_vol, vol_eps))``. An explicit
      ``target_vol`` wins. Otherwise the backtester fills the target with
      the filter's median training ``pred_vol``.

    The backtester clips again, so a strategy that forgets to clip cannot
    publish a position outside ``[-1, 1]``.
    """

    def __init__(
        self,
        *,
        surprise_scale: float = 1.0,
        surprise_threshold: float = 0.0,
        regime_cutoff: float | None = None,
        target_vol: float | None = None,
        vol_eps: float = 1e-8,
    ) -> None:
        if surprise_scale < 0.0:
            raise ValueError("surprise_scale must be non-negative")
        if regime_cutoff is not None and not 0.0 <= regime_cutoff <= 1.0:
            raise ValueError("regime_cutoff must be in [0, 1] when set")
        if target_vol is not None and target_vol <= 0.0:
            raise ValueError("target_vol must be positive")
        if vol_eps <= 0.0:
            raise ValueError("vol_eps must be positive")
        self.surprise_scale = float(surprise_scale)
        self.surprise_threshold = float(surprise_threshold)
        self.regime_cutoff = None if regime_cutoff is None else float(regime_cutoff)
        self.target_vol = None if target_vol is None else float(target_vol)
        self.vol_eps = float(vol_eps)
        self._runtime_target_vol: float | None = None

    def bind_runtime_vol_target(self, reference_vol: float | None) -> None:
        """Select this run's inverse-vol target.

        ``target_vol`` is the caller's override. Otherwise a positive
        ``reference_vol`` from the fitted filter is the calibrated target.
        """

        if self.target_vol is not None:
            self._runtime_target_vol = self.target_vol
        elif reference_vol is not None and reference_vol > 0.0:
            self._runtime_target_vol = float(reference_vol)
        else:
            self._runtime_target_vol = None

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
        surprise_multiplier = 1.0 / (1.0 + self.surprise_scale * excess)
        target = self.target_vol if self.target_vol is not None else self._runtime_target_vol
        if target is None:
            vol_multiplier = 1.0
        else:
            vol_multiplier = min(1.0, target / max(state.pred_vol, self.vol_eps))
        return float(np.clip(raw_position * surprise_multiplier * vol_multiplier, -1.0, 1.0))
