"""Rolling spread z-score pairs strategy."""

from __future__ import annotations

import numpy as np

from lewm_harness.strategies.base import Strategy
from lewm_harness.types import Bar, FilterState


class StatArbStrategy(Strategy):
    """Mean-revert the contemporaneous log spread in ``obs.extras``.

    The spread has to be known at the same timestamp as the bar. The
    backtester copies extra columns from the current row only, and the
    pair simulator stores ``log(close_a) - log(close_b)`` there. A high
    spread produces a negative position, which shorts leg A against leg B
    when ``ret`` is the equal-weight pair return.

    The z-score window includes the current spread. The signal is 0 until
    the window is full.
    """

    def __init__(
        self,
        window: int = 30,
        z_scale: float = 2.0,
        *,
        surprise_scale: float = 1.0,
        surprise_threshold: float = 0.0,
        regime_cutoff: float | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if z_scale == 0.0:
            raise ValueError("z_scale must be non-zero")
        super().__init__(
            surprise_scale=surprise_scale,
            surprise_threshold=surprise_threshold,
            regime_cutoff=regime_cutoff,
        )
        self.window = window
        self.z_scale = float(z_scale)
        self._spreads: list[float] = []

    def reset(self) -> None:
        self._spreads = []

    def step(self, obs: Bar, state: FilterState) -> float:
        if obs.extras is None or "spread" not in obs.extras:
            raise KeyError("StatArbStrategy requires obs.extras['spread']")
        spread = float(obs.extras["spread"])
        self._spreads.append(spread)
        if len(self._spreads) < self.window:
            raw = 0.0
        else:
            window = np.asarray(self._spreads[-self.window :], dtype=np.float64)
            scale = float(window.std(ddof=0))
            if scale == 0.0:
                raw = 0.0
            else:
                z_score = (spread - float(window.mean())) / scale
                raw = float(np.clip(-z_score / self.z_scale, -1.0, 1.0))
        return self.apply_overlay(raw, state)
