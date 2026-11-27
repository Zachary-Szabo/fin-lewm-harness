"""Rolling z-score mean-reversion strategy."""

from __future__ import annotations

import numpy as np

from lewm_harness.strategies.base import Strategy
from lewm_harness.types import Bar, FilterState


class MeanReversionStrategy(Strategy):
    """Fade deviations of price from its trailing mean.

    The z-score uses the last ``window`` closes, including the current
    close, which is known at decision time. The position earns the *next*
    return inside the backtester. Until the window is full the signal is 0.
    A positive z-score (price above its mean) produces a negative position.
    """

    def __init__(
        self,
        window: int = 20,
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
        self._closes: list[float] = []

    def reset(self) -> None:
        self._closes = []

    def step(self, obs: Bar, state: FilterState) -> float:
        self._closes.append(float(obs.close))
        if len(self._closes) < self.window:
            raw = 0.0
        else:
            window = np.asarray(self._closes[-self.window :], dtype=np.float64)
            scale = float(window.std(ddof=0))
            if scale == 0.0:
                raw = 0.0
            else:
                z_score = (float(obs.close) - float(window.mean())) / scale
                raw = float(np.clip(-z_score / self.z_scale, -1.0, 1.0))
        return self.apply_overlay(raw, state)
