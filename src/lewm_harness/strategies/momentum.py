"""Fast/slow EWMA crossover momentum strategy."""

from __future__ import annotations

import numpy as np

from lewm_harness.strategies.base import Strategy
from lewm_harness.types import Bar, FilterState


def _ewma(previous: float, value: float, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    return alpha * value + (1.0 - alpha) * previous


class MomentumStrategy(Strategy):
    """Go with a fast EWMA of price when it leads the slow EWMA.

    Both averages update from the current close only. The raw signal is
    ``(fast - slow) / (|slow| * signal_scale)``, clipped to ``[-1, 1]``.
    ``signal_scale`` is the relative gap that maps to a full position
    (default 1%). The signal stays 0 until ``slow`` observations have been
    seen, so the startup transient is not traded.
    """

    def __init__(
        self,
        fast: int = 8,
        slow: int = 24,
        signal_scale: float = 0.01,
        *,
        surprise_scale: float = 1.0,
        surprise_threshold: float = 0.0,
        regime_cutoff: float | None = None,
    ) -> None:
        if fast < 1 or slow <= fast:
            raise ValueError("require 1 <= fast < slow")
        if signal_scale == 0.0:
            raise ValueError("signal_scale must be non-zero")
        super().__init__(
            surprise_scale=surprise_scale,
            surprise_threshold=surprise_threshold,
            regime_cutoff=regime_cutoff,
        )
        self.fast = fast
        self.slow = slow
        self.signal_scale = float(signal_scale)
        self._fast: float | None = None
        self._slow: float | None = None
        self._count = 0

    def reset(self) -> None:
        self._fast = None
        self._slow = None
        self._count = 0

    def step(self, obs: Bar, state: FilterState) -> float:
        price = float(obs.close)
        self._count += 1
        if self._fast is None or self._slow is None:
            self._fast = price
            self._slow = price
            raw = 0.0
        else:
            self._fast = _ewma(self._fast, price, self.fast)
            self._slow = _ewma(self._slow, price, self.slow)
            if self._count < self.slow or self._slow == 0.0:
                raw = 0.0
            else:
                gap = (self._fast - self._slow) / (abs(self._slow) * self.signal_scale)
                raw = float(np.clip(gap, -1.0, 1.0))
        return self.apply_overlay(raw, state)
