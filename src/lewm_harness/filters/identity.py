"""Passthrough baseline filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lewm_harness.filters.base import RegimeFilter
from lewm_harness.types import Bar, FilterState


class IdentityFilter(RegimeFilter):
    """Report the raw close and a trailing return volatility.

    ``filtered_price`` is ``obs.close``. ``regime_prob`` and ``surprise``
    are always 0, so the strategy overlay does not react to this filter.
    ``pred_vol`` is the population standard deviation of the last
    ``vol_window`` simple returns, including the current bar. Fewer than
    two returns yields 0.
    """

    def __init__(self, vol_window: int = 20) -> None:
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        self.vol_window = vol_window
        self._returns: list[float] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """No parameters. The argument is accepted to honor the interface."""

        del train_df

    def reset(self) -> None:
        self._returns = []

    def update(self, obs: Bar) -> FilterState:
        self._returns.append(float(obs.ret))
        window = self._returns[-self.vol_window :]
        if len(window) < 2:
            pred_vol = 0.0
        else:
            pred_vol = float(np.std(np.asarray(window, dtype=np.float64), ddof=0))
        return FilterState(
            filtered_price=float(obs.close),
            regime_prob=0.0,
            surprise=0.0,
            pred_vol=pred_vol,
        )
