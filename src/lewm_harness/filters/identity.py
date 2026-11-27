"""Passthrough baseline filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lewm_harness.filters.base import RegimeFilter
from lewm_harness.types import Bar, FilterState


class IdentityFilter(RegimeFilter):
    """Report the raw close and a trailing return volatility.

    ``filtered_price`` is ``obs.close``. ``regime_prob`` is 0. The raw
    surprise is ``|ret| / pred_vol`` (0 until two returns exist). ``fit``
    replays the training frame only and replaces that raw score with its
    training z-score. ``pred_vol`` is the population standard deviation of
    the last ``vol_window`` simple returns, including the current bar.
    Nothing in ``update`` reads a future bar.
    """

    def __init__(self, vol_window: int = 20) -> None:
        super().__init__()
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        self.vol_window = vol_window
        self._returns: list[float] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """Calibrate surprise and the vol target from a causal train replay."""

        self._calibrate_online(train_df)

    def reset(self) -> None:
        self._returns = []

    def update(self, obs: Bar) -> FilterState:
        self._returns.append(float(obs.ret))
        window = self._returns[-self.vol_window :]
        if len(window) < 2:
            pred_vol = 0.0
            raw_surprise = 0.0
        else:
            pred_vol = float(np.std(np.asarray(window, dtype=np.float64), ddof=0))
            raw_surprise = 0.0 if pred_vol <= 1e-12 else abs(float(obs.ret)) / pred_vol
        return FilterState(
            filtered_price=float(obs.close),
            regime_prob=0.0,
            surprise=self.standardize_surprise(raw_surprise),
            pred_vol=pred_vol,
        )
