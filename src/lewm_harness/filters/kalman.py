"""Causal one-dimensional local-level Kalman filter on log price."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lewm_harness.filters.base import RegimeFilter
from lewm_harness.types import Bar, FilterState


class KalmanFilter(RegimeFilter):
    """Local-level filter: a random walk observed in Gaussian noise.

    State and observation, with ``y_t = log(close_t)``::

        x_t = x_{t-1} + w_t,   w_t ~ N(0, q)
        y_t = x_t + v_t,       v_t ~ N(0, r)

    ``fit`` estimates ``q`` and ``r`` from the training log-price
    increments only. For a local-level model the increment variance and
    lag-1 autocovariance identify the noises::

        Var(dy) = q + 2 r,    Cov(dy_t, dy_{t-1}) = -r

    ``update`` is the standard predict/correct step. It never revises
    past states. The raw surprise is the absolute innovation
    ``|y_t - x_{t|t-1}| / sqrt(S_t)``; ``fit`` turns that into a z-score
    against a causal replay of the training prices only. ``pred_vol`` is
    the innovation standard deviation ``sqrt(S_t)``. ``regime_prob`` is 0
    because this model has no regime latent. ``filtered_price`` is
    ``exp(x_{t|t})``.
    """

    def __init__(self, process_var: float = 1e-6, obs_var: float = 1e-4) -> None:
        super().__init__()
        if process_var <= 0.0 or obs_var <= 0.0:
            raise ValueError("process_var and obs_var must be positive")
        self._q = float(process_var)
        self._r = float(obs_var)
        self._x: float | None = None
        self._p: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        close = train_df["close"].to_numpy(dtype=np.float64)
        if close.size >= 5 and np.all(close > 0.0):
            increments = np.diff(np.log(close))
            if increments.size >= 5:
                centered = increments - increments.mean()
                variance = float(np.dot(centered, centered) / centered.size)
                lag = float(np.dot(centered[1:], centered[:-1]) / centered.size)
                obs_var = max(-lag, 1e-12)
                process_var = max(variance - 2.0 * obs_var, 1e-12)
                self._q = process_var
                self._r = obs_var
        if close.size == 0 or np.any(close <= 0.0):
            return
        self._calibrate_online(train_df)

    def reset(self) -> None:
        self._x = None
        self._p = None

    def update(self, obs: Bar) -> FilterState:
        if obs.close <= 0.0:
            raise ValueError("KalmanFilter requires a positive close")
        y = math.log(obs.close)
        if self._x is None or self._p is None:
            self._x = y
            self._p = self._r
            return FilterState(
                filtered_price=float(obs.close),
                regime_prob=0.0,
                surprise=self.standardize_surprise(0.0),
                pred_vol=math.sqrt(self._r),
            )
        p_pred = self._p + self._q
        innovation_var = p_pred + self._r
        innovation = y - self._x
        gain = p_pred / innovation_var
        self._x = self._x + gain * innovation
        self._p = (1.0 - gain) * p_pred
        raw_surprise = abs(innovation) / math.sqrt(innovation_var)
        return FilterState(
            filtered_price=math.exp(self._x),
            regime_prob=0.0,
            surprise=self.standardize_surprise(raw_surprise),
            pred_vol=math.sqrt(innovation_var),
        )
