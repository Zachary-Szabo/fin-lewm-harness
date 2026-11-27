"""Causal regime-filter interface.

A filter may read a training frame only inside ``fit``. After ``reset``,
``update`` is a pure online step: the current :class:`Bar` plus whatever
state the filter itself stored from earlier bars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from lewm_harness.types import Bar, FilterState


class RegimeFilter(ABC):
    """Online filter that maps one bar to one :class:`FilterState`.

    ``fit`` may replay ``train_df`` causally to freeze two scalars: the
    mean and standard deviation of that filter's raw surprise, and the
    median ``pred_vol``. ``update`` then reports surprise as a z-score
    against those training moments, so Identity, Kalman, HMM, and a later
    LeWM checkpoint share one overlay scale. The replay never sees rows
    outside ``train_df``, and ``reset`` does not drop the frozen moments.
    """

    def __init__(self) -> None:
        self._surprise_mean: float | None = None
        self._surprise_std: float | None = None
        self.reference_vol_: float | None = None

    def standardize_surprise(self, raw: float) -> float:
        """Map a raw surprise to the training z-score.

        Before :meth:`calibrate_surprise` this returns ``raw`` unchanged,
        which is what the training replay collects. A degenerate training
        sample (zero standard deviation) reports 0.
        """

        if self._surprise_mean is None or self._surprise_std is None:
            return float(raw)
        if self._surprise_std <= 1e-12:
            return 0.0
        return (float(raw) - self._surprise_mean) / self._surprise_std

    def calibrate_surprise(self, raw_surprise: np.ndarray, pred_vol: np.ndarray) -> None:
        """Freeze training moments. Both arrays are one causal replay of the train slice."""

        surprise = np.asarray(raw_surprise, dtype=np.float64).reshape(-1)
        volatility = np.asarray(pred_vol, dtype=np.float64).reshape(-1)
        if surprise.size == 0:
            self._surprise_mean = 0.0
            self._surprise_std = 0.0
            self.reference_vol_ = None
            return
        self._surprise_mean = float(surprise.mean())
        self._surprise_std = float(surprise.std(ddof=0))
        positive = volatility[np.isfinite(volatility) & (volatility > 0.0)]
        self.reference_vol_ = float(np.median(positive)) if positive.size else None

    def _calibrate_online(self, train_df: pd.DataFrame) -> None:
        """Replay ``train_df`` from a reset state and freeze surprise and vol scales.

        Moments are cleared first so a second ``fit`` does not z-score the
        replay with the previous sample.
        """

        self._surprise_mean = None
        self._surprise_std = None
        self.reference_vol_ = None
        self.reset()
        n_rows = len(train_df)
        if n_rows == 0 or "close" not in train_df.columns or "ret" not in train_df.columns:
            return
        timestamps = train_df["ts"].to_numpy()
        closes = train_df["close"].to_numpy(dtype=np.float64)
        returns = train_df["ret"].to_numpy(dtype=np.float64)
        raw = np.empty(n_rows, dtype=np.float64)
        vols = np.empty(n_rows, dtype=np.float64)
        for t in range(n_rows):
            state = self.update(
                Bar(ts=int(timestamps[t]), close=float(closes[t]), ret=float(returns[t]))
            )
            raw[t] = state.surprise
            vols[t] = state.pred_vol
        self.calibrate_surprise(raw, vols)
        self.reset()

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Estimate parameters from ``train_df`` only.

        Implementations must not retain ``train_df`` for use during
        ``update``. Online state belongs in ``reset`` / ``update``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear online state. Fitted parameters stay in place."""

    @abstractmethod
    def update(self, obs: Bar) -> FilterState:
        """Ingest the current bar and return the filtered state."""
