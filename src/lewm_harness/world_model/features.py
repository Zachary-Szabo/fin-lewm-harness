"""Causal OHLCV features for the world model.

Channel order is fixed:

0. log return of the current bar
1. high-low range ``(high - low) / close`` (0 when high/low are absent)
2. close-open return ``(close - open) / open`` (0 when open is absent)
3. log-volume z-score, using mean and std fit on the training split only
4. trailing realized volatility of log returns

Every channel at time ``t`` is a function of bars ``0..t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "hl_range",
    "co_return",
    "log_volume_z",
    "realized_vol",
)
N_FEATURES = len(FEATURE_NAMES)


def volume_log_stats(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """Mean and std of ``log(volume + 1e-8)`` on ``frame`` only.

    Returns ``(None, None)`` when the frame has no volume column. Callers
    pass the training slice so the test split cannot move the z-score.
    """

    if "volume" not in frame.columns or len(frame) == 0:
        return None, None
    logged = np.log(frame["volume"].to_numpy(dtype=np.float64) + 1e-8)
    logged = logged[np.isfinite(logged)]
    if logged.size == 0:
        return None, None
    std = float(logged.std(ddof=0))
    return float(logged.mean()), std


class OnlineFeatureState:
    """One-bar-at-a-time feature builder. The history is only the past."""

    def __init__(
        self,
        vol_window: int,
        log_volume_mean: float | None,
        log_volume_std: float | None,
    ) -> None:
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        self.vol_window = vol_window
        self.log_volume_mean = log_volume_mean
        self.log_volume_std = log_volume_std
        self._log_returns: list[float] = []

    def reset(self) -> None:
        self._log_returns = []

    def update(
        self,
        *,
        ret: float,
        close: float,
        high: float | None = None,
        low: float | None = None,
        open_: float | None = None,
        volume: float | None = None,
    ) -> np.ndarray:
        log_return = float(np.log1p(ret)) if ret > -1.0 else 0.0
        self._log_returns.append(log_return)
        window = self._log_returns[-self.vol_window :]
        if len(window) < 2:
            realized = 0.0
        else:
            realized = float(np.std(np.asarray(window, dtype=np.float64), ddof=0))
        if high is not None and low is not None and close != 0.0:
            hl_range = (high - low) / close
        else:
            hl_range = 0.0
        if open_ is not None and open_ != 0.0:
            co_return = (close - open_) / open_
        else:
            co_return = 0.0
        if (
            volume is not None
            and self.log_volume_mean is not None
            and self.log_volume_std is not None
            and self.log_volume_std > 1e-8
        ):
            log_volume_z = (float(np.log(volume + 1e-8)) - self.log_volume_mean) / self.log_volume_std
        else:
            log_volume_z = 0.0
        return np.array(
            [log_return, hl_range, co_return, log_volume_z, realized],
            dtype=np.float64,
        )


def _optional_column(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in frame.columns:
        return None
    return frame[name].to_numpy(dtype=np.float64)


def causal_feature_frame(
    frame: pd.DataFrame,
    vol_window: int,
    log_volume_mean: float | None,
    log_volume_std: float | None,
) -> np.ndarray:
    """Stack :class:`OnlineFeatureState` over ``frame`` in time order.

    The result has shape ``(len(frame), 5)`` and matches stepping the same
    bars one at a time.
    """

    state = OnlineFeatureState(vol_window, log_volume_mean, log_volume_std)
    returns = frame["ret"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    highs = _optional_column(frame, "high")
    lows = _optional_column(frame, "low")
    opens = _optional_column(frame, "open")
    volumes = _optional_column(frame, "volume")
    rows = np.empty((len(frame), N_FEATURES), dtype=np.float64)
    for i in range(len(frame)):
        rows[i] = state.update(
            ret=float(returns[i]),
            close=float(closes[i]),
            high=None if highs is None else float(highs[i]),
            low=None if lows is None else float(lows[i]),
            open_=None if opens is None else float(opens[i]),
            volume=None if volumes is None else float(volumes[i]),
        )
    return rows


def prepare_feature_matrix(
    frame: pd.DataFrame,
    train_end: int,
    vol_window: int,
) -> dict[str, np.ndarray | float | None]:
    """Scale features with statistics fit on ``frame.iloc[:train_end]`` only.

    Volume z-score moments and the per-channel scaler both ignore the test
    tail. The scaled matrix still covers the whole frame so a later split
    can read it.
    """

    if train_end < 2 or train_end > len(frame):
        raise ValueError("train_end must lie inside the frame and cover at least 2 rows")
    log_volume_mean, log_volume_std = volume_log_stats(frame.iloc[:train_end])
    raw = causal_feature_frame(frame, vol_window, log_volume_mean, log_volume_std)
    train_block = raw[:train_end]
    feature_mean = train_block.mean(axis=0)
    feature_std = train_block.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    scaled = (raw - feature_mean) / feature_std
    return {
        "raw": raw,
        "scaled": scaled,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "log_volume_mean": log_volume_mean,
        "log_volume_std": log_volume_std,
    }
