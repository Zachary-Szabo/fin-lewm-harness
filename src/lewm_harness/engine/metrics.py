"""Performance statistics for a realized PnL path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    """Summary of one backtest.

    ``total_return`` is the sum of step PnL (arithmetic, not compounded).
    ``sharpe`` is the mean step PnL over its sample standard deviation,
    scaled by ``sqrt(periods_per_year)``. Synthetic steps are not calendar
    days; the default of 252 only puts numbers on a familiar scale.
    ``max_drawdown`` is the largest peak-to-trough loss of the equity
    curve ``1 + cumsum(pnl)``, expressed as a positive fraction of the
    peak. ``turnover`` is the mean absolute position change, including the
    move from a flat start. ``win_rate`` is the fraction of non-zero PnL
    steps that are positive, or 0 when every step is flat.
    """

    total_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    win_rate: float


def compute_metrics(
    pnl_series: np.ndarray,
    pos_series: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """Summarize ``pnl_series`` and the positions that produced it."""

    pnl = np.asarray(pnl_series, dtype=np.float64)
    positions = np.asarray(pos_series, dtype=np.float64)
    if pnl.ndim != 1 or positions.ndim != 1:
        raise ValueError("pnl_series and pos_series must be one-dimensional")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    if pnl.size == 0:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    total_return = float(pnl.sum())
    if pnl.size < 2 or float(pnl.std(ddof=1)) == 0.0:
        sharpe = 0.0
    else:
        sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(periods_per_year))

    equity = 1.0 + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    denominator = np.where(peak > 0.0, peak, 1.0)
    max_drawdown = float(np.max((peak - equity) / denominator))

    changes = np.abs(np.diff(positions, prepend=0.0))
    turnover = float(changes.mean()) if changes.size else 0.0
    active = pnl != 0.0
    if np.any(active):
        win_rate = float(np.mean(pnl[active] > 0.0))
    else:
        win_rate = 0.0
    return PerformanceMetrics(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        turnover=turnover,
        win_rate=win_rate,
    )
