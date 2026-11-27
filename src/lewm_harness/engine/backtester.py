"""Strictly causal step-by-step backtest loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lewm_harness.engine.bars import EXCLUDED_COLUMNS, make_bar
from lewm_harness.engine.metrics import PerformanceMetrics
from lewm_harness.filters.base import RegimeFilter
from lewm_harness.strategies.base import Strategy
from lewm_harness.types import FilterState

# Re-exported so existing readers of this module still see the column rule.
_EXCLUDED_COLUMNS = EXCLUDED_COLUMNS


@dataclass
class BacktestResult:
    """Aligned online outputs of one causal pass.

    ``states[i]`` and ``positions[i]`` are the decision at bar ``i``.
    ``pnl[i]`` is the return that decision earned on bar ``i + 1``, net of
    the turnover fee charged when the position was set. The last bar is
    not a decision: it only supplies ``ret`` for the previous position.
    """

    pnl: np.ndarray
    positions: np.ndarray
    states: list[FilterState]
    metrics: PerformanceMetrics | None = None


def run_backtest(
    df: pd.DataFrame,
    filt: RegimeFilter,
    strategy: Strategy,
    fee_bps: float = 1.0,
) -> BacktestResult:
    """Run ``filt`` and ``strategy`` one bar at a time.

    At step ``t`` the engine builds ``Bar(t)`` from row ``t`` only, calls
    ``filt.update`` and ``strategy.step``, and clips the position to
    ``[-1, 1]``. Realized PnL is

        pos_t * ret_{t+1} - fee_bps * 1e-4 * abs(pos_t - pos_{t-1})

    with ``pos_{-1} = 0``. ``ret_{t+1}`` is applied as the outcome. It is
    never placed on the bar that the filter or strategy sees.

    ``fit`` is not called here. Walk-forward and Monte Carlo fit on the
    training slice first; ``run_backtest`` only resets online state so a
    fitted filter can be replayed from a flat start.
    """

    if fee_bps < 0.0:
        raise ValueError("fee_bps must be non-negative")
    missing = {"ts", "close", "ret"} - set(df.columns)
    if missing:
        raise ValueError(f"backtest frame is missing columns: {sorted(missing)}")
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("backtest frame must contain at least 2 rows")

    returns = df["ret"].to_numpy(dtype=np.float64)

    filt.reset()
    strategy.reset()
    strategy.bind_runtime_vol_target(getattr(filt, "reference_vol_", None))
    n_decisions = n_rows - 1
    positions = np.zeros(n_decisions, dtype=np.float64)
    pnl = np.zeros(n_decisions, dtype=np.float64)
    states: list[FilterState] = []
    previous = 0.0
    fee_rate = fee_bps * 1e-4

    for t in range(n_decisions):
        bar = make_bar(df, t)
        state = filt.update(bar)
        position = float(np.clip(strategy.step(bar, state), -1.0, 1.0))
        pnl[t] = position * float(returns[t + 1]) - fee_rate * abs(position - previous)
        positions[t] = position
        states.append(state)
        previous = position

    return BacktestResult(pnl=pnl, positions=positions, states=states)
