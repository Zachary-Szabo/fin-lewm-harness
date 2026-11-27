"""Rolling train/test evaluation.

Each fold fits a new filter on its training slice and then runs the
causal backtester on the following test slice only. Test rows are never
passed to ``fit``. Folds step forward by ``test_size`` and each test
window starts from a flat position.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from lewm_harness.engine.backtester import BacktestResult, run_backtest
from lewm_harness.filters.base import RegimeFilter
from lewm_harness.strategies.base import Strategy
from lewm_harness.types import FilterState


def run_walk_forward(
    df: pd.DataFrame,
    filter_factory: Callable[[], RegimeFilter],
    strategy_factory: Callable[[], Strategy],
    train_size: int,
    test_size: int,
    fee_bps: float = 1.0,
) -> BacktestResult:
    """Concatenate out-of-sample backtests over rolling train/test folds.

    A fold starting at ``s`` trains on ``[s, s + train_size)`` and tests
    on ``[s + train_size, s + train_size + test_size)``. The next fold
    starts at ``s + test_size``. The returned PnL, positions, and states
    are the test decisions only.
    """

    if train_size < 2 or test_size < 2:
        raise ValueError("train_size and test_size must both be at least 2")
    n_rows = len(df)
    if train_size + test_size > n_rows:
        raise ValueError("frame is shorter than one train/test fold")

    pnl_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    states: list[FilterState] = []
    start = 0
    while start + train_size + test_size <= n_rows:
        train = df.iloc[start : start + train_size]
        test = df.iloc[start + train_size : start + train_size + test_size]
        filt = filter_factory()
        strategy = strategy_factory()
        filt.fit(train)
        fold = run_backtest(test, filt, strategy, fee_bps=fee_bps)
        pnl_parts.append(fold.pnl)
        position_parts.append(fold.positions)
        states.extend(fold.states)
        start += test_size

    return BacktestResult(
        pnl=np.concatenate(pnl_parts),
        positions=np.concatenate(position_parts),
        states=states,
    )
