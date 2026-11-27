"""Monte Carlo evaluation on seeded regime-switching paths.

Each path is split once. ``fit`` sees only the training prefix. The
reported Sharpe and drawdown come from the untouched suffix, so a filter
cannot improve its score by training on the returns it is judged on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lewm_harness.data.markov import generate_regime_series
from lewm_harness.engine.backtester import run_backtest
from lewm_harness.engine.metrics import compute_metrics
from lewm_harness.filters.base import RegimeFilter
from lewm_harness.strategies.base import Strategy

SeriesGenerator = Callable[[int, int], pd.DataFrame]


@dataclass(frozen=True)
class MonteCarloSummary:
    """Distribution of out-of-sample Sharpe and drawdown across paths.

    ``sharpe_p5`` is the 5th percentile (a bad-path Sharpe, not a typical
    one). ``worst_max_drawdown`` is the largest drawdown observed on any
    path. Drawdowns are positive fractions.
    """

    sharpe_mean: float
    sharpe_std: float
    sharpe_p5: float
    worst_max_drawdown: float
    n_paths: int


def run_monte_carlo(
    n_paths: int,
    n_steps: int,
    filter_factory: Callable[[], RegimeFilter],
    strategy_factory: Callable[[], Strategy],
    *,
    seed: int = 0,
    train_frac: float = 0.5,
    fee_bps: float = 1.0,
    periods_per_year: float = 252.0,
    generator: SeriesGenerator = generate_regime_series,
) -> MonteCarloSummary:
    """Aggregate out-of-sample metrics over ``n_paths`` synthetic paths.

    Path ``i`` is ``generator(n_steps, seed + i)``. The first
    ``train_frac`` of each path is the fit sample; the rest is the test.
    """

    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie in (0, 1)")
    cut = int(n_steps * train_frac)
    if cut < 2 or n_steps - cut < 2:
        raise ValueError("train_frac split leaves a train or test slice shorter than 2")

    sharpes = np.empty(n_paths, dtype=np.float64)
    drawdowns = np.empty(n_paths, dtype=np.float64)
    for i in range(n_paths):
        path = generator(n_steps, seed + i)
        if len(path) != n_steps:
            raise ValueError(f"generator returned {len(path)} rows, expected {n_steps}")
        filt = filter_factory()
        strategy = strategy_factory()
        filt.fit(path.iloc[:cut])
        result = run_backtest(path.iloc[cut:], filt, strategy, fee_bps=fee_bps)
        metrics = compute_metrics(
            result.pnl,
            result.positions,
            periods_per_year=periods_per_year,
        )
        sharpes[i] = metrics.sharpe
        drawdowns[i] = metrics.max_drawdown

    sharpe_std = float(sharpes.std(ddof=1)) if n_paths > 1 else 0.0
    return MonteCarloSummary(
        sharpe_mean=float(sharpes.mean()),
        sharpe_std=sharpe_std,
        sharpe_p5=float(np.percentile(sharpes, 5)),
        worst_max_drawdown=float(drawdowns.max()),
        n_paths=n_paths,
    )
