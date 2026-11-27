"""Backtester accounting: fees, position limits, and a flat book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lewm_harness.engine.backtester import run_backtest
from lewm_harness.engine.metrics import compute_metrics
from lewm_harness.filters.identity import IdentityFilter
from lewm_harness.strategies.base import Strategy
from lewm_harness.strategies.mean_reversion import MeanReversionStrategy
from lewm_harness.strategies.momentum import MomentumStrategy
from lewm_harness.types import Bar, FilterState


def _frame(returns: list[float]) -> pd.DataFrame:
    rets = np.asarray(returns, dtype=np.float64)
    # close[0] = 100 and close[t] = close[t-1] * (1 + ret[t]).
    close = np.empty(rets.shape[0], dtype=np.float64)
    close[0] = 100.0
    for t in range(1, rets.shape[0]):
        close[t] = close[t - 1] * (1.0 + rets[t])
    return pd.DataFrame(
        {
            "ts": np.arange(rets.shape[0], dtype=np.int64),
            "close": close,
            "ret": rets,
        }
    )


class _Scripted(Strategy):
    """Replay a position list. ``raw`` skips the overlay so the engine clip is visible."""

    def __init__(self, positions: list[float], *, raw: bool = False) -> None:
        super().__init__(surprise_scale=0.0)
        self._positions = positions
        self._raw = raw
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def step(self, obs: Bar, state: FilterState) -> float:
        value = self._positions[self._index]
        self._index += 1
        if self._raw:
            return value
        return self.apply_overlay(value, state)


def test_fee_accounting_matches_the_causal_formula() -> None:
    frame = _frame([0.0, 0.02, -0.01, 0.04])
    fee_bps = 10.0
    result = run_backtest(
        frame,
        IdentityFilter(),
        _Scripted([0.5, -0.5, 0.0]),
        fee_bps=fee_bps,
    )
    fee = fee_bps * 1e-4
    expected = np.array(
        [
            0.5 * 0.02 - fee * abs(0.5 - 0.0),
            -0.5 * -0.01 - fee * abs(-0.5 - 0.5),
            0.0 * 0.04 - fee * abs(0.0 - (-0.5)),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.pnl, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.positions, [0.5, -0.5, 0.0], atol=0.0, rtol=0.0)


def test_positions_are_clipped_to_unit_interval() -> None:
    frame = _frame([0.0, 0.01, -0.02, 0.03])
    result = run_backtest(
        frame,
        IdentityFilter(),
        _Scripted([5.0, -4.0, 0.25], raw=True),
        fee_bps=0.0,
    )
    np.testing.assert_allclose(result.positions, [1.0, -1.0, 0.25], atol=0.0, rtol=0.0)
    assert result.positions.min() >= -1.0
    assert result.positions.max() <= 1.0


def test_flat_position_has_zero_pnl_and_zero_fees() -> None:
    frame = _frame([0.0, 0.5, -0.4, 0.2, -0.1])
    result = run_backtest(
        frame,
        IdentityFilter(),
        _Scripted([0.0, 0.0, 0.0, 0.0]),
        fee_bps=25.0,
    )
    np.testing.assert_allclose(result.pnl, np.zeros_like(result.pnl), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.positions, np.zeros_like(result.positions), atol=0.0, rtol=0.0)


def test_metrics_on_a_known_path() -> None:
    pnl = np.array([0.10, -0.05, -0.05, 0.20], dtype=np.float64)
    positions = np.array([1.0, 1.0, 0.0, 0.5], dtype=np.float64)
    metrics = compute_metrics(pnl, positions, periods_per_year=1.0)
    assert metrics.total_return == pytest.approx(float(pnl.sum()))
    # Equity: 1.10, 1.05, 1.00, 1.20. Worst loss from the 1.10 peak is 0.10.
    assert metrics.max_drawdown == pytest.approx(0.10 / 1.10)
    assert metrics.turnover == pytest.approx((1.0 + 0.0 + 1.0 + 0.5) / 4.0)
    assert metrics.win_rate == pytest.approx(0.5)
    sample_std = float(pnl.std(ddof=1))
    assert metrics.sharpe == pytest.approx(float(pnl.mean()) / sample_std)


def test_true_regime_label_is_not_given_to_the_strategy() -> None:
    frame = _frame([0.0, 0.01, -0.02, 0.03])
    frame["true_regime"] = np.array([0, 1, 1, 0], dtype=np.int64)
    seen: list[dict[str, float] | None] = []

    class _Probe(Strategy):
        def reset(self) -> None:
            return None

        def step(self, obs: Bar, state: FilterState) -> float:
            del state
            seen.append(obs.extras)
            return 0.0

    run_backtest(frame, IdentityFilter(), _Probe(), fee_bps=0.0)
    assert seen
    assert all(extras is None or "true_regime" not in extras for extras in seen)


def test_overlay_sizes_on_vol_and_pauses_only_mean_reversion() -> None:
    calm = FilterState(filtered_price=100.0, regime_prob=0.0, surprise=0.0, pred_vol=0.01)
    stressed = FilterState(filtered_price=100.0, regime_prob=0.95, surprise=0.0, pred_vol=0.05)
    fade = MeanReversionStrategy(surprise_scale=0.0, target_vol=0.01, regime_cutoff=0.85)
    assert fade.apply_overlay(1.0, calm) == pytest.approx(1.0)
    assert fade.apply_overlay(1.0, stressed) == pytest.approx(0.0)
    trend = MomentumStrategy()
    assert trend.regime_cutoff is None
    trend.target_vol = 0.01
    assert trend.apply_overlay(0.5, stressed) == pytest.approx(0.5 * 0.01 / 0.05)
    surprised = FilterState(filtered_price=100.0, regime_prob=0.0, surprise=3.0, pred_vol=0.01)
    scaled = MeanReversionStrategy(
        surprise_scale=1.0,
        surprise_threshold=1.0,
        regime_cutoff=None,
        target_vol=None,
    )
    assert scaled.apply_overlay(1.0, surprised) == pytest.approx(1.0 / 3.0)


def test_lewm_filter_requires_a_checkpoint_file() -> None:
    pytest.importorskip("torch")
    from lewm_harness.filters.lewm import LeWMFilter

    with pytest.raises(FileNotFoundError):
        LeWMFilter("phase2.pt")
