"""Bit-exact causality checks.

Series B matches series A through ``t_cut`` and diverges afterward. Every
filter state and strategy position on ``0..t_cut`` must match exactly.
That fails if a filter smooths, refits online on the whole sample, or a
strategy reads a future bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from lewm_harness.data.markov import generate_pair_series, generate_regime_series
from lewm_harness.engine.backtester import run_backtest
from lewm_harness.engine.walk_forward import run_walk_forward
from lewm_harness.filters.base import RegimeFilter
from lewm_harness.filters.hmm import HMMFilter
from lewm_harness.filters.identity import IdentityFilter
from lewm_harness.filters.kalman import KalmanFilter
from lewm_harness.strategies.mean_reversion import MeanReversionStrategy
from lewm_harness.strategies.momentum import MomentumStrategy
from lewm_harness.strategies.stat_arb import StatArbStrategy
from lewm_harness.types import Bar, FilterState

_FILTERS = [IdentityFilter, KalmanFilter, HMMFilter]
_STRATEGIES = [MeanReversionStrategy, MomentumStrategy, StatArbStrategy]


def _perturb_tail(frame: pd.DataFrame, t_cut: int) -> pd.DataFrame:
    """Return a copy whose rows after ``t_cut`` are heavily changed."""

    perturbed = frame.copy(deep=True)
    tail = slice(t_cut + 1, None)
    perturbed.loc[perturbed.index[tail], "close"] = (
        perturbed["close"].to_numpy()[tail] * 4.0 + 25.0
    )
    perturbed.loc[perturbed.index[tail], "ret"] = np.linspace(
        -0.2, 0.35, perturbed["ret"].to_numpy()[tail].shape[0]
    )
    perturbed.loc[perturbed.index[tail], "spread"] = (
        perturbed["spread"].to_numpy()[tail] + np.linspace(-3.0, 4.0, perturbed["spread"].to_numpy()[tail].shape[0])
    )
    perturbed.loc[perturbed.index[tail], "close_b"] = (
        perturbed["close_b"].to_numpy()[tail] * 0.2
    )
    return perturbed


def _state_matrix(states: list[FilterState]) -> np.ndarray:
    return np.array(
        [
            [state.filtered_price, state.regime_prob, state.surprise, state.pred_vol]
            for state in states
        ],
        dtype=np.float64,
    )


def _assert_prefix_invariant(make_filter, strategy_cls: type, n_steps: int = 180, t_cut: int = 100) -> None:
    original = generate_pair_series(n_steps, seed=7)
    perturbed = _perturb_tail(original, t_cut)
    pd.testing.assert_frame_equal(original.iloc[: t_cut + 1], perturbed.iloc[: t_cut + 1])

    def run(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        filt = make_filter()
        strategy = strategy_cls(surprise_scale=1.0, surprise_threshold=0.5, regime_cutoff=0.8)
        filt.fit(frame.iloc[: t_cut + 1])
        if isinstance(filt, HMMFilter):
            assert filt.stds_ is not None
            assert filt.stds_[1] >= filt.stds_[0]
        result = run_backtest(frame, filt, strategy, fee_bps=1.5)
        return _state_matrix(result.states), result.positions.copy()

    states_a, positions_a = run(original)
    states_b, positions_b = run(perturbed)
    prefix = slice(0, t_cut + 1)
    tail = slice(t_cut + 1, None)
    np.testing.assert_allclose(states_a[prefix], states_b[prefix], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(positions_a[prefix], positions_b[prefix], atol=0.0, rtol=0.0)
    assert states_a[tail].shape[0] > 0
    assert not np.allclose(states_a[tail], states_b[tail], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("filter_cls", _FILTERS)
@pytest.mark.parametrize("strategy_cls", _STRATEGIES)
def test_prefix_is_invariant_to_future(filter_cls: type, strategy_cls: type) -> None:
    _assert_prefix_invariant(filter_cls, strategy_cls)


@pytest.fixture(scope="module")
def tiny_lewm_checkpoint(tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("torch")
    from lewm_harness.world_model.train import TrainConfig, resolve_device, run_training, synthetic_frame

    path = tmp_path_factory.mktemp("lewm") / "checkpoint.pt"
    config = TrainConfig(epochs=1, n_steps=256, batch_size=64, seed=0)
    run_training(config, synthetic_frame(config.n_steps, config.seed), path, resolve_device("cpu"), False)
    return path


@pytest.mark.parametrize("strategy_cls", _STRATEGIES)
def test_lewm_prefix_is_invariant_to_future(strategy_cls: type, tiny_lewm_checkpoint) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from lewm_harness.filters.lewm import LeWMFilter

    _assert_prefix_invariant(lambda: LeWMFilter(tiny_lewm_checkpoint), strategy_cls)


def _batch_forward_and_smooth(
    returns: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference forward messages and forward-backward smoothed probabilities.

    The library filter must match the forward column and must not match the
    smoothed column on a path that actually changes regime.
    """

    n_steps = returns.shape[0]
    log_alpha = np.zeros((n_steps, 2), dtype=np.float64)
    log_emit = np.zeros((n_steps, 2), dtype=np.float64)
    log_pred = np.log(startprob)
    log_trans = np.log(transmat)
    for t in range(n_steps):
        emit = norm.logpdf(float(returns[t]), loc=means, scale=stds)
        log_emit[t] = emit
        log_joint = log_pred + emit
        log_norm = logsumexp(log_joint)
        log_alpha[t] = log_joint - log_norm
        log_pred = logsumexp(log_alpha[t][:, None] + log_trans, axis=0)
    log_beta = np.zeros((n_steps, 2), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_emit[t + 1] + log_beta[t + 1], axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return np.exp(log_alpha[:, 1]), np.exp(log_gamma[:, 1])


def test_hmm_matches_forward_algorithm_not_smoother() -> None:
    frame = generate_regime_series(500, seed=3)
    filt = HMMFilter()
    filt.fit(frame)
    online = np.empty(len(frame), dtype=np.float64)
    for t in range(len(frame)):
        bar = Bar(
            ts=int(frame["ts"].iloc[t]),
            close=float(frame["close"].iloc[t]),
            ret=float(frame["ret"].iloc[t]),
        )
        online[t] = filt.update(bar).regime_prob
    assert filt.means_ is not None
    assert filt.stds_ is not None
    assert filt.stds_[1] >= filt.stds_[0]
    assert filt.transmat_ is not None
    assert filt.startprob_ is not None
    forward, smoothed = _batch_forward_and_smooth(
        frame["ret"].to_numpy(dtype=np.float64),
        filt.means_,
        filt.stds_,
        filt.transmat_,
        filt.startprob_,
    )
    np.testing.assert_allclose(online, forward, atol=0.0, rtol=0.0)
    assert float(np.max(np.abs(forward - smoothed))) > 1e-3
    assert np.all((online >= 0.0) & (online <= 1.0))


def test_walk_forward_fit_never_sees_the_test_slice() -> None:
    frame = generate_regime_series(120, seed=11)
    created: list[_RecordingFilter] = []

    def factory() -> _RecordingFilter:
        filt = _RecordingFilter()
        created.append(filt)
        return filt

    run_walk_forward(
        frame,
        factory,
        MeanReversionStrategy,
        train_size=40,
        test_size=20,
        fee_bps=0.0,
    )
    assert created, "walk-forward did not build a filter"
    for filt in created:
        fit_ts = filt.fit_timestamps
        update_ts = np.asarray(filt.update_timestamps, dtype=np.int64)
        assert fit_ts.size > 0
        assert update_ts.size > 0
        assert int(fit_ts.max()) < int(update_ts.min())
        assert set(fit_ts.tolist()).isdisjoint(update_ts.tolist())


class _RecordingFilter(RegimeFilter):
    """Spy that records which timestamps fit and update actually observed."""

    def __init__(self) -> None:
        self.fit_timestamps = np.array([], dtype=np.int64)
        self.update_timestamps: list[int] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        self.fit_timestamps = train_df["ts"].to_numpy(dtype=np.int64).copy()

    def reset(self) -> None:
        self.update_timestamps = []

    def update(self, obs: Bar) -> FilterState:
        self.update_timestamps.append(obs.ts)
        return FilterState(float(obs.close), 0.0, 0.0, 0.0)
