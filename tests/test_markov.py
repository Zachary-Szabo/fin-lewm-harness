"""Synthetic Markov paths switch regimes and regime 1 is more volatile."""

from __future__ import annotations

import numpy as np
import pytest

from lewm_harness.data.markov import generate_pair_series, generate_regime_series


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_series_switches_and_regime_one_is_more_volatile(seed: int) -> None:
    frame = generate_regime_series(8_000, seed)
    regime = frame["true_regime"].to_numpy()
    returns = frame["ret"].to_numpy()[1:]
    regime = regime[1:]
    switches = int(np.sum(regime[1:] != regime[:-1]))
    assert switches > 5
    assert set(np.unique(regime).tolist()) == {0, 1}
    calm = float(returns[regime == 0].std(ddof=0))
    turbulent = float(returns[regime == 1].std(ddof=0))
    assert turbulent > calm * 1.5
    assert np.all(frame["close"].to_numpy() > 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pair_series_is_cointegrated_and_spread_regime_switches(seed: int) -> None:
    frame = generate_pair_series(8_000, seed)
    regime = frame["true_regime"].to_numpy()[1:]
    spread = frame["spread"].to_numpy()
    spread_change = np.diff(spread)
    assert int(np.sum(regime[1:] != regime[:-1])) > 5
    assert set(np.unique(regime).tolist()) == {0, 1}
    calm = float(spread_change[regime == 0].std(ddof=0))
    turbulent = float(spread_change[regime == 1].std(ddof=0))
    assert turbulent > calm
    log_a = np.log(frame["close"].to_numpy())
    log_b = np.log(frame["close_b"].to_numpy())
    correlation = float(np.corrcoef(log_a, log_b)[0, 1])
    assert correlation > 0.8
    np.testing.assert_allclose(spread, log_a - log_b, atol=1e-12, rtol=0.0)
    # The traded return is the equal-weight long/short of the two legs.
    ret_a = np.zeros(len(frame))
    ret_b = np.zeros(len(frame))
    close_a = frame["close"].to_numpy()
    close_b = frame["close_b"].to_numpy()
    ret_a[1:] = close_a[1:] / close_a[:-1] - 1.0
    ret_b[1:] = close_b[1:] / close_b[:-1] - 1.0
    np.testing.assert_allclose(frame["ret"].to_numpy(), 0.5 * (ret_a - ret_b), atol=1e-12, rtol=0.0)
