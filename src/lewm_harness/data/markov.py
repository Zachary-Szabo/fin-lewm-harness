"""Two-state Markov regime-switching price simulators.

Regime 0 is calm and mean-reverting: low volatility and negative return
autocorrelation (an Ornstein-Uhlenbeck pull on the return). Regime 1 is
turbulent and trending: higher volatility, a positive drift, and positive
return autocorrelation.

The simulators are for evaluation only. They are not used inside filters,
so a realized ``true_regime`` label never enters ``fit`` or ``update``
unless a caller deliberately puts it there. The backtester drops that
column before building bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def generate_regime_series(
    n_steps: int,
    seed: int,
    *,
    p_stay: tuple[float, float] = (0.98, 0.95),
    mu: tuple[float, float] = (0.0, 0.0008),
    phi: tuple[float, float] = (-0.20, 0.35),
    sigma: tuple[float, float] = (0.004, 0.018),
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Simulate a single price path with Markov regime switching.

    Log returns follow a regime-dependent AR(1)::

        r_t = mu[k_t] + phi[k_t] * r_{t-1} + sigma[k_t] * eps_t

    Regime 0 uses ``phi < 0`` and a small ``sigma``. Regime 1 uses
    ``phi > 0``, a positive drift, and a larger ``sigma``. Simple returns
    and a positive price path are recovered with ``expm1`` / ``exp``, so
    prices stay positive for any finite shock.

    Parameters
    ----------
    n_steps:
        Length of the path, including the t = 0 anchor bar (return 0).
    seed:
        Seed for ``numpy.random.Generator``.
    p_stay:
        Probability of remaining in regime 0 and regime 1.
    mu, phi, sigma:
        AR(1) intercept, lag coefficient, and shock scale per regime.
        ``abs(phi)`` must be strictly less than 1.
    start_price:
        Positive initial price level.

    Returns
    -------
    pd.DataFrame
        Columns ``ts, close, ret, true_regime``. ``ret`` is the simple
        return realized on arrival at that bar.
    """

    _validate_spec(n_steps, p_stay, phi, sigma, start_price)
    rng = np.random.default_rng(seed)
    regime = _simulate_regimes(n_steps, rng, p_stay)
    log_ret = _ar1_returns(regime, rng, mu, phi, sigma)
    close = start_price * np.exp(np.cumsum(log_ret))
    simple = np.expm1(log_ret)
    return _frame(close, simple, regime)


def generate_pair_series(
    n_steps: int,
    seed: int,
    *,
    p_stay: tuple[float, float] = (0.98, 0.95),
    common_sigma: float = 0.02,
    phi_spread: tuple[float, float] = (0.85, 0.92),
    sigma_spread: tuple[float, float] = (0.01, 0.05),
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Simulate two cointegrated prices whose spread regime-switches.

    A shared log-price random walk keeps the legs cointegrated. The log
    spread is an AR(1) whose shock scale (and persistence) jumps with the
    Markov regime: tight and calmer in regime 0, wider in regime 1.

    ``close`` is leg A. ``ret`` is the equal-weight pair return
    ``0.5 * (ret_a - ret_b)``, which is what a unit spread position earns.
    ``spread`` is ``log(close_a) - log(close_b)`` and ``close_b`` is leg B.
    Both are contemporaneous with the bar, so a pairs strategy can read
    them from ``Bar.extras`` without looking ahead.
    """

    _validate_spec(n_steps, p_stay, phi_spread, sigma_spread, start_price)
    if common_sigma <= 0.0:
        raise ValueError("common_sigma must be positive")
    rng = np.random.default_rng(seed)
    regime = _simulate_regimes(n_steps, rng, p_stay)

    common_shock = rng.normal(0.0, common_sigma, size=n_steps)
    common_shock[0] = 0.0
    common = start_price * np.exp(np.cumsum(common_shock))
    spread = _ar1_returns(regime, rng, (0.0, 0.0), phi_spread, sigma_spread)

    close_a = common * np.exp(0.5 * spread)
    close_b = common * np.exp(-0.5 * spread)
    ret_a = _simple_returns(close_a)
    ret_b = _simple_returns(close_b)
    pair_ret = 0.5 * (ret_a - ret_b)
    # Reconstruct the log spread from the legs so the column matches the
    # prices a strategy would actually observe.
    log_spread = np.log(close_a) - np.log(close_b)
    frame = _frame(close_a, pair_ret, regime)
    frame["spread"] = log_spread
    frame["close_b"] = close_b
    return frame


def _validate_spec(
    n_steps: int,
    p_stay: tuple[float, float],
    phi: tuple[float, float],
    sigma: tuple[float, float],
    start_price: float,
) -> None:
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if start_price <= 0.0:
        raise ValueError("start_price must be positive")
    for name, pair in (("p_stay", p_stay), ("phi", phi), ("sigma", sigma)):
        if len(pair) != 2:
            raise ValueError(f"{name} must have two regime values")
    if not all(0.0 < p <= 1.0 for p in p_stay):
        raise ValueError("p_stay values must lie in (0, 1]")
    if not all(abs(value) < 1.0 for value in phi):
        raise ValueError("phi must be inside (-1, 1) so each regime is stationary")
    if not all(value > 0.0 for value in sigma):
        raise ValueError("sigma values must be positive")


def _simulate_regimes(
    n_steps: int,
    rng: np.random.Generator,
    p_stay: tuple[float, float],
) -> np.ndarray:
    """Draw a 2-state Markov chain. The path starts in the calm regime."""

    regime = np.zeros(n_steps, dtype=np.int64)
    for t in range(1, n_steps):
        previous = int(regime[t - 1])
        if rng.random() >= p_stay[previous]:
            regime[t] = 1 - previous
        else:
            regime[t] = previous
    return regime


def _ar1_returns(
    regime: np.ndarray,
    rng: np.random.Generator,
    mu: tuple[float, float],
    phi: tuple[float, float],
    sigma: tuple[float, float],
) -> np.ndarray:
    """Causal AR(1) driven by ``regime[t]``. Index 0 is the zero anchor."""

    shocks = rng.normal(0.0, 1.0, size=regime.shape[0])
    values = np.zeros(regime.shape[0], dtype=np.float64)
    for t in range(1, regime.shape[0]):
        k = int(regime[t])
        values[t] = mu[k] + phi[k] * values[t - 1] + sigma[k] * shocks[t]
    return values


def _simple_returns(close: np.ndarray) -> np.ndarray:
    out = np.zeros(close.shape[0], dtype=np.float64)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def _frame(close: np.ndarray, ret: np.ndarray, regime: np.ndarray) -> pd.DataFrame:
    n_steps = close.shape[0]
    return pd.DataFrame(
        {
            "ts": np.arange(n_steps, dtype=np.int64),
            "close": np.asarray(close, dtype=np.float64),
            "ret": np.asarray(ret, dtype=np.float64),
            "true_regime": np.asarray(regime, dtype=np.int64),
        }
    )


__all__ = ["generate_pair_series", "generate_regime_series"]
