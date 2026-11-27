"""Shared value types passed across the harness.

``Bar`` is the only observation a filter or strategy may see at time ``t``.
``FilterState`` is the causal summary the strategy may condition on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    """One market observation known at decision time ``ts``.

    ``ret`` is the simple return that *ended* at this bar
    (``close_t / close_{t-1} - 1``), not the forward return the position
    is about to earn. ``extras`` carries contemporaneous features such as
    a pairs spread. It is ``None`` when the series has no extra columns.
    """

    ts: int
    close: float
    ret: float
    extras: dict[str, float] | None = None


@dataclass(frozen=True)
class FilterState:
    """Causal filter output after observing the current bar.

    ``regime_prob`` is the filtered probability of the high-volatility
    state. Filters without a regime latent (identity, Kalman) report 0.
    ``surprise`` is the training-set z-score of that filter's raw score
    (absolute return over trailing vol, absolute Kalman innovation, or
    HMM negative log likelihood). ``pred_vol`` is that filter's
    volatility estimate in its native units.
    """

    filtered_price: float
    regime_prob: float
    surprise: float
    pred_vol: float
