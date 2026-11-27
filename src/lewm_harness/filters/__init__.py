"""Regime filters.

``LeWMFilter`` is resolved lazily so importing this package does not import
PyTorch. Use ``from lewm_harness.filters.lewm import LeWMFilter`` when the
``ml`` extra is installed.
"""

from lewm_harness.filters.base import RegimeFilter
from lewm_harness.filters.hmm import HMMFilter
from lewm_harness.filters.identity import IdentityFilter
from lewm_harness.filters.kalman import KalmanFilter

__all__ = [
    "HMMFilter",
    "IdentityFilter",
    "KalmanFilter",
    "LeWMFilter",
    "RegimeFilter",
]


def __getattr__(name: str):
    if name == "LeWMFilter":
        from lewm_harness.filters.lewm import LeWMFilter

        return LeWMFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
