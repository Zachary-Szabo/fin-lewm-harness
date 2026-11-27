"""Leak-free backtesting harness for market regime filters.

Phase 1 ships causal classical filters (identity, Kalman, HMM) and a
placeholder for the LeWM world model. Filters and strategies observe one
bar at a time. Training frames are passed only to ``fit``.
"""

from lewm_harness.types import Bar, FilterState

__all__ = ["Bar", "FilterState"]
__version__ = "0.1.0"
