"""Causal regime-filter interface.

A filter may read a training frame only inside ``fit``. After ``reset``,
``update`` is a pure online step: the current :class:`Bar` plus whatever
state the filter itself stored from earlier bars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from lewm_harness.types import Bar, FilterState


class RegimeFilter(ABC):
    """Online filter that maps one bar to one :class:`FilterState`."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Estimate parameters from ``train_df`` only.

        Implementations must not retain ``train_df`` for use during
        ``update``. Online state belongs in ``reset`` / ``update``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear online state. Fitted parameters stay in place."""

    @abstractmethod
    def update(self, obs: Bar) -> FilterState:
        """Ingest the current bar and return the filtered state."""
