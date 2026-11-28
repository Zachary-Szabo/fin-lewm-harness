"""Build one causal bar from a single frame row."""

from __future__ import annotations

import pandas as pd

from lewm_harness.types import Bar

# Evaluation labels stay off the bar. Every other column is a same-timestamp extra.
EXCLUDED_COLUMNS = frozenset({"ts", "close", "ret", "true_regime"})


def make_bar(frame: pd.DataFrame, index: int) -> Bar:
    """Return the bar at ``index`` using only that row.

    Extra columns become ``Bar.extras``. ``true_regime`` is omitted so a
    strategy cannot read the simulator label.
    """

    row = frame.iloc[index]
    extras: dict[str, float] = {}
    for column in frame.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        extras[str(column)] = float(row[column])
    return Bar(
        ts=int(row["ts"]),
        close=float(row["close"]),
        ret=float(row["ret"]),
        extras=extras or None,
    )
