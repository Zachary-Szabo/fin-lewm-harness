"""Compatibility import path for :class:`LeWMFilter`.

The implementation lives in :mod:`lewm_harness.filters.lewm` and imports
PyTorch. Attribute access is lazy so Phase 1 tests can import the filter
package without loading that stack.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "LeWMFilter":
        from lewm_harness.filters.lewm import LeWMFilter

        return LeWMFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
