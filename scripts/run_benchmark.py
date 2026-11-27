"""CLI comparison of strategies and classical regime filters.

Fits each filter on training data only, then scores it out of sample.
Monte Carlo uses seeded Markov paths (pair paths for the spread strategy).
Walk-forward rolls a train/test window across one longer path.

Run from the repository root::

    python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_benchmark.py` without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lewm_harness.data.markov import generate_pair_series, generate_regime_series
from lewm_harness.engine.metrics import compute_metrics
from lewm_harness.engine.monte_carlo import run_monte_carlo
from lewm_harness.engine.walk_forward import run_walk_forward
from lewm_harness.filters.hmm import HMMFilter
from lewm_harness.filters.identity import IdentityFilter
from lewm_harness.filters.kalman import KalmanFilter
from lewm_harness.strategies.mean_reversion import MeanReversionStrategy
from lewm_harness.strategies.momentum import MomentumStrategy
from lewm_harness.strategies.stat_arb import StatArbStrategy

# Surprise is a training z-score, so one threshold applies to every filter.
# Vol targeting is left unset and filled from each filter's median train pred_vol.
# Mean reversion and pairs pause in a high-vol regime; momentum does not.
_SURPRISE = {
    "surprise_scale": 1.0,
    "surprise_threshold": 1.0,
}
_REGIME_CUTOFF = {
    "mean_reversion": 0.85,
    "momentum": None,
    "stat_arb": 0.85,
}

_FILTERS = {
    "identity": IdentityFilter,
    "kalman": KalmanFilter,
    "hmm": HMMFilter,
}

_STRATEGIES = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "stat_arb": StatArbStrategy,
}


def _strategy_factory(name: str):
    cls = _STRATEGIES[name]

    def factory():
        return cls(**_SURPRISE, regime_cutoff=_REGIME_CUTOFF[name])

    return factory


def _generator_for(strategy: str):
    if strategy == "stat_arb":
        return generate_pair_series
    return generate_regime_series


def _format_table(rows: list[dict[str, float | str]]) -> str:
    headers = [
        "strategy",
        "filter",
        "mc_sharpe",
        "mc_sharpe_std",
        "mc_sharpe_p5",
        "mc_worst_dd",
        "wf_sharpe",
        "wf_return",
        "wf_max_dd",
    ]
    string_rows: list[list[str]] = []
    for row in rows:
        string_rows.append(
            [
                str(row["strategy"]),
                str(row["filter"]),
                f"{row['mc_sharpe']:.4f}",
                f"{row['mc_sharpe_std']:.4f}",
                f"{row['mc_sharpe_p5']:.4f}",
                f"{row['mc_worst_dd']:.4f}",
                f"{row['wf_sharpe']:.4f}",
                f"{row['wf_return']:.4f}",
                f"{row['wf_max_dd']:.4f}",
            ]
        )
    widths = [len(header) for header in headers]
    for cells in string_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.rjust(width) for cell, width in zip(cells, widths))

    separator = "  ".join("-" * width for width in widths)
    body = [line(headers), separator]
    body.extend(line(cells) for cells in string_rows)
    return "\n".join(body)


def run(args: argparse.Namespace) -> str:
    rows: list[dict[str, float | str]] = []
    for strategy_name in _STRATEGIES:
        generator = _generator_for(strategy_name)
        walk_path = generator(args.wf_steps, args.seed + 10_000)
        for filter_name, filter_cls in _FILTERS.items():
            summary = run_monte_carlo(
                args.paths,
                args.steps,
                filter_cls,
                _strategy_factory(strategy_name),
                seed=args.seed,
                train_frac=0.5,
                fee_bps=args.fee_bps,
                periods_per_year=args.periods_per_year,
                generator=generator,
            )
            walk = run_walk_forward(
                walk_path,
                filter_cls,
                _strategy_factory(strategy_name),
                train_size=args.wf_train,
                test_size=args.wf_test,
                fee_bps=args.fee_bps,
                periods_per_year=args.periods_per_year,
            )
            metrics = walk.metrics
            if metrics is None:
                metrics = compute_metrics(
                    walk.pnl,
                    walk.positions,
                    periods_per_year=args.periods_per_year,
                )
            rows.append(
                {
                    "strategy": strategy_name,
                    "filter": filter_name,
                    "mc_sharpe": summary.sharpe_mean,
                    "mc_sharpe_std": summary.sharpe_std,
                    "mc_sharpe_p5": summary.sharpe_p5,
                    "mc_worst_dd": summary.worst_max_drawdown,
                    "wf_sharpe": metrics.sharpe,
                    "wf_return": metrics.total_return,
                    "wf_max_dd": metrics.max_drawdown,
                }
            )
    header = (
        f"lewm-market-filter phase 1  paths={args.paths} steps={args.steps} "
        f"train_frac=0.5 fee_bps={args.fee_bps} seed={args.seed}\n"
        f"walk-forward steps={args.wf_steps} train={args.wf_train} "
        f"test={args.wf_test} periods_per_year={args.periods_per_year:g}\n"
        "Monte Carlo and walk-forward scores are out of sample. "
        f"Sharpe uses sqrt({args.periods_per_year:g}). "
        "Surprise is the train-set z-score; vol target is each filter's "
        "median training pred_vol. Regime pause at 0.85 for "
        "mean_reversion and stat_arb; momentum pause is off.\n"
        "mean_reversion and momentum use single-asset regime paths; "
        "stat_arb uses cointegrated pair paths."
    )
    return header + "\n" + _format_table(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=50, help="Monte Carlo paths")
    parser.add_argument("--steps", type=int, default=2000, help="steps per Monte Carlo path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument(
        "--periods-per-year",
        type=float,
        default=252.0,
        help="Sharpe annualization. Use 252 for daily bars, or 252*390 for 1-minute equity bars.",
    )
    parser.add_argument("--wf-steps", type=int, default=4000)
    parser.add_argument("--wf-train", type=int, default=1000)
    parser.add_argument("--wf-test", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)))


if __name__ == "__main__":
    main()
