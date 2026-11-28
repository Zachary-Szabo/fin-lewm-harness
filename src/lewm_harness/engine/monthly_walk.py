"""Month-by-month walk-forward on a real (or synthetic) OHLCV frame.

Each window trains on its own past months. The test months are not passed to
``fit``, to the feature scaler, or to LeWM checkpoint selection. Surprise
z-scores are a causal replay of that same train slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lewm_harness.data.binance import periods_per_year
from lewm_harness.engine.backtester import run_backtest
from lewm_harness.engine.metrics import compute_metrics
from lewm_harness.filters.hmm import HMMFilter
from lewm_harness.filters.identity import IdentityFilter
from lewm_harness.filters.kalman import KalmanFilter
from lewm_harness.filters.lewm import LeWMFilter
from lewm_harness.strategies.mean_reversion import MeanReversionStrategy
from lewm_harness.strategies.momentum import MomentumStrategy
from lewm_harness.strategies.stat_arb import StatArbStrategy
from lewm_harness.world_model.train import TrainConfig, run_training

_STRATEGIES = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "stat_arb": StatArbStrategy,
}


@dataclass(frozen=True)
class WalkForwardOutputs:
    windows_path: Path
    summary_path: Path
    summary_text: str


def ensure_spread(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add a causal trailing price deviation when the frame has no pair spread."""

    if "spread" in frame.columns:
        return frame
    close = frame["close"].to_numpy(dtype=np.float64)
    spread = np.zeros(len(frame), dtype=np.float64)
    for index in range(len(frame)):
        start = max(0, index - window + 1)
        spread[index] = close[index] - float(close[start : index + 1].mean())
    out = frame.copy()
    out["spread"] = spread
    return out


def month_labels(frame: pd.DataFrame) -> pd.Series:
    """Calendar month of each bar. ``ts`` is Unix time in milliseconds."""

    stamps = pd.to_datetime(frame["ts"].to_numpy(dtype=np.int64), unit="ms", utc=True)
    return pd.Series(stamps.tz_convert("UTC").tz_localize(None).to_period("M"), index=frame.index)


def infer_periods_per_year(frame: pd.DataFrame, interval: str | None) -> float:
    if interval:
        return periods_per_year(interval)
    stored = frame.attrs.get("interval")
    if isinstance(stored, str):
        return periods_per_year(stored)
    if len(frame) < 3:
        return periods_per_year("5m")
    delta_ms = float(np.median(np.diff(frame["ts"].to_numpy(dtype=np.int64))))
    minutes = max(delta_ms / 60_000.0, 1e-6)
    return 365.0 * 24.0 * 60.0 / minutes


def run_monthly_walk_forward(
    frame: pd.DataFrame,
    out_dir: str | Path,
    config: TrainConfig,
    seeds: list[int],
    train_months: int,
    test_months: int,
    step_months: int,
    device,
    use_amp: bool,
    fee_bps: float = 1.0,
    interval: str | None = None,
    reuse_checkpoints: bool = False,
) -> WalkForwardOutputs:
    """Train and score every rolling month window. Test months stay out of fit."""

    if train_months < 1 or test_months < 1 or step_months < 1:
        raise ValueError("month counts must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    working = ensure_spread(frame)
    labels = month_labels(working)
    months = list(dict.fromkeys(labels.tolist()))
    if len(months) < train_months + test_months:
        raise ValueError(
            f"need at least {train_months + test_months} months, found {len(months)}"
        )
    bar_year = infer_periods_per_year(working, interval)
    rows: list[dict[str, object]] = []
    start = 0
    window_id = 0
    while start + train_months + test_months <= len(months):
        train_keys = set(months[start : start + train_months])
        test_keys = set(months[start + train_months : start + train_months + test_months])
        train_df = working.loc[labels.isin(train_keys)].reset_index(drop=True)
        test_df = working.loc[labels.isin(test_keys)].reset_index(drop=True)
        for seed in seeds:
            checkpoint = _train_window(
                train_df,
                out / "checkpoints" / f"w{window_id}_s{seed}",
                config,
                seed,
                device,
                use_amp,
                reuse_checkpoints,
            )
            rows.extend(
                _score_window(
                    window_id,
                    seed,
                    train_df,
                    test_df,
                    checkpoint,
                    bar_year,
                    fee_bps,
                )
            )
        window_id += 1
        start += step_months
    windows = pd.DataFrame(rows)
    windows_path = out / "windows.csv"
    windows.to_csv(windows_path, index=False)
    summary = _summarize(windows)
    summary_path = out / "summary.csv"
    summary.to_csv(summary_path, index=False)
    text = _format_summary(summary, _bootstrap_deltas(windows))
    (out / "summary.txt").write_text(text)
    return WalkForwardOutputs(windows_path, summary_path, text)


def _train_window(
    train_df: pd.DataFrame,
    directory: Path,
    config: TrainConfig,
    seed: int,
    device,
    use_amp: bool,
    reuse_checkpoints: bool,
) -> Path:
    best = directory / "best.pt"
    if reuse_checkpoints and best.is_file():
        return best
    window_config = TrainConfig(
        seq_len=config.seq_len,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        pred_hidden_dim=config.pred_hidden_dim,
        vol_window=config.vol_window,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        lambda_reg=config.lambda_reg,
        ema_decay=config.ema_decay,
        train_frac=1.0,
        val_frac=config.val_frac,
        seed=seed,
        pred_hop=config.pred_hop,
        patience=config.patience,
        amp=use_amp,
    )
    result = run_training(window_config, train_df, directory, device, use_amp)
    return result.checkpoint_path


def _score_window(
    window_id: int,
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    checkpoint: Path,
    periods: float,
    fee_bps: float,
) -> list[dict[str, object]]:
    builders = {
        "identity": lambda: IdentityFilter(),
        "kalman": lambda: KalmanFilter(),
        "hmm": lambda: HMMFilter(random_state=seed),
        "lewm": lambda: LeWMFilter(checkpoint),
    }
    rows: list[dict[str, object]] = []
    for strategy_name, strategy_cls in _STRATEGIES.items():
        for filter_name, factory in builders.items():
            filt = factory()
            filt.fit(train_df)
            result = run_backtest(test_df, filt, strategy_cls(), fee_bps=fee_bps)
            metrics = compute_metrics(result.pnl, result.positions, periods_per_year=periods)
            rows.append(
                {
                    "window": window_id,
                    "seed": seed,
                    "strategy": strategy_name,
                    "filter": filter_name,
                    "sharpe": metrics.sharpe,
                    "total_return": metrics.total_return,
                    "max_drawdown": metrics.max_drawdown,
                    "turnover": metrics.turnover,
                    "win_rate": metrics.win_rate,
                    "periods_per_year": periods,
                }
            )
    return rows


def _summarize(windows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (strategy, filt), group in windows.groupby(["strategy", "filter"], sort=False):
        sharpe = group["sharpe"].to_numpy(dtype=np.float64)
        records.append(
            {
                "strategy": strategy,
                "filter": filt,
                "sharpe_mean": float(sharpe.mean()),
                "sharpe_std": float(sharpe.std(ddof=1)) if sharpe.size > 1 else 0.0,
                "sharpe_p5": float(np.percentile(sharpe, 5)),
                "max_drawdown": float(group["max_drawdown"].max()),
                "turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(records)


def _bootstrap_deltas(windows: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """LeWM minus the best non-LeWM Sharpe, with a 95% bootstrap CI over windows.

    Seeds are averaged inside each window first, so the bootstrap does not
    treat repeated seeds as extra independent samples.
    """

    baselines = ["identity", "kalman", "hmm"]
    rows = []
    rng = np.random.default_rng(seed)
    for strategy, group in windows.groupby("strategy", sort=False):
        per_window = (
            group.groupby(["window", "filter"], sort=False)["sharpe"].mean().unstack("filter")
        )
        if "lewm" not in per_window.columns or not set(baselines).issubset(per_window.columns):
            continue
        best_name = str(per_window[baselines].mean().idxmax())
        delta = (per_window["lewm"] - per_window[best_name]).to_numpy(dtype=np.float64)
        if delta.size == 0:
            continue
        draws = rng.choice(delta, size=(n_boot, delta.size), replace=True).mean(axis=1)
        rows.append(
            {
                "strategy": strategy,
                "baseline": best_name,
                "delta_mean": float(delta.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _format_summary(summary: pd.DataFrame, deltas: pd.DataFrame) -> str:
    headers = ["strategy", "filter", "sharpe_mean", "sharpe_std", "sharpe_p5", "max_drawdown", "turnover"]
    lines = ["  ".join(f"{name:>12}" for name in headers)]
    for _, row in summary.iterrows():
        lines.append(
            "  ".join(
                [
                    f"{row['strategy']:>12}",
                    f"{row['filter']:>12}",
                    f"{row['sharpe_mean']:12.4f}",
                    f"{row['sharpe_std']:12.4f}",
                    f"{row['sharpe_p5']:12.4f}",
                    f"{row['max_drawdown']:12.4f}",
                    f"{row['turnover']:12.4f}",
                ]
            )
        )
    lines.append("")
    lines.append("lewm_minus_best_baseline")
    if deltas.empty:
        lines.append("(no comparison)")
    else:
        for _, row in deltas.iterrows():
            lines.append(
                f"{row['strategy']}: lewm - {row['baseline']} = {row['delta_mean']:.4f} "
                f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
    return "\n".join(lines) + "\n"
