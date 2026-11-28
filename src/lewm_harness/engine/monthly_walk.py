"""Month-by-month walk-forward on a real (or synthetic) OHLCV frame.

Each window trains on its own past months. The test months are not passed to
``fit``, to the feature scaler, or to LeWM checkpoint selection. Surprise
z-scores and ``reference_vol_`` are estimated on that window's validation
tail, not on the bars the encoder trained on.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lewm_harness.data.binance import interval_milliseconds, periods_per_year
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
_FILTERS = ("identity", "kalman", "hmm", "lewm")
_BASELINES = ("identity", "kalman", "hmm")
_WINDOW_COLUMNS = [
    "window",
    "seed",
    "strategy",
    "filter",
    "sharpe",
    "total_return",
    "max_drawdown",
    "turnover",
    "win_rate",
    "flat_fraction",
    "periods_per_year",
]


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


def resolve_bar_clock(frame: pd.DataFrame, interval: str | None) -> tuple[float, float]:
    """Return ``(periods_per_year, bar_spacing_ms)`` from the median timestamp gap.

    When ``interval`` is set, its nominal length must agree with that gap
    within 1%. Otherwise the annualization would not describe these bars.
    """

    if len(frame) < 3:
        raise ValueError("need at least 3 bars to measure spacing")
    spacing_ms = float(np.median(np.diff(frame["ts"].to_numpy(dtype=np.int64))))
    if spacing_ms <= 0.0:
        raise ValueError("bar timestamps are not increasing")
    chosen = interval
    if chosen is None and isinstance(frame.attrs.get("interval"), str):
        chosen = str(frame.attrs["interval"])
    if chosen:
        expected = interval_milliseconds(chosen)
        if abs(spacing_ms - expected) / expected > 0.01:
            raise ValueError(
                f"--interval {chosen} is {expected:.0f} ms but median bar spacing is {spacing_ms:.3f} ms"
            )
        return periods_per_year(chosen), spacing_ms
    minutes = spacing_ms / 60_000.0
    return 365.0 * 24.0 * 60.0 / minutes, spacing_ms


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
    bar_year, spacing_ms = resolve_bar_clock(working, interval)
    windows_path = out / "windows.csv"
    if not reuse_checkpoints and windows_path.exists():
        windows_path.unlink()
    done = _completed_pairs(windows_path) if reuse_checkpoints else set()
    start = 0
    window_id = 0
    while start + train_months + test_months <= len(months):
        train_keys = set(months[start : start + train_months])
        test_keys = set(months[start + train_months : start + train_months + test_months])
        train_df = working.loc[labels.isin(train_keys)].reset_index(drop=True)
        test_df = working.loc[labels.isin(test_keys)].reset_index(drop=True)
        for seed in seeds:
            directory = out / "checkpoints" / f"w{window_id}_s{seed}"
            payload = _reuse_payload(config, train_df, test_df, seed, use_amp)
            matches = _config_matches(directory, payload)
            if reuse_checkpoints and (window_id, seed) in done and matches:
                continue
            _drop_pair(windows_path, window_id, seed)
            checkpoint = _train_window(
                train_df,
                directory,
                config,
                seed,
                device,
                use_amp,
                reuse_checkpoints and matches,
                payload,
            )
            rows = _score_window(
                window_id,
                seed,
                train_df,
                test_df,
                checkpoint,
                bar_year,
                fee_bps,
                device,
            )
            _append_windows(windows_path, rows)
        window_id += 1
        start += step_months
    windows = pd.read_csv(windows_path) if windows_path.exists() else pd.DataFrame(columns=_WINDOW_COLUMNS)
    summary = _summarize(windows)
    summary_path = out / "summary.csv"
    _atomic_csv(summary, summary_path)
    text = _format_summary(summary, _baseline_deltas(windows), bar_year, spacing_ms)
    _atomic_text(out / "summary.txt", text)
    return WalkForwardOutputs(windows_path, summary_path, text)


def _train_window(
    train_df: pd.DataFrame,
    directory: Path,
    config: TrainConfig,
    seed: int,
    device,
    use_amp: bool,
    reuse: bool,
    payload: dict,
) -> Path:
    best = directory / "best.pt"
    if reuse and best.is_file():
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
    _atomic_json(directory / "config.json", payload)
    return result.checkpoint_path


def _score_window(
    window_id: int,
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    checkpoint: Path,
    periods: float,
    fee_bps: float,
    device,
) -> list[dict[str, object]]:
    filters = {
        "identity": IdentityFilter(),
        "kalman": KalmanFilter(),
        "hmm": HMMFilter(random_state=seed),
        "lewm": LeWMFilter(checkpoint, device=device),
    }
    for filt in filters.values():
        filt.fit(train_df)
    rows: list[dict[str, object]] = []
    for strategy_name, strategy_cls in _STRATEGIES.items():
        for filter_name, filt in filters.items():
            filt.reset()
            result = run_backtest(test_df, filt, strategy_cls(), fee_bps=fee_bps)
            metrics = compute_metrics(result.pnl, result.positions, periods_per_year=periods)
            flat = float(np.mean(result.positions == 0.0)) if result.positions.size else 0.0
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
                    "flat_fraction": flat,
                    "periods_per_year": periods,
                }
            )
    return rows


def _summarize(windows: pd.DataFrame) -> pd.DataFrame:
    records = []
    if windows.empty:
        return pd.DataFrame(
            columns=["strategy", "filter", "sharpe_mean", "sharpe_std", "sharpe_p5", "max_drawdown", "turnover"]
        )
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


def _baseline_deltas(windows: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """LeWM minus each baseline, with per-window deltas and a sign count.

    Seeds are averaged inside each window. The bootstrap 95% CI is omitted
    when fewer than 8 windows are available.
    """

    if windows.empty or "lewm" not in set(windows["filter"]):
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(seed)
    for strategy, group in windows.groupby("strategy", sort=False):
        per_window = group.groupby(["window", "filter"], sort=False)["sharpe"].mean().unstack("filter")
        if "lewm" not in per_window.columns:
            continue
        n_windows = int(len(per_window))
        for baseline in _BASELINES:
            if baseline not in per_window.columns:
                continue
            delta = (per_window["lewm"] - per_window[baseline]).to_numpy(dtype=np.float64)
            record = {
                "strategy": strategy,
                "baseline": baseline,
                "n_windows": n_windows,
                "delta_mean": float(delta.mean()) if delta.size else float("nan"),
                "n_positive": int(np.sum(delta > 0.0)),
                "n_negative": int(np.sum(delta < 0.0)),
                "n_zero": int(np.sum(delta == 0.0)),
                "per_window": json.dumps([float(value) for value in delta]),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            if n_windows >= 8 and delta.size:
                draws = rng.choice(delta, size=(n_boot, delta.size), replace=True).mean(axis=1)
                record["ci_low"] = float(np.quantile(draws, 0.025))
                record["ci_high"] = float(np.quantile(draws, 0.975))
            rows.append(record)
    return pd.DataFrame(rows)


def _format_summary(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    periods: float,
    spacing_ms: float,
) -> str:
    headers = ["strategy", "filter", "sharpe_mean", "sharpe_std", "sharpe_p5", "max_drawdown", "turnover"]
    lines = [
        f"periods_per_year={periods:.6f}",
        f"bar_spacing_ms={spacing_ms:.6f}",
        "",
        "  ".join(f"{name:>12}" for name in headers),
    ]
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
    lines.append("lewm_minus_baseline")
    if deltas.empty:
        lines.append("(no comparison)")
    else:
        for _, row in deltas.iterrows():
            values = json.loads(row["per_window"])
            rendered = ", ".join(f"w{index}={value:.4f}" for index, value in enumerate(values))
            lines.append(
                f"{row['strategy']}: lewm - {row['baseline']} N={int(row['n_windows'])} "
                f"mean={row['delta_mean']:.4f} "
                f"signs +{int(row['n_positive'])}/-{int(row['n_negative'])}/0={int(row['n_zero'])}"
            )
            lines.append(f"  per-window deltas: {rendered}")
            if int(row["n_windows"]) >= 8 and np.isfinite(row["ci_low"]):
                lines.append(f"  95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
            else:
                lines.append("  bootstrap CI skipped (N < 8)")
    return "\n".join(lines) + "\n"


def _reuse_payload(
    config: TrainConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    use_amp: bool,
) -> dict:
    body = {
        "train_config": asdict(config),
        "train_start_ts": int(train_df["ts"].iloc[0]),
        "train_end_ts": int(train_df["ts"].iloc[-1]),
        "test_start_ts": int(test_df["ts"].iloc[0]),
        "test_end_ts": int(test_df["ts"].iloc[-1]),
        "data_fingerprint": _fingerprint(train_df),
        "seed": int(seed),
        "amp": bool(use_amp),
    }
    body["train_config"]["seed"] = int(seed)
    body["train_config"]["amp"] = bool(use_amp)
    body["train_config"]["train_frac"] = 1.0
    body["hash"] = _hash(body)
    return body


def _config_matches(directory: Path, payload: dict) -> bool:
    path = directory / "config.json"
    best = directory / "best.pt"
    if not path.is_file() or not best.is_file():
        return False
    saved = json.loads(path.read_text())
    return saved.get("hash") == payload["hash"]


def _fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for column in ("ts", "open", "high", "low", "close", "volume", "ret", "spread"):
        if column not in frame.columns:
            continue
        values = np.ascontiguousarray(frame[column].to_numpy())
        digest.update(column.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _hash(payload: dict) -> str:
    encoded = json.dumps({key: value for key, value in payload.items() if key != "hash"}, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _completed_pairs(path: Path) -> set[tuple[int, int]]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    expected = len(_STRATEGIES) * len(_FILTERS)
    counts = frame.groupby(["window", "seed"]).size()
    return {(int(window), int(seed)) for (window, seed), count in counts.items() if int(count) >= expected}


def _drop_pair(path: Path, window_id: int, seed: int) -> None:
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    keep = ~((frame["window"] == window_id) & (frame["seed"] == seed))
    if bool(keep.all()):
        return
    _atomic_csv(frame.loc[keep].reset_index(drop=True), path)


def _append_windows(path: Path, rows: list[dict[str, object]]) -> None:
    fresh = pd.DataFrame(rows, columns=_WINDOW_COLUMNS)
    if path.is_file():
        existing = pd.read_csv(path)
        fresh = pd.concat([existing, fresh], ignore_index=True)
    _atomic_csv(fresh, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
