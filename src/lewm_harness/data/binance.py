"""Binance USDⓈ-M perpetual kline archive.

Monthly files live at::

    https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip

``load_ohlcv_parquet`` returns a frame the walk-forward runner can score.
Pass :func:`periods_per_year` for the bar size (``365 * 24 * 12`` for 5-minute
crypto bars, which trade all week).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

_INTERVAL_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}


def interval_milliseconds(interval: str) -> float:
    """Nominal bar length for a Binance interval, in milliseconds."""

    if interval not in _INTERVAL_MINUTES:
        known = ", ".join(sorted(_INTERVAL_MINUTES))
        raise ValueError(f"unsupported interval {interval!r}; expected one of {known}")
    return float(_INTERVAL_MINUTES[interval]) * 60_000.0


def assert_single_symbol(path: str | Path) -> None:
    """Reject a data root that holds more than one symbol directory.

    ``data/BTCUSDT/5m`` is one symbol. ``data/`` with both ``BTCUSDT`` and
    ``ETHUSDT`` is not, because concatenating them would invent returns at
    the symbol boundary.
    """

    root = Path(path)
    if not root.is_dir() or root.name in _INTERVAL_MINUTES:
        return
    children = [child for child in root.iterdir() if child.is_dir()]
    if children and all(child.name in _INTERVAL_MINUTES for child in children):
        return
    symbols = [child.name for child in children if child.name not in _INTERVAL_MINUTES]
    if len(symbols) > 1:
        names = ", ".join(sorted(symbols))
        raise ValueError(f"--data contains more than one symbol folder: {names}")


def periods_per_year(interval: str) -> float:
    """Bar count in a 365-day year for a Binance kline interval.

    5-minute bars are ``365 * 24 * 12``. Crypto books do not close on
    weekends, so this is not the equity-session ``252 * 78``.
    """

    if interval not in _INTERVAL_MINUTES:
        known = ", ".join(sorted(_INTERVAL_MINUTES))
        raise ValueError(f"unsupported interval {interval!r}; expected one of {known}")
    minutes = _INTERVAL_MINUTES[interval]
    return 365.0 * 24.0 * 60.0 / minutes


def kline_zip_url(symbol: str, interval: str, year: int, month: int) -> str:
    """Public archive URL for one monthly USDⓈ-M kline zip."""

    name = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    return (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/{interval}/{name}"
    )


def iter_months(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive ``YYYY-MM`` range as ``(year, month)`` pairs."""

    start_year, start_month = _parse_month(start)
    end_year, end_month = _parse_month(end)
    if (start_year, start_month) > (end_year, end_month):
        raise ValueError("start month is after end month")
    months: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def parse_kline_zip(payload: bytes) -> pd.DataFrame:
    """Parse one Binance monthly kline zip into the harness schema."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            raise ValueError("kline zip did not contain a csv")
        raw = archive.read(names[0])
    return _parse_kline_csv(raw)


def frame_from_klines(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Binance kline columns to ``ts, open, high, low, close, volume, ret``."""

    frame = pd.DataFrame(
        {
            "ts": raw["open_time"].to_numpy(dtype=np.int64),
            "open": raw["open"].to_numpy(dtype=np.float64),
            "high": raw["high"].to_numpy(dtype=np.float64),
            "low": raw["low"].to_numpy(dtype=np.float64),
            "close": raw["close"].to_numpy(dtype=np.float64),
            "volume": raw["volume"].to_numpy(dtype=np.float64),
        }
    )
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last")
    close = frame["close"].to_numpy(dtype=np.float64)
    returns = np.zeros(len(frame), dtype=np.float64)
    if len(frame) > 1:
        previous = close[:-1]
        valid = previous != 0.0
        returns[1:] = np.where(valid, close[1:] / previous - 1.0, 0.0)
    frame["ret"] = returns
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame.reset_index(drop=True)


def load_ohlcv_parquet(path: str | Path, interval: str | None = None) -> pd.DataFrame:
    """Load one parquet file or every parquet file under a directory.

    The frame is sorted, de-duplicated on ``ts``, and free of NaNs. When
    ``interval`` is set (or can be read from a parent folder such as ``5m``),
    ``frame.attrs["periods_per_year"]`` is the value to pass into
    ``run_walk_forward`` / ``compute_metrics``.
    """

    root = Path(path)
    if root.is_dir():
        files = sorted(root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {root}")
        frames = [_read_parquet(file) for file in files]
        frame = pd.concat(frames, ignore_index=True)
        if interval is None:
            interval = _interval_from_path(root)
    else:
        frame = _read_parquet(root)
        if interval is None:
            interval = _interval_from_path(root)
    frame = frame_from_klines(frame) if "open_time" in frame.columns else _finalize_loaded(frame)
    if interval is not None and interval in _INTERVAL_MINUTES:
        frame.attrs["periods_per_year"] = periods_per_year(interval)
        frame.attrs["interval"] = interval
    return frame


def download_monthly_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    out_dir: str | Path,
) -> list[Path]:
    """Download monthly zips and cache each month as parquet.

    Existing parquet files are left in place. A missing month (HTTP 404)
    is skipped. Returns the paths that are present after the call.
    """

    import requests

    destination = Path(out_dir) / symbol / interval
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year, month in iter_months(start, end):
        target = destination / f"{symbol}-{interval}-{year:04d}-{month:02d}.parquet"
        if target.exists():
            written.append(target)
            continue
        url = kline_zip_url(symbol, interval, year, month)
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        frame = parse_kline_zip(response.content)
        frame.to_parquet(target, index=False)
        written.append(target)
    return written


def _parse_month(text: str) -> tuple[int, int]:
    year_text, month_text = text.split("-")
    year, month = int(year_text), int(month_text)
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month in {text!r}")
    return year, month


def _parse_kline_csv(raw: bytes) -> pd.DataFrame:
    sample = raw.lstrip()[:1]
    header = None if sample.isdigit() or sample.startswith(b"-") else 0
    frame = pd.read_csv(io.BytesIO(raw), header=header)
    if header is None:
        if frame.shape[1] < 6:
            raise ValueError("kline csv has fewer than 6 columns")
        frame = frame.iloc[:, : len(_KLINE_COLUMNS)]
        frame.columns = list(_KLINE_COLUMNS[: frame.shape[1]])
    else:
        frame = frame.rename(columns={frame.columns[0]: "open_time"})
    return frame_from_klines(frame)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _finalize_loaded(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, drop bad prices, then rebuild ``ret`` from ``close``.

    Each monthly parquet stores ``ret`` of its own first bar as 0. After
    concatenation that bar is no longer the start of the series, so the
    return is computed again from the previous close.
    """

    required = {"ts", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"parquet is missing columns: {sorted(missing)}")
    cleaned = frame.sort_values("ts").drop_duplicates("ts", keep="last")
    price_columns = [
        name for name in ("ts", "open", "high", "low", "close", "volume") if name in cleaned.columns
    ]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=price_columns)
    returns = cleaned["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cleaned = cleaned.copy()
    cleaned["ret"] = returns.to_numpy(dtype=np.float64)
    return cleaned.reset_index(drop=True)


def _interval_from_path(path: Path) -> str | None:
    parts = [path.name, *path.parts]
    for part in parts:
        if part in _INTERVAL_MINUTES:
            return part
    return None
