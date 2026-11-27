"""Binance kline parsing and the crypto bar-count helper."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lewm_harness.data.binance import (
    frame_from_klines,
    kline_zip_url,
    load_ohlcv_parquet,
    parse_kline_zip,
    periods_per_year,
)


def test_periods_per_year_for_five_minute_crypto() -> None:
    assert periods_per_year("5m") == 365 * 24 * 12
    assert periods_per_year("1m") == 365 * 24 * 60


def test_monthly_archive_url() -> None:
    assert kline_zip_url("BTCUSDT", "5m", 2023, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/5m/BTCUSDT-5m-2023-01.zip"
    )


def test_zip_round_trip_and_parquet_loader(tmp_path: Path) -> None:
    csv = (
        "1609459200000,100,101,99,100.5,12,1609459499999,1,1,1,1,0\n"
        "1609459500000,100.5,102,100,101.0,15,1609459799999,1,1,1,1,0\n"
        "1609459200000,100,101,99,100.5,12,1609459499999,1,1,1,1,0\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-5m-2023-01.csv", csv)
    frame = parse_kline_zip(buffer.getvalue())
    assert list(frame.columns) == ["ts", "open", "high", "low", "close", "volume", "ret"]
    assert len(frame) == 2
    assert frame["ts"].is_monotonic_increasing
    assert not frame.isna().any().any()
    expected_ret = 101.0 / 100.5 - 1.0
    assert frame["ret"].iloc[1] == pytest.approx(expected_ret)
    target = tmp_path / "5m" / "BTCUSDT-5m-2023-01.parquet"
    target.parent.mkdir(parents=True)
    frame.to_parquet(target, index=False)
    loaded = load_ohlcv_parquet(tmp_path, interval="5m")
    assert loaded.attrs["periods_per_year"] == 365 * 24 * 12
    assert len(loaded) == 2


def test_ret_matches_close_ratio() -> None:
    raw = pd.DataFrame(
        {
            "open_time": [1, 2],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.0, 12.0],
            "volume": [1.0, 1.0],
        }
    )
    frame = frame_from_klines(raw)
    np.testing.assert_allclose(frame["ret"].to_numpy(), [0.0, 0.2], atol=1e-12, rtol=0.0)
