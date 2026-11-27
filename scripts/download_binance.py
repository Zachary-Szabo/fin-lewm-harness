"""Download Binance USDⓈ-M monthly klines into a parquet cache.

Example::

    python scripts/download_binance.py --symbols BTCUSDT ETHUSDT --interval 5m --start 2023-01 --end 2025-12 --out data/binance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lewm_harness.data.binance import download_monthly_klines, periods_per_year


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", required=True, help="YYYY-MM, inclusive")
    parser.add_argument("--end", required=True, help="YYYY-MM, inclusive")
    parser.add_argument("--out", default="data/binance")
    args = parser.parse_args(argv)
    print(f"periods_per_year={periods_per_year(args.interval):.0f} interval={args.interval}")
    for symbol in args.symbols:
        paths = download_monthly_klines(symbol, args.interval, args.start, args.end, args.out)
        print(f"{symbol}: {len(paths)} monthly parquet file(s) in {args.out}/{symbol}/{args.interval}")


if __name__ == "__main__":
    main()
