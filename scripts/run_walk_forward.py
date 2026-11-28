"""Rolling monthly walk-forward of Identity, Kalman, HMM, and LeWM.

Example (Kaggle T4)::

    python scripts/run_walk_forward.py --data /kaggle/working/data/BTCUSDT/5m --device cuda --amp --epochs 20 --seeds 0 1 2 --train-months 6 --test-months 1 --step-months 1 --out /kaggle/working/walk_forward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lewm_harness.data.binance import load_ohlcv_parquet
from lewm_harness.engine.monthly_walk import run_monthly_walk_forward
from lewm_harness.world_model.train import TrainConfig, resolve_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="OHLCV parquet file or directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--train-months", type=int, required=True)
    parser.add_argument("--test-months", type=int, required=True)
    parser.add_argument("--step-months", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--pred-hop", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--interval", default=None, help="Override the bar size, e.g. 5m")
    args = parser.parse_args(argv)
    if (args.seq_len - 4) % 4 != 0:
        parser.error("--seq-len must satisfy (seq_len - 4) % 4 == 0 for the default patch size")
    frame = load_ohlcv_parquet(args.data, interval=args.interval)
    config = TrainConfig(
        epochs=args.epochs,
        seq_len=args.seq_len,
        pred_hop=args.pred_hop,
        patience=args.patience,
        amp=args.amp,
    )
    result = run_monthly_walk_forward(
        frame,
        args.out,
        config,
        seeds=args.seeds,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        device=resolve_device(args.device),
        use_amp=args.amp,
        fee_bps=args.fee_bps,
        interval=args.interval or frame.attrs.get("interval"),
        reuse_checkpoints=args.reuse_checkpoints,
    )
    print(result.summary_text, end="")
    print(f"windows={result.windows_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
