"""Train the financial LeWM and optionally smoke-test it in the causal harness.

``--synthetic-smoke`` trains on a short Markov path so the checkpoint, Gate 1
probes, and ``LeWMFilter`` load can be checked on CPU without a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lewm_harness.data.binance import load_ohlcv_parquet
from lewm_harness.world_model.train import (
    TrainConfig,
    resolve_device,
    run_smoke_benchmark,
    run_training,
    synthetic_frame,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--data", default=None, help="OHLCV parquet file or directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoint.pt")
    parser.add_argument("--smoke-benchmark", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic_smoke and not args.data:
        parser.error("pass --synthetic-smoke or --data")
    device = resolve_device(args.device)
    config = TrainConfig(epochs=args.epochs, n_steps=args.n_steps, seed=args.seed, amp=args.amp)
    if args.synthetic_smoke:
        frame = synthetic_frame(config.n_steps, config.seed)
    else:
        frame = load_ohlcv_parquet(args.data)
    result = run_training(config, frame, args.out, device, use_amp=args.amp)
    report = result.report
    print(
        "gate1 "
        f"vol_r2={report.vol_r2:.4f} "
        f"baseline_vol_r2={report.baseline_vol_r2:.4f} "
        f"regime_auc={report.regime_auc:.4f} "
        f"baseline_regime_auc={report.baseline_regime_auc:.4f} "
        f"n_train={report.n_train} n_test={report.n_test} "
        f"device={result.device}"
    )
    print(f"checkpoint={result.checkpoint_path}")
    if args.smoke_benchmark:
        rows = run_smoke_benchmark(result.checkpoint_path, seed=args.seed)
        print("filter  mc_sharpe  mc_sharpe_std  mc_sharpe_p5  mc_worst_dd")
        for row in rows:
            print(
                f"{row['filter']:>8}  {row['mc_sharpe']:9.4f}  {row['mc_sharpe_std']:13.4f}  "
                f"{row['mc_sharpe_p5']:12.4f}  {row['mc_worst_dd']:11.4f}"
            )


if __name__ == "__main__":
    main()
