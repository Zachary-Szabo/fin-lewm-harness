# fin-lewm-harness

Phase 1 of `lewm-market-filter`: a leak-free backtest that will benchmark a JEPA-style market world model (LeWM) against a Kalman local-level filter and a two-state Gaussian HMM. The world model is a stub. Identity, Kalman, and HMM run today, on the same causal contract the checkpoint will have to honor.

## Causal contract

At decision time `t` the engine builds one `Bar` from row `t` and passes it to `filter.update` and `strategy.step`. The position earns `ret_{t+1}` and pays `fee_bps * 1e-4 * |pos_t - pos_{t-1}|`. That next return is an outcome. It is not on the bar.

`fit` sees only the training slice. Walk-forward and Monte Carlo both split that way. `reset` clears online state (Kalman mean, HMM forward message, trailing windows) and keeps fitted parameters, including the surprise moments below. The simulator column `true_regime` is dropped before a bar is built.

The HMM reports `P(high-vol | observations through t)` from the forward algorithm. There is no backward pass and no Viterbi decode. `stds_[0]` is the low-vol emission and `stds_[1]` is the high-vol emission.

`LeWMFilter` loads a PyTorch checkpoint (`pip install -e ".[ml]"`). `fit` does not train the network. It replays the training slice and calls `calibrate_surprise`, so latent prediction error uses the same z-score as Kalman and HMM. `pred_vol` and `regime_prob` come from the frozen Gate 1 linear heads. The rolling window in `update` contains only bars up to the current one.

## Overlay

Each filter keeps its own raw surprise, then `fit` replays the training frame and freezes the mean and standard deviation of that score:

| Filter | Raw surprise | `pred_vol` |
| --- | --- | --- |
| Identity | absolute return divided by trailing return std | trailing return std |
| Kalman | absolute innovation of the log-price local level | innovation std `sqrt(S_t)` |
| HMM | negative log likelihood of the current return | filtered mixture std |
| LeWM (later) | one-step model surprise | model volatility |

`FilterState.surprise` is the z-score of that raw score. A value of `1` means one training standard deviation above the training mean, for every filter. The replay that estimates those moments uses only `train_df`, then `reset` starts the test from a flat online state.

`apply_overlay` then does three things, in order:

1. If `regime_cutoff` is set and `regime_prob` exceeds it, the position is `0`. Mean reversion and the pairs fade default to `0.85`, because a high-vol trend is where those books get run over. Momentum defaults to no cutoff, because that is when it is supposed to stay on.
2. Otherwise the raw signal is multiplied by `1 / (1 + surprise_scale * max(0, surprise - threshold))`.
3. If a volatility target is active, the result is multiplied by `min(1, target_vol / max(pred_vol, eps))`. Set `target_vol` on the strategy to force one number. Leave it unset and the backtester uses that filter's median training `pred_vol`, so Identity, Kalman, HMM, and later LeWM share the sizing rule without sharing units.

The backtester clips the published position to `[-1, 1]`.

## Metrics

`compute_metrics(..., periods_per_year=252.0)` returns:

- `total_return`: sum of step PnL (additive, not compounded).
- `sharpe`: mean step PnL over its sample standard deviation, times `sqrt(periods_per_year)`.
- `max_drawdown`: largest peak-to-trough loss of the equity curve `1 + cumsum(pnl)`, as a positive fraction of the peak. A long losing path can print a drawdown above 1.
- `turnover`: mean absolute position change, including the move off a flat start.
- `win_rate`: fraction of non-zero PnL steps that are positive.

`periods_per_year` is plumbed through `run_monte_carlo` and `run_walk_forward`. Daily bars use `252`. A 1-minute equity session can pass `252 * 390`. The synthetic benchmark below uses `252` only as a common scale. These steps are not calendar days.

## Run

From the repository root. The benchmark script puts `src` on `sys.path`; pytest does the same via `pyproject.toml`.

```bash
python3 -m pytest -v
python3 scripts/run_benchmark.py
```

The world model is an optional extra. It trains on CPU without a GPU:

```bash
python3 -m pip install -e ".[ml]"
python3 scripts/train_lewm.py --synthetic-smoke --device cpu --epochs 2 --pred-hop 1 --out checkpoint.pt
python3 scripts/download_binance.py --symbols BTCUSDT ETHUSDT --interval 5m --start 2023-01 --end 2025-12 --out data/binance
```

`load_ohlcv_parquet` sets `frame.attrs["periods_per_year"]` (`365 * 24 * 12` for 5-minute bars). Pass that into `run_walk_forward` and `compute_metrics`.

`--pred-hop` (default 1) is the bar shift from a context window to the JEPA target window. Gate 1 does not use that hop. It scores the next full window: the `seq_len` bars that start when the context window ends. Those two horizons are different on purpose. Training writes `best.pt` from validation vol R² (tie-break: validation loss) and `last.pt` from the final epoch. A history CSV records `epoch, train_loss, val_loss, val_vol_r2, val_regime_auc`. The test tail is not used to pick the epoch. `--patience` (default 5) stops after that many epochs without a better validation score.

Optional benchmark flags: `--paths`, `--steps`, `--seed`, `--fee-bps`, `--wf-steps`, `--wf-train`, `--wf-test`, `--periods-per-year`.

## Benchmark

`python3 scripts/run_benchmark.py` with the defaults: 50 Monte Carlo paths of 2,000 steps (fit on the first half, score the rest), walk-forward of 4,000 steps with train 1,000 / test 500, and a 1 bp fee. Mean reversion and momentum use single-asset regime paths. Stat-arb uses cointegrated pair paths, so its edge is the synthetic spread.

```
      strategy    filter  mc_sharpe  mc_sharpe_std  mc_sharpe_p5  mc_worst_dd  wf_sharpe  wf_return  wf_max_dd
--------------  --------  ---------  -------------  ------------  -----------  ---------  ---------  ---------
mean_reversion  identity    -0.4735         0.5939       -1.3548       0.7134    -0.4935    -0.2969     0.3455
mean_reversion    kalman    -0.6042         0.5945       -1.5050       1.2252    -0.5371    -0.5891     0.6397
mean_reversion       hmm    -0.0920         0.6951       -1.3560       0.7209    -0.1114    -0.0908     0.2157
      momentum  identity     0.2275         0.5948       -0.8145       0.3773     0.1513     0.0929     0.1456
      momentum    kalman     0.2296         0.5742       -0.7560       0.5482     0.1172     0.1574     0.2710
      momentum       hmm     0.1846         0.5651       -0.6947       0.5169    -0.0457    -0.0454     0.4189
      stat_arb  identity     1.9254         0.4904        1.0783       0.1351     2.1743     1.4951     0.0444
      stat_arb    kalman     1.7884         0.4036        1.1548       0.1978     1.9403     2.8965     0.0583
      stat_arb       hmm     1.7824         0.5328        1.1239       0.1758     1.8588     1.5400     0.0866
```

## Phase 3: Real-data walk-forward

Each window trains only on its past months. A validation tail (`val_frac` 0.1 by default) is carved from the end of that train block, with an embargo of `2 * seq_len` bars so train labels and JEPA targets do not touch validation. The test months are held out of the scaler, the optimizer, the probe fit, and the epoch choice. LeWM surprise z-scores and `reference_vol_` are fit on that validation tail, not on the bars the encoder trained on. If the file has no `spread` column, the runner adds a trailing close deviation that uses only past bars, so the pairs strategy can run on a single symbol. `--data` must not contain more than one symbol folder.

On a Kaggle T4, write outputs under `/kaggle/working` and turn on mixed precision (`--amp` uses fp16 on a T4 and bf16 on Ampere or newer):

```bash
python scripts/download_binance.py --symbols BTCUSDT ETHUSDT --interval 5m --start 2023-01 --end 2025-12 --out /kaggle/working/data
python scripts/train_lewm.py --data /kaggle/working/data/BTCUSDT/5m --device cuda --amp --epochs 20 --pred-hop 1 --patience 5 --out /kaggle/working/lewm
python scripts/run_walk_forward.py --data /kaggle/working/data/BTCUSDT/5m --device cuda --amp --epochs 20 --seeds 0 1 2 --train-months 6 --test-months 1 --step-months 1 --pred-hop 1 --out /kaggle/working/walk_forward
```

`--reuse-checkpoints` skips a finished `(window, seed)` only when `config.json` hashes to the same `TrainConfig`, window start/end timestamps, train-data fingerprint, and seed. Checkpoints and CSV files are replaced atomically. The summary is mean and std Sharpe across windows and seeds, the 5th percentile Sharpe, the worst max drawdown, and mean turnover. It prints LeWM minus each baseline, with N, the per-window deltas, and a sign count. A bootstrap 95% interval is included only when N is at least 8. `summary.txt` also records `periods_per_year` and the median bar spacing. `--interval` must agree with that spacing within 1%.

Results below are a placeholder until that Kaggle run is filled in.

| strategy | filter | sharpe_mean | sharpe_std | sharpe_p5 | max_drawdown | turnover |
| --- | --- | --- | --- | --- | --- | --- |
| mean_reversion | identity | — | — | — | — | — |
| mean_reversion | kalman | — | — | — | — | — |
| mean_reversion | hmm | — | — | — | — | — |
| mean_reversion | lewm | — | — | — | — | — |
| momentum | identity | — | — | — | — | — |
| momentum | kalman | — | — | — | — | — |
| momentum | hmm | — | — | — | — | — |
| momentum | lewm | — | — | — | — | — |
| stat_arb | identity | — | — | — | — | — |
| stat_arb | kalman | — | — | — | — | — |
| stat_arb | hmm | — | — | — | — | — |
| stat_arb | lewm | — | — | — | — | — |
