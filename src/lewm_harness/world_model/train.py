"""Training loop for :class:`FinancialLeWM`.

The synthetic smoke path builds a short Markov series, trains on CPU for a
couple of epochs, fits Gate 1 probes on a time split, and writes a checkpoint
the causal filter can load.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lewm_harness.data.markov import generate_regime_series
from lewm_harness.world_model.encoder import PatchTSTEncoder
from lewm_harness.world_model.features import prepare_feature_matrix
from lewm_harness.world_model.model import FinancialLeWM
from lewm_harness.world_model.probes import Gate1Report, LinearHead, evaluate_gate1


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 16
    patch_len: int = 4
    stride: int = 4
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    pred_hidden_dim: int = 32
    vol_window: int = 8
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    lambda_reg: float = 0.1
    ema_decay: float = 0.99
    train_frac: float = 0.7
    seed: int = 0
    n_steps: int = 512
    amp: bool = False


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    report: Gate1Report
    device: str


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA, then MPS, then CPU."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 on Ampere and newer, fp16 on older CUDA (T4). CPU and MPS stay off."""

    if device.type != "cuda":
        return None
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def build_model(config: TrainConfig) -> FinancialLeWM:
    encoder = PatchTSTEncoder(
        in_channels=5,
        seq_len=config.seq_len,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=0.0,
    )
    return FinancialLeWM(
        encoder,
        latent_dim=config.d_model,
        pred_hidden_dim=config.pred_hidden_dim,
        ema_decay=config.ema_decay,
        lambda_reg=config.lambda_reg,
    )


def train_model(
    config: TrainConfig,
    scaled: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> FinancialLeWM:
    """Fit the world model on windows that end before ``train_end`` is applied by the caller.

    ``scaled`` here is already the training-portion matrix. Windows never
    read past its last row.
    """

    torch.manual_seed(config.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    cast_dtype = amp_dtype(device) if use_amp else None
    scaler = torch.amp.GradScaler("cuda") if cast_dtype == torch.float16 else None
    max_start = len(scaled) - config.seq_len - 1
    if max_start < 1:
        raise ValueError("scaled series is shorter than two windows")
    starts = np.arange(0, max_start + 1)
    model.train()
    for _epoch in range(config.epochs):
        order = np.random.default_rng(config.seed + _epoch).permutation(starts)
        for offset in range(0, len(order), config.batch_size):
            batch_starts = order[offset : offset + config.batch_size]
            if batch_starts.size < 2:
                continue
            x_t, x_next = _batch_windows(scaled, batch_starts, config.seq_len, device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=cast_dtype)
                if cast_dtype is not None
                else nullcontext()
            )
            with context:
                loss, _parts = model.forward_loss(x_t, x_next)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            model.update_target()
    model.eval()
    return model


def run_training(
    config: TrainConfig,
    frame,
    out_path: str | Path,
    device: torch.device,
    use_amp: bool,
) -> TrainResult:
    """Time-split ``frame``, train, probe, and write ``checkpoint.pt``."""

    train_end = int(len(frame) * config.train_frac)
    prepared = prepare_feature_matrix(frame, train_end, config.vol_window)
    scaled = np.asarray(prepared["scaled"], dtype=np.float64)
    raw = np.asarray(prepared["raw"], dtype=np.float64)
    model = train_model(config, scaled[:train_end], device, use_amp)
    report = evaluate_gate1(model, scaled, raw, train_end, config.seq_len, device)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, model, config, prepared, report)
    return TrainResult(checkpoint_path=path, report=report, device=str(device))


def synthetic_frame(n_steps: int, seed: int):
    return generate_regime_series(n_steps, seed)


def save_checkpoint(
    path: Path,
    model: FinancialLeWM,
    config: TrainConfig,
    prepared: dict,
    report: Gate1Report,
) -> None:
    feature_mean = np.asarray(prepared["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(prepared["feature_std"], dtype=np.float64)
    blob = {
        "version": 2,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": {
            "seq_len": config.seq_len,
            "patch_len": config.patch_len,
            "stride": config.stride,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "pred_hidden_dim": config.pred_hidden_dim,
            "vol_window": config.vol_window,
            "ema_decay": config.ema_decay,
            "lambda_reg": config.lambda_reg,
        },
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "log_volume_mean": prepared["log_volume_mean"],
        "log_volume_std": prepared["log_volume_std"],
        "vol_head": {"coef": report.vol_head.coef.tolist(), "intercept": report.vol_head.intercept},
        "regime_head": {
            "coef": report.regime_head.coef.tolist(),
            "intercept": report.regime_head.intercept,
        },
    }
    torch.save(blob, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[FinancialLeWM, dict]:
    """Rebuild the model and return ``(model, blob)`` on CPU by default."""

    target = torch.device("cpu") if device is None else device
    try:
        blob = torch.load(path, map_location=target, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=target)
    config = TrainConfig(
        seq_len=int(blob["config"]["seq_len"]),
        patch_len=int(blob["config"]["patch_len"]),
        stride=int(blob["config"]["stride"]),
        d_model=int(blob["config"]["d_model"]),
        n_heads=int(blob["config"]["n_heads"]),
        n_layers=int(blob["config"]["n_layers"]),
        pred_hidden_dim=int(blob["config"]["pred_hidden_dim"]),
        vol_window=int(blob["config"]["vol_window"]),
        ema_decay=float(blob["config"]["ema_decay"]),
        lambda_reg=float(blob["config"]["lambda_reg"]),
    )
    model = build_model(config).to(target)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def head_from_blob(payload: dict) -> LinearHead:
    return LinearHead(
        coef=np.asarray(payload["coef"], dtype=np.float64),
        intercept=float(payload["intercept"]),
    )


def run_smoke_benchmark(checkpoint: str | Path, n_paths: int = 3, n_steps: int = 240, seed: int = 0) -> list[dict[str, float | str]]:
    """Short out-of-sample comparison of Identity, Kalman, HMM, and LeWM.

    One strategy (mean reversion) keeps the run small. LeWM is loaded from
    ``checkpoint`` and still calibrates surprise only on each path's train prefix.
    """

    from lewm_harness.engine.monte_carlo import run_monte_carlo
    from lewm_harness.filters.hmm import HMMFilter
    from lewm_harness.filters.identity import IdentityFilter
    from lewm_harness.filters.kalman import KalmanFilter
    from lewm_harness.filters.lewm import LeWMFilter
    from lewm_harness.strategies.mean_reversion import MeanReversionStrategy

    def strategy_factory():
        return MeanReversionStrategy(regime_cutoff=0.85, surprise_scale=1.0, surprise_threshold=1.0)

    filters = {
        "identity": IdentityFilter,
        "kalman": KalmanFilter,
        "hmm": HMMFilter,
        "lewm": lambda: LeWMFilter(checkpoint),
    }
    rows: list[dict[str, float | str]] = []
    for name, factory in filters.items():
        summary = run_monte_carlo(
            n_paths,
            n_steps,
            factory,
            strategy_factory,
            seed=seed,
            train_frac=0.5,
            fee_bps=1.0,
            periods_per_year=252.0,
        )
        rows.append(
            {
                "filter": name,
                "mc_sharpe": summary.sharpe_mean,
                "mc_sharpe_std": summary.sharpe_std,
                "mc_sharpe_p5": summary.sharpe_p5,
                "mc_worst_dd": summary.worst_max_drawdown,
            }
        )
    return rows


def _batch_windows(
    scaled: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = np.stack([scaled[int(s) : int(s) + seq_len] for s in starts])
    nxt = np.stack([scaled[int(s) + 1 : int(s) + 1 + seq_len] for s in starts])
    return (
        torch.as_tensor(current, dtype=torch.float32, device=device),
        torch.as_tensor(nxt, dtype=torch.float32, device=device),
    )
