"""Training loop for :class:`FinancialLeWM`.

The synthetic smoke path builds a short Markov series, trains on CPU for a
couple of epochs, fits Gate 1 probes on a time split, and writes a checkpoint
the causal filter can load.
"""

from __future__ import annotations

import os
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
    val_frac: float = 0.1
    seed: int = 0
    n_steps: int = 512
    amp: bool = False
    pred_hop: int = 1
    patience: int = 5


@dataclass(frozen=True)
class TimeSplit:
    """Train, embargo, validation, and an untouched test tail.

    Rows ``[0, train_limit)`` are the only rows the optimizer, the feature
    scaler, and the probe fit may use. ``[val_start, test_start)`` is the
    validation tail of that training block. ``[test_start, n_rows)`` is never
    read while choosing an epoch.
    """

    train_limit: int
    val_start: int
    test_start: int
    n_rows: int
    embargo: int


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_vol_r2: float
    val_regime_auc: float


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    report: Gate1Report
    device: str
    best_epoch: int
    history: tuple[EpochLog, ...]
    history_path: Path
    last_checkpoint_path: Path
    split: TimeSplit


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


def make_time_split(
    n_rows: int,
    seq_len: int,
    pred_hop: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> TimeSplit:
    """Place validation at the end of the pre-test block, behind an embargo.

    The embargo is ``2 * seq_len`` bars. No training window, including its
    JEPA target and its Gate 1 label window, can touch the validation rows.
    ``train_frac == 1`` means the caller already removed the test slice.
    """

    if n_rows < 4:
        raise ValueError("series is too short")
    if not 0.0 < train_frac <= 1.0:
        raise ValueError("train_frac must lie in (0, 1]")
    if pred_hop < 1:
        raise ValueError("pred_hop must be at least 1")
    if train_frac == 1.0:
        test_start = n_rows
    else:
        test_start = int(n_rows * train_frac)
        test_start = min(max(test_start, 1), n_rows)
    embargo = 2 * seq_len
    develop = test_start
    min_block = 2 * seq_len + pred_hop + 2
    room = develop - embargo - min_block
    if room < min_block:
        raise ValueError(
            f"series length {n_rows} is too short for seq_len={seq_len}, "
            f"pred_hop={pred_hop}, and an embargo of {embargo} bars"
        )
    val_len = min(max(min_block, int(develop * val_frac)), room)
    val_start = develop - val_len
    train_limit = val_start - embargo
    split = TimeSplit(train_limit, val_start, test_start, n_rows, embargo)
    if len(jepa_starts(0, train_limit, seq_len, pred_hop)) < 4:
        raise ValueError("not enough training windows after the embargo")
    if len(jepa_starts(val_start, test_start, seq_len, pred_hop)) < 2:
        raise ValueError("not enough validation windows")
    if len(label_starts(0, train_limit, seq_len)) < 4:
        raise ValueError("not enough train probe windows")
    if len(label_starts(val_start, test_start, seq_len)) < 2:
        raise ValueError("not enough validation probe windows")
    return split


def jepa_starts(region_start: int, region_end: int, seq_len: int, pred_hop: int) -> np.ndarray:
    """Starts whose context and hop-shifted target both lie in ``[start, end)``."""

    last = region_end - (seq_len + pred_hop)
    if last < region_start:
        return np.array([], dtype=np.int64)
    return np.arange(region_start, last + 1, dtype=np.int64)


def label_starts(region_start: int, region_end: int, seq_len: int) -> np.ndarray:
    """Starts whose context and next full label window both lie in the region."""

    last = region_end - 2 * seq_len
    if last < region_start:
        return np.array([], dtype=np.int64)
    return np.arange(region_start, last + 1, dtype=np.int64)


def train_model(
    config: TrainConfig,
    scaled: np.ndarray,
    raw: np.ndarray,
    split: TimeSplit,
    device: torch.device,
    use_amp: bool,
) -> tuple[FinancialLeWM, list[EpochLog], int, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Train on the pre-embargo rows and keep the best epoch by validation R².

    Validation windows are the time-ordered tail of the training block.
    They are never shuffled. Rows at and after ``split.test_start`` are not
    indexed by this function.
    """

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
    )
    cast_dtype = amp_dtype(device) if use_amp else None
    scaler = torch.amp.GradScaler("cuda") if cast_dtype == torch.float16 else None
    train_starts = jepa_starts(0, split.train_limit, config.seq_len, config.pred_hop)
    val_starts = jepa_starts(split.val_start, split.test_start, config.seq_len, config.pred_hop)
    # Probe code reads only these slices. The test tail is not among them.
    visible = split.test_start
    best_r2 = float("nan")
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    since_best = 0
    history: list[EpochLog] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            scaled[:visible],
            train_starts,
            config,
            device,
            cast_dtype,
            scaler,
            optimizer,
            seed=config.seed + epoch,
            train=True,
        )
        val_loss = _run_epoch(
            model,
            scaled[:visible],
            val_starts,
            config,
            device,
            cast_dtype,
            scaler,
            optimizer,
            seed=config.seed + epoch,
            train=False,
        )
        val_report = evaluate_gate1(
            model,
            scaled[:visible],
            raw[:visible],
            split.train_limit,
            config.seq_len,
            device,
            eval_start=split.val_start,
            eval_end=split.test_start,
        )
        row = EpochLog(epoch, train_loss, val_loss, val_report.vol_r2, val_report.regime_auc)
        history.append(row)
        if _is_better(val_report.vol_r2, val_loss, best_r2, best_loss):
            best_r2 = val_report.vol_r2 if np.isfinite(val_report.vol_r2) else -np.inf
            best_loss = val_loss
            best_epoch = epoch
            since_best = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no validation epoch")
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, last_state, best_state


def run_training(
    config: TrainConfig,
    frame,
    out_path: str | Path,
    device: torch.device,
    use_amp: bool,
) -> TrainResult:
    """Train with a held-out tail, then score Gate 1 on that tail from ``best`` weights.

    The feature scaler is fit on the pre-embargo train rows only. Checkpoint
    choice uses validation loss and validation vol R². The test tail is read
    only after the best epoch has been loaded.
    """

    split = make_time_split(
        len(frame),
        config.seq_len,
        config.pred_hop,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
    )
    prepared = prepare_feature_matrix(frame, split.train_limit, config.vol_window)
    scaled = np.asarray(prepared["scaled"], dtype=np.float64)
    raw = np.asarray(prepared["raw"], dtype=np.float64)
    model, history, best_epoch, last_state, best_state = train_model(
        config, scaled, raw, split, device, use_amp
    )
    paths = _output_paths(Path(out_path))
    _write_history(paths["history"], history)
    # last.pt uses the final epoch. Its probe is fit on train rows and scored
    # on validation, so this write does not read the test tail.
    model.load_state_dict(last_state)
    last_report = evaluate_gate1(
        model,
        scaled[: split.test_start],
        raw[: split.test_start],
        split.train_limit,
        config.seq_len,
        device,
        eval_start=split.val_start,
        eval_end=split.test_start,
    )
    save_checkpoint(paths["last"], model, config, prepared, last_report)
    model.load_state_dict(best_state)
    model.eval()
    if split.test_start < split.n_rows:
        report = evaluate_gate1(
            model,
            scaled,
            raw,
            split.train_limit,
            config.seq_len,
            device,
            eval_start=split.test_start,
            eval_end=split.n_rows,
        )
    else:
        report = evaluate_gate1(
            model,
            scaled[: split.test_start],
            raw[: split.test_start],
            split.train_limit,
            config.seq_len,
            device,
            eval_start=split.val_start,
            eval_end=split.test_start,
        )
    save_checkpoint(paths["best"], model, config, prepared, report)
    if paths["best_alias"] != paths["best"]:
        save_checkpoint(paths["best_alias"], model, config, prepared, report)
    return TrainResult(
        checkpoint_path=paths["best"],
        report=report,
        device=str(device),
        best_epoch=best_epoch,
        history=tuple(history),
        history_path=paths["history"],
        last_checkpoint_path=paths["last"],
        split=split,
    )


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
            "pred_hop": config.pred_hop,
            "val_frac": config.val_frac,
        },
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "log_volume_mean": prepared["log_volume_mean"],
        "log_volume_std": prepared["log_volume_std"],
        "vol_head": _head_payload(report.vol_head),
        "regime_head": _head_payload(report.regime_head),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(blob, temporary)
    os.replace(temporary, path)


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
        pred_hop=int(blob["config"].get("pred_hop", 1)),
        val_frac=float(blob["config"].get("val_frac", 0.1)),
    )
    model = build_model(config).to(target)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def head_from_blob(payload: dict) -> LinearHead:
    return LinearHead(
        coef=np.asarray(payload["coef"], dtype=np.float64),
        intercept=float(payload["intercept"]),
        mean=np.asarray(payload["mean"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def _head_payload(head: LinearHead) -> dict[str, list[float] | float]:
    return {
        "coef": head.coef.tolist(),
        "intercept": head.intercept,
        "mean": head.mean.tolist(),
        "scale": head.scale.tolist(),
    }


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
    pred_hop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = np.stack([scaled[int(s) : int(s) + seq_len] for s in starts])
    nxt = np.stack([scaled[int(s) + pred_hop : int(s) + pred_hop + seq_len] for s in starts])
    return (
        torch.as_tensor(current, dtype=torch.float32, device=device),
        torch.as_tensor(nxt, dtype=torch.float32, device=device),
    )


def _run_epoch(
    model: FinancialLeWM,
    scaled: np.ndarray,
    starts: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    cast_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    optimizer: torch.optim.Optimizer,
    seed: int,
    train: bool,
) -> float:
    """Mean JEPA loss. Training shuffles; validation stays in time order."""

    if train:
        model.train()
        order = np.random.default_rng(seed).permutation(starts)
    else:
        model.eval()
        order = starts
    total = 0.0
    seen = 0
    for offset in range(0, len(order), config.batch_size):
        batch_starts = order[offset : offset + config.batch_size]
        if batch_starts.size < 2:
            continue
        x_t, x_next = _batch_windows(scaled, batch_starts, config.seq_len, config.pred_hop, device)
        context = (
            torch.autocast(device_type="cuda", dtype=cast_dtype)
            if cast_dtype is not None
            else nullcontext()
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
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
        else:
            with torch.no_grad(), context:
                loss, _parts = model.forward_loss(x_t, x_next)
        total += float(loss.detach()) * int(batch_starts.size)
        seen += int(batch_starts.size)
    if seen == 0:
        raise RuntimeError("no batches of size at least 2")
    return total / seen


def _is_better(vol_r2: float, val_loss: float, best_r2: float, best_loss: float) -> bool:
    """Higher validation vol R² wins. Equal R² keeps the lower validation loss.

    A non-finite R² only wins when the incumbent is also non-finite and the
    validation loss improved. The test slice is not an input to this comparison.
    """

    r2_ok = bool(np.isfinite(vol_r2))
    best_ok = bool(np.isfinite(best_r2))
    if r2_ok and not best_ok:
        return True
    if not r2_ok and not best_ok:
        return val_loss < best_loss
    if not r2_ok:
        return False
    if vol_r2 > best_r2 + 1e-12:
        return True
    if abs(vol_r2 - best_r2) <= 1e-12 and val_loss < best_loss:
        return True
    return False


def _output_paths(out_path: Path) -> dict[str, Path]:
    """``best.pt``, ``last.pt``, and a history CSV.

    A ``.pt`` path stays the best checkpoint so older callers can load ``--out``
    directly. The CSV and ``last.pt`` are written beside it.
    """

    if out_path.suffix == ".pt":
        directory = out_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        best = out_path
        last = directory / f"{out_path.stem}_last.pt"
        history = directory / f"{out_path.stem}_history.csv"
        alias = directory / "best.pt"
        return {"best": best, "last": last, "history": history, "best_alias": alias}
    out_path.mkdir(parents=True, exist_ok=True)
    best = out_path / "best.pt"
    return {"best": best, "last": out_path / "last.pt", "history": out_path / "history.csv", "best_alias": best}


def _write_history(path: Path, history: list[EpochLog]) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_vol_r2", "val_regime_auc"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "val_loss": row.val_loss,
                    "val_vol_r2": row.val_vol_r2,
                    "val_regime_auc": row.val_regime_auc,
                }
            )
