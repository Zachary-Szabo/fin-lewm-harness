"""Gate 1 linear probes on a frozen world-model latent.

Both heads are fit on the training time split only and scored on a later
split. Each head is a ``StandardScaler`` plus a linear model. The scaler is
fit on that same training split and stored with the coefficients so
``LeWMFilter`` applies the identical map. The baseline is the same pipeline
on the raw feature row at the end of the window, not on ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lewm_harness.world_model.model import FinancialLeWM

_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class LinearHead:
    """Scaled linear map saved inside the checkpoint.

    ``predict`` applies the training ``StandardScaler`` and then
    ``coef @ z_scaled + intercept``. That is the same number the probe
    pipeline produces.
    """

    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray

    def predict(self, z: np.ndarray) -> float:
        values = np.asarray(z, dtype=np.float64).reshape(-1)
        scaled = (values - self.mean) / self.scale
        return float(self.coef @ scaled + self.intercept)


@dataclass(frozen=True)
class Gate1Report:
    vol_r2: float
    baseline_vol_r2: float
    regime_auc: float
    baseline_regime_auc: float
    n_train: int
    n_test: int
    vol_head: LinearHead
    regime_head: LinearHead


def evaluate_gate1(
    model: FinancialLeWM,
    scaled: np.ndarray,
    raw: np.ndarray,
    train_end: int,
    seq_len: int,
    device: torch.device,
) -> Gate1Report:
    """Probe next-window realized volatility and a high-vol indicator.

    A window starting at ``s`` uses rows ``[s, s + seq_len)``. Its label is
    the realized volatility of the following window, which is the prediction
    target, not an input. Training windows lie entirely inside the train
    split, including their label window.
    """

    model.eval()
    train_starts = np.arange(0, train_end - 2 * seq_len + 1)
    test_starts = np.arange(train_end, len(scaled) - 2 * seq_len + 1)
    if train_starts.size < 4 or test_starts.size < 2:
        raise ValueError("not enough rows for a train and test probe split")
    z_train, y_train, base_train = _design(model, scaled, raw, train_starts, seq_len, device)
    z_test, y_test, base_test = _design(model, scaled, raw, test_starts, seq_len, device)
    threshold = float(np.median(y_train))
    regime_train = (y_train > threshold).astype(np.int64)
    regime_test = (y_test > threshold).astype(np.int64)

    vol_model = _ridge_pipeline()
    vol_model.fit(z_train, y_train)
    baseline_vol = _ridge_pipeline()
    baseline_vol.fit(base_train, y_train)
    vol_r2 = _r2(y_test, vol_model.predict(z_test))
    baseline_vol_r2 = _r2(y_test, baseline_vol.predict(base_test))

    regime_auc, regime_head = _fit_auc(z_train, regime_train, z_test, regime_test)
    baseline_regime_auc, _baseline_regime_head = _fit_auc(
        base_train, regime_train, base_test, regime_test
    )
    vol_head = _head_from_pipeline(vol_model)
    return Gate1Report(
        vol_r2=vol_r2,
        baseline_vol_r2=baseline_vol_r2,
        regime_auc=regime_auc,
        baseline_regime_auc=baseline_regime_auc,
        n_train=int(train_starts.size),
        n_test=int(test_starts.size),
        vol_head=vol_head,
        regime_head=regime_head,
    )


def _design(
    model: FinancialLeWM,
    scaled: np.ndarray,
    raw: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latents: list[np.ndarray] = []
    targets: list[float] = []
    baseline: list[np.ndarray] = []
    batch = 256
    log_returns = raw[:, 0]
    with torch.no_grad():
        for offset in range(0, len(starts), batch):
            chunk = starts[offset : offset + batch]
            windows = np.stack([scaled[int(s) : int(s) + seq_len] for s in chunk]).astype(np.float32)
            z = model.encode(torch.as_tensor(windows, device=device)).detach().cpu().numpy()
            latents.append(z)
    for start in starts:
        nxt = log_returns[int(start) + seq_len : int(start) + 2 * seq_len]
        targets.append(float(np.std(nxt, ddof=0)) if nxt.size >= 2 else 0.0)
        baseline.append(raw[int(start) + seq_len - 1])
    return np.concatenate(latents, axis=0), np.asarray(targets, dtype=np.float64), np.stack(baseline)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if float(np.var(y_true)) == 0.0:
        return 0.0
    return float(r2_score(y_true, y_pred))


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=list(_RIDGE_ALPHAS))),
        ]
    )


def _logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=400)),
        ]
    )


def _head_from_pipeline(pipeline: Pipeline) -> LinearHead:
    scaler: StandardScaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    return LinearHead(
        coef=np.asarray(model.coef_, dtype=np.float64).reshape(-1),
        intercept=float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0]),
        mean=np.asarray(scaler.mean_, dtype=np.float64).reshape(-1),
        scale=np.asarray(scaler.scale_, dtype=np.float64).reshape(-1),
    )


def _fit_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, LinearHead]:
    width = x_train.shape[1]
    if len(np.unique(y_train)) < 2:
        zeros = np.zeros(width, dtype=np.float64)
        ones = np.ones(width, dtype=np.float64)
        return float("nan"), LinearHead(coef=zeros, intercept=0.0, mean=zeros, scale=ones)
    pipeline = _logistic_pipeline()
    pipeline.fit(x_train, y_train)
    head = _head_from_pipeline(pipeline)
    if len(np.unique(y_test)) < 2:
        return float("nan"), head
    scores = pipeline.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores)), head
