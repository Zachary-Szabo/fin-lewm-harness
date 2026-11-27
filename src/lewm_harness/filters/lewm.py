"""Causal LeWM filter backed by a trained JEPA checkpoint.

``update`` appends the current bar to a rolling window and never reads a
future row. ``fit`` does not train the network. It replays the training
slice and freezes the same surprise z-score the Kalman and HMM filters use.
``pred_vol`` and ``regime_prob`` come from the frozen Gate 1 linear heads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lewm_harness.engine.bars import make_bar
from lewm_harness.filters.base import RegimeFilter
from lewm_harness.types import Bar, FilterState
from lewm_harness.world_model.features import OnlineFeatureState
from lewm_harness.world_model.probes import LinearHead
from lewm_harness.world_model.train import head_from_blob, load_checkpoint


class LeWMFilter(RegimeFilter):
    """Online world-model filter loaded from ``checkpoint.pt``."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"LeWM checkpoint not found: {path}")
        self.checkpoint_path = path
        self.model, blob = load_checkpoint(path)
        self.model.eval()
        config = blob["config"]
        self.seq_len = int(config["seq_len"])
        self.vol_window = int(config["vol_window"])
        self.feature_mean = np.asarray(blob["feature_mean"], dtype=np.float64)
        self.feature_std = np.asarray(blob["feature_std"], dtype=np.float64)
        self.log_volume_mean = _optional_float(blob.get("log_volume_mean"))
        self.log_volume_std = _optional_float(blob.get("log_volume_std"))
        self.vol_head: LinearHead = head_from_blob(blob["vol_head"])
        self.regime_head: LinearHead = head_from_blob(blob["regime_head"])
        self._features = OnlineFeatureState(self.vol_window, self.log_volume_mean, self.log_volume_std)
        self._rows: list[np.ndarray] = []
        self._prev_online_z: torch.Tensor | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Z-score latent surprise on ``train_df`` only. Weights stay frozen."""

        if "ret" not in train_df.columns or "close" not in train_df.columns:
            raise ValueError("LeWMFilter.fit expected close and ret columns")
        self._surprise_mean = None
        self._surprise_std = None
        self.reference_vol_ = None
        self.reset()
        raw = np.empty(len(train_df), dtype=np.float64)
        vols = np.empty(len(train_df), dtype=np.float64)
        for index in range(len(train_df)):
            state = self.update(make_bar(train_df, index))
            raw[index] = state.surprise
            vols[index] = state.pred_vol
        self.calibrate_surprise(raw, vols)
        self.reset()

    def reset(self) -> None:
        """Drop the rolling window and the previous latent. Calibration stays."""

        self._features.reset()
        self._rows = []
        self._prev_online_z = None

    def update(self, obs: Bar) -> FilterState:
        extras = obs.extras or {}
        row = self._features.update(
            ret=float(obs.ret),
            close=float(obs.close),
            high=_maybe(extras, "high"),
            low=_maybe(extras, "low"),
            open_=_maybe(extras, "open"),
            volume=_maybe(extras, "volume"),
        )
        self._rows.append(row)
        if len(self._rows) > self.seq_len:
            self._rows = self._rows[-self.seq_len :]
        window = self._scaled_window()
        features = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z_online = self.model.encode(features)
            z_target = self.model.target_encoder(features)
            if self._prev_online_z is None:
                raw_surprise = 0.0
            else:
                predicted = self.model.predictor(self._prev_online_z)
                error = predicted - z_target
                raw_surprise = float(torch.linalg.vector_norm(error, ord=2).item())
            latent = z_online.squeeze(0).detach().cpu().numpy().astype(np.float64)
            self._prev_online_z = z_online.detach()
        pred_vol = max(self.vol_head.predict(latent), 1e-8)
        logit = float(np.clip(self.regime_head.predict(latent), -40.0, 40.0))
        regime_prob = float(1.0 / (1.0 + np.exp(-logit)))
        return FilterState(
            filtered_price=float(obs.close),
            regime_prob=regime_prob,
            surprise=self.standardize_surprise(raw_surprise),
            pred_vol=pred_vol,
        )

    def _scaled_window(self) -> np.ndarray:
        rows = list(self._rows)
        if len(rows) < self.seq_len:
            pad = np.array(self.feature_mean, dtype=np.float64, copy=True)
            rows = [pad for _ in range(self.seq_len - len(rows))] + rows
        raw = np.stack(rows).astype(np.float64)
        return (raw - self.feature_mean) / self.feature_std


def _maybe(extras: dict[str, float], key: str) -> float | None:
    if key not in extras:
        return None
    return float(extras[key])


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
