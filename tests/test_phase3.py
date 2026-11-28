"""Phase 3: validation-only checkpointing and a leak-free monthly walk-forward."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch

from lewm_harness.data.binance import assert_single_symbol
from lewm_harness.data.markov import generate_regime_series
from lewm_harness.engine.monthly_walk import month_labels, resolve_bar_clock, run_monthly_walk_forward
from lewm_harness.filters.lewm import LeWMFilter
from lewm_harness.world_model.train import (
    TrainConfig,
    jepa_starts,
    make_time_split,
    resolve_device,
    run_training,
)


def _tiny_config(**overrides) -> TrainConfig:
    values = dict(epochs=2, n_steps=220, seq_len=8, patience=5, seed=0, pred_hop=1, batch_size=32)
    values.update(overrides)
    return TrainConfig(**values)


def test_time_split_keeps_an_embargo_before_validation() -> None:
    split = make_time_split(220, seq_len=8, pred_hop=1, train_frac=0.7)
    assert split.embargo == 16
    assert split.val_start - split.train_limit == split.embargo
    assert split.test_start > split.val_start
    last_train = int(jepa_starts(0, split.train_limit, 8, 1)[-1])
    assert last_train + 8 + 1 <= split.train_limit
    assert split.val_start >= last_train + 8 + 1 + split.embargo - 8


def test_checkpoint_choice_ignores_the_test_slice(tmp_path: Path) -> None:
    frame = generate_regime_series(220, seed=4)
    config = _tiny_config(epochs=3)
    device = resolve_device("cpu")
    original = run_training(config, frame, tmp_path / "a" / "best.pt", device, False)
    test_start = original.split.test_start
    perturbed = frame.copy(deep=True)
    perturbed.loc[test_start:, "close"] = perturbed.loc[test_start:, "close"].to_numpy() * 4.0 + 7.0
    perturbed.loc[test_start:, "ret"] = np.linspace(-0.2, 0.4, len(perturbed) - test_start)
    other = run_training(config, perturbed, tmp_path / "b" / "best.pt", device, False)
    assert original.best_epoch == other.best_epoch
    assert [(row.epoch, row.val_loss, row.val_vol_r2) for row in original.history] == [
        (row.epoch, row.val_loss, row.val_vol_r2) for row in other.history
    ]
    first = torch.load(original.checkpoint_path, map_location="cpu", weights_only=False)["model"]
    second = torch.load(other.checkpoint_path, map_location="cpu", weights_only=False)["model"]
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key])


def test_walk_forward_train_end_does_not_change_lewm_fit(tmp_path: Path) -> None:
    frame = generate_regime_series(200, seed=5)
    frame["volume"] = np.linspace(5.0, 9.0, len(frame))
    config = _tiny_config(epochs=1, n_steps=200)
    device = resolve_device("cpu")
    original = run_training(config, frame, tmp_path / "a", device, False)
    cut = original.split.test_start
    perturbed = frame.copy(deep=True)
    perturbed.loc[cut:, "close"] = 50.0
    perturbed.loc[cut:, "ret"] = -0.25
    perturbed.loc[cut:, "volume"] = 0.01
    other = run_training(config, perturbed, tmp_path / "b", device, False)
    left = torch.load(original.checkpoint_path, map_location="cpu", weights_only=False)
    right = torch.load(other.checkpoint_path, map_location="cpu", weights_only=False)
    for key in left["model"]:
        assert torch.equal(left["model"][key], right["model"][key])
    np.testing.assert_array_equal(np.asarray(left["feature_mean"]), np.asarray(right["feature_mean"]))
    np.testing.assert_array_equal(np.asarray(left["feature_std"]), np.asarray(right["feature_std"]))
    assert left["log_volume_mean"] == right["log_volume_mean"]
    filt_left = LeWMFilter(original.checkpoint_path)
    filt_right = LeWMFilter(other.checkpoint_path)
    filt_left.fit(frame.iloc[:cut])
    filt_right.fit(perturbed.iloc[:cut])
    assert filt_left._surprise_mean == filt_right._surprise_mean
    assert filt_left._surprise_std == filt_right._surprise_std


def test_synthetic_walk_forward_script_finishes(tmp_path: Path) -> None:
    frame = generate_regime_series(160, seed=6)
    stamps: list[int] = []
    for month in range(4):
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.DateOffset(months=month)
        for step in range(40):
            stamps.append(int((start + pd.Timedelta(minutes=5 * step)).timestamp() * 1000))
    frame["ts"] = stamps
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] * 1.001
    frame["low"] = frame["close"] * 0.999
    frame["volume"] = 1.0
    data = tmp_path / "5m" / "sample.parquet"
    data.parent.mkdir(parents=True)
    frame.to_parquet(data, index=False)
    out = tmp_path / "walk"
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_walk_forward.py",
            "--data",
            str(data),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--seeds",
            "0",
            "--train-months",
            "2",
            "--test-months",
            "1",
            "--step-months",
            "1",
            "--seq-len",
            "8",
            "--patience",
            "2",
            "--out",
            str(out),
            "--interval",
            "5m",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    summary = (out / "summary.txt").read_text()
    assert "lewm" in summary
    assert "identity" in summary
    assert "lewm_minus_baseline" in summary
    assert "bootstrap CI skipped" in summary
    assert "periods_per_year=" in summary
    assert "bar_spacing_ms=" in summary
    windows = pd.read_csv(out / "windows.csv")
    assert set(windows["filter"]) == {"identity", "kalman", "hmm", "lewm"}
    assert set(windows["strategy"]) == {"mean_reversion", "momentum", "stat_arb"}
    assert "flat_fraction" in windows.columns


def _monthly_frame(seed: int) -> pd.DataFrame:
    frame = generate_regime_series(160, seed=seed)
    stamps: list[int] = []
    for month in range(4):
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.DateOffset(months=month)
        for step in range(40):
            stamps.append(int((start + pd.Timedelta(minutes=5 * step)).timestamp() * 1000))
    frame["ts"] = stamps
    frame["volume"] = np.linspace(5.0, 9.0, len(frame))
    return frame


def test_monthly_walk_forward_ignores_perturbed_test_months(tmp_path: Path) -> None:
    frame = _monthly_frame(7)
    config = _tiny_config(epochs=1, n_steps=160, patience=2)
    device = resolve_device("cpu")
    kwargs = dict(
        config=config,
        seeds=[0],
        train_months=2,
        test_months=1,
        step_months=1,
        device=device,
        use_amp=False,
        interval="5m",
    )
    run_monthly_walk_forward(frame, tmp_path / "a", **kwargs)
    labels = month_labels(frame)
    months = list(dict.fromkeys(labels.tolist()))
    perturbed = frame.copy(deep=True)
    mask = labels.isin([months[2]]).to_numpy()
    perturbed.loc[mask, "close"] = perturbed.loc[mask, "close"].to_numpy() * 3.0 + 11.0
    perturbed.loc[mask, "ret"] = np.linspace(-0.4, 0.5, int(mask.sum()))
    perturbed.loc[mask, "volume"] = 0.02
    run_monthly_walk_forward(perturbed, tmp_path / "b", **kwargs)

    left = torch.load(tmp_path / "a" / "checkpoints" / "w0_s0" / "best.pt", map_location="cpu", weights_only=False)
    right = torch.load(tmp_path / "b" / "checkpoints" / "w0_s0" / "best.pt", map_location="cpu", weights_only=False)
    for key in left["model"]:
        assert torch.equal(left["model"][key], right["model"][key])
    np.testing.assert_array_equal(np.asarray(left["feature_mean"]), np.asarray(right["feature_mean"]))
    np.testing.assert_array_equal(np.asarray(left["feature_std"]), np.asarray(right["feature_std"]))

    train_mask = labels.isin(months[:2]).to_numpy()
    train = frame.loc[train_mask].reset_index(drop=True)
    filt_left = LeWMFilter(tmp_path / "a" / "checkpoints" / "w0_s0" / "best.pt")
    filt_right = LeWMFilter(tmp_path / "b" / "checkpoints" / "w0_s0" / "best.pt")
    filt_left.fit(train)
    filt_right.fit(train)
    assert filt_left._surprise_mean == filt_right._surprise_mean
    assert filt_left._surprise_std == filt_right._surprise_std
    assert filt_left.reference_vol_ == filt_right.reference_vol_

    scored_a = pd.read_csv(tmp_path / "a" / "windows.csv")
    scored_b = pd.read_csv(tmp_path / "b" / "windows.csv")
    window_a = scored_a.loc[scored_a["window"] == 0, "total_return"].to_numpy()
    window_b = scored_b.loc[scored_b["window"] == 0, "total_return"].to_numpy()
    assert window_a.shape == window_b.shape
    assert not np.allclose(window_a, window_b)


def test_interval_mismatch_and_multi_symbol_root_raise(tmp_path: Path) -> None:
    frame = _monthly_frame(1)
    frame["ts"] = frame["ts"].iloc[0] + np.arange(len(frame)) * 12 * 60 * 60 * 1000
    with pytest.raises(ValueError, match="median bar spacing"):
        resolve_bar_clock(frame, "5m")
    (tmp_path / "BTCUSDT").mkdir()
    (tmp_path / "ETHUSDT").mkdir()
    with pytest.raises(ValueError, match="more than one symbol"):
        assert_single_symbol(tmp_path)
    assert_single_symbol(tmp_path / "BTCUSDT")
