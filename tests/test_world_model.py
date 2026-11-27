"""World-model shapes, the synthetic training script, and causal features."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from lewm_harness.data.markov import generate_regime_series
from lewm_harness.filters.lewm import LeWMFilter
from lewm_harness.world_model.encoder import PatchTSTEncoder
from lewm_harness.world_model.features import causal_feature_frame, prepare_feature_matrix
from lewm_harness.world_model.model import FinancialLeWM
from lewm_harness.world_model.train import TrainConfig, resolve_device, run_training, synthetic_frame


def test_encoder_and_jepa_loss_backward_on_cpu() -> None:
    torch.manual_seed(0)
    encoder = PatchTSTEncoder(
        in_channels=5,
        seq_len=16,
        patch_len=4,
        stride=4,
        d_model=32,
        n_heads=4,
        n_layers=1,
    )
    model = FinancialLeWM(encoder, latent_dim=32, pred_hidden_dim=32)
    current = torch.randn(8, 16, 5)
    nxt = torch.randn(8, 16, 5)
    latent = model.encode(current)
    assert latent.shape == (8, 32)
    loss, parts = model.forward_loss(current, nxt)
    assert torch.isfinite(loss)
    assert parts["pred"] >= 0.0
    loss.backward()
    encoder_grads = [parameter.grad for parameter in model.encoder.parameters() if parameter.grad is not None]
    predictor_grads = [parameter.grad for parameter in model.predictor.parameters() if parameter.grad is not None]
    assert encoder_grads and predictor_grads
    assert all(torch.isfinite(grad).all() for grad in encoder_grads + predictor_grads)
    surprise = model.surprise(current, nxt)
    assert surprise.shape == (8,)
    assert torch.isfinite(surprise).all()


def test_features_do_not_depend_on_the_future() -> None:
    frame = generate_regime_series(80, seed=1)
    frame["open"] = frame["close"] * 0.999
    frame["high"] = frame["close"] * 1.01
    frame["low"] = frame["close"] * 0.99
    frame["volume"] = np.linspace(10.0, 20.0, len(frame))
    t_cut = 40
    prepared = prepare_feature_matrix(frame, t_cut, vol_window=8)
    perturbed = frame.copy(deep=True)
    perturbed.loc[t_cut:, "close"] = perturbed.loc[t_cut:, "close"] * 3.0
    perturbed.loc[t_cut:, "ret"] = 0.2
    perturbed.loc[t_cut:, "volume"] = 1.0
    perturbed_prepared = prepare_feature_matrix(perturbed, t_cut, vol_window=8)
    np.testing.assert_allclose(prepared["raw"][:t_cut], perturbed_prepared["raw"][:t_cut], atol=0.0, rtol=0.0)
    assert not np.allclose(prepared["raw"][t_cut:], perturbed_prepared["raw"][t_cut:])
    direct = causal_feature_frame(frame, 8, prepared["log_volume_mean"], prepared["log_volume_std"])
    np.testing.assert_allclose(direct, prepared["raw"], atol=0.0, rtol=0.0)


def test_synthetic_smoke_script_saves_a_checkpoint_and_loads(tmp_path: Path) -> None:
    out = tmp_path / "checkpoint.pt"
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_lewm.py",
            "--synthetic-smoke",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--n-steps",
            "400",
            "--out",
            str(out),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gate1" in completed.stdout
    assert "vol_r2=" in completed.stdout
    assert "regime_auc=" in completed.stdout
    assert out.is_file()
    frame = synthetic_frame(64, seed=2)
    filt = LeWMFilter(out)
    filt.fit(frame.iloc[:40])
    state = None
    from lewm_harness.engine.bars import make_bar

    for index in range(40):
        state = filt.update(make_bar(frame, index))
    assert state is not None
    assert np.isfinite(state.surprise)
    assert np.isfinite(state.pred_vol)
    assert 0.0 <= state.regime_prob <= 1.0


def test_smoke_training_report_is_finite(tmp_path: Path) -> None:
    config = TrainConfig(epochs=1, n_steps=256, seed=1)
    result = run_training(
        config,
        synthetic_frame(config.n_steps, config.seed),
        tmp_path / "checkpoint.pt",
        resolve_device("cpu"),
        False,
    )
    report = result.report
    assert np.isfinite(report.vol_r2)
    assert np.isfinite(report.baseline_vol_r2)
    assert report.n_train > 0 and report.n_test > 0
    if np.isfinite(report.regime_auc):
        assert 0.0 <= report.regime_auc <= 1.0
