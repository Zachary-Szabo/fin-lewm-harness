"""JEPA latent prediction loss and SIGReg.

SIGReg keeps the online latent from collapsing to a constant. Random
one-dimensional projections are matched to a standard normal (mean 0,
variance 1), and the off-diagonal covariance of ``z`` is driven toward 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg(z: torch.Tensor, n_proj: int = 32) -> torch.Tensor:
    """Sliced isotropic-Gaussian penalty on a batch of latents ``(batch, dim)``."""

    if z.ndim != 2:
        raise ValueError("sigreg expected (batch, dim)")
    if z.shape[0] < 2:
        return z.new_zeros(())
    batch, dim = z.shape
    directions = torch.randn(dim, n_proj, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-6)
    projected = z @ directions
    mean = projected.mean(dim=0)
    variance = projected.var(dim=0, unbiased=False)
    sliced = mean.square().mean() + (variance - 1.0).square().mean()
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = (centered.transpose(0, 1) @ centered) / batch
    eye = torch.eye(dim, device=z.device, dtype=z.dtype)
    off_diagonal = covariance.square() * (1.0 - eye)
    return sliced + off_diagonal.sum() / dim


def jepa_sigreg_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    lambda_reg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``MSE(pred, stop_grad(target)) + lambda_reg * sigreg(z)``."""

    prediction = F.mse_loss(predicted, target.detach())
    regularizer = sigreg(z)
    total = prediction + float(lambda_reg) * regularizer
    return total, {
        "pred": float(prediction.detach()),
        "reg": float(regularizer.detach()),
        "total": float(total.detach()),
    }
