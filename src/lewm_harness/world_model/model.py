"""JEPA world model: an online encoder, an EMA target encoder, and a latent predictor.

The predictor maps ``z_t`` to the next-window latent. The training target is
the stop-gradient (EMA) encoding of that next window. ``surprise`` is the
latent error ``||pred(z_t) - z_{t+1}||_2``.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from lewm_harness.world_model.encoder import SequenceEncoder
from lewm_harness.world_model.sigreg import jepa_sigreg_loss


class FinancialLeWM(nn.Module):
    """Latent next-window predictor on top of a pooled sequence encoder."""

    def __init__(
        self,
        encoder: SequenceEncoder,
        latent_dim: int,
        pred_hidden_dim: int,
        ema_decay: float = 0.99,
        lambda_reg: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder.out_dim != latent_dim:
            raise ValueError("encoder.out_dim must equal latent_dim")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, pred_hidden_dim),
            nn.GELU(),
            nn.Linear(pred_hidden_dim, latent_dim),
        )
        self.latent_dim = latent_dim
        self.ema_decay = float(ema_decay)
        self.lambda_reg = float(lambda_reg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Online latent ``z`` for windows ``x`` of shape ``(batch, length, channels)``."""

        return self.encoder(x)

    def forward_loss(self, x_t: torch.Tensor, x_next: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """JEPA loss for a batch of current windows and the following windows."""

        z_t = self.encoder(x_t)
        with torch.no_grad():
            z_next = self.target_encoder(x_next)
        predicted = self.predictor(z_t)
        return jepa_sigreg_loss(predicted, z_next, z_t, self.lambda_reg)

    def surprise(self, x_t: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """Latent prediction error ``||pred(z_t) - z_{t+1}||_2`` per batch row.

        ``z_t`` comes from the online encoder. ``z_{t+1}`` comes from the EMA
        target encoder, matching the training loss. No gradient is recorded.
        """

        with torch.no_grad():
            z_t = self.encoder(x_t)
            z_next = self.target_encoder(x_next)
            predicted = self.predictor(z_t)
            return torch.linalg.vector_norm(predicted - z_next, ord=2, dim=-1)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA step from the online encoder into the target encoder."""

        decay = self.ema_decay
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)
        for online, target in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target.data.copy_(online.data)
