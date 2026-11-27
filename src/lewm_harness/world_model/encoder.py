"""Sequence encoders. Phase 4 can replace :class:`PatchTSTEncoder` with a TCN or MLP.

Every encoder maps ``(batch, length, channels)`` to a pooled latent
``(batch, out_dim)``. :class:`FinancialLeWM` only depends on that contract.
"""

from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """Pooled sequence encoder. Subclasses set ``out_dim``."""

    out_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PatchTSTEncoder(SequenceEncoder):
    """PatchTST-style encoder: patch the time axis, then a Transformer.

    ``x`` is ``(batch, seq_len, channels)``. Patches are non-overlapping
    when ``stride == patch_len``. The pooled latent is the mean of the
    patch tokens, shape ``(batch, d_model)``.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if patch_len < 1 or stride < 1:
            raise ValueError("patch_len and stride must be positive")
        if seq_len < patch_len:
            raise ValueError("seq_len must be at least patch_len")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.out_dim = d_model
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.input_proj = nn.Linear(in_channels * patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.register_buffer("positional", _sinusoidal(self.n_patches, d_model), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("PatchTSTEncoder expected (batch, length, channels)")
        if x.shape[1] != self.seq_len or x.shape[2] != self.in_channels:
            raise ValueError(
                f"expected x shape (batch, {self.seq_len}, {self.in_channels}), got {tuple(x.shape)}"
            )
        patches = x.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 2, 3, 1).contiguous()
        flat = patches.reshape(x.shape[0], self.n_patches, self.patch_len * self.in_channels)
        tokens = self.input_proj(flat) + self.positional.unsqueeze(0)
        encoded = self.transformer(tokens)
        return encoded.mean(dim=1)


def _sinusoidal(n_positions: int, d_model: int) -> torch.Tensor:
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(d_model, dtype=torch.float32)
    exponent = torch.div(index, 2, rounding_mode="floor") * 2.0 / d_model
    angle = position / torch.pow(10_000.0, exponent.unsqueeze(0))
    encoding = torch.zeros(n_positions, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angle[:, 0::2])
    encoding[:, 1::2] = torch.cos(angle[:, 1::2])
    return encoding
