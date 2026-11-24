"""Residual table: a base map times a free, momentum-interpolated log-factor per grid cell (plan D7, M2B).

    log P(ln p) = log_softmax( log P_base(ln p) + delta(ln p) ),
    delta(ln p) = linear interpolation in ln p of free tables delta_k (N_Z x N_R) at knots ln p_k,
                  constant beyond the first / last knot,

with every delta_k initialised to zero, so at initialisation the model *is* the base map. Used as a non-parametric
ceiling probe: how much any 128 x 32 map near the base could still gain under the metric. By default the base is
frozen (its parameters get ``requires_grad=False``) and only the tables train.

``n_params()`` counts every constant of the model: the base's plus n_knots x N_Z x N_R. ``n_trainable()`` counts
only what trains.
"""
from __future__ import annotations

import sys
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402

from .ensemble import count_params  # noqa: E402

__all__ = ["ResidualTable"]


class ResidualTable(nn.Module):
    """ResidualTable(base_model, lnp_knots, freeze_base=True)."""

    def __init__(self, base_model: nn.Module, lnp_knots: Sequence[float], freeze_base: bool = True):
        super().__init__()
        knots = torch.as_tensor(lnp_knots, dtype=torch.float32).reshape(-1)
        if knots.numel() < 1 or not torch.isfinite(knots).all():
            raise ValueError("ResidualTable: need at least one finite knot")
        if knots.numel() > 1 and not bool((knots[1:] > knots[:-1]).all()):
            raise ValueError("ResidualTable: knots must be strictly increasing")
        self.base = base_model
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        try:
            dev = next(base_model.parameters()).device
        except StopIteration:
            dev = knots.device
        self.register_buffer("knots", knots.to(dev))
        self.delta = nn.Parameter(torch.zeros(knots.numel(), C.N_Z, C.N_R, device=dev))

    def factor(self, lnp: torch.Tensor) -> torch.Tensor:
        """delta(ln p), shape (B, N_Z, N_R)."""
        K = self.knots.numel()
        if K == 1:
            return self.delta[0].expand(len(lnp), -1, -1)
        x = lnp.clamp(self.knots[0], self.knots[-1])
        i = torch.searchsorted(self.knots, x.contiguous()).clamp(1, K - 1)
        t = ((x - self.knots[i - 1]) / (self.knots[i] - self.knots[i - 1]))[:, None, None]
        return self.delta[i - 1] * (1 - t) + self.delta[i] * t

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        lp = self.base(lnp) + self.factor(lnp)
        return F.log_softmax(lp.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)

    def n_params(self) -> int:
        return count_params(self.base) + self.delta.numel()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
