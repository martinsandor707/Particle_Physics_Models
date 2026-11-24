"""Probability-average ensemble of maps (plan B1, D8).

    log P(ln p) = logsumexp_m ( log w_m + log P_m(ln p) ),   w = softmax(log_weights)  (uniform by default)

Every member is a normalised map, so the mixture is normalised for any weights. ``n_params()`` is the sum over the
members (the plan reports a 5-member ensemble as 5 x 3,841 weights); trainable mixture weights, if requested, add
M - 1 free constants (one is fixed by the normalisation).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["Ensemble", "count_params"]


def count_params(model: nn.Module) -> int:
    """``model.n_params()`` if it has one, else the number of parameter elements."""
    return int(model.n_params()) if hasattr(model, "n_params") else sum(p.numel() for p in model.parameters())


class Ensemble(nn.Module):
    """Ensemble(members, log_weights=None, trainable_weights=False): mixture of member maps in probability space."""

    def __init__(self, members: Sequence[nn.Module], log_weights: Optional[Sequence[float]] = None,
                 trainable_weights: bool = False):
        super().__init__()
        members = list(members)
        if not members:
            raise ValueError("Ensemble: no members")
        self.members = nn.ModuleList(members)
        lw = torch.zeros(len(members)) if log_weights is None else torch.as_tensor(log_weights, dtype=torch.float32)
        if lw.shape != (len(members),) or not torch.isfinite(lw).all():
            raise ValueError("Ensemble: log_weights must be finite, one per member")
        try:
            dev = next(members[0].parameters()).device
        except StopIteration:
            dev = lw.device
        lw = lw.to(dev)
        self.trainable_weights = bool(trainable_weights)
        if self.trainable_weights:
            self.log_weights = nn.Parameter(lw)
        else:
            self.register_buffer("log_weights", lw)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, 0)

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        lw = torch.log_softmax(self.log_weights, 0)
        lps = torch.stack([m(lnp) for m in self.members], 0)                 # (M, B, N_Z, N_R)
        return torch.logsumexp(lps + lw.reshape(-1, 1, 1, 1), 0)

    def n_params(self) -> int:
        return sum(count_params(m) for m in self.members) + (len(self.members) - 1 if self.trainable_weights else 0)
