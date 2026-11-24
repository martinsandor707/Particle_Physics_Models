"""Field models: ``common.FieldModel`` variants, the analytic-acceptance wrapper ``FieldAcc`` and a data-free
function-space warm start.

FieldNet(hidden, n_layers, n_freq)
    ``common.FieldModel`` plus ``n_params()`` and ``logits()``; same state_dict keys, so ``mlp_std.pt`` loads into it.
mlp_std_start(device)
    FieldNet(32, 4) with the weights of the fine-tuned benchmark ``mlp_std.pt`` (key "state", read only), as a
    starting point for continued training (plan B0, B2, B4). Same tensors as ``sfm.legacy.load_mlp_std``.
FieldAcc(base_field, logA)
    log P = log_softmax(logits + ln A), A floored at 1e-6 (plan B4): the network models the density before the
    detector acceptance template A(z', r) cuts it. ``warm_start_from(target, lnp_grid, steps)`` fits the wrapped
    model's map to an existing model's map by KL (no data), so training starts from (nearly) the same function.
acceptance_template(kind)
    "stored" (``abar.pt``, front + back faces), "front" (front face only) or "front_back" (recomputed, identical to
    the stored one); the last two from ``results/m0_parity/`` (Gate 0).

Teacher maps below the training momentum range are refused by default: the benchmarks are finite but unphysical
there (Gate 0, recommendation 5), so ``kl_warm_start`` raises for grid points below p = 0.0895 GeV.
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402

__all__ = ["FieldNet", "field_model", "mlp_std_start", "FieldAcc", "kl_warm_start", "acceptance_template",
           "default_lnp_grid", "TRAIN_P_MIN", "TRAIN_P_MAX", "ACC_FLOOR"]

TRAIN_P_MIN = 0.08951619267463684        # GeV, lowest training momentum after cuts (cache meta, Gate 0 row 34)
TRAIN_P_MAX = 19.9999942779541           # GeV, highest training momentum
ACC_FLOOR = 1e-6


class FieldNet(C.FieldModel):
    """``common.FieldModel`` (MLP(z', r, ln p) -> logit per cell -> softmax) with ``logits`` and ``n_params``."""

    def __init__(self, hidden: int = 64, n_layers: int = 3, n_freq: int = 4):
        super().__init__(hidden=hidden, n_layers=n_layers, n_freq=n_freq)
        self.hidden, self.n_layers, self.n_freq = int(hidden), int(n_layers), int(n_freq)

    def logits(self, lnp: torch.Tensor) -> torch.Tensor:
        """Unnormalised log density per cell (B, N_Z, N_R): ``FieldModel.forward`` before the softmax."""
        x = torch.cat([self.static[None].expand(len(lnp), -1, -1, -1),
                       (lnp / 3)[:, None, None, None].expand(-1, C.N_Z, C.N_R, 1)], -1)
        return self.mlp(x).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def field_model(hidden: int = 64, n_layers: int = 3, n_freq: int = 4, device=None) -> FieldNet:
    """A freshly initialised FieldNet (seed the global torch RNG before calling for a reproducible init)."""
    return FieldNet(hidden, n_layers, n_freq).to(device or C.DEVICE)


def mlp_std_start(device=None) -> FieldNet:
    """FieldNet(32, 4) carrying the weights of the benchmark ``mlp_std.pt`` (read-only file), in train mode."""
    device = device or C.DEVICE
    net = FieldNet(hidden=32, n_layers=4).to(device)
    ck = torch.load(paths.PION_RESULTS_DIR / "mlp_std.pt", map_location=device, weights_only=False)
    net.load_state_dict(ck["state"])
    return net.train()


def acceptance_template(kind: str = "stored", device=None) -> torch.Tensor:
    """Acceptance template A(z', r) (N_Z, N_R): "stored" | "front" | "front_back" (see the module docstring)."""
    device = device or C.DEVICE
    files = {"stored": paths.PION_RESULTS_DIR / "abar.pt", "front": paths.M0_DIR / "abar_front.pt",
             "front_back": paths.M0_DIR / "abar_front_back.pt"}
    if kind not in files:
        raise ValueError(f"acceptance template must be one of {sorted(files)}")
    A = torch.load(files[kind], map_location=device, weights_only=False)
    if A.shape != (C.N_Z, C.N_R):
        raise ValueError(f"acceptance template {kind}: shape {tuple(A.shape)}, expected {(C.N_Z, C.N_R)}")
    return A.float()


def default_lnp_grid(n: int = 64, device=None) -> torch.Tensor:
    """n points uniform in ln p over the training momentum range [0.0895, 20] GeV."""
    return torch.linspace(math.log(TRAIN_P_MIN), math.log(TRAIN_P_MAX), int(n), device=device or C.DEVICE)


def _model_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def kl_warm_start(model: nn.Module, target: nn.Module, lnp_grid: Optional[torch.Tensor] = None, steps: int = 500,
                  lr: float = 1e-3, weights: Optional[torch.Tensor] = None, allow_below_train_range: bool = False,
                  log=None) -> dict:
    """Data-free function-space fit: minimise the mean over the grid of KL(P_target(.|ln p) || P_model(.|ln p)).

    KL_g = sum_cells P_t (ln P_t - ln P_m), weighted over grid points by ``weights`` (uniform by default). Adam with
    learning rate ``lr`` on the trainable parameters of ``model``; ``target`` is evaluated once without gradients.
    Returns {"kl_initial", "kl_final", "steps"} (weighted mean KL in nat)."""
    dev = next(model.parameters()).device
    lnp = (default_lnp_grid(device=dev) if lnp_grid is None
           else torch.as_tensor(lnp_grid, dtype=torch.float32, device=dev).reshape(-1))
    if not allow_below_train_range and float(lnp.min()) < math.log(TRAIN_P_MIN) - 1e-6:
        raise ValueError(f"kl_warm_start: grid reaches below the training momentum range (p < {TRAIN_P_MIN} GeV); "
                         "the benchmark maps are not valid teachers there (pass allow_below_train_range=True to override)")
    w = torch.ones(len(lnp), device=dev) if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=dev)
    if w.shape != lnp.shape or not (w >= 0).all() or float(w.sum()) <= 0:
        raise ValueError("kl_warm_start: weights must be non-negative, one per grid point")
    w = w / w.sum()
    was = target.training
    target.eval()
    with torch.no_grad():
        logPt = target(lnp).float()
        Pt = logPt.exp()
    target.train(was)

    def kl():
        logPm = model(lnp)
        return (w * (Pt * (logPt - logPm)).sum((1, 2))).sum()

    with torch.no_grad():
        k0 = float(kl())
    params = _model_params(model)
    if steps > 0 and params:
        model.train()
        opt = torch.optim.Adam(params, lr=lr)
        for step in range(int(steps)):
            loss = kl()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"kl_warm_start: non-finite KL at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if log and (step % max(1, steps // 5) == 0):
                log(f"  warm start step {step:5d}  KL {float(loss.detach()):.3e}")
    with torch.no_grad():
        k1 = float(kl())
    return {"kl_initial": k0, "kl_final": k1, "steps": int(steps)}


class FieldAcc(nn.Module):
    """log P = log_softmax(logits(ln p) + ln A): a field model times a fixed acceptance template A (floored at 1e-6).

    FieldAcc(base_field, logA=None, *, A=None, floor=1e-6)
      base_field  a FieldNet (its ``logits`` are used) or any model lnp -> log P (a log-softmax differs from the logits
                  by a per-event constant, which the outer softmax removes)
      logA        ln A as a tensor (N_Z, N_R), clamped below at ln(floor); or give ``A`` itself
    ``n_params()`` counts the base's parameters (ln A is a buffer).
    """

    def __init__(self, base_field: nn.Module, logA: Optional[torch.Tensor] = None, *, A: Optional[torch.Tensor] = None,
                 floor: float = ACC_FLOOR):
        super().__init__()
        if (logA is None) == (A is None):
            raise ValueError("FieldAcc: give exactly one of logA and A")
        if A is not None:
            logA = torch.log(torch.as_tensor(A, dtype=torch.float32).clamp_min(floor))
        else:
            logA = torch.as_tensor(logA, dtype=torch.float32).clamp_min(math.log(floor))
        if logA.shape != (C.N_Z, C.N_R):
            raise ValueError(f"FieldAcc: ln A must have shape {(C.N_Z, C.N_R)}, got {tuple(logA.shape)}")
        if not torch.isfinite(logA).all():
            raise ValueError("FieldAcc: ln A must be finite")
        self.base = base_field
        self.register_buffer("logA", logA.to(next(base_field.parameters()).device))

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        z = self.base.logits(lnp) if hasattr(self.base, "logits") else self.base(lnp)
        z = z + self.logA
        return F.log_softmax(z.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)

    def n_params(self) -> int:
        b = self.base
        return int(b.n_params()) if hasattr(b, "n_params") else sum(p.numel() for p in b.parameters())

    def warm_start_from(self, model: nn.Module, lnp_grid: Optional[torch.Tensor] = None, steps: int = 500,
                        lr: float = 1e-3, **kw) -> dict:
        """Fit this wrapper's map to ``model``'s map by KL (data-free), so that training starts from the same
        function. See ``kl_warm_start``."""
        return kl_warm_start(self, model, lnp_grid, steps, lr, **kw)
