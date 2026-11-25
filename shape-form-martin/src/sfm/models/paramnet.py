"""Neural parameter provider: the momentum-dependent raw parameters of a closed-form family from a small MLP.

Contract (shared with ``sfm/families/params.py``, written in parallel by the families worker)::

    class ParamProvider(nn.Module):
        names: list[str]                                  # raw (unconstrained) parameter names, fixed order
        def forward(self, lnp: Tensor) -> Tensor: ...     # (B,) -> (B, len(names)) raw values

``sfm/families/params.py`` did not exist when this module was written, so ``ParamProvider`` below is a local
definition of exactly that contract; any module with ``names`` and that ``forward`` is accepted everywhere here
(``ParamNet.fit_to`` only uses those two). When the families package lands, both definitions describe the same
interface; ``is_param_provider`` checks it structurally.

ParamNet (PLAN.md section 5, rung A1)
    input       x = (ln T - v0) / vs,  T = p^2 / (sqrt(p^2 + m^2) + m),  m = 0.139570 GeV (pion), p = exp(ln p).
                ``v0, vs`` come from the training range (``centring``); x is *not* clamped (as in the component
                formula, Gate 0 row 35), so extrapolation below the training minimum is the MLP's own.
    names_net   outputs of an MLP  x -> (width) -> ... -> len(names_net), ``depth`` hidden layers of ``width``
                units with ``activation`` (tanh by default). The last layer starts with zero weights and the
                bias at ``init`` (0 by default), so a fresh ParamNet is the constant function ``init``.
    names_const scalar ``nn.Parameter`` each (momentum independent).
    output      raw values in the order ``names`` (default names_net + names_const, or ``order``).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["M_PION", "ParamProvider", "ParamNet", "is_param_provider", "kinetic", "centring"]

M_PION = 0.139570                                        # GeV, as particle_shape.MASS["pion"] and sfm.data.M_PION

_ACT = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "softplus": nn.Softplus}


class ParamProvider(nn.Module):
    """Local definition of the parameter-provider contract (see the module docstring)."""

    names: list

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:          # pragma: no cover - interface
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def is_param_provider(obj) -> bool:
    """Structural check of the contract: an ``nn.Module`` with a list of unique string ``names``."""
    names = getattr(obj, "names", None)
    return (isinstance(obj, nn.Module) and isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names)
            and len(set(names)) == len(names))


def kinetic(lnp: torch.Tensor, m: float | torch.Tensor = M_PION) -> torch.Tensor:
    """T = p^2 / (sqrt(p^2 + m^2) + m) with p = exp(ln p): ``particle_shape.kinetic``, the same float operations."""
    p = torch.exp(lnp)
    return p * p / (torch.sqrt(p * p + m * m) + m)


def centring(lnp_train, mass: float = M_PION) -> dict:
    """Centring of the closed-form family, computed exactly as the component formula does.

    lnp0, lnp_s = middle and half-width of the training ln p range (``shapelab.lnp_center_scale``, numpy float32);
    v0, vs      = middle and half-width of ln T at lnp0 -+ lnp_s (``ParticleShape.__init__``, torch float32).
    On the cached pion training set this gives v0 = -0.32585955, vs = 3.31461334 (the checkpoint buffers).
    """
    lnp = lnp_train.detach().cpu().numpy() if torch.is_tensor(lnp_train) else np.asarray(lnp_train)
    lnp0, lnp_s = float((lnp.max() + lnp.min()) / 2), float((lnp.max() - lnp.min()) / 2)
    lo, hi = [float(torch.log(kinetic(torch.tensor(float(lnp0) + sg * float(lnp_s)), float(mass)))) for sg in (-1.0, 1.0)]
    return {"lnp0": lnp0, "lnp_s": lnp_s, "v0": float(torch.tensor((hi + lo) / 2)), "vs": float(torch.tensor((hi - lo) / 2))}


class ParamNet(ParamProvider):
    """MLP parameter provider: raw values of ``names_net`` from an MLP of x, scalar parameters for ``names_const``.

    ParamNet(names_net, names_const, width=16, depth=2, activation="tanh", v0=..., vs=..., mass=M_PION,
             init=None, order=None)

    init   {name: starting raw value} for any name (net outputs: the bias of the last layer; constants: their value)
    order  the order of ``names`` (a permutation of names_net + names_const); default names_net + names_const
    """

    def __init__(self, names_net: Sequence[str], names_const: Sequence[str] = (), width: int = 16, depth: int = 2,
                 activation: str = "tanh", v0: float = 0.0, vs: float = 1.0, mass: float = M_PION,
                 init: Optional[Mapping[str, float]] = None, order: Optional[Sequence[str]] = None):
        super().__init__()
        self.names_net, self.names_const = [str(n) for n in names_net], [str(n) for n in names_const]
        allnames = self.names_net + self.names_const
        if len(set(allnames)) != len(allnames):
            raise ValueError("ParamNet: names_net and names_const must be unique and disjoint")
        if not self.names_net and not self.names_const:
            raise ValueError("ParamNet: no parameter names")
        if any("." in n for n in self.names_const):
            raise ValueError("ParamNet: constant names must not contain '.'")
        self.names = list(order) if order is not None else list(allnames)
        if sorted(self.names) != sorted(allnames):
            raise ValueError("ParamNet: order must be a permutation of names_net + names_const")
        if activation not in _ACT:
            raise ValueError(f"ParamNet: activation must be one of {sorted(_ACT)}")
        if int(depth) < 0 or int(width) < 1:
            raise ValueError("ParamNet: depth >= 0 and width >= 1 required")
        if not float(vs) > 0:
            raise ValueError("ParamNet: vs must be positive")
        init = dict(init or {})
        unknown = set(init) - set(allnames)
        if unknown:
            raise ValueError(f"ParamNet: init for unknown names {sorted(unknown)}")
        self.width, self.depth, self.activation = int(width), int(depth), activation
        self.register_buffer("v0", torch.tensor(float(v0)))
        self.register_buffer("vs", torch.tensor(float(vs)))
        self.register_buffer("mass", torch.tensor(float(mass)))
        if self.names_net:
            layers, d = [], 1
            for _ in range(self.depth):
                layers += [nn.Linear(d, self.width), _ACT[activation]()]
                d = self.width
            last = nn.Linear(d, len(self.names_net))
            with torch.no_grad():
                last.weight.zero_()
                last.bias.copy_(torch.tensor([float(init.get(n, 0.0)) for n in self.names_net]))
            self.mlp = nn.Sequential(*layers, last)
        else:
            self.mlp = None
        self.const = nn.ParameterDict({n: nn.Parameter(torch.tensor(float(init.get(n, 0.0)))) for n in self.names_const})
        pos = {n: i for i, n in enumerate(allnames)}
        self.register_buffer("perm", torch.tensor([pos[n] for n in self.names], dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------------------------------------------------------
    def x(self, lnp: torch.Tensor) -> torch.Tensor:
        """x = (ln T - v0) / vs (unclamped)."""
        return (torch.log(kinetic(lnp, self.mass)) - self.v0) / self.vs

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        B = lnp.shape[0]
        cols = []
        if self.mlp is not None:
            cols.append(self.mlp(self.x(lnp)[:, None]))
        if self.names_const:
            cols.append(torch.stack([self.const[n] for n in self.names_const]).expand(B, -1))
        raw = torch.cat(cols, 1) if len(cols) > 1 else cols[0]
        return raw.index_select(1, self.perm)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (f"net={len(self.names_net)} outputs, const={len(self.names_const)}, width={self.width}, "
                f"depth={self.depth}, activation={self.activation}, v0={float(self.v0):.6g}, vs={float(self.vs):.6g}")

    # ------------------------------------------------------------------------------------------------------------------
    # Scale floor of the column normalisation (see ``fit_to``): a column whose weighted standard deviation over the
    # grid is at most SCALE_FLOOR_REL * max(1, |weighted mean|) is "constant in momentum" at float32 resolution
    # (2**-20 = 8 float32 ulps of a number of magnitude 1, 2**-23 being the float32 machine epsilon).
    SCALE_FLOOR_REL = 2.0 ** -20

    def fit_to(self, provider: nn.Module, lnp_grid: torch.Tensor, weights: Optional[torch.Tensor] = None,
               steps: int = 2000, lr: float = 1e-2, lbfgs_iter: int = 500, log=None) -> dict:
        """Function-space warm start: weighted least squares on another provider's trajectories.

        target[g, j] = provider(lnp_grid)[g, name_j] for every name of this ParamNet (matched by name; the provider
        may have more names). Weights ``w[g, j]``:
          * ``None`` or shape (G,)  - per grid point (e.g. the momentum density); every column is then divided by
            the weighted variance of its target, so parameters of different scale count equally (each column's
            term is its fraction of unexplained variance) - except for constant columns, below;
          * shape (G, len(names)) - per point and name, in the order of ``self.names``, used as given (e.g. a
            diagonal Fisher information, which makes the loss ~ 2 x the NLL increase).
        Constants get the closed-form weighted mean of their target; the MLP starts from the weighted mean
        (zero last-layer weights), is fitted with Adam on the full grid for ``steps`` steps (cosine decay from ``lr``)
        and then polished with up to ``lbfgs_iter`` L-BFGS iterations (kept only if they lower the loss). On random
        cubic trajectories a width-16, depth-2 tanh net reaches a weighted rms of ~1e-3 of each trajectory's spread
        this way (Adam alone: 1-3e-2); more iterations or float64 do not go further - that is the net's own limit.
        Seed the global torch RNG before building the ParamNet for a reproducible result.

        Constant columns (scale floor; only with normalised weights, i.e. ``None`` or shape (G,)). With p_g the
        point weights normalised to sum 1, the column's float64 weighted mean mu_j = sum_g p_g t_gj and standard
        deviation s_j = sqrt(sum_g p_g (t_gj - mu_j)^2). Column j is constant iff
            s_j <= SCALE_FLOOR_REL * max(1, |mu_j|)          (SCALE_FLOOR_REL = 2^-20).
        For such a column the variance normalisation 1/s_j^2 is meaningless (s_j is float rounding: 1/s_j^2 ~ 1e15
        was the earlier failure). It is treated as follows, which is exact for an exactly constant target t_gj = c:
          * its value is set to round_f32(mu_j) (= c bit for bit when the target is bitwise constant): as a scalar
            (names_const), or as the MLP output with last-layer row 0 and bias round_f32(mu_j) (names_net);
          * it gets weight 0 in the MLP objective and in the reported loss (scale = infinity, not the floor). For a
            net column the column's own term sum_g p_g (b_j + W_j h(x_g) - c)^2 is >= 0 and is 0 at (b_j, W_j) =
            (c, 0) for EVERY value of the hidden layers h, and (b_j, W_j) enter no other column's term; so for any
            weight lambda_j in [0, inf) the minimisers of the full objective are (c, 0) x the minimisers of the
            remaining columns' objective. Weight 0 gives the same solution without the conditioning problem; the
            row and the bias then get exactly zero gradient (Adam moves an entry with zero gradient by 0; L-BFGS
            directions are linear combinations of past gradients and steps, all 0 in those entries), and they are
            reset to (0, round_f32(mu_j)) at the end in any case.
        Columns above the floor are normalised exactly as before (same float32 operations), so when no constant
        column is present the result is bit-for-bit the earlier one.

        L-BFGS guard: every chunk of at most 100 iterations runs inside try/except (an overflowing line search
        raised "value cannot be converted to type float without overflow"); after each chunk the objective must be
        finite. On an exception or a non-finite value the best state seen so far (start of the polish included) is
        restored and the polish stops. The kept state is the last one whose loss is <= every earlier one (in the
        normal, monotone case: the final state, as before). A non-finite loss during Adam raises
        ``FloatingPointError`` (it would leave NaN weights).

        Returns {"rms": {name: weighted rms error}, "max_abs": {name: max |error|}, "loss": final loss,
        "constant_columns": [names], "near_floor_columns": {name: s_j / (floor * max(1,|mu_j|))} for non-constant
        columns with that ratio below 1e3, "scale_floor_rel": SCALE_FLOOR_REL, "lbfgs": {...}}.
        """
        if not is_param_provider(provider):
            raise TypeError("fit_to: provider must be an nn.Module with a list of unique names")
        missing = [n for n in self.names if n not in provider.names]
        if missing:
            raise KeyError(f"fit_to: provider has no trajectories for {missing}")
        dev = self.v0.device
        lnp_grid = torch.as_tensor(lnp_grid, dtype=torch.float32, device=dev).reshape(-1)
        G = lnp_grid.numel()
        if G < 1:
            raise ValueError("fit_to: empty grid")
        was_training = provider.training
        provider.eval()
        with torch.no_grad():
            full = provider(lnp_grid).to(dev, torch.float32)
        provider.train(was_training)
        idx = torch.tensor([provider.names.index(n) for n in self.names], device=dev)
        target = full.index_select(1, idx)                                    # (G, P) in self.names order
        if not torch.isfinite(target).all():
            raise ValueError("fit_to: non-finite target trajectories")
        P = len(self.names)
        if weights is None:
            w = torch.ones(G, P, device=dev)
            normalise = True
        else:
            w = torch.as_tensor(weights, dtype=torch.float32, device=dev)
            if w.shape == (G,):
                w, normalise = w[:, None].expand(G, P).clone(), True
            elif w.shape == (G, P):
                normalise = False
            else:
                raise ValueError(f"fit_to: weights must have shape ({G},) or ({G}, {P})")
        if not (torch.isfinite(w).all() and (w >= 0).all() and (w.sum(0) > 0).all()):
            raise ValueError("fit_to: weights must be finite, non-negative and not all zero per column")
        w = w / w.sum(0, keepdim=True)
        wp = w                                                                # point weights (sum 1 per column)
        mean = (w * target).sum(0)                                            # (P,)
        const_mask = torch.zeros(P, dtype=torch.bool, device=dev)
        near_floor = {}
        if normalise:
            var = (w * (target - mean) ** 2).sum(0)
            # scale floor (docstring): detection in float64, so that the summation rounding of the float32 mean
            # cannot make a bitwise-constant column look varying
            p64, t64 = wp.double(), target.double()
            p64 = p64 / p64.sum(0, keepdim=True)                             # sum 1 in float64 (float32 1/G is not)
            mu64 = (p64 * t64).sum(0)
            s64 = torch.sqrt((p64 * (t64 - mu64) ** 2).sum(0))
            floor64 = self.SCALE_FLOOR_REL * torch.clamp(mu64.abs(), min=1.0)
            const_mask = s64 <= floor64
            ratio = s64 / floor64
            near_floor = {n: float(ratio[j]) for j, n in enumerate(self.names)
                          if not bool(const_mask[j]) and float(ratio[j]) < 1e3}
            w = w / torch.where(var > 0, var, torch.ones_like(var))         # unchanged for non-constant columns
            if bool(const_mask.any()):
                w = torch.where(const_mask[None, :], torch.zeros_like(w), w)
                mean = torch.where(const_mask, mu64.to(torch.float32), mean)
        constant_columns = [n for j, n in enumerate(self.names) if bool(const_mask[j])]
        col = {n: j for j, n in enumerate(self.names)}
        net_idx = [col[n] for n in self.names_net]
        with torch.no_grad():
            for n in self.names_const:
                self.const[n].copy_(mean[col[n]])
            if self.mlp is not None:
                last = self.mlp[-1]
                last.weight.zero_()
                last.bias.copy_(mean[net_idx])
        net_cols = torch.tensor(net_idx, device=dev, dtype=torch.long)
        lbfgs_rep = None
        if self.mlp is not None and (steps > 0 or lbfgs_iter > 0):
            xg = self.x(lnp_grid)[:, None]
            tw, tt = w.index_select(1, net_cols), target.index_select(1, net_cols)

            def objective():
                return (tw * (self.mlp(xg) - tt) ** 2).sum()

            self.train()
            if steps > 0:
                opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps), eta_min=lr * 0.05)
                for step in range(int(steps)):
                    loss = objective()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"fit_to: non-finite weighted loss at Adam step {step} "
                                                 f"(constant columns {constant_columns})")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    if log and (step % max(1, steps // 5) == 0 or step == steps - 1):
                        log(f"  fit_to Adam step {step:5d}  weighted loss {float(loss.detach()):.3e}")
            if lbfgs_iter > 0:                     # full-batch least squares: polish with L-BFGS, keep it only if better
                saved = {k: v.detach().clone() for k, v in self.mlp.state_dict().items()}
                with torch.no_grad():
                    before = float(objective())
                best_state, best_loss = saved, before
                opt2 = torch.optim.LBFGS(self.mlp.parameters(), lr=1.0, max_iter=min(100, int(lbfgs_iter)),
                                         history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                         tolerance_change=1e-15)

                def closure():
                    opt2.zero_grad(set_to_none=True)
                    loss = objective()
                    loss.backward()
                    return loss

                stopped, chunk_losses = None, []
                for _ in range(math.ceil(int(lbfgs_iter) / 100)):
                    try:
                        opt2.step(closure)
                        with torch.no_grad():
                            cur = float(objective())
                    except (RuntimeError, OverflowError, ValueError, FloatingPointError) as err:
                        stopped = f"{type(err).__name__}: {err}"
                        break
                    chunk_losses.append(cur)
                    if not math.isfinite(cur):
                        stopped = "non-finite objective after an L-BFGS chunk"
                        break
                    if cur <= best_loss:
                        best_state = {k: v.detach().clone() for k, v in self.mlp.state_dict().items()}
                        best_loss = cur
                if not (math.isfinite(best_loss) and best_loss <= before):           # cannot happen; kept for clarity
                    best_state, best_loss = saved, before
                self.mlp.load_state_dict(best_state)
                after = best_loss
                lbfgs_rep = {"before": before, "after": after, "chunk_losses": chunk_losses, "stopped": stopped}
                if log:
                    log(f"  fit_to L-BFGS: weighted loss {before:.3e} -> {after:.3e}"
                        + (f" (stopped: {stopped})" if stopped else ""))
            if constant_columns:                   # exact for constant net columns (zero gradient: a no-op anyway)
                with torch.no_grad():
                    last = self.mlp[-1]
                    for n in constant_columns:
                        if n in self.names_net:
                            k = self.names_net.index(n)
                            last.weight[k].zero_()
                            last.bias[k] = mean[col[n]]
        with torch.no_grad():
            err = self(lnp_grid) - target
            if not torch.isfinite(err).all():
                raise FloatingPointError("fit_to: the fitted ParamNet gives non-finite values on the grid")
            wn = (w / w.sum(0, keepdim=True))
            if constant_columns:                   # zero-weight columns: report the error with the point weights
                wn = torch.where(const_mask[None, :], wp, wn)
            rms = torch.sqrt((wn * err ** 2).sum(0))
            mx = err.abs().max(0).values
            final = float((w * err ** 2).sum())
        return {"rms": {n: float(rms[j]) for j, n in enumerate(self.names)},
                "max_abs": {n: float(mx[j]) for j, n in enumerate(self.names)},
                "loss": final if math.isfinite(final) else float("nan"),
                "constant_columns": constant_columns, "near_floor_columns": near_floor,
                "scale_floor_rel": self.SCALE_FLOOR_REL, "lbfgs": lbfgs_rep}
