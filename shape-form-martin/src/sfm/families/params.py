"""Parameter providers: the map  ln p -> raw (unconstrained) parameters  of a closed-form family.

Contract (shared with ``sfm.models.paramnet``; any ``nn.Module`` with these two members is accepted everywhere)::

    class ParamProvider(nn.Module):
        names: list[str]                                  # raw parameter names, fixed order, unique, no '.'
        def forward(self, lnp: Tensor) -> Tensor: ...     # (B,) -> (B, len(names)) raw values

A provider returns *raw* numbers only. Every constraint (softplus / exp / sigmoid / stick-breaking / ordering) is
applied by the family code that consumes them, never here; a provider may therefore be a polynomial, a network or a
table without any knowledge of the density it feeds.

Kinetic variable (identical float operations to ``particle_shape.kinetic`` / ``ParticleShape.params``)::

    T = p^2 / (sqrt(p^2 + m^2) + m),   p = exp(ln p),   x = (ln T - v0) / vs

``T`` is the kinetic energy written without cancellation. Nothing floors ``T`` or ``x``: at p = 0.0005 GeV the pion
has T = 8.96e-7 GeV and x = -4.10 (training range x in [-1, 1]); in float32 the expression stays finite down to
p ~ 1e-19 GeV (p*p underflows below that).

Providers here
    PolyProvider    polynomial in x with a per-parameter degree mask (``shapelab.PolyFormula``); ``from_legacy``
                    copies a fitted ``ParticleShape`` and returns exactly its raw parameters
    ConstProvider   momentum-independent scalars
    ConcatProvider  several providers side by side (e.g. a network for 22 names + scalars for 21)
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["M_PION", "V0_PION", "VS_PION", "ParamProvider", "PolyProvider", "ConstProvider", "ConcatProvider",
           "kinetic", "lnT", "x_of_lnp", "is_param_provider", "raw_dict"]

M_PION = 0.139570            # GeV (particle_shape.MASS["pion"], sfm.data.M_PION)
V0_PION = -0.325860          # centre of ln T over the training momentum range (checkpoint buffer v0 = -0.32585955)
VS_PION = 3.314613           # half-width of ln T over the training momentum range (checkpoint buffer vs = 3.31461334)


def kinetic(lnp: torch.Tensor, mass=M_PION) -> torch.Tensor:
    """T [GeV] = p^2 / (sqrt(p^2 + m^2) + m) = sqrt(p^2 + m^2) - m without cancellation; p = exp(lnp) in GeV."""
    p = torch.exp(lnp)
    return p * p / (torch.sqrt(p * p + mass * mass) + mass)


def lnT(lnp: torch.Tensor, mass=M_PION) -> torch.Tensor:
    """ln(T / GeV), the same float operations as ``ParticleShape.lnT`` (no floor)."""
    return torch.log(kinetic(lnp, mass))


def x_of_lnp(lnp: torch.Tensor, v0=V0_PION, vs=VS_PION, mass=M_PION) -> torch.Tensor:
    """x = (ln T - v0) / vs; x in [-1, 1] over the training range p in [0.0895, 20] GeV, x = -4.10 at 0.0005 GeV."""
    return (lnT(lnp, mass) - v0) / vs


def is_param_provider(obj) -> bool:
    """Structural check of the contract: an ``nn.Module`` with a list of unique string ``names``."""
    names = getattr(obj, "names", None)
    return (isinstance(obj, nn.Module) and isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names)
            and len(set(names)) == len(names))


def raw_dict(provider, lnp: torch.Tensor) -> dict:
    """{name: (B,) raw values} of a provider."""
    raw = provider(lnp)
    if raw.dim() != 2 or raw.shape[1] != len(provider.names):
        raise ValueError(f"provider returned shape {tuple(raw.shape)}, expected (B, {len(provider.names)})")
    return {n: raw[:, i] for i, n in enumerate(provider.names)}


class ParamProvider(nn.Module):
    """Interface: ``names`` (raw parameter names, fixed order) and ``forward(lnp) -> (B, len(names))``."""

    names: list

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:          # pragma: no cover - interface
        raise NotImplementedError

    def n_params(self) -> int:
        """Number of free constants (default: all trainable tensor elements)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PolyProvider(ParamProvider):
    """raw_i(p) = sum_k c_ik x^k, k <= deg_i, with x = (ln T - v0)/vs.

    PolyProvider(names, degrees, init=None, v0=V0_PION, vs=VS_PION, mass=M_PION)

    degrees   int (all names) or {name: degree}; names missing from the dict get ``default_degree``
    init      {name: c_i0} or {name: sequence of coefficients c_i0, c_i1, ...}

    ``coef`` has shape (n, D + 1) with D the largest degree; ``mask`` zeroes the coefficients above each parameter's
    own degree (they never receive a gradient through ``forward``). ``n_params()`` counts the unmasked coefficients:
    101 for the final pion formula.
    """

    def __init__(self, names: Sequence[str], degrees=0, init: Optional[Mapping] = None, v0: float = V0_PION,
                 vs: float = VS_PION, mass: float = M_PION, default_degree: int = 0):
        super().__init__()
        self.names = [str(n) for n in names]
        if len(set(self.names)) != len(self.names):
            raise ValueError("PolyProvider: names must be unique")
        if isinstance(degrees, int):
            deg = {n: int(degrees) for n in self.names}
        else:
            unknown = set(degrees) - set(self.names)
            if unknown:
                raise ValueError(f"PolyProvider: degrees for unknown names {sorted(unknown)}")
            deg = {n: int(degrees.get(n, default_degree)) for n in self.names}
        if min(deg.values()) < 0:
            raise ValueError("PolyProvider: degrees must be >= 0")
        self.deg = deg
        self.degree = max(deg.values())
        coef = torch.zeros(len(self.names), self.degree + 1)
        mask = torch.zeros_like(coef)
        init = dict(init or {})
        unknown = set(init) - set(self.names)
        if unknown:
            raise ValueError(f"PolyProvider: init for unknown names {sorted(unknown)}")
        for i, n in enumerate(self.names):
            mask[i, :deg[n] + 1] = 1.0
            if n in init:
                v = torch.as_tensor(init[n], dtype=coef.dtype).reshape(-1)
                if len(v) > deg[n] + 1:
                    raise ValueError(f"PolyProvider: {len(v)} coefficients for '{n}' of degree {deg[n]}")
                coef[i, :len(v)] = v
        self.coef = nn.Parameter(coef)
        self.register_buffer("mask", mask)
        self.register_buffer("v0", torch.tensor(float(v0)))
        self.register_buffer("vs", torch.tensor(float(vs)))
        self.register_buffer("mass", torch.tensor(float(mass)))

    # ---- evaluation (the float operations of ParticleShape.params, in the same order) ----
    def x(self, lnp: torch.Tensor) -> torch.Tensor:
        return (lnT(lnp, self.mass) - self.v0) / self.vs

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        x = self.x(lnp)
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ (self.coef * self.mask).T

    def n_params(self) -> int:
        return int(self.mask.sum().item())

    def index(self, name: str) -> int:
        return self.names.index(name)

    # ---- construction from, and comparison with, the legacy formula ----
    @classmethod
    def from_legacy(cls, core, names: Optional[Sequence[str]] = None, defaults: Optional[Mapping] = None,
                    degrees: Optional[Mapping] = None, default_degree: Optional[int] = None) -> "PolyProvider":
        """Provider with the coefficients, degree masks, v0, vs and mass of a fitted ``ParticleShape`` (``core``).

        names     parameter names of the new provider (default: ``core.names``, the same order). Names that the
                  legacy formula has keep its coefficients and degree; other names start at ``defaults[name]``
                  (a constant) with degree ``degrees[name]`` (default ``default_degree``, itself defaulting to the
                  legacy maximum degree).
        With ``names=None`` the provider returns the legacy raw parameters ``core.params(lnp)`` bit for bit on the
        same device and dtype.
        """
        lnames = list(core.names)
        lcoef = (core.coef * core.mask).detach()
        lmask = core.mask.detach()
        names = lnames if names is None else [str(n) for n in names]
        defaults, degrees = dict(defaults or {}), dict(degrees or {})
        dd = int(core.degree) if default_degree is None else int(default_degree)
        deg, init = {}, {}
        for n in names:
            if n in lnames:
                i = lnames.index(n)
                deg[n] = int(lmask[i].sum().item()) - 1
                init[n] = lcoef[i, :deg[n] + 1].cpu()
            else:
                if n not in defaults:
                    raise KeyError(f"from_legacy: '{n}' is not a legacy parameter and has no default")
                deg[n] = int(degrees.get(n, dd))
                init[n] = [float(defaults[n])]
        out = cls(names, deg, init, v0=float(core.v0), vs=float(core.vs), mass=float(core.mass))
        with torch.no_grad():                                   # exact buffers (float() of a float32 is exact)
            out.v0.copy_(core.v0.detach().cpu())
            out.vs.copy_(core.vs.detach().cpu())
            out.mass.copy_(core.mass.detach().cpu())
        return out.to(core.coef.device)

    @torch.no_grad()
    def set_rows_lstsq(self, names: Sequence[str], lnp: torch.Tensor, values: torch.Tensor, weight=None):
        """Least-squares polynomials (each row's own degree) through ``values`` (N, len(names)) given at ``lnp`` (N,).
        Function-space initialisation of selected rows; the other rows are untouched."""
        x = self.x(lnp).double()
        w = torch.ones_like(x) if weight is None else weight.double().sqrt()
        for j, n in enumerate(names):
            i, d = self.index(n), self.deg[n]
            V = torch.stack([x ** k for k in range(d + 1)], 1) * w[:, None]
            sol = torch.linalg.lstsq(V, (values[:, j].double() * w)[:, None]).solution[:, 0]
            self.coef[i].zero_()
            self.coef[i, :d + 1] = sol.to(self.coef.dtype)
        return self

    def row_mask(self, names: Sequence[str]) -> torch.Tensor:
        """(n, 1) mask selecting the rows of ``names`` (to restrict an optimiser step: ``coef.grad *= row_mask``)."""
        m = torch.zeros(len(self.names), 1, device=self.coef.device, dtype=self.coef.dtype)
        for n in names:
            m[self.index(n)] = 1.0
        return m


class ConstProvider(ParamProvider):
    """Momentum-independent scalars: forward(lnp) repeats ``value`` for every batch element."""

    def __init__(self, names: Sequence[str], init: Optional[Mapping[str, float]] = None, trainable: bool = True):
        super().__init__()
        self.names = [str(n) for n in names]
        init = dict(init or {})
        v = torch.tensor([float(init.get(n, 0.0)) for n in self.names])
        if trainable:
            self.value = nn.Parameter(v)
        else:
            self.register_buffer("value", v)

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        return self.value[None, :].expand(lnp.shape[0], -1)


class ConcatProvider(ParamProvider):
    """Several providers side by side; names are concatenated (they must stay unique)."""

    def __init__(self, providers: Sequence[nn.Module]):
        super().__init__()
        self.parts = nn.ModuleList(providers)
        self.names = [n for p in providers for n in p.names]
        if len(set(self.names)) != len(self.names):
            raise ValueError("ConcatProvider: duplicate parameter names")

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        return torch.cat([p(lnp) for p in self.parts], 1)

    def n_params(self) -> int:
        return sum(p.n_params() if hasattr(p, "n_params") else sum(q.numel() for q in p.parameters() if q.requires_grad)
                   for p in self.parts)


class FunctionProvider(ParamProvider):
    """Wrap a plain function ``fn(lnp) -> (B, n)`` (tests, frozen tables). No free constants."""

    def __init__(self, names: Sequence[str], fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.names, self.fn = [str(n) for n in names], fn

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        return self.fn(lnp)

    def n_params(self) -> int:
        return 0
