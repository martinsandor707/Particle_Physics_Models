"""fam - model classes handed over by lane G2_reference (reference ladders). No side effects at import.

All classes are torch.nn.Module with forward(lnp) -> log P (B, 128, 32) and n_params() = number of free coefficients.
`C` is the common namespace returned by shapelab.setup(...) (the unchanged 00_common definitions).

GFlashRef(C, degree, lnp0, lnp_s)          the report's Formula("gamma", "core_tail") (baseline family), 9 + 6 n parameters
GFlashAcc(C, degree, lnp0, lnp_s, A=None)   the same formula multiplied by a fixed, parameter-free acceptance map A(z', r)
MLPRef(C, hidden, n_layers)                 the report's FieldModel (MLP upper bound)
MLPAcc(C, hidden, n_layers, A=None)         FieldModel multiplied by a fixed acceptance map (not used for the reference numbers)

Rebuild a checkpoint written by shapelab.evaluate:
    m = fam.GFlashAcc(C, degree=6, lnp0=c, lnp_s=s).to(C["DEVICE"]); m.load_state_dict(torch.load(path)["state"])
(the acceptance map is a buffer, so it comes back with the state dict; c, s = shapelab.lnp_center_scale(data)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FormulaWrap:
    """Helpers for the wrappers around the report's Formula (attribute `base`)."""

    def warm_start(self, prev):
        """Same function as `prev` (same family, lower degree): the new coefficients are zero (as in select_degree)."""
        with torch.no_grad():
            self.base.coef.zero_()
            self.base.coef[:, :prev.base.coef.shape[1]] = prev.base.coef
            self.base.slope.copy_(prev.base.slope)
        return self

    def init_from_hits(self, h):
        """Starting values of select_degree: Formula.init_from(z, r, w) of the training hits (h = data['train'].hits())."""
        self.base.init_from(h.z.values, h.r.values, h.w.values)
        return self


class GFlashRef(_FormulaWrap, nn.Module):
    """rho(z', r | p) = Gamma(z') x [core + tail Cauchy](r | z'): the report's GFlash baseline, every parameter a polynomial of
    degree n in L = (ln p - lnp0) / lnp_s, lateral parameters with a depth slope. Parameters: 6 (n + 1) + 3."""

    def __init__(self, C, degree=1, lnp0=0.0, lnp_s=1.0, long="gamma", lat="core_tail"):
        super().__init__()
        self.base = C["Formula"](long, lat, degree=degree, lnp0=lnp0, lnp_s=lnp_s)
        self.degree = degree

    def forward(self, lnp):
        return self.base(lnp)

    def n_params(self):
        return sum(p.numel() for p in self.base.parameters())

    def describe(self, title=""):
        return self.base.describe(title)


class _Acc(nn.Module):
    """base(z', r | p) x A(z', r), renormalised on the grid. A is fixed geometry (a buffer, no free parameter)."""

    def _set_acceptance(self, C, A, floor):
        A = torch.ones(C["N_Z"], C["N_R"]) if A is None else A.detach().float().cpu()
        self.register_buffer("logA", torch.log(A.clamp_min(floor)))

    def forward(self, lnp):
        lp = self.base(lnp) + self.logA
        return F.log_softmax(lp.reshape(len(lnp), -1), 1).reshape(lp.shape)

    def n_params(self):
        return sum(p.numel() for p in self.base.parameters())


class GFlashAcc(_FormulaWrap, _Acc):
    """The report's GFlash formula (body of revolution) cut by the event-averaged azimuthal acceptance of the tilted detector
    face(s): rho ∝ Gamma(z') x T(r | z') x Abar(z', r).  Abar = shapelab.mean_acceptance(C, data['train'], front, back):
    parameter-free, computed from the incidence angles (and back-face depths) of the training events only."""

    def __init__(self, C, degree=1, lnp0=0.0, lnp_s=1.0, A=None, floor=1e-6, long="gamma", lat="core_tail"):
        super().__init__()
        self.base = C["Formula"](long, lat, degree=degree, lnp0=lnp0, lnp_s=lnp_s)
        self.degree = degree
        self._set_acceptance(C, A, floor)

    def describe(self, title=""):
        return self.base.describe(title) + "\n\nmultiplied by the fixed acceptance map Abar(z', r) (buffer `logA`)."


class MLPRef(nn.Module):
    """The report's FieldModel: MLP(z', r, ln p) -> logit per grid cell -> softmax. All weights are free parameters."""

    def __init__(self, C, hidden=16, n_layers=3):
        super().__init__()
        self.base = C["FieldModel"](hidden, n_layers)

    def forward(self, lnp):
        return self.base(lnp)

    def n_params(self):
        return sum(p.numel() for p in self.base.parameters())


class MLPAcc(_Acc):
    """FieldModel x fixed acceptance map."""

    def __init__(self, C, hidden=16, n_layers=3, A=None, floor=1e-6):
        super().__init__()
        self.base = C["FieldModel"](hidden, n_layers)
        self._set_acceptance(C, A, floor)


class GFlashAccP(_FormulaWrap, nn.Module):
    """GFlash formula x momentum-dependent acceptance map: A has shape (n_bins, N_Z, N_R), one map per momentum bin
    (bin edges in ln p in the buffer `edges`, n_bins + 1 values). Used only to test whether a p-dependent Abar matters."""

    def __init__(self, C, degree=1, lnp0=0.0, lnp_s=1.0, A=None, edges=None, floor=1e-6, long="gamma", lat="core_tail", n_bins=20):
        super().__init__()
        self.base = C["Formula"](long, lat, degree=degree, lnp0=lnp0, lnp_s=lnp_s)
        self.degree = degree
        A = torch.ones(n_bins, C["N_Z"], C["N_R"]) if A is None else A.detach().float().cpu()
        edges = torch.linspace(0.0, 1.0, len(A) + 1) if edges is None else torch.as_tensor(edges, dtype=torch.float32).cpu()
        self.register_buffer("logA", torch.log(A.clamp_min(floor)))
        self.register_buffer("edges", edges)

    def forward(self, lnp):
        idx = torch.bucketize(lnp.contiguous(), self.edges[1:-1].contiguous())
        lp = self.base(lnp) + self.logA[idx]
        return F.log_softmax(lp.reshape(len(lnp), -1), 1).reshape(lp.shape)

    def n_params(self):
        return sum(p.numel() for p in self.base.parameters())


class GFlashSlabMarg(_FormulaWrap, nn.Module):
    """Exact (z', r | p) marginal of the slab-normalised body of revolution:
        P(z', r | p) = (1/K) sum_k  P0(z', r | p) A_k(z', r) / N_k(p),      N_k(p) = sum_cells P0(. | p) A_k(.),
    P0 = the report's GFlash formula, A_k = azimuthal acceptance of the detector slab for K representative training events
    (incidence angle theta_k and, if the back face is included, its depth D_k). Unlike `GFlashAcc` (plain event average of A)
    every event geometry is weighted by 1/N_k(p), i.e. the acceptance factor becomes momentum dependent through the shape itself.
    A: (K, N_Z, N_R) buffer, parameter-free geometry."""

    def __init__(self, C, degree=1, lnp0=0.0, lnp_s=1.0, A=None, floor=1e-9, long="gamma", lat="core_tail", K=64):
        super().__init__()
        self.base = C["Formula"](long, lat, degree=degree, lnp0=lnp0, lnp_s=lnp_s)
        self.degree = degree
        self.floor = floor
        self.register_buffer("A", torch.ones(K, C["N_Z"], C["N_R"]) if A is None else A.detach().float().cpu())

    def forward(self, lnp):
        lp = self.base(lnp)
        A = self.A.reshape(len(self.A), -1)                                   # (K, cells)
        N = (lp.exp().reshape(len(lnp), -1) @ A.T).clamp_min(1e-12)           # (B, K) slab mass of every event geometry
        mix = ((1.0 / N) @ A) / len(A)                                        # (B, cells)
        out = lp.reshape(len(lnp), -1) + torch.log(mix.clamp_min(self.floor))
        return F.log_softmax(out, 1).reshape(lp.shape)

    def n_params(self):
        return sum(p.numel() for p in self.base.parameters())
