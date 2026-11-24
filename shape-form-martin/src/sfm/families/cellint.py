"""Exact cell masses on the (z', r) ring grid, and the numerically careful primitives they need.

Grid (``common.py``): z' in [-200, 1450] mm in 128 uniform cells (DZ = 12.890625 mm); ring edges
r_i = 700 (i/32)^1.7 mm, i = 0..32; cell volume V_ij = DZ pi (r_{j+1}^2 - r_j^2). A model returns the normalised
log-probability log P_ij of the 4096 ring cells; the likelihood looks up the piecewise-constant density P_ij / V_ij.

Cell mass of a term  L(z') T(r | z')  with radial survival function S(r | z') = 1 - F(r | z') (energy fraction outside
radius r, so dE/dr = -dS/dr = 2 pi r T):

    P_ij = int_{z_i}^{z_{i+1}} L(z) [S(r_j | z) - S(r_{j+1} | z)] dz
         ~ sum_q w_q L(z_iq) [S(r_j | z_iq) - S(r_{j+1} | z_iq)]                    (Gauss-Legendre in z, Q nodes)

    ln[S(r_j) - S(r_{j+1})] = ln S(r_j) + log1mexp(ln S(r_{j+1}) - ln S(r_j))      (no cancellation, no underflow)

The 2 pi r Jacobian is exact, so nothing has to be regularised at the axis. Mass inside the grid disc:
1 - S(700 mm | z) (``log_disc_mass``).

Two-level rule for sharp depth profiles (``Grid.integrate_z_fine``): the depth profile is evaluated on n_zf = 16
fine nodes per cell, the cross-section only at the Q coarse nodes; ln T_j is carried to the fine nodes by the
degree-(Q - 1) polynomial through its coarse values. Used for the track (4 mm entry edge, range edge, Bragg cusp)
and for every shower depth family except the smooth plateau.

Accuracy (measured in tests/test_families_cellint.py; relative errors of single cell masses, over the cells that
hold more than 1e-12 of the largest one, and the L1 error of the whole profile in brackets)
  * radial ring masses from a closed-form ln S: rounding only. float64 < 2e-11; float32 <= 3e-5 for widths of
    5-300 mm (a ring mass is a difference of two rounded ln S values: heavy power-law tails on the outer rings are
    the worst case), <= 2e-4 for any width from 0.5 mm to 1e6 mm and a Weibull exponent down to 0.3.
  * z quadrature, Gauss-Legendre with Q nodes per 12.89 mm cell (geometric convergence for analytic integrands):
        integrand                      Q = 5              Q = 9      16 fine nodes
        plateau, w1 = 60 mm            1e-15
        logistic edge, w = 4 mm        1e-7  (1e-9)
        logistic edge, w = 2 mm        7e-5  (2e-6)       3e-8
        gamma soft start, delta = 2 mm 2e-2  (2e-9)       1e-6       7e-13
        Gaussian, sigma = 12.9 mm      2e-5  (1e-10)
        Gaussian, sigma = 6.4 mm       1e-3  (4e-7)
        Gaussian, sigma = 4.8 mm       4e-3  (4e-6)                  4e-15
        Gaussian, sigma = 3.2 mm       9e-3  (2e-4)
        Gaussian, sigma = 2.0 mm       4e-1  (4e-3)                  1e-10
    Gaussians in z (spike, end blob) therefore use the closed form ``gauss_cell_logmass`` (difference of normal
    CDFs): 1e-6 relative in float32, 1e-11 in float64, for every sigma from 1e-3 mm to 1e4 mm.
    A kink (``clamp`` in the legacy track and far-halo profiles, the Bragg cusp) reduces the convergence to
    O(n^-2) in the one cell that contains it: this, not the smooth part, limits the exact mode (see
    tests/test_families_components.py for the map-level numbers).
  * generic radial quadrature (``Grid.ring_logmass_quad``, forms without a closed-form survival function):
    Gauss-Legendre in u = ln(r + 1 mm), which removes the 1/(r + eps) pole of the legacy halos. For
    [r/(r + eps)] exp(-r/lam)/lam with n nodes per ring:
        n = 3: 4e-4 (lam = 5 mm), 1e-5 (lam >= 20 mm);   n = 4 (default): 3e-6, 6e-8;   n = 6: 1e-11.
    Always finite and positive, whatever the accuracy (a sum of positive terms in log space).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = ["LN2", "log1mexp", "logdiffexp", "log_ndtr", "log_erfcx", "gauss_legendre", "ring_logmass",
           "log_disc_mass", "gauss_cell_logmass", "Grid"]

LN2 = math.log(2.0)
_SQRT2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_SQRT_PI = math.sqrt(math.pi)


# ----------------------------------------------------------------------------------------------------------------------
# log(1 - e^x), log(e^a - e^b)
# ----------------------------------------------------------------------------------------------------------------------
class _Log1mexp(torch.autograd.Function):
    """log(1 - exp(x)), x <= 0 (Maechler 2012: log(-expm1(x)) for x > -ln 2, log1p(-exp(x)) otherwise).

    A custom Function so that only ``x`` is kept for the backward pass (the two-branch ``where`` would keep five
    intermediates of the same size) and the derivative is the cancellation-free  d/dx = -1 / expm1(-x)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x > -LN2, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (-1.0 / torch.expm1(-x))


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x <= 0, accurate over the whole range.

    x -> 0-: ~ log(-x) (returns -inf at x = 0, derivative -inf); x -> -inf: ~ -exp(x) (derivative -> -0).
    x > 0 is outside the domain and gives NaN. Relative accuracy of the *result* ~1 ulp for all x < 0."""
    return _Log1mexp.apply(x)


def logdiffexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log(exp(a) - exp(b)) for a >= b:  a + log1mexp(b - a)."""
    return a + log1mexp(b - a)


# ----------------------------------------------------------------------------------------------------------------------
# log Phi(x) and log erfcx(x) with derivatives that stay accurate in the far tails
# ----------------------------------------------------------------------------------------------------------------------
class _LogNdtr(torch.autograd.Function):
    """log Phi(x). The value is ``torch.special.log_ndtr``; the derivative is the inverse Mills ratio written as
    sqrt(2/pi) / erfcx(-x / sqrt 2), which has full relative accuracy for all x. (torch's own derivative,
    exp(-(log Phi + x^2/2)) / sqrt(2 pi), cancels two large numbers: in float32 it is 4 % off at x = -1000 and off by
    a factor 177 at x = -2e4.)"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.log_ndtr(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (_SQRT_2_OVER_PI / torch.special.erfcx(-x / _SQRT2))


def log_ndtr(x: torch.Tensor) -> torch.Tensor:
    """log of the standard normal CDF with a tail-stable derivative (see ``_LogNdtr``)."""
    return _LogNdtr.apply(x)


class _LogErfcx(torch.autograd.Function):
    """log erfcx(y) for y >= 0 (erfcx(y) = exp(y^2) erfc(y) in (0, 1]).  d/dy = 2y - 2 / (sqrt(pi) erfcx(y)); for
    large y the two terms cancel (the result is ~ -1/y, the relative error of the difference ~ 2 y^2 ulp), so beyond
    y = 10 (float32) / 50 (float64) the derivative is taken from the asymptotic series
    1 / (sqrt(pi) y erfcx(y)) = 1 / (1 - a),  a = 1/(2y^2) - 3/(4y^4) + 15/(8y^6) - 105/(16y^8) + 945/(32y^10)
    - 10395/(64y^12)  (truncation 2e-9 at y = 10, 1e-17 at y = 50)."""

    @staticmethod
    def forward(ctx, y):
        ctx.save_for_backward(y)
        return torch.log(torch.special.erfcx(y))

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        direct = 2.0 * y - 2.0 / (_SQRT_PI * torch.special.erfcx(y))
        y0 = 50.0 if y.dtype == torch.float64 else 10.0
        yb = y.clamp_min(y0)
        u = 1.0 / (yb * yb)
        a = u * (0.5 - u * (0.75 - u * (1.875 - u * (6.5625 - u * (29.53125 - u * 162.421875)))))
        return g * torch.where(y > y0, -2.0 * yb * a / (1.0 - a), direct)


def log_erfcx(y: torch.Tensor) -> torch.Tensor:
    """log erfcx(y), intended for y >= 0 (overflows for y < -9.3 in float32, y < -26.6 in float64)."""
    return _LogErfcx.apply(y)


# ----------------------------------------------------------------------------------------------------------------------
# quadrature nodes
# ----------------------------------------------------------------------------------------------------------------------
def gauss_legendre(n: int):
    """Gauss-Legendre nodes in (0, 1) and weights summing to 1 (float64 tensors)."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return torch.tensor(0.5 * (x + 1.0), dtype=torch.float64), torch.tensor(0.5 * w, dtype=torch.float64)


# ----------------------------------------------------------------------------------------------------------------------
# radial ring masses from ln S
# ----------------------------------------------------------------------------------------------------------------------
def _tiny(dtype) -> float:
    """Smallest relative ring mass that is kept: 1e8 x the smallest normal number (1.2e-30 float32, 2.2e-300 float64)."""
    return float(torch.finfo(dtype).tiny) * 1e8


def ring_logmass(log_sf_edges: torch.Tensor, tiny: float = None) -> torch.Tensor:
    """ln[S(r_j) - S(r_{j+1})] from ln S at the ring edges (..., N_R + 1) -> (..., N_R).

    ``ln S`` must be non-increasing along the last axis. If two neighbouring values coincide in floating point, or
    differ by less than ``tiny`` (a term so wide or so narrow that it has no resolvable mass in that ring), the
    difference is clamped at -tiny: the ring gets the relative mass ``tiny`` (default 1e-30 in float32, 2e-300 in
    float64) and a zero gradient instead of -inf / NaN."""
    tiny = _tiny(log_sf_edges.dtype) if tiny is None else tiny
    a = log_sf_edges[..., :-1]
    d = (log_sf_edges[..., 1:] - a).clamp_max(-tiny)
    return a + log1mexp(d)


def log_disc_mass(log_sf_rmax: torch.Tensor, tiny: float = None) -> torch.Tensor:
    """ln[1 - S(R_max)]: the mass of a plane-normalised term inside the grid disc."""
    return log1mexp(log_sf_rmax.clamp_max(-(_tiny(log_sf_rmax.dtype) if tiny is None else tiny)))


# ----------------------------------------------------------------------------------------------------------------------
# exact z-cell masses of a Gaussian
# ----------------------------------------------------------------------------------------------------------------------
def gauss_cell_logmass(z_edges: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """ln[Phi((z_{i+1} - mu)/sigma) - Phi((z_i - mu)/sigma)] for every z cell.

    z_edges (N + 1,), mu and sigma (..., 1) -> (..., N). Evaluated in float64 internally (N + 1 numbers per batch
    element, so the cost is nil) on the tail where both CDF values are small, so the result keeps full relative
    accuracy for cells hundreds of sigma away and for sigma much larger or smaller than a cell."""
    dt = mu.dtype
    ze, m, s = z_edges.double(), mu.double(), sigma.double()
    a, b = (ze[:-1] - m) / s, (ze[1:] - m) / s                      # a < b
    flip = (a + b) > 0                                              # use the upper tail: Phi(b) - Phi(a) = Phi(-a) - Phi(-b)
    lo, hi = torch.where(flip, -b, a), torch.where(flip, -a, b)
    lhi = log_ndtr(hi)
    return (lhi + log1mexp((log_ndtr(lo) - lhi).clamp_max(-1e-300))).to(dt)


# ----------------------------------------------------------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------------------------------------------------------
class Grid(nn.Module):
    """The 128 x 32 ring grid of ``common.py`` with quadrature nodes.

    Grid(n_z=5, n_r=4, r0=1.0)   Q = n_z Gauss-Legendre nodes per z cell; n_r nodes per ring in u = ln(r + r0)
                                 for the generic radial fallback.
    Buffers (dtype float32 by default; ``.double()`` for reference computations):
        z_edges (129,)  r_edges (33,)  zc (128,)  rc (32,) = sqrt(mean of r^2 edges)   dz
        log_vol (128, 32) = ln(DZ pi (r_{j+1}^2 - r_j^2))      log_area (32,) = log_vol - ln DZ
        z_nodes (128 * Q,)  log_wz (Q,) = ln(w_q DZ)
        zf_nodes (128 * n_zf,)  log_wzf (n_zf,)  lag (n_zf, Q): fine nodes for sharp depth profiles
        r_nodes (32 * n_r,)  log_wr (32 * n_r,) = ln(w_q du_j (r_q + r0))        [int f(r) dr = sum w f(r_q)]
    All values are computed in float64 and rounded once; ``from_common`` instead copies the float32 grid tensors of
    ``common.py`` bit for bit (cell-centre parity with the legacy formula).
    """

    def __init__(self, n_z: int = 5, n_r: int = 4, r0: float = 1.0, z_min: float = -200.0, z_max: float = 1450.0,
                 N_Z: int = 128, r_max: float = 700.0, N_R: int = 32, power: float = 1.7, dtype=torch.float32,
                 n_zf: int = 16, _edges=None):
        super().__init__()
        self.N_Z, self.N_R, self.n_z, self.n_r, self.r0 = int(N_Z), int(N_R), int(n_z), int(n_r), float(r0)
        self.n_zf = int(n_zf)
        self.z_min, self.z_max, self.r_max = float(z_min), float(z_max), float(r_max)
        self.dz = (self.z_max - self.z_min) / self.N_Z
        if _edges is None:
            ze = torch.linspace(self.z_min, self.z_max, self.N_Z + 1, dtype=torch.float64)
            re = self.r_max * (torch.arange(self.N_R + 1, dtype=torch.float64) / self.N_R) ** power
            zc = 0.5 * (ze[1:] + ze[:-1])
            rc = torch.sqrt(0.5 * (re[1:] ** 2 + re[:-1] ** 2))
            log_area = torch.log(math.pi * (re[1:] ** 2 - re[:-1] ** 2))
            log_vol = (math.log(self.dz) + log_area)[None, :].expand(self.N_Z, self.N_R).clone()
        else:
            ze, re, zc, rc, log_vol = (t.detach().cpu() for t in _edges)
            log_area = (log_vol - math.log(self.dz))[0].clone()        # as ProtonShape.LOG_AREA (float32 arithmetic)
        xz, wz = gauss_legendre(self.n_z)
        z_nodes = (ze[:-1, None].double() + self.dz * xz[None, :]).reshape(-1)
        log_wz = torch.log(wz * self.dz)
        xf, wf = gauss_legendre(self.n_zf)
        zf_nodes = (ze[:-1, None].double() + self.dz * xf[None, :]).reshape(-1)
        log_wzf = torch.log(wf * self.dz)
        lag = torch.ones(self.n_zf, self.n_z, dtype=torch.float64)        # Lagrange basis of the coarse nodes at the fine nodes
        for q in range(self.n_z):
            for m in range(self.n_z):
                if m != q:
                    lag[:, q] = lag[:, q] * (xf - xz[m]) / (xz[q] - xz[m])
        xr, wr = gauss_legendre(self.n_r)
        ue = torch.log(re.double() + self.r0)
        du = (ue[1:] - ue[:-1])[:, None]
        u = ue[:-1, None] + du * xr[None, :]
        r_nodes = (torch.exp(u) - self.r0).clamp_min(0.0)
        log_wr = torch.log(wr[None, :] * du) + u                       # ln(w du (r + r0))
        for k, v in (("z_edges", ze), ("r_edges", re), ("zc", zc), ("rc", rc), ("log_vol", log_vol),
                     ("log_area", log_area), ("z_nodes", z_nodes), ("log_wz", log_wz),
                     ("zf_nodes", zf_nodes), ("log_wzf", log_wzf), ("lag", lag),
                     ("r_nodes", r_nodes.reshape(-1)), ("log_wr", log_wr.reshape(-1))):
            self.register_buffer(k, v.to(dtype).contiguous(), persistent=False)

    @classmethod
    def from_common(cls, C=None, n_z: int = 5, n_r: int = 4, r0: float = 1.0, n_zf: int = 16) -> "Grid":
        """Grid with the float32 edge / centre / volume tensors of ``common.py`` copied bit for bit."""
        if C is None:
            import common as C                                         # noqa: N811  (shape-form-martin/common.py)
        C = C if isinstance(C, dict) else vars(C)
        return cls(n_z, n_r, r0, C["Z_MIN"], C["Z_MAX"], C["N_Z"], C["R_MAX"], C["N_R"], dtype=torch.float32,
                   n_zf=n_zf, _edges=(C["Z_EDGES"], C["R_EDGES"], C["ZC"], C["RC"], C["LOG_VOL"]))

    @property
    def shape(self):
        return self.N_Z, self.N_R

    # ---- z ----
    def integrate_z(self, logf: torch.Tensor) -> torch.Tensor:
        """ln int_cell f dz from ln f at the z nodes: (B, N_Z * Q, ...) -> (B, N_Z, ...)."""
        s = logf.shape
        w = self.log_wz.reshape((1, 1, self.n_z) + (1,) * (len(s) - 2))
        return torch.logsumexp(logf.reshape((s[0], self.N_Z, self.n_z) + s[2:]) + w, 2)

    def integrate_zf(self, logf: torch.Tensor) -> torch.Tensor:
        """ln int_cell f dz from ln f at the fine z nodes: (B, N_Z * n_zf) -> (B, N_Z)."""
        return torch.logsumexp(logf.reshape(logf.shape[0], self.N_Z, self.n_zf) + self.log_wzf, 2)

    def integrate_z_fine(self, logf_fine: torch.Tensor, loglat: torch.Tensor) -> torch.Tensor:
        """Two-level rule for  int_cell f(z) T_j(z) dz  with a sharp depth profile f and a smooth cross-section T_j.

        logf_fine (B, N_Z * n_zf): ln f at the fine nodes; loglat (B, N_Z * n_z, N_R) or (B, 1, N_R): ln T_j at the
        coarse nodes. ln T_j is carried to the fine nodes by the degree-(n_z - 1) polynomial through its coarse
        values (in log space: positive, and exact whenever ln T_j is a polynomial of that degree in z), then

            ln P_ij = logsumexp_f[ln(w_f DZ) + ln f(z_if) + ln T_j(z_if)].

        The expensive ring masses are still computed at n_z nodes per cell only; the extra cost is one tensor of
        shape (B, N_Z, n_zf, N_R)."""
        B = logf_fine.shape[0]
        lf = logf_fine.reshape(B, self.N_Z, self.n_zf) + self.log_wzf
        if loglat.shape[1] == 1:
            return torch.logsumexp(lf, 2)[:, :, None] + loglat
        lt = torch.einsum("fq,bzqr->bzfr", self.lag, loglat.reshape(B, self.N_Z, self.n_z, -1))
        return torch.logsumexp(lt + lf[:, :, :, None], 2)

    def gauss_z(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Exact ln z-cell masses of N(mu, sigma^2): mu, sigma (B, 1) -> (B, N_Z)."""
        return gauss_cell_logmass(self.z_edges, mu, sigma)

    # ---- r ----
    def ring_logmass_quad(self, log_dEdr) -> torch.Tensor:
        """Generic ring masses ln int_{r_j}^{r_{j+1}} (dE/dr) dr by quadrature.

        ``log_dEdr(r)`` receives the radial nodes (N_R * n_r,) and returns ln(dE/dr) = ln(2 pi r T(r)) with shape
        (..., N_R * n_r); the result has shape (..., N_R). Always finite and positive (a sum of positive terms in
        log space), whatever the accuracy."""
        lf = log_dEdr(self.r_nodes) + self.log_wr
        return torch.logsumexp(lf.reshape(lf.shape[:-1] + (self.N_R, self.n_r)), -1)

    @staticmethod
    def normalise(logmass: torch.Tensor) -> torch.Tensor:
        """log_softmax over all cells: (B, N_Z, N_R) log masses -> normalised log P."""
        B = logmass.shape[0]
        return torch.log_softmax(logmass.reshape(B, -1), 1).reshape(logmass.shape)
