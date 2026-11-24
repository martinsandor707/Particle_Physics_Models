"""Detector-frame halo: a diffuse component that attenuates with depth in the *detector*, not along the shower axis.

Geometry (``shapelab.SLAB_GEOM_SRC``). The shower frame has z' along the direction of incidence (polar angle theta
to the detector axis) and its origin at the centroid of the first hit layer. A point at depth z', radius r and
azimuth phi about the shower axis (phi = 0 towards the tilt direction) lies at the detector depth

    z_t = z' cos(theta) - r cos(phi) sin(theta)

and inside the detector slab iff  -layer/2 <= z_t <= D  (front face half a layer in front of the origin, back face at
D = 59.5 layers = 1219.75 mm for a shower that starts in the first layer).

Model. Energy carried by slow / neutral secondaries (albedo into the first layers, neutron and photon transport) is
taken to fall off with detector depth and to be azimuthally symmetric about the shower axis in r:

    h(z', r, phi | theta) = 1{-layer/2 <= z_t <= D} exp(-z_t / L) g(r; R)

A model of rho(z', r | p) can depend on neither theta nor phi, so the component is the average over the incidence
angles of the training events and over the azimuth:

    H(z', r) = g(r; R) E_theta[ (1/pi) int_{phi_1}^{phi_2} exp(-(z' c - r s cos phi)/L) dphi ]
    phi_1 = arccos(clamp((z' c + layer/2)/(r s), -1, 1)),   phi_2 = arccos(clamp((z' c - D)/(r s), -1, 1))

(c = cos theta, s = sin theta; the bracket is an incomplete modified Bessel integral, I_0(r s / L) exp(-z' c / L) when
the whole ring is inside). For z' < 0 the bracket is non-zero only on the wedge r >= (|z'| c - layer/2)/s: the
"backward cone" of tilted events, here weighted towards small detector depth.

Cell masses. For fixed (r, theta, phi) the slab condition is an interval in z', and the z' integral of the
exponential over (cell) n (interval) is elementary:

    int_{z_i}^{z_{i+1}} 1{-layer/2 <= z_t <= D} exp(-z_t / L) dz' = (L / c) [exp(-t_1 / L) - exp(-t_2 / L)],
    t_1 = clamp(z_i c - r s cos phi, -layer/2, D),   t_2 = clamp(z_{i+1} c - r s cos phi, -layer/2, D)

so the indicator is integrated exactly and only continuous integrands are left for the quadratures: Gauss-Legendre
in phi on [0, pi] (n_phi nodes), in r (n_r nodes per ring) and in the probability of the angle sample (n_theta
nodes at its quantiles, ``theta_bins``).
t_1 and t_2 depend on the geometry only and are precomputed once; a forward pass is

    ln P_ij = logsumexp_nodes[ ln w + ln(L/c) - t_1/L + log1mexp(-(t_2 - t_1)/L) + ln(dE/dr)(r; R) ]

(differentiable in L and R) and is evaluated once per optimiser step, independent of the batch.

Assumptions (each is a modelling choice, none is exact)
  1. theta-blind average: the map is E_theta of the unnormalised h, i.e. every event contributes the same halo
     energy regardless of its angle and of how much of the halo its slab cuts away. The per-event structure (the
     wedge at z' < 0 exists only on one side in phi and grows with theta) is averaged out, as it must be for a
     theta-blind model; it is therefore a template, not a prediction for a given event.
  2. phi is uniform and independent of theta (true for an axially symmetric deposit; the rings of the grid already
     integrate over phi).
  3. An infinite slab with one D: the lateral edges of the detector are ignored, and so is a late shower start
     (D = z_b cos(theta) is within 0.6 mm of 1219.75 mm for 97.3 % of the training events and smaller for the rest,
     down to 1097 mm at the 0.5 % quantile). ``D=None`` drops the back face (use it with a front-only acceptance
     template; the metric's M_e already cuts at z_b).
  4. g depends on r in the shower frame although the attenuation is in the detector frame: the source is taken to be
     the shower axis.
  5. The component has its own acceptance built in. It must be added *after* the acceptance template A-bar of the
     other components (``ComponentModel(det_halo=...)`` does that); multiplying it by A-bar would count the slab
     twice.
The parameters L and R (and the weight) are momentum independent here.

Accuracy (tests/test_families_halo.py)
  * against a brute-force Monte Carlo (2e6 points): block masses agree within the Monte-Carlo error (rms pull 1.0-1.1);
  * default quadrature (n_theta = 24, n_r = 2, n_phi = 24; 4.7e6 nodes) against a finer one: KL 2e-6 ... 4e-6,
    max |d ln P| 2e-2 ... 6e-2 in single cells cut by the wedge boundary (cells holding 99.9 % of the mass); the
    depth integral is exact, the error is the phi / r quadrature of a kinked integrand and the theta nodes;
  * geometry: ``acceptance()`` (the L -> infinity limit) equals the closed-form azimuthal slab acceptance averaged over
    the same angles to 6e-3 (max), and with the training events' angles it reproduces the stored template
    ``abar.pt`` to 0.015 (max) / 0.002 (mean) although that template uses every event's own D.
Cost on the GPU: 54 MiB of buffers, 126 MiB peak, 5 ms for a forward + backward.

Use::

    sets, extras = data.load_sets(("train",))                      # training events only
    halo = DetHalo.from_samples(extras["train"]["theta"])          # 24 quantile nodes of the training angles
    model = ComponentModel(provider, logA, det_halo=halo)          # adds the raw parameter "logit_dh"
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cellint import Grid, gauss_legendre, log1mexp
from .lateral import make_lat
from .longitudinal import raw_of

__all__ = ["DetHalo", "LAYER_MM", "D_BACK_MM", "theta_bins"]

LAYER_MM = 20.5
D_BACK_MM = 59.5 * LAYER_MM                # 1219.75 mm


def theta_bins(theta, n: int = 24, rule: str = "gl"):
    """n nodes and probabilities representing the distribution of an angle sample.

    rule="gl"        Gauss-Legendre in the probability u: theta_q = quantile(u_q), weight w_q (E[f(theta)] =
                     int_0^1 f(Q(u)) du; exact for f(Q(u)) polynomial of degree 2n - 1 in u)
    rule="midpoint"  equal-probability bins represented by their medians
    """
    th = np.sort(np.asarray(theta, dtype=np.float64).reshape(-1))
    if rule == "gl":
        x, w = np.polynomial.legendre.leggauss(int(n))
        return np.quantile(th, 0.5 * (x + 1.0)), 0.5 * w
    return np.quantile(th, (np.arange(n) + 0.5) / n), np.full(n, 1.0 / n)


class DetHalo(nn.Module):
    """H(z', r) = E_{theta, phi}[1{-layer/2 <= z_t <= D} exp(-z_t / L)] g(r; R) as a normalised ln P map (128, 32).

    DetHalo(theta, weights=None, L=300.0, R=150.0, radial="exp", layer=20.5, D=1219.75, grid=None,
            n_r=2, n_phi=24, trainable=True)

    theta, weights  angles [rad] and probabilities of the incidence-angle distribution (``theta_bins(sample, n)``
                    for a sample of the TRAINING events; a histogram works the same way). Nothing is hard-coded.
    L, R            starting values [mm]; L = 10 mm + 100 mm softplus(raw_L), R = 5 mm + 100 mm softplus(raw_R)
    radial          radial family of ``lateral.py`` giving dE/dr = 2 pi r g(r; R) ("exp": exp(-r/R)/R); shape
                    parameters of the family, if any, are further scalars of this module
    D               back-face depth, None = no back face
    ``forward()`` returns the normalised ln P (N_Z, N_R); ``log_mass()`` the unnormalised ln masses;
    ``acceptance()`` the theta-averaged geometric acceptance of the cells (the L -> infinity limit without g).
    """

    L_FLOOR, R_FLOOR, UNIT = 10.0, 5.0, 100.0
    LOG_ZERO = -80.0            # ln weight given to (cell, node) pairs outside the slab (instead of -inf)

    def __init__(self, theta, weights=None, L: float = 300.0, R: float = 150.0, radial="exp",
                 layer: float = LAYER_MM, D: Optional[float] = D_BACK_MM, grid: Optional[Grid] = None,
                 n_r: int = 2, n_phi: int = 24, trainable: bool = True):
        super().__init__()
        grid = grid if grid is not None else Grid()
        self.N_Z, self.N_R, self.n_r, self.n_phi = grid.N_Z, grid.N_R, int(n_r), int(n_phi)
        self.layer, self.D, self.dz = float(layer), (None if D is None else float(D)), float(grid.dz)
        self.family = make_lat(radial)
        theta = torch.as_tensor(np.asarray(theta, dtype=np.float64)).reshape(-1)
        w = torch.full_like(theta, 1.0 / len(theta)) if weights is None else torch.as_tensor(np.asarray(weights, dtype=np.float64)).reshape(-1)
        w = w / w.sum()
        self.n_theta = len(theta)
        # nodes (float64)
        ze, re = grid.z_edges.double().cpu(), grid.r_edges.double().cpu()
        xr, wr = gauss_legendre(self.n_r)
        xp, wp = gauss_legendre(self.n_phi)
        dr = (re[1:] - re[:-1])[:, None]
        r = (re[:-1, None] + dr * xr[None, :]).reshape(-1)                   # (Nr n_r,)
        lwr = torch.log(dr * wr[None, :]).reshape(-1)
        c, sn = torch.cos(theta), torch.sin(theta)
        a = r[:, None, None] * sn[None, :, None] * torch.cos(math.pi * xp)[None, None, :]      # (R, T, P): r s cos(phi)
        hi = self.D if self.D is not None else float("inf")
        t1 = (ze[:-1, None, None, None] * c[None, None, :, None] - a[None]).clamp(-self.layer / 2, hi)
        t2 = (ze[1:, None, None, None] * c[None, None, :, None] - a[None]).clamp(-self.layer / 2, hi)
        dt_ = (t2 - t1).clamp_min(0.0)                                       # detector-depth extent inside the slab
        lw = (lwr[:, None, None] + torch.log(w / c)[None, :, None] + torch.log(wp)[None, None, :])[None]
        lw = lw + torch.where(dt_ > 0, 0.0, self.LOG_ZERO)                   # finite: no NaN gradient in empty cells
        n = (self.N_Z, len(r), -1)
        dt = grid.zc.dtype
        self.register_buffer("t1", t1.reshape(n).to(dt), persistent=False)                    # (N_Z, Nr n_r, T P)
        self.register_buffer("dt", dt_.reshape(n).clamp_min(1e-6).to(dt), persistent=False)
        self.register_buffer("lw", lw.reshape(n).to(dt), persistent=False)
        self.register_buffer("r", r.to(dt), persistent=False)
        self.register_buffer("lwr", lwr.to(dt), persistent=False)
        raw = {"raw_L": raw_of(L, self.L_FLOOR, self.UNIT), "raw_R": raw_of(R, self.R_FLOOR, self.UNIT)}
        raw.update({"shape_" + n_: float(v) for n_, v in self.family.SHAPES})
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.tensor(float(v), dtype=dt), requires_grad=trainable)
                                     for k, v in raw.items()})

    @classmethod
    def from_samples(cls, theta_samples, n_theta: int = 24, rule: str = "gl", **kw) -> "DetHalo":
        """Template for the angle distribution of a sample (use the training events' theta)."""
        th, w = theta_bins(theta_samples, n_theta, rule)
        return cls(th, w, **kw)

    @property
    def L(self):
        return self.L_FLOOR + self.UNIT * F.softplus(self.raw["raw_L"])

    @property
    def R(self):
        return self.R_FLOOR + self.UNIT * F.softplus(self.raw["raw_R"])

    def _rings(self, a):
        """(N_Z, Nr n_r) ln values at the radial nodes -> (N_Z, N_R): logsumexp over the nodes of each ring."""
        return torch.logsumexp(a.reshape(self.N_Z, self.N_R, self.n_r), -1)

    def acceptance(self):
        """Theta-averaged geometric acceptance of every cell (N_Z, N_R): the fraction of the ring cell (area
        weighted in r) inside the slab, E_theta[A]. Equals the event mean of ``shapelab.slab_accept``."""
        frac = torch.logsumexp(self.lw + torch.log(self.dt / self.dz), -1)   # sum_{theta, phi} w (t2 - t1) / (c DZ)
        lr = torch.log(self.r)
        norm = torch.logsumexp((self.lwr + lr).reshape(self.N_R, self.n_r), -1)
        return torch.exp(self._rings(frac + lr[None, :]) - norm[None, :])

    def depth_factor(self, L=None):
        """ln of the cell integral of E_{theta, phi}[1{in slab} exp(-z_t / L)] per unit r at the radial nodes:
        (N_Z, Nr n_r), without g and without the radial weights."""
        L = self.L if L is None else L
        return torch.logsumexp(self.lw + torch.log(L) - self.t1 / L + log1mexp(-self.dt / L), -1) - self.lwr[None, :]

    def log_mass(self):
        """Unnormalised ln cell masses (N_Z, N_R)."""
        sh = self.family.shapes({n: self.raw["shape_" + n] for n, _ in self.family.SHAPES})
        L = self.L
        a = torch.logsumexp(self.lw + torch.log(L) - self.t1 / L + log1mexp(-self.dt / L), -1)
        return self._rings(a + self.family.log_dEdr(self.r, self.R, sh)[None, :])

    def forward(self):
        m = self.log_mass()
        return torch.log_softmax(m.reshape(-1), 0).reshape(m.shape)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
