"""sfm.families.halo_det: the theta-averaged detector-frame halo against brute-force Monte Carlo and an independent
closed form of the slab acceptance (CPU, synthetic angle sample: no data).

Measured accuracy of the default quadrature (n_theta = 24 Gauss-Legendre quantile nodes, n_r = 2, n_phi = 24;
4.7e6 precomputed nodes) against n_theta = 40, n_r = 3, n_phi = 40:
    L, R [mm]            max |d ln P| in the cells holding 99.9 % of the mass     KL
    300, 150             2.4e-2                                                   1.8e-6
     60,  80             6.1e-2                                                   3.7e-6
    1200, 400, D = None  1.8e-2                                                   1.5e-6
The depth integral is analytic; the error is in single cells cut by the wedge boundary (phi and r quadrature of a
kinked integrand). For a component with a weight of ~1 % this is <= 6e-4 in ln P of those cells.
GPU: 54 MiB of buffers, 126 MiB peak and 5 ms for a forward + backward.
"""
import math

import numpy as np
import pytest
import torch

from sfm.families import cellint as ci
from sfm.families import halo_det as hd

F64 = torch.float64
THETA = np.random.default_rng(0).uniform(0.0524, 0.6283, 70000)             # the pion sample spans 0.052-0.628 rad
HALF = hd.LAYER_MM / 2


def _mc(L, R, D, n=2_000_000, seed=1, theta=THETA):
    """Brute force: sample (theta, phi, z', r), weight 1{in slab} exp(-z_t / L); r drawn from dE/dr ~ exp(-r/R)."""
    g = ci.Grid(dtype=F64)
    rng = np.random.default_rng(seed)
    th = rng.choice(theta, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-200.0, 1450.0, n)
    r = -R * np.log1p(-rng.uniform(0, 1, n) * (1 - math.exp(-700.0 / R)))
    zt = z * np.cos(th) - r * np.cos(phi) * np.sin(th)
    inside = (zt >= -HALF) & ((zt <= D) if D is not None else True)
    w = inside * np.exp(-zt / L)
    iz = np.clip(((z + 200.0) / g.dz).astype(int), 0, 127)
    ir = np.searchsorted(g.r_edges.numpy()[1:-1], r, side="right")
    H, H2 = np.zeros((128, 32)), np.zeros((128, 32))
    np.add.at(H, (iz, ir), w)
    np.add.at(H2, (iz, ir), w * w)
    return H / H.sum(), np.sqrt(H2) / H.sum()


@pytest.mark.parametrize("L,R,D", [(300.0, 150.0, hd.D_BACK_MM), (60.0, 80.0, hd.D_BACK_MM), (1200.0, 400.0, None)])
def test_template_against_brute_force_monte_carlo(L, R, D):
    P, E = _mc(L, R, D)
    m = hd.DetHalo.from_samples(THETA, L=L, R=R, D=D, grid=ci.Grid(dtype=F64))
    with torch.no_grad():
        Q = m().exp().numpy()
    assert abs(Q.sum() - 1) < 1e-12
    blk = lambda a: a.reshape(16, 8, 8, 4).sum((1, 3))                         # noqa: E731  blocks of 8 x 4 cells
    Pb, Qb, Eb = blk(P), blk(Q), np.sqrt(blk(E ** 2))
    msk = Pb > 1e-4
    pull = (Qb - Pb)[msk] / Eb[msk]
    assert msk.sum() > 40
    assert np.sqrt((pull ** 2).mean()) < 1.4 and np.abs(pull).max() < 5.0     # measured rms 1.0-1.1
    assert np.abs(Qb[msk] / Pb[msk] - 1)[Pb[msk] > 1e-2].max() < 0.04         # well-populated blocks (MC error ~1 %)
    # the backward wedge: the Monte Carlo and the template agree on the energy in front of the origin
    assert abs(Q[:15].sum() - P[:15].sum()) < 5 * np.sqrt((E[:15] ** 2).sum()) + 2e-4
    assert Q[:15].sum() > 1e-3
    # a cell in front of the detector for every angle is empty: z' < -(layer/2)/cos(theta), r -> 0
    assert Q[:13, 0].sum() < 1e-25 and P[:13, 0].sum() == 0.0


def _slab_acceptance_reference(theta, D, nsub=24):
    """E_theta of the azimuthal slab acceptance A = [arccos(lo) - arccos(hi)] / pi (shapelab.SLAB_GEOM_SRC),
    averaged over nsub x nsub points per cell (uniform in z' and in r^2)."""
    g = ci.Grid(dtype=F64)
    k = (np.arange(nsub) + 0.5) / nsub
    ze, re = g.z_edges.numpy(), g.r_edges.numpy()
    zs = ze[:-1, None] + g.dz * k
    rs = np.sqrt(re[:-1, None] ** 2 + (re[1:, None] ** 2 - re[:-1, None] ** 2) * k)
    out = np.zeros((128, 32))
    for t, w in zip(*theta):
        zc = zs[:, :, None, None] * math.cos(t)
        den = np.maximum(rs[None, None, :, :] * math.sin(t), 1e-9)
        lo = np.clip((zc - D) / den, -1, 1) if D is not None else -np.ones_like(zc / den)
        hi = np.clip((zc + HALF) / den, -1, 1)
        out += w * ((np.arccos(lo) - np.arccos(hi)) / np.pi).mean((1, 3))
    return out


@pytest.mark.parametrize("D", [hd.D_BACK_MM, None])
def test_geometry_reduces_to_the_slab_acceptance(D):
    """L -> infinity without g: the template's geometric factor is the event-mean azimuthal acceptance, the quantity
    the acceptance template A-bar is made of (independent closed form in phi, sampled in z' and r)."""
    th = hd.theta_bins(THETA, 12)
    m = hd.DetHalo(*th, D=D, grid=ci.Grid(dtype=F64), n_r=6, n_phi=96)
    acc = m.acceptance().numpy()
    ref = _slab_acceptance_reference(th, D)
    assert acc.min() >= 0 and acc.max() <= 1 + 1e-9
    assert np.abs(acc - ref).max() < 6e-3 and np.abs(acc - ref).mean() < 3e-4
    assert np.abs(acc[40:80, :10] - 1).max() < 1e-9                            # deep inside, r < 100 mm: fully accepted
    assert acc[40, -1] < 0.99                                                 # r ~ 680 mm at z' = 320 mm reaches the front face
    assert acc[:8, :8].max() < 1e-30                                          # far in front, near the axis: empty (floor e^-80)
    if D is not None:
        assert acc[-1, 0] < 0.5 < acc[80, 0]                                  # the back face cuts the last rows
    # depth factor at very large L tends to the same geometry (times the cell length)
    with torch.no_grad():
        df = m.depth_factor(torch.tensor(1e9, dtype=F64)).exp()               # per unit r at the radial nodes
    cell = df.reshape(128, 32, 6).mean(-1).numpy() / m.dz
    inner = (ref > 0.02) & (ref < 0.98)
    assert np.abs(cell - ref)[~inner].max() < 2e-2 and np.abs(cell - ref).mean() < 2e-3


def test_default_quadrature_against_a_finer_one():
    g = ci.Grid()
    for (L, R, D), tol_max, tol_kl in (((300.0, 150.0, hd.D_BACK_MM), 5e-2, 1e-5), ((60.0, 80.0, hd.D_BACK_MM), 1.2e-1, 2e-5)):
        with torch.no_grad():
            ref = hd.DetHalo.from_samples(THETA, 40, L=L, R=R, D=D, grid=g, n_r=3, n_phi=40)().double()
            out = hd.DetHalo.from_samples(THETA, L=L, R=R, D=D, grid=g)().double()
            coarse = hd.DetHalo.from_samples(THETA, 12, L=L, R=R, D=D, grid=g, n_phi=12)().double()
        srt = torch.sort(ref.exp().reshape(-1), descending=True)
        keep = srt.indices[: int((srt.values.cumsum(0) < 0.999).sum()) + 1]
        d = (out - ref).abs().reshape(-1)[keep]
        kl = float((ref.exp() * (ref - out)).sum())
        assert float(d.max()) < tol_max and 0 <= kl < tol_kl, (L, float(d.max()), kl)
        assert float((ref.exp() * (ref - coarse)).sum()) > 3 * kl             # it does converge with the node counts


def test_gradients_in_L_and_R_and_extremes():
    g = ci.Grid(dtype=F64)
    th = hd.theta_bins(THETA, 6)
    m = hd.DetHalo(*th, L=250.0, R=120.0, grid=g, n_r=1, n_phi=6)
    w = torch.softmax(torch.randn(4096, dtype=F64, generator=torch.Generator().manual_seed(0)), 0).reshape(128, 32)

    def f(rl, rr):
        m.raw["raw_L"].data.copy_(rl.detach())
        m.raw["raw_R"].data.copy_(rr.detach())
        return (m() * w).sum()

    rl0, rr0 = m.raw["raw_L"].detach().clone(), m.raw["raw_R"].detach().clone()
    loss = (m() * w).sum()
    gl, gr = (x.detach() for x in torch.autograd.grad(loss, [m.raw["raw_L"], m.raw["raw_R"]]))
    h = 1e-5
    with torch.no_grad():
        nl = (f(rl0 + h, rr0) - f(rl0 - h, rr0)) / (2 * h)
        nr = (f(rl0, rr0 + h) - f(rl0, rr0 - h)) / (2 * h)
        f(rl0, rr0)
    assert abs(float(gl) - float(nl)) < 1e-6 * max(1.0, abs(float(nl))) and abs(float(gr) - float(nr)) < 1e-6 * max(1.0, abs(float(nr)))
    assert abs(float(m.L.detach()) - 250.0) < 1e-9 and abs(float(m.R.detach()) - 120.0) < 1e-9 and m.n_params() == 2
    # float32, raw parameters far out: L from 10 mm (the floor) to 1e5 mm stays finite with finite gradients
    m32 = hd.DetHalo(*th, grid=ci.Grid(), n_r=1, n_phi=6)
    for rl, rr in ((-50.0, -50.0), (-50.0, 1e3), (1e3, -50.0), (1e3, 1e3)):
        with torch.no_grad():
            m32.raw["raw_L"].fill_(rl)
            m32.raw["raw_R"].fill_(rr)
        out = m32()
        gs = torch.autograd.grad(out[60, 5] + out[3, 20], list(m32.parameters()))
        assert torch.isfinite(out).all() and all(torch.isfinite(x).all() for x in gs), (rl, rr)
        assert abs(float(out.detach().double().exp().sum()) - 1) < 1e-5


def test_theta_representation_and_radial_families():
    th, w = hd.theta_bins(THETA, 24)
    assert abs(w.sum() - 1) < 1e-12 and (np.diff(th) > 0).all() and th[0] > THETA.min() and th[-1] < THETA.max()
    for fn in (np.cos, lambda t: np.tan(t) ** 2, lambda t: np.exp(-200.0 * np.cos(t) / 300.0)):
        assert abs((w * fn(th)).sum() - fn(THETA).mean()) < 2e-4               # smooth functions of theta: exact to MC noise
    thm, wm = hd.theta_bins(THETA, 24, "midpoint")
    assert np.allclose(wm, 1 / 24) and abs((wm * np.cos(thm)).sum() - np.cos(THETA).mean()) < 1e-3
    # a histogram is accepted the same way as a sample; nothing about the angles is hard-coded
    cnt, edges = np.histogram(THETA, bins=20)
    hist = hd.DetHalo(0.5 * (edges[1:] + edges[:-1]), cnt, n_r=1, n_phi=8)
    assert hist.n_theta == 20 and torch.isfinite(hist()).all()
    one = hd.DetHalo([0.0], n_r=2, n_phi=4, L=300.0, R=150.0, grid=ci.Grid(dtype=F64))      # normal incidence
    with torch.no_grad():
        o = one().exp()
    # normal incidence: z_t = z', so H = 1{-layer/2 <= z' <= D} exp(-z'/L) x ring masses of exp(-r/R)
    zprof = o.sum(1)
    ze = ci.Grid(dtype=F64).z_edges
    a, b = ze[:-1].clamp(-HALF, hd.D_BACK_MM), ze[1:].clamp(-HALF, hd.D_BACK_MM)
    want = torch.exp(-a / 300.0) - torch.exp(-b / 300.0)
    assert torch.allclose(zprof, want / want.sum(), atol=1e-12)
    for radial in ("exp_reg", "weibull", "gauss", "burr"):
        mm = hd.DetHalo(th[::4], w[::4], radial=radial, n_r=2, n_phi=8)
        out = mm()
        assert torch.isfinite(out).all() and mm.n_params() == 2 + len(mm.family.SHAPES)
        out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in mm.parameters())
