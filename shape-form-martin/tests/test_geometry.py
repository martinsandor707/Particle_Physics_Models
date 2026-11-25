"""Tests of sfm.geometry: rotation convention and origin identity (synthetic Parquet file), the volume constants
against the training columns, the acceptance maps against closed forms, limiting cases and a Monte Carlo with an
independent point-in-volume test, and the evaluation helper against ``sfm.metric``.

No test opens a split file of the campaign. The only data file read is
``results/m2c_theta/geometry/explore/train_columns.npz`` (the 8,752 cell columns seen in TRAINING hits); tests that
need it are skipped when it is missing. Everything runs on the CPU except the last test, which needs the device of
``common`` (the metric's grid tensors live there) and takes ``sfm.gpulock.gpu_lock`` when that is a GPU.
"""
import math

import numpy as np
import pytest
import torch

import common as C
from sfm import geometry as G
from sfm import metric as M
from sfm import paths

COLUMNS = paths.RESULTS_DIR / "m2c_theta" / "geometry" / "explore" / "train_columns.npz"
needs_columns = pytest.mark.skipif(not COLUMNS.exists(), reason="training column list not available")
CPU = torch.device("cpu")


def rot(theta, phi):
    return torch.tensor(C.rotation(np.atleast_1d(np.float64(theta)), np.atleast_1d(np.float64(phi))))


# ----------------------------------------------------------------------------------------------------------------------
# Convention and origins
# ----------------------------------------------------------------------------------------------------------------------
def test_to_global_is_R_transpose():
    rng = np.random.default_rng(0)
    th, ph = rng.uniform(0.05, 0.63, 50), rng.uniform(-np.pi, np.pi, 50)
    loc = rng.normal(0, 300, (50, 3))
    R = C.rotation(th, ph).astype(np.float64)
    ref = np.einsum("nji,nj->ni", R, loc)                                   # R^T . local
    assert np.abs(G.to_global(loc, th, ph) - ref).max() < 1e-4              # R is float32
    wrong = np.einsum("nij,nj->ni", R, loc)
    assert np.abs(G.to_global(loc, th, ph) - wrong).max() > 10              # the transposed convention is far off
    # the shower axis (0, 0, 1) points along (sin t cos p, sin t sin p, cos t)
    ax = G.to_global(np.array([[0.0, 0.0, 1.0]]), np.array([0.4]), np.array([1.1]))[0]
    assert np.allclose(ax, [math.sin(0.4) * math.cos(1.1), math.sin(0.4) * math.sin(1.1), math.cos(0.4)])


def _synthetic_parquet(path, seed=1, n_events=7, corrupt=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    evn = np.array([11, 5, 42, 7, 99, 23, 64])[:n_events]
    cols = {k: [] for k in ("event_number", "x", "y", "z", "x_local", "y_local", "z_local", "incoming_theta",
                            "incoming_phi", "energy")}
    truth = {}
    for e in evn:
        n = int(rng.integers(3, 40))
        th, ph = rng.uniform(0.05, 0.63), rng.uniform(-np.pi, np.pi)
        origin = np.array([rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 3662.4 + 20.5 * rng.integers(0, 5)])
        loc = rng.normal(0, 200, (n, 3))
        g = origin + G.to_global(loc, np.full(n, th), np.full(n, ph))
        if corrupt == e:
            g[0, 0] += 0.01
        truth[int(e)] = (origin, th, ph, n)
        for k, v in (("event_number", np.full(n, e, np.int32)), ("x", g[:, 0]), ("y", g[:, 1]), ("z", g[:, 2]),
                     ("x_local", loc[:, 0]), ("y_local", loc[:, 1]), ("z_local", loc[:, 2]),
                     ("incoming_theta", np.full(n, th)), ("incoming_phi", np.full(n, ph)), ("energy", rng.random(n))):
            cols[k].append(v)
    tbl = pa.table({k: np.concatenate(v) for k, v in cols.items()})
    perm = rng.permutation(tbl.num_rows)                                    # rows of an event need not be contiguous
    pq.write_table(tbl.take(pa.array(perm)), path, row_group_size=37)
    return truth


def test_origins_from_synthetic_parquet(tmp_path):
    f = tmp_path / "hits.parquet"
    truth = _synthetic_parquet(f)
    want = np.array([42, 5, 64, 11, 23])                                    # 7 and 99 play the locked events
    seen = set()

    def hook(index, cols):
        seen.update(np.unique(cols["event_number"]).tolist())
        assert np.array_equal(want[index], cols["event_number"])

    res = G.origins_from_parquet(want, f, row_hook=hook, batch_rows=16)
    assert seen == set(want.tolist())                                       # no row of 7 or 99 was returned
    assert np.array_equal(res["event_number"], want)
    for j, e in enumerate(want):
        origin, th, ph, n = truth[int(e)]
        assert np.abs(res["origin"][j] - origin).max() < 1e-9
        assert res["rows"][j] == n and res["theta"][j] == th and res["phi"][j] == ph
    assert res["spread"].max() < 1e-9
    with pytest.raises(ValueError, match="no row"):
        G.origins_from_parquet(np.array([42, 1000]), f)
    with pytest.raises(ValueError, match="duplicates"):
        G.origins_from_parquet(np.array([42, 42]), f)
    g = tmp_path / "bad.parquet"
    _synthetic_parquet(g, corrupt=23)
    with pytest.raises(ValueError, match="not unique"):
        G.origins_from_parquet(want, g)
    assert G.origins_from_parquet(np.array([42, 5]), g)["spread"].max() < 1e-9   # the bad event is not read


def test_origin_file_roundtrip_and_split_guard(tmp_path):
    f = tmp_path / "hits.parquet"
    _synthetic_parquet(f)
    tr, va = np.array([42, 5, 64]), np.array([11, 23])
    per = {"train": G.origins_from_parquet(tr, f), "val": G.origins_from_parquet(va, f)}
    meta = G.save_origins(per, cache_dir=tmp_path)
    assert meta["splits"]["train"]["events"] == 3 and meta["contains_test_events"] is False
    assert np.array_equal(G.load_origins("val", va, cache_dir=tmp_path), per["val"]["origin"])
    with pytest.raises(ValueError, match="differ"):
        G.load_origins("val", va[::-1], cache_dir=tmp_path)
    with pytest.raises(PermissionError):
        G.load_origins("test", cache_dir=tmp_path)
    with pytest.raises(PermissionError):
        G.save_origins({"test": per["val"]}, cache_dir=tmp_path)
    (tmp_path / f"{G.GEOMETRY_STEM}.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="SHA-256"):
        G.load_origins("val", va, cache_dir=tmp_path)


def test_straight_line_origin():
    o = G.straight_line_origin([0.3], [0.7], [1221.0 / math.cos(0.3) + 100.0])
    z0 = G.Z_BACK - (1221.0 / math.cos(0.3) + 100.0) * math.cos(0.3)
    assert np.allclose(o[0], [z0 * math.tan(0.3) * math.cos(0.7), z0 * math.tan(0.3) * math.sin(0.7), z0])


# ----------------------------------------------------------------------------------------------------------------------
# Volume
# ----------------------------------------------------------------------------------------------------------------------
def test_volume_constants_are_consistent():
    assert G.Z_FRONT == pytest.approx(3652.15) and G.Z_BACK == pytest.approx(4882.15)
    assert len(G.X_LAST) == 26 and G.Y_MAX == pytest.approx(26 * G.MODULE_PITCH_Y - 1.9)
    steps = np.diff(G.X_LAST[10:])
    assert np.allclose(steps, -100.9445, atol=2e-3)                         # one module (two cells) per module row
    with pytest.raises(ValueError):
        G.Volume("sphere")
    v = G.Volume()
    X = torch.tensor([2.5, -100.0, 2622.0, 2630.0, 2500.0, 0.0, 0.0])
    Y = torch.tensor([500.0, 0.0, 0.0, 0.0, 2000.0, 2620.0, 2630.0])
    assert v.inside_xy(X, Y).tolist() == [True, False, True, False, False, True, False]
    assert G.Volume("stepped", hole=False).inside_xy(X, Y).tolist() == [True, True, True, False, False, True, False]
    assert G.Volume("box").inside_xy(X, Y).tolist() == [True, True, True, False, True, True, False]


@needs_columns
def test_volume_matches_training_columns():
    c = np.load(COLUMNS)
    ux, uy = c["ux"], c["uy"]
    assert len(ux) == 8752
    v = G.Volume()
    eps = 1e-3
    for dx in (-G.HALF_CELL + eps, 0.0, G.HALF_CELL - eps):                 # the whole fold box of every column
        for dy in (-G.HALF_CELL + eps, 0.0, G.HALF_CELL - eps):
            assert v.inside_xy_np(ux + dx, uy + dy).all()
    # outermost column of every cell row: just beyond its outer edge is outside
    for y in np.unique(np.round(uy, 3)):
        xs = ux[np.abs(uy - y) < 1e-2]
        assert not v.inside_xy_np(xs.max() + G.HALF_CELL + 0.01, y) and not v.inside_xy_np(xs.min() - G.HALF_CELL - 0.01, y)
        assert xs.min() + xs.max() == pytest.approx(2 * G.X_CENTRE, abs=2e-3)
    # the model and the independent column-based test agree on random points
    rng = np.random.default_rng(3)
    X, Y = rng.uniform(-2700, 2700, 400_000), rng.uniform(-2700, 2700, 400_000)
    a, b = v.inside_xy_np(X, Y), G.columns_inside(X, Y, ux, uy)
    assert (a != b).mean() < 2e-5                                           # only rounding of the edge constants (<= 1 um)
    assert 0.79 < a.mean() * 5400 ** 2 / (2 * G.BOX_HALF) ** 2 < 0.82       # ~19 % of the audit's box has no cell
    # the simpler alternatives
    assert G.Volume("box").inside_xy_np(ux, uy).all()
    octa = G.Volume("octagon")
    assert octa.inside_xy_np(ux, uy).all()                                  # every cell centre is kept ...
    corner = ~octa.inside_xy_np(ux + np.sign(ux - G.X_CENTRE) * 24.0, uy + np.sign(uy) * 24.0)
    assert 40 <= corner.sum() <= 160                                        # ... but the outer corner of each step is cut
    hole = ~v.inside_xy_np(X, Y) & G.Volume(hole=False).inside_xy_np(X, Y)
    assert X[hole].min() > -407 and X[hole].max() < 210 and np.abs(Y[hole]).max() < 303


# ----------------------------------------------------------------------------------------------------------------------
# Acceptance
# ----------------------------------------------------------------------------------------------------------------------
def analytic_slab(theta, D, half, nsub, front=True, back=True):
    """shapelab.slab_accept re-typed for one event on the CPU (closed form in the azimuth)."""
    ze, re = C.Z_EDGES.cpu(), C.R_EDGES.cpu()
    k = (torch.arange(nsub, dtype=torch.float32) + 0.5) / nsub
    zs = ze[:-1, None] + C.DZ * k
    rs = torch.sqrt(re[:-1, None] ** 2 + (re[1:, None] ** 2 - re[:-1, None] ** 2) * k)
    zc = zs[:, :, None, None] * math.cos(theta)
    den = (rs[None, None, :, :] * math.sin(theta)).clamp_min(1e-9)
    lo = ((zc - D) / den).clamp(-1, 1) if back else -torch.ones_like(zc / den)
    hi = ((zc + half) / den).clamp(-1, 1) if front else torch.ones_like(zc / den)
    return ((torch.acos(lo) - torch.acos(hi)) / math.pi).mean((1, 3))


@pytest.mark.parametrize("theta,phi", [(0.08, 0.3), (0.35, -2.0), (0.62, 2.9)])
def test_front_and_back_match_the_closed_form(theta, phi):
    origin = torch.tensor([[900.0, -1200.0, G.Z_LAYER0 + 2 * G.LAYER]])     # far from the sides and the hole
    zb = (G.Z_BACK - float(origin[0, 2])) / math.cos(theta)
    A = G.acceptance(rot(theta, phi), origin, G.Volume(), {"front": ("front",), "fb": ("front", "back"),
                                                           "fs": ("front", "side"), "all": G.FACES}, nsub=2, nphi=720)
    ref_f = analytic_slab(theta, zb * math.cos(theta), G.LAYER / 2, 2, back=False)
    ref_fb = analytic_slab(theta, zb * math.cos(theta), G.LAYER / 2, 2)
    assert (A["front"][0] - ref_f).abs().max() < 4e-3 and (A["front"][0] - ref_f).abs().mean() < 2e-4
    assert (A["fb"][0] - ref_fb).abs().max() < 4e-3
    assert torch.equal(A["fs"], A["front"]) and torch.equal(A["all"], A["fb"])   # no side face within 700 mm
    assert float(A["front"].min()) >= 0 and float(A["front"].max()) <= 1


def test_limiting_cases():
    # (1) normal incidence: a step in z' at -LAYER/2, the same for every ring
    origin = torch.tensor([[900.0, 900.0, G.Z_LAYER0]])
    A = G.acceptance(rot(1e-6, 0.0), origin, G.Volume(), nsub=8, nphi=16)
    zs, _, _ = G.ring_nodes(8, 16)
    step = (zs >= -G.LAYER / 2).float().reshape(C.N_Z, 8).mean(1)
    assert torch.allclose(A["front"][0], step[:, None].expand(C.N_Z, C.N_R))
    assert torch.equal(A["front_side"], A["front"])
    # (2) "detector" front plane equals the first-hit-layer plane when the first layer is hit, and is looser otherwise
    o2 = torch.tensor([[900.0, 900.0, G.Z_LAYER0 + 5 * G.LAYER]])
    a1 = G.acceptance(rot(0.4, 1.0), o2, G.Volume(), {"front": ("front",)}, nphi=64)["front"]
    a2 = G.acceptance(rot(0.4, 1.0), o2, G.Volume(), {"front": ("front",)}, nphi=64, front_plane="detector")["front"]
    a3 = G.acceptance(rot(0.4, 1.0), origin, G.Volume(), {"front": ("front",)}, nphi=64, front_plane="detector")["front"]
    a4 = G.acceptance(rot(0.4, 1.0), origin, G.Volume(), {"front": ("front",)}, nphi=64)["front"]
    assert (a2 >= a1).all() and float((a2 - a1).max()) > 0.1 and torch.equal(a3, a4)
    # (3) an origin beyond the outline: nothing near the axis is accepted by the side faces
    o3 = torch.tensor([[2900.0, 0.0, G.Z_LAYER0]])
    As = G.acceptance(rot(0.05, math.pi / 2), o3, G.Volume(), {"side": ("side",)}, nphi=64)["side"][0]
    assert float(As[:, :12].max()) == 0.0 and float(As[:, -1].max()) > 0.2
    with pytest.raises(ValueError):
        G.acceptance(rot(0.1, 0.0), origin, G.Volume(), {"x": ("top",)})


def test_normal_incidence_back_face():
    """theta -> 0: the back face is the plane z' = zb; with zb inside the grid the 'all' map is a slab in z'."""
    origin = torch.tensor([[900.0, 900.0, G.Z_BACK - 800.0]])
    A = G.acceptance(rot(1e-6, 0.0), origin, G.Volume(), nsub=8, nphi=16)["all"][0]
    zs, _, _ = G.ring_nodes(8, 16)
    slab = ((zs >= -G.LAYER / 2) & (zs <= 800.0)).float().reshape(C.N_Z, 8).mean(1)
    assert torch.allclose(A, slab[:, None].expand(C.N_Z, C.N_R))


@needs_columns
@pytest.mark.parametrize("origin,theta,phi", [((2450.0, 150.0, 3662.4), 0.45, 0.2),      # side face
                                              ((1900.0, 1700.0, 3662.4), 0.55, 0.9),     # staircase
                                              ((-520.0, 120.0, 3682.9), 0.30, 0.1),      # into the beam hole
                                              ((300.0, -2400.0, 3662.4), 0.20, -1.2)])   # bottom edge
def test_side_acceptance_against_monte_carlo(origin, theta, phi):
    c = np.load(COLUMNS)
    R = rot(theta, phi)
    A = G.acceptance(R, torch.tensor([origin]), G.Volume(), {"side": ("side",)}, nsub=4, nphi=256)["side"][0].numpy()
    rng = np.random.default_rng(11)
    Mpts = 300
    ze, re = C.Z_EDGES.cpu().numpy().astype(np.float64), C.R_EDGES.cpu().numpy().astype(np.float64)
    z = ze[:-1, None, None] + C.DZ * rng.random((C.N_Z, C.N_R, Mpts))
    r = np.sqrt(re[None, :-1, None] ** 2 + (re[None, 1:, None] ** 2 - re[None, :-1, None] ** 2) * rng.random((C.N_Z, C.N_R, Mpts)))
    a = rng.uniform(0, 2 * np.pi, (C.N_Z, C.N_R, Mpts))
    g = G.to_global(np.stack([r * np.cos(a), r * np.sin(a), z], -1), theta, phi) + np.asarray(origin)
    mc = G.columns_inside(g[..., 0], g[..., 1], c["ux"], c["uy"]).mean(2)   # independent of the constants in sfm.geometry
    assert 0.05 < mc.mean() < 0.98                                          # the case does probe a boundary
    sig = np.sqrt(np.clip(mc * (1 - mc), 1e-4, None) / Mpts)
    zscore = (A - mc) / np.hypot(sig, 2e-3)                                 # 2e-3: quadrature error allowance
    assert abs((A - mc).mean()) < 1.5e-3
    assert np.sqrt((zscore ** 2).mean()) < 1.3 and np.abs(zscore).max() < 6


def test_mean_acceptance_is_the_event_mean():
    rng = np.random.default_rng(2)
    n = 9
    R = torch.tensor(C.rotation(rng.uniform(0.05, 0.63, n), rng.uniform(-np.pi, np.pi, n)))
    o = np.stack([rng.uniform(-2500, 2500, n), rng.uniform(-2500, 2500, n), np.full(n, G.Z_LAYER0)], 1)
    A = G.event_acceptance(R, o, G.Volume(), nphi=16, chunk=4)
    T = G.mean_acceptance(R, o, G.Volume(), nphi=16, chunk=4)
    for k in G.VARIANTS:
        assert T[k].shape == (C.N_Z, C.N_R) and torch.allclose(T[k], A[k].mean(0), atol=1e-6)


def test_axis_exit():
    v = G.Volume()
    th = 0.5
    R = C.rotation(np.array([th, th, 0.1]), np.array([0.0, math.pi / 2, 0.0]))
    o = np.array([[2200.0, 600.0, 3662.4], [1000.0, 600.0, 3662.4], [-500.0, 50.0, 3662.4]])
    z = G.axis_exit(R, o, v, step=1.0)
    assert z[0] == pytest.approx((G.X_LAST[5] + G.HALF_CELL - 2200.0) / math.sin(th), abs=1.5)   # row a = 5 (y = 600)
    assert np.isinf(z[1])                                # along +Y: 600 + 1450 sin(0.5) = 1295 < the staircase at x = 1000
    assert z[2] == pytest.approx((G.HOLE_LEFT[0] + G.HALF_CELL + 500.0) / math.sin(0.1), abs=1.5)  # into the hole
    assert G.axis_exit(R[:1], np.array([[-100.0, 0.0, 3662.4]]), v)[0] == 0.0                   # starts in the hole


# ----------------------------------------------------------------------------------------------------------------------
# Applying and scoring
# ----------------------------------------------------------------------------------------------------------------------
def test_apply_acceptance():
    g = torch.Generator().manual_seed(0)
    logP = torch.log_softmax(torch.randn(3, C.N_Z * C.N_R, generator=g), 1).reshape(3, C.N_Z, C.N_R)
    one = torch.ones_like(logP)
    assert torch.allclose(G.apply_acceptance(logP, one), logP, atol=1e-6)
    A = one.clone()
    A[:, :10] = 0.0
    A[:, 10:20] = 0.5
    out = G.apply_acceptance(logP, A, floor=1e-3)
    assert torch.isfinite(out).all()
    assert torch.allclose(torch.logsumexp(out.reshape(3, -1), 1), torch.zeros(3), atol=1e-5)
    d = out - logP                                                           # ln A - ln sum(P A), cell by cell
    lnS = torch.log((logP.exp() * A.clamp_min(1e-3)).sum((1, 2)))
    assert torch.allclose(d[:, 30], (-lnS)[:, None].expand(3, C.N_R), atol=1e-5)
    assert torch.allclose(d[:, 15], (math.log(0.5) - lnS)[:, None].expand(3, C.N_R), atol=1e-5)
    assert torch.allclose(d[:, 5], (math.log(1e-3) - lnS)[:, None].expand(3, C.N_R), atol=1e-5)
    # ratio form: the event's acceptance replaces the mean one; A == template leaves the map unchanged
    T = torch.full((C.N_Z, C.N_R), 0.5)
    T[:10] = 0.0
    same = G.apply_acceptance(logP, T.expand_as(logP), 1e-3, template=T)
    assert torch.allclose(same, logP, atol=1e-6)
    rat = G.apply_acceptance(logP, A, 1e-3, template=T)
    d2 = rat - logP
    assert torch.allclose(d2[:, 15] - d2[:, 5], torch.zeros(3, C.N_R), atol=1e-5)          # 0.5/0.5 and floor/floor
    assert torch.allclose(d2[:, 30] - d2[:, 15], torch.full((3, C.N_R), math.log(2.0)), atol=1e-5)
    assert torch.allclose(torch.logsumexp(rat.reshape(3, -1), 1), torch.zeros(3), atol=1e-5)
    with pytest.raises(ValueError):
        G.apply_acceptance(logP, A, template=T[:, :5])
    with pytest.raises(ValueError):
        G.apply_acceptance(logP, A, floor=0.0)
    with pytest.raises(ValueError):
        G.apply_acceptance(logP, A[:, :5])


def _showers(n_ev, seed):
    rng = np.random.default_rng(seed)
    cnt = rng.integers(20, 120, n_ev)
    H = int(cnt.sum())
    theta, phi = rng.uniform(0.05, 0.63, n_ev), rng.uniform(-np.pi, np.pi, n_ev)
    p = np.exp(rng.uniform(np.log(0.2), np.log(20.0), n_ev)).astype(np.float32)
    zb = np.where(rng.random(n_ev) < 0.8, 1221.0 / np.cos(theta), rng.uniform(1230, 1800, n_ev))
    z = np.clip(rng.gamma(2.0, 150.0, H) - 60.0, -199, 1449)
    r = np.clip(rng.exponential(60.0, H), 0, 699)
    a = rng.uniform(0, 2 * np.pi, H)
    xyz = np.stack([r * np.cos(a), r * np.sin(a), z], 1).astype(np.float32)
    owner = np.repeat(np.arange(n_ev), cnt)
    e = rng.exponential(1.0, H)
    w = (e / np.bincount(owner, weights=e)[owner]).astype(np.float32)
    ev = {"p": torch.tensor(p), "bin": torch.tensor(np.floor(p).astype(np.int64)),
          "zb": torch.tensor(zb, dtype=torch.float32), "R": torch.tensor(C.rotation(theta, phi)), "evis": torch.ones(n_ev)}
    ev["lnp"] = torch.log(ev["p"])
    return C.Showers(ev, torch.tensor(np.concatenate([[0], np.cumsum(cnt)])), torch.tensor(xyz), torch.tensor(w),
                     torch.tensor((48.6, 48.6, 20.5), dtype=torch.float32))


def _model(lnp):
    a = torch.sigmoid(lnp)[:, None, None]
    logd = -C.RR / (25 + 20 * a) - ((C.ZZ - 250 - 200 * a) / 350) ** 2 + 0.3 * torch.sin(C.ZZ / 40 + 6 * a)
    return torch.log_softmax((logd + C.LOG_VOL).reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)


def test_score_variants_reproduces_the_metric():
    from sfm.gpulock import gpu_lock
    import contextlib
    lock = gpu_lock("pytest test_geometry", timeout=1800) if C.DEVICE.type == "cuda" else contextlib.nullcontext()
    with lock:
        data = _showers(300, 5).to(C.DEVICE)
        n = len(data)
        origin = torch.tensor(np.stack([np.linspace(-2400, 2400, n), np.linspace(2300, -2300, n), np.full(n, 3662.4)], 1))
        A = G.event_acceptance(data.ev["R"].cpu(), origin, G.Volume(), nphi=32, device=C.DEVICE)
        assert A["all"].shape == (n, C.N_Z, C.N_R) and A["all"].device.type == "cpu"
        assert (A["all"] <= A["front_side"] + 1e-6).all() and (A["front_side"] <= A["front"] + 1e-6).all()

        def make(b, idx):
            lp = _model(b.lnp)
            return {"base": lp, "ones": G.apply_acceptance(lp, torch.ones_like(lp)),
                    "all": G.apply_acceptance(lp, A["all"][idx.cpu()].to(lp.device), 1e-3)}

        res = G.score_variants(make, data)
        ref = M.test_nll(_model, data, deterministic_sum=True)
        assert np.array_equal(res["base"]["nll"].astype(np.float32), ref)       # the bits of the metric
        assert np.abs(res["ones"]["nll"] - res["base"]["nll"]).max() < 2e-4
        for k in ("base", "all"):
            r = res[k]
            assert np.abs(r["nll"] - (res["_shared"]["weight_sum"] * r["log_mass"] - r["hit"])).max() < 2e-4
            assert np.abs(r["total"] - 1).max() < 1e-5
        assert np.isfinite(res["all"]["nll"]).all() and (res["all"]["beyond_zb"] >= 0).all()
        idx = np.flatnonzero(data.ev["zb"].cpu().numpy() >= 1221.0)
        sub = data.subset(torch.as_tensor(idx, device=data.off.device))
        Asub = A["all"][idx]
        bulk = G.score_variants(lambda b, i: {"base": _model(b.lnp),
                                              "all": G.apply_acceptance(_model(b.lnp), Asub[i.cpu()].to(b.lnp.device))},
                                sub, bulk=M.Bulk(short="raise"))
        refb = M.test_nll(_model, sub, bulk=M.Bulk(short="raise"), deterministic_sum=True)
        assert np.array_equal(bulk["base"]["nll"].astype(np.float32), refb)
        assert "retained" in bulk["_shared"] and "beyond_zb" not in bulk["base"]
