"""sfm.sr.targets on synthetic inputs: sampling design, masks, gauge, seed-null, Fisher weights (expected Hessian),
file schema; the event-level estimators in a CPU child process on synthetic showers. No real data is read."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sfm.sr import operators as O
from sfm.sr import search as S
from sfm.sr import targets as TG
from sfm.sr import translate as T

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent
NZ, NR, NS = 40, 12, 4
Z = np.linspace(-190.0, 1440.0, NZ)
R = 700.0 * ((np.arange(NR) + 0.5) / NR) ** 1.7
P = np.array([0.2, 1.0, 5.0, 18.0])


def delta_true(zt, rh, x):
    return 0.4 * np.tanh(1.2 * zt - 1.0) - 0.3 / (1.0 + (0.2 * rh) ** 2) * (1.0 + 0.5 * x)


def synthetic(m=6, noise=0.0, seed=0, shift=None):
    rng = np.random.default_rng(seed)
    zz, rr = np.meshgrid(Z, R, indexing="ij")
    xs = O.x_of_p(P)
    base = np.stack([-0.5 * ((zz - 300 - 100 * x) / 280) ** 2 - rr / 90 for x in xs])
    base -= np.log(np.exp(base).sum((1, 2), keepdims=True))
    d = np.stack([delta_true(zz / 260, rr / 21.9, x) for x in xs])
    teacher = base[None] + d[None] + noise * rng.normal(size=(m, NS, NZ, NR))
    if shift is not None:
        teacher = teacher + np.asarray(shift)[None, :, None, None]
    energy = np.exp(base) * np.exp(d)
    energy /= energy.sum((1, 2), keepdims=True)
    hits = np.round(energy * 4.0e5).astype(np.int64)
    abar = np.clip(1.2 - np.abs(zz - 600) / 900, 0, 1)
    return dict(log_rho_teacher=teacher, log_rho_base=base, slice_p=P, energy_w=energy * 7000, n_hits=hits,
                z_mm=Z, r_mm=R, a_bar=abar)


# ----------------------------------------------------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------------------------------------------------
def test_inclusion_probabilities_and_fixed_size():
    rng = np.random.default_rng(0)
    size = rng.lognormal(0, 2.5, 500)
    pi = TG.inclusion_probabilities(size, 80)
    assert pi.sum() == pytest.approx(80) and pi.max() <= 1.0 and (pi == 1.0).sum() >= 1 and pi.min() > 0
    free = pi < 1
    assert np.allclose(pi[free] / size[free], (pi[free] / size[free])[0])          # proportional outside certainty
    assert TG.inclusion_probabilities(size[:50], 80).sum() == 50                    # fewer units than the size: all
    for s in range(20):
        idx, _ = TG.ht_sample(size / size.sum(), 80, gamma=1.0, rng=s)
        assert len(idx) == 80 == len(set(idx.tolist())) and set(np.flatnonzero(pi == 1.0)) <= set(idx.tolist())


def test_horvitz_thompson_total_is_unbiased():
    rng = np.random.default_rng(1)
    omega = rng.lognormal(0, 2.0, 400)
    omega /= omega.sum()
    g = rng.normal(size=400) + 3.0 * omega * 400
    truth = np.sum(omega * g)
    est = []
    for s in range(600):
        idx, pi = TG.ht_sample(omega, 60, gamma=0.5, rng=1000 + s)
        est.append(np.sum(omega[idx] / pi[idx] * g[idx]))
    est = np.array(est)
    assert abs(est.mean() - truth) < 4 * est.std() / math.sqrt(len(est))
    naive = [np.mean(g[TG.ht_sample(omega, 60, 0.5, rng=s)[0]]) for s in range(200)]   # without the weights: biased
    assert abs(np.mean(naive) - truth) > 10 * est.std() / math.sqrt(len(est))


def test_event_slices():
    p = np.exp(np.random.default_rng(2).uniform(np.log(0.09), np.log(20), 12000))
    sid, med = TG.event_slices(p, 12)
    assert np.bincount(sid).tolist() == [1000] * 12 and np.all(np.diff(med) > 0)
    assert all(np.median(p[sid == s]) == med[s] for s in range(12))


# ----------------------------------------------------------------------------------------------------------------------
# target (iii)
# ----------------------------------------------------------------------------------------------------------------------
def test_residual_target_layout_masks_weights():
    inp = synthetic()
    t = TG.build_residual_target(**inp, n_per_slice=120, seed=3)
    X, y, w, sid = t["X"], t["y"], t["weights"], t["slice_id"]
    meta = json.loads(str(t["meta_json"]))
    assert X.shape == (len(y), 3) and w.shape == y.shape and list(t["variable_names"]) == ["zt", "rh", "x"]
    assert np.all(np.diff(sid) >= 0) and np.bincount(sid).tolist() == [120] * NS          # contiguous, fixed size
    for s in range(NS):
        m = sid == s
        assert np.all(X[m, 2] == t["slice_x"][s]) and t["slice_x"][s] == O.x_of_p(t["slice_p"][s])
        assert np.sum(w[m] * y[m]) == pytest.approx(0.0, abs=1e-12)                       # gauge: weighted mean zero
        assert 0.6 < w[m].sum() < 1.6                                                      # HT estimate of sum omega = 1
    assert np.all(np.diff(t["slice_x"]) > 0)
    iz, ir = t["cell_iz"], t["cell_ir"]
    assert np.all(Z[iz] < 1150.0) and np.all(iz > 0) and np.all(ir < NR - 1)             # bulk, first row, last ring
    assert np.all(inp["a_bar"][iz, ir] > 0.05)
    order = np.argsort(O.x_of_p(P))
    assert np.all(inp["n_hits"][order[sid], iz, ir] >= 30)
    assert np.all(w > 0) and np.allclose(w, t["omega"] / t["pi"]) and np.all((t["pi"] > 0) & (t["pi"] <= 1))
    assert np.allclose(X[:, 0], Z[iz] / 260.0) and np.allclose(X[:, 1], R[ir] / 21.9)    # no response: cell centres
    assert meta["masks"]["z_max_mm"] == 1150.0 and meta["se_mask_applied"] and meta["counts"]["per_slice"][0]["sampled"] == 120
    assert str(t["loss"]) == "gauge" and str(t["kind"]) == "residual"


def test_residual_target_reproduces_delta_and_gauge_loss_is_invariant():
    t = TG.build_residual_target(**synthetic(), n_per_slice=150, seed=4)
    X, y, w = t["X"], t["y"], t["weights"]
    f = delta_true(X[:, 0], X[:, 1], X[:, 2])
    null = TG.null_loss(t)
    assert null == pytest.approx(O.wl2_loss_numpy(np.zeros_like(y), y, w), rel=1e-12)     # centred: the two agree
    assert null > 1e-3
    assert O.gauge_loss_numpy(f, y, w, X[:, 2]) < 1e-25                                   # the truth, in any gauge
    for c in (lambda x: 3.0 * x ** 2, lambda x: np.exp(-x) - 7.0, lambda x: 0.0 * x + 1e3):
        g = f + c(X[:, 2])
        assert O.gauge_loss_numpy(g, y, w, X[:, 2]) < 1e-18                               # invariant under f + c(x)
        assert O.wl2_loss_numpy(g, y, w) > 1e-3                                           # the plain L2 is not
    h = 0.5 * f
    assert O.gauge_loss_numpy(h + 5.0 * X[:, 2], y, w, X[:, 2]) == pytest.approx(
        O.gauge_loss_numpy(h, y, w, X[:, 2]), rel=1e-10)
    # the stored target itself does not depend on the gauge of the maps (a constant per slice in teacher or baseline)
    t2 = TG.build_residual_target(**synthetic(shift=[3.0, -1.0, 0.5, 10.0]), n_per_slice=150, seed=4)
    assert np.array_equal(t2["cell_iz"], t["cell_iz"]) and np.allclose(t2["y"], y, atol=1e-12)


def test_residual_target_refusals_and_single_teacher():
    inp = synthetic()
    with pytest.raises(ValueError, match="training minimum"):
        TG.build_residual_target(**{**inp, "slice_p": np.array([0.05, 1.0, 5.0, 18.0])})
    with pytest.raises(ValueError, match="distinct"):
        TG.build_residual_target(**{**inp, "slice_p": np.array([1.0, 1.0, 5.0, 18.0])})
    one = dict(inp, log_rho_teacher=inp["log_rho_teacher"][0])
    t = TG.build_residual_target(**one, n_per_slice=50)
    meta = json.loads(str(t["meta_json"]))
    assert meta["single_teacher"] and not meta["se_mask_applied"] and np.all(t["seed_se"] == 0)
    noisy = synthetic(noise=3.0, seed=5)                    # se = 3 / sqrt(6) = 1.2 > 0.05: nothing survives
    with pytest.raises(ValueError, match="fewer than two cells"):
        TG.build_residual_target(**noisy)
    t = TG.build_residual_target(**noisy, max_se=1e3, n_per_slice=10 ** 6)                 # more than there are
    assert np.all(t["pi"] == 1.0) and np.allclose([t["weights"][t["slice_id"] == s].sum() for s in range(NS)], 1.0)


def test_seed_null_has_the_noise_of_the_full_teacher():
    assert TG.seed_null_kappa(6, 3, 3) == 0.5 and TG.seed_null_kappa(5, 2, 3) == pytest.approx(math.sqrt(0.2 / (1 / 2 + 1 / 3)))
    inp = synthetic(m=6, noise=0.05, seed=6)
    t = TG.build_residual_target(**inp, n_per_slice=10 ** 6, max_se=1.0, halves=([0, 1, 2], [3, 4, 5]), seed=1)
    X, y, yn, w = t["X"], t["y"], t["y_null"], t["weights"]
    noise = y - delta_true(X[:, 0], X[:, 1], X[:, 2])       # teacher noise (up to the gauge), sd 0.05 / sqrt(6)
    sd = lambda v: math.sqrt(O.gauge_loss_numpy(v, np.zeros_like(v), w, X[:, 2]))   # noqa: E731
    assert sd(noise) == pytest.approx(0.05 / math.sqrt(6), rel=0.25)
    assert sd(yn) == pytest.approx(sd(noise), rel=0.25)     # same noise level, no signal
    assert TG.null_loss(t, "y_null") < 0.05 * TG.null_loss(t)
    with pytest.raises(ValueError):
        TG.build_residual_target(**inp, halves=([0, 1], [1, 2]))


def test_response_smoothing_and_tilted_centroid():
    inp = synthetic(m=1)
    C = NZ * NR
    Rm = np.zeros((C, C))                                    # each cell spreads over its two radial neighbours
    for iz in range(NZ):
        for ir in range(NR):
            for d, wgt in ((-1, 0.25), (0, 0.5), (1, 0.25)):
                j = min(max(ir + d, 0), NR - 1)
                Rm[iz * NR + ir, iz * NR + j] += wgt
    t = TG.build_residual_target(**inp, response=[Rm] * NS, n_per_slice=10 ** 6)
    order = np.argsort(O.x_of_p(P))
    s = 0
    m = t["slice_id"] == s
    cells = t["cell_iz"][m] * NR + t["cell_ir"][m]
    rb = np.exp(inp["log_rho_base"][order[s]]).reshape(-1)
    rt = np.exp(inp["log_rho_teacher"][0, order[s]]).reshape(-1)
    direct = np.log(Rm @ rt) - np.log(Rm @ rb)
    yy = direct[cells] - np.sum(t["weights"][m] * direct[cells]) / t["weights"][m].sum()
    assert np.allclose(t["y"][m], yy, atol=1e-12)
    rbar = (Rm @ (rb * np.tile(R, NZ))) / (Rm @ rb)
    assert np.allclose(t["X"][m, 1], rbar[cells] / 21.9)
    inner = (t["cell_ir"][m] > 0)
    assert np.all(t["X"][m, 1][inner] * 21.9 < R[t["cell_ir"][m][inner]])                  # falling density: pulled in
    assert np.allclose(t["X"][m, 0], Z[t["cell_iz"][m]] / 260.0)


def test_file_round_trip_through_the_runner(tmp_path):
    t = TG.build_residual_target(**synthetic(), n_per_slice=60, halves=([0, 1, 2], [3, 4, 5]))
    path = TG.save_target(tmp_path / "res.npz", t)
    tgt = S.load_target(path)
    assert tgt.nout == 1 and tgt.X.shape == (240, 3) and tgt.weights.shape == (240,)
    assert tgt.variable_names == ["zt", "rh", "x"]
    back = TG.load_target_arrays(path)
    assert str(back["schema_version"]) == TG.SCHEMA_VERSION and "y_null" in back
    bad = dict(t)
    perm = np.random.default_rng(0).permutation(len(t["y"]))
    bad.update(X=t["X"][perm], y=t["y"][perm], weights=t["weights"][perm])
    with pytest.raises(ValueError, match="contiguous"):
        TG.save_target(tmp_path / "bad.npz", bad)


def test_effective_sample_size_is_stored_per_slice_for_the_pilot_log():
    """User confirmation 4: n_eff of the Horvitz-Thompson weights per momentum slice under the size measure
    omega^0.5, with the gamma = 1 value of an equal-size sample for comparison."""
    assert TG.effective_sample_size(np.ones(50)) == 50.0 and TG.effective_sample_size([1.0, 0.0, 0.0]) == 1.0
    t = TG.build_residual_target(**synthetic(), n_per_slice=120, seed=3)
    sid, w = t["slice_id"], t["weights"]
    meta = json.loads(str(t["meta_json"]))
    assert float(t["sampling_gamma"]) == 0.5 and meta["gamma"] == 0.5
    for s in range(NS):
        ne = w[sid == s].sum() ** 2 / np.sum(w[sid == s] ** 2)
        assert t["slice_n_eff"][s] == pytest.approx(ne, rel=1e-12) and 1.0 < ne <= 120.0
        assert meta["counts"]["per_slice"][s]["n_eff"] == pytest.approx(ne, rel=1e-12)
        assert t["slice_n_sampled"][s] == 120 and t["slice_n_masked"][s] >= 120
        assert 1.0 < t["slice_n_eff_gamma1"][s] <= 120.0 and t["slice_n_eff_population"][s] > 1.0
    # under strict proportionality the weights of the non-certain cells are equal: a larger n_eff per sampled row,
    # bought with fewer distinct tail cells; omega^0.5 spreads the rows and pays in n_eff
    assert np.all(t["slice_n_eff_gamma1"] > t["slice_n_eff"])
    rows = TG.n_eff_report(t)
    assert [r["slice"] for r in rows] == list(range(NS)) and all(r["design_effect"] >= 1.0 for r in rows)
    assert rows[0]["n_eff"] == pytest.approx(float(t["slice_n_eff"][0])) and rows[0]["gamma"] == 0.5
    txt = TG.format_n_eff(t)
    assert "design-consistent" in txt and "not unbiased" in txt and len(txt.splitlines()) == 2 + NS
    assert "design-consistent" in meta["estimator_note"] and "n_eff" in meta["n_eff_note"]
    t1 = TG.build_residual_target(**synthetic(), n_per_slice=120, seed=3, gamma=1.0)
    assert float(t1["sampling_gamma"]) == 1.0 and np.all(t1["slice_n_eff"] > t["slice_n_eff"])
    again = TG.build_residual_target(**synthetic(), n_per_slice=120, seed=3)       # the comparison draw is separate
    assert np.array_equal(again["cell_iz"], t["cell_iz"]) and np.array_equal(again["weights"], t["weights"])
    bad = dict(t, slice_n_eff=t["slice_n_eff"] * 1.5)
    with pytest.raises(ValueError, match="does not match"):
        TG.n_eff_report(bad)


def test_the_loss_estimate_is_consistent_not_unbiased():
    """The sampled within-slice weighted variance against the population value: the mean over many designs is
    close, the relative bias is of order 1 / n_eff (it shrinks when the sample grows), and it is not zero - while
    the Horvitz-Thompson estimate of a linear total has no bias at all."""
    rng = np.random.default_rng(12)
    N = 1500
    omega = rng.lognormal(0, 1.6, N)
    omega /= omega.sum()
    err = 0.2 * rng.normal(size=N) + 0.6 * (omega * N > 3)                 # the error is larger in the core
    pop = np.sum(omega * (err - np.sum(omega * err)) ** 2)

    def mean_estimate(n, draws):
        est, neff = [], []
        for k in range(draws):
            idx, pi = TG.ht_sample(omega, n, 0.5, rng=50_000 + k)
            w, e = omega[idx] / pi[idx], err[idx]
            est.append(np.sum(w * (e - np.sum(w * e) / w.sum()) ** 2) / w.sum())
            neff.append(TG.effective_sample_size(w))
        return np.mean(est), np.std(est) / np.sqrt(draws), np.mean(neff)
    small, se_small, ne_small = mean_estimate(25, 3000)
    large, se_large, ne_large = mean_estimate(400, 600)
    assert abs(large / pop - 1.0) < 0.02 and ne_large > 5 * ne_small       # consistent
    assert (pop - small) > 4 * se_small                                     # biased low at small n: not unbiased
    assert abs(small / pop - 1.0) > 2 * abs(large / pop - 1.0)             # and the bias shrinks with n_eff
    assert abs(small / pop - 1.0) < 6.0 / ne_small                         # of relative order 1 / n_eff


# ----------------------------------------------------------------------------------------------------------------------
# target (i)
# ----------------------------------------------------------------------------------------------------------------------
def test_fisher_is_the_expected_hessian_not_the_score_covariance():
    """Hits z_i ~ N(mu, sigma) with energy weights w_i (sum 1): the event loss is -sum_i w_i l_i. Its Hessian in mu
    is sum_i w_i / sigma^2 = 1 / sigma^2. E[sum_i w_i (d l_i)^2] equals it; the squared event score does not."""
    rng = np.random.default_rng(0)
    E, H, mu, sig = 4000, 50, 0.3, 0.7
    z = rng.normal(mu, sig, (E, H))
    w = rng.exponential(1.0, (E, H))
    w /= w.sum(1, keepdims=True)
    ev = np.repeat(np.arange(E), H)
    dl = ((z - mu) / sig ** 2).reshape(-1, 1)
    F = TG.fisher_from_hit_scores(dl, w.reshape(-1), ev, E)
    C2 = TG.score_covariance_from_hit_scores(dl, w.reshape(-1), ev, E)

    def loss(m):
        return np.mean(np.sum(w * 0.5 * ((z - m) / sig) ** 2, 1))
    h = 1e-3
    hess = (loss(mu + h) - 2 * loss(mu) + loss(mu - h)) / h ** 2
    assert hess == pytest.approx(1 / sig ** 2, rel=1e-6)
    assert F.mean() == pytest.approx(hess, rel=0.02)
    assert C2.mean() == pytest.approx(hess * np.mean(np.sum(w ** 2, 1)), rel=0.1)          # smaller by sum w_i^2
    assert C2.mean() < 0.06 * F.mean()


def test_trajectory_target():
    rng = np.random.default_rng(1)
    x_ev = rng.uniform(-1, 1, 6000)
    F_ev = np.column_stack([2.0 + x_ev, np.full(6000, 0.5), 10.0 * np.exp(x_ev)]) * rng.uniform(0.8, 1.2, (6000, 3))
    tw = TG.trajectory_weights(x_ev, F_ev, n_points=30)
    assert tw["x"].shape == (30,) and tw["n"].sum() == pytest.approx(1.0) and tw["weights"].shape == (30, 3)
    vals = np.column_stack([0.3 + 0.5 * tw["x"] - 0.2 * tw["x"] ** 2 + 0.1 * tw["x"] ** 3, np.tanh(2 * tw["x"]),
                            0.1 * tw["x"]])
    t = TG.build_trajectory_target(tw["x"], vals, tw["n"], tw["fisher"], ["a", "b", "c"])
    assert t["X"].shape == (30, 1) and t["y"].shape == (30, 3) and t["weights"].shape == (30, 3)   # (N, n_out)
    assert np.allclose(t["weights"].mean(0), 1.0) and list(t["variable_names"]) == ["x"]
    d = rng.normal(size=(30, 3)) * 0.01                    # a perturbation: weights * nat_factor give n F Delta^2
    assert np.allclose((t["weights"] * d ** 2).sum(0) * t["nat_factor"], (tw["weights"] * d ** 2).sum(0))
    guesses = json.loads(str(t["guesses_json"]))
    assert isinstance(guesses, list) and len(guesses) == 3 and all(isinstance(g, list) and len(g) == 1 for g in guesses)
    e = T.parse(guesses[0][0], ("x",))
    assert T.complexity(e) == 13 and T.n_constants(e) == 4 and T.complexity(e) <= O.MAXSIZE_1D
    assert np.allclose(T.constants(e), [0.1, -0.2, 0.5, 0.3], atol=1e-9)                   # Horner order
    assert np.allclose(T.evaluate(e, x=t["X"][:, 0]), t["y"][:, 0], atol=1e-10)
    with pytest.raises(ValueError, match="training range"):
        TG.build_trajectory_target(tw["x"] * 3, vals, tw["n"], tw["fisher"], ["a", "b", "c"])
    with pytest.raises(ValueError, match="without Fisher"):
        TG.build_trajectory_target(tw["x"], vals, tw["n"], tw["fisher"] * np.array([1, 0, 1]), ["a", "b", "c"])


def test_trajectory_file_is_what_the_runner_reads(tmp_path):
    x = np.linspace(-1, 1, 20)
    t = TG.build_trajectory_target(x, np.column_stack([x, x ** 2]), np.full(20, 0.05), np.ones((20, 2)), ["a", "b"])
    tgt = S.load_target(TG.save_target(tmp_path / "traj.npz", t))
    assert tgt.nout == 2 and tgt.weights.shape == (20, 2) and tgt.variable_names == ["x"]
    assert TG.null_loss(t) == [pytest.approx(np.mean(x ** 2)), pytest.approx(np.mean(x ** 4))]


def test_import_loads_no_torch():
    code = ("import sys; sys.path.insert(0, %r); import sfm.sr.targets as TG; "
            "bad = [m for m in ('torch', 'pysr', 'juliacall') if m in sys.modules]; assert not bad, bad"
            % str(MARTIN / "src"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ----------------------------------------------------------------------------------------------------------------------
# event-level estimators (CPU child process, synthetic showers)
# ----------------------------------------------------------------------------------------------------------------------
def _cpu_suite(out_json: str) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import common as C
    from sfm.families.params import PolyProvider
    from test_sr_refit import sample_showers
    from test_train import make_showers

    assert C.DEVICE.type == "cpu"
    rep = {}
    sh = make_showers(120, 3, torch.device("cpu"))
    sid, med = TG.event_slices(sh.ev["p"].numpy(), 3)
    est = TG.estimate_response(sh, sid, 3, K=8, seed=1)
    rows = [np.asarray(m.sum(1)).reshape(-1) for m in est["R"]]
    has = [est["energy_w"][s].reshape(-1) > 0 for s in range(3)]
    rep["response"] = {"row_sums_ok": all(np.allclose(r[h], 1.0) and np.all(r[~h] == 0) for r, h in zip(rows, has)),
                       "energy_per_slice": est["energy_w"].sum((1, 2)).tolist(), "n_events": est["n_events"].tolist(),
                       "n_hits": int(est["n_hits"].sum()), "hits": int(len(sh.w)),
                       "offdiag": float(1 - sum(m.diagonal().sum() for m in est["R"]) / sum(m.sum() for m in est["R"]))}

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.provider = PolyProvider(["mu", "lw"], 1, init={"mu": [1.0, 0.3], "lw": [0.0, 0.1]})

        def forward(self, lnp):
            raw = self.provider(lnp)
            z0 = 260.0 * raw[:, 0, None, None]
            wd = 260.0 * torch.exp(raw[:, 1, None, None])
            logit = -0.5 * ((C.ZZ[None] - z0) / wd) ** 2 - C.RR[None] / 80.0
            return F.log_softmax(logit.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)

    toy = Toy().eval()
    with torch.inference_mode():
        b = next(sh.batches(40))
        g1 = torch.Generator().manual_seed(5)
        g2 = torch.Generator().manual_seed(5)
        ll, pts = TG.hit_loglik(toy, b, sh.box, 16, g1)
        nll = torch.zeros(len(b.lnp)).index_add_(0, b.ev, -b.w * ll)
        ref = C.event_nll(toy, b, sh.box, 16, g2)[0]
        rep["hit_loglik_diff"] = float((nll - ref).abs().max())
    data = sample_showers(toy, 300, 200, 7)
    fs = TG.fisher_scores(toy, data, [0, 1], eps=1e-3, K=1, seed=0)

    def mean_nll(col, shift):
        prov = toy.provider
        toy.provider = TG.ShiftedProvider(prov, col, shift)
        try:
            with torch.inference_mode():
                gen = torch.Generator().manual_seed(0)
                v = torch.cat([C.event_nll(toy, bb, data.box, 1, gen)[0] for bb in data.batches(128)])
        finally:
            toy.provider = prov
        return float(v.double().mean())
    h = 0.02
    hess = [(mean_nll(j, h) - 2 * mean_nll(j, 0.0) + mean_nll(j, -h)) / h ** 2 for j in (0, 1)]
    rep["fisher"] = {"mean": fs["fisher"].mean(0).tolist(), "hessian": hess, "score_sq": fs["score_sq"].mean(0).tolist(),
                     "shape": list(fs["fisher"].shape), "x_range": [float(fs["x"].min()), float(fs["x"].max())],
                     "provider_restored": isinstance(toy.provider, PolyProvider)}
    Path(out_json).write_text(json.dumps(rep))


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("sr_targets") / "report.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", SFM_ALLOW_CPU="1")   # CPU unit test
    code = (f"import sys; sys.dont_write_bytecode = True; sys.path[:0] = [{str(MARTIN / 'src')!r}, {str(MARTIN)!r}, "
            f"{str(HERE)!r}]; import test_sr_targets as M; M._cpu_suite({str(out)!r})")
    proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    return json.loads(out.read_text())


def test_estimated_response(suite):
    r = suite["response"]
    assert r["row_sums_ok"] and r["n_hits"] == r["hits"] and sum(r["n_events"]) == 120
    assert np.allclose(r["energy_per_slice"], r["n_events"], rtol=1e-5)                    # weights sum to 1 per event
    assert 0.2 < r["offdiag"] < 1.0                                                        # a 48 mm box leaves the cell


def test_hit_loglik_is_the_metric(suite):
    assert suite["hit_loglik_diff"] < 2e-5


def test_fisher_scores_match_the_curvature_of_the_loss(suite):
    r = suite["fisher"]
    assert r["shape"] == [300, 2] and r["provider_restored"]
    for f, h, s in zip(r["mean"], r["hessian"], r["score_sq"]):
        assert f == pytest.approx(h, rel=0.15), r           # expected Hessian = curvature of the mean NLL
        assert s < 0.1 * f                                  # the squared event score is 1/hits of it
