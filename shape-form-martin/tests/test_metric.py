"""Tests of sfm.metric against the metric of record (common.event_nll / common.test_nll) and of its extensions.

All tests but the last use small synthetic ``common.Showers`` objects (random events, a few hundred hits each,
rotation matrices from ``common.rotation``, back faces inside and beyond the grid, the 48.6 x 48.6 x 20.5 mm cell)
on ``common.DEVICE``; GPU memory stays below ~350 MB. The last test compares with ``common.test_nll`` on the cached
validation split and is skipped when the cache file does not exist. No test opens a test-split file.

"Bitwise" comparisons run inside ``metric.deterministic()``: on CUDA the last operation of ``common.event_nll``
(``index_add_``) is not run-to-run deterministic, so two calls of ``common.test_nll`` itself differ by ~2e-5 per
event outside that context (see ``test_default_mode_agrees_within_the_noise_of_common``).
"""
import math

import numpy as np
import pytest
import torch

import common as C
from sfm import metric as M
from sfm import paths

DEV = C.DEVICE
BOX = (48.6, 48.6, 20.5)
R_LAST = float(C.R_EDGES[C.N_R - 1])                      # inner edge of the last ring, 663.2 mm


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic showers and models
# ----------------------------------------------------------------------------------------------------------------------
def make_showers(n_ev, seed, hits=(30, 300), zb="mixed", z_range=(-199.0, 1449.0), r_max=699.0, tails=True):
    """Random events. zb: "mixed" = front-face events (1221 / cos theta: inside and beyond the grid), late starts
    down to 200 mm and far back faces up to 1800 mm; "deep" = all between 1200 and 1700 mm."""
    rng = np.random.default_rng(seed)
    cnt = rng.integers(hits[0], hits[1], n_ev)
    cnt[0] = 1                                                             # single-hit events exist in the data
    H = int(cnt.sum())
    theta, phi = rng.uniform(0.05, 0.63, n_ev), rng.uniform(-np.pi, np.pi, n_ev)
    p = np.exp(rng.uniform(np.log(0.2), np.log(20.0), n_ev)).astype(np.float32)
    if zb == "mixed":
        kind = rng.random(n_ev)
        zb = np.where(kind < 0.6, 1221.0 / np.cos(theta), np.where(kind < 0.8, rng.uniform(200, 1150, n_ev),
                                                                   rng.uniform(1150, 1800, n_ev)))
    else:
        zb = rng.uniform(1200, 1700, n_ev)
    z = rng.gamma(2.0, 150.0, H) - 60.0
    r = rng.exponential(60.0, H)
    if tails:                                                              # populate the deep tail and the last ring
        sel = rng.random(H)
        z = np.where(sel < 0.05, rng.uniform(1100, 1449, H), z)
        r = np.where(sel > 0.97, rng.uniform(600, 699, H), r)
    z, r = np.clip(z, *z_range), np.clip(r, 0.0, r_max)
    a = rng.uniform(0, 2 * np.pi, H)
    xyz = np.stack([r * np.cos(a), r * np.sin(a), z], 1).astype(np.float32)
    owner = np.repeat(np.arange(n_ev), cnt)
    e = rng.exponential(1.0, H)
    w = (e / np.bincount(owner, weights=e)[owner]).astype(np.float32)
    ev = {"p": torch.tensor(p), "bin": torch.tensor(np.floor(p).astype(np.int64)),
          "zb": torch.tensor(zb, dtype=torch.float32), "R": torch.tensor(C.rotation(theta, phi)),
          "evis": torch.ones(n_ev)}
    ev["lnp"] = torch.log(ev["p"])
    return C.Showers(ev, torch.tensor(np.concatenate([[0], np.cumsum(cnt)])), torch.tensor(xyz), torch.tensor(w),
                     torch.tensor(BOX, dtype=torch.float32)).to(DEV)


def sharp_map(lnp, amp=1.0):
    """A deliberately rough but bounded map: smooth envelope times a cell-to-cell checkerboard (factor e^+-amp
    between neighbouring grid cells), momentum dependent. Large variance inside a detector cell, light tails."""
    a = torch.sigmoid(lnp)[:, None, None]
    iz = torch.arange(C.N_Z, device=lnp.device)[:, None]
    ir = torch.arange(C.N_R, device=lnp.device)[None, :]
    logd = (-C.RR / (25 + 20 * a) - ((C.ZZ - 250 - 200 * a) / 350) ** 2
            + amp * torch.cos(math.pi * iz) * torch.cos(math.pi * ir) * (0.5 + a) + 0.5 * amp * torch.sin(C.ZZ / 40 + 6 * a))
    return torch.log_softmax((logd + C.LOG_VOL).reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)


def whole(data):
    return data.batch(torch.arange(len(data), device=DEV))


def gen_at(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def seg_sum(values, ev, n):
    return torch.zeros(n, dtype=torch.float64, device=values.device).index_add_(0, ev, values.double())


@pytest.fixture(scope="module")
def data():
    return make_showers(150, seed=0)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    mlp = C.FieldModel(16, 2).to(DEV).eval()
    formula = C.Formula("gamma", "core_tail", 1).to(DEV).eval()
    with torch.no_grad():
        mlp.mlp[-1].weight.mul_(8.0)                                       # random weights, visible structure
        formula.coef[:, 1].normal_(0, 0.1)
        formula.slope.normal_(0, 0.1)
    return {"mlp": mlp, "formula": formula}


# ----------------------------------------------------------------------------------------------------------------------
# (1) equality with common
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["mlp", "formula"])
def test_event_nll_bitwise_equals_common(data, models, name):
    model = models[name]
    with torch.no_grad(), M.deterministic():
        for K in (C.N_FOLD, C.N_FOLD_EVAL):
            g_ref, g_new = gen_at(12345), gen_at(12345)
            for b in data.batches(64):
                ref, ref_logP = C.event_nll(model, b, data.box, K, g_ref)
                new, new_logP = M.event_nll(model, b, data.box, K, g_new)
                assert new.dtype == ref.dtype and torch.equal(new, ref) and torch.equal(new_logP, ref_logP)
                assert torch.isfinite(new).all()
            # same consumption of the generator: the next numbers drawn from both are identical
            assert torch.equal(torch.rand(7, device=DEV, generator=g_ref), torch.rand(7, device=DEV, generator=g_new))
        # defaults (K = N_FOLD, global generator) and a precomputed log P
        b = next(iter(data.batches(64)))
        torch.manual_seed(5)
        ref = C.event_nll(model, b, data.box)[0]
        torch.manual_seed(5)
        assert torch.equal(M.event_nll(model, b, data.box)[0], ref)
        torch.manual_seed(5)
        assert torch.equal(M.event_nll(model(b.lnp), b, data.box)[0], ref)


@pytest.mark.parametrize("name", ["mlp", "formula"])
def test_test_nll_bitwise_equals_common(data, models, name):
    model = models[name]
    with M.deterministic():
        ref = C.test_nll(model, data)
        new = M.test_nll(model, data)
        assert new.dtype == ref.dtype == np.float32 and new.shape == (len(data),)
        assert np.array_equal(new, ref) and np.isfinite(ref).all()
        assert np.array_equal(M.test_nll(model, data, C.N_FOLD), C.test_nll(model, data, C.N_FOLD))   # K = 16
        assert np.array_equal(M.test_nll(model, data, inference=False), ref)      # no_grad instead of inference_mode
        assert np.array_equal(M.test_nll(model, data, chunk=64), ref)             # chunk >= K is the same code path
        assert not np.array_equal(M.test_nll(model, data, seed=1), ref)           # the seed matters
    # the option does what the context does
    with M.deterministic():
        ref = C.test_nll(model, data)
    assert np.array_equal(M.test_nll(model, data, deterministic_sum=True), ref)


def test_deterministic_context_restores_the_setting():
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    with M.deterministic():
        assert torch.are_deterministic_algorithms_enabled()
    assert (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) == before
    with pytest.raises(RuntimeError):
        with M.deterministic():
            raise RuntimeError("boom")
    assert torch.are_deterministic_algorithms_enabled() == before[0]
    with M.deterministic():                                                # nesting, and an already strict session
        with M.deterministic():
            assert torch.are_deterministic_algorithms_enabled()
        assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with M.deterministic():
            pass
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


def test_default_mode_agrees_within_the_noise_of_common(data, models):
    """Without deterministic(): equal on CPU; on CUDA equal up to the accumulation order of index_add_, which
    also separates two runs of common.test_nll. A float32 sum of a few hundred terms is uncertain by a few ulp,
    i.e. ~1e-6 relative (measured: up to 2e-5 on an NLL of ~20); bound 1e-5 relative."""
    model = models["formula"]
    ref, again, new = C.test_nll(model, data), C.test_nll(model, data), M.test_nll(model, data)
    scale = np.maximum(1.0, np.abs(ref))
    assert (np.abs(new - ref) / scale).max() <= 1e-5 and (np.abs(again - ref) / scale).max() <= 1e-5
    assert abs(float(new.astype(np.float64).mean() - ref.astype(np.float64).mean())) < 1e-5
    if DEV.type == "cpu":
        assert np.array_equal(new, ref)


def test_gradient_matches_common(data, models):
    """The loss is usable for training exactly like common.event_nll."""
    model = models["mlp"]
    b = next(iter(data.batches(48)))
    grads = []
    with M.deterministic():
        for fn in (C.event_nll, M.event_nll):
            model.zero_grad()
            fn(model, b, data.box, 16, gen_at(3))[0].mean().backward()
            grads.append(torch.cat([p.grad.flatten().clone() for p in model.parameters()]))
    assert torch.isfinite(grads[0]).all() and float(grads[0].abs().max()) > 0
    assert torch.allclose(grads[0], grads[1], rtol=1e-5, atol=1e-7)
    for kw in ({"fold": "stratified"}, {"chunk": 5}, {"bulk": True}, {"bulk": True, "chunk": 5, "fold": "stratified"}):
        model.zero_grad()
        nll = M.event_nll(model, b, data.box, 16, gen_at(3), **kw)[0]
        nll[torch.isfinite(nll)].mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters()), kw
    model.zero_grad()


# ----------------------------------------------------------------------------------------------------------------------
# (2) several models on the same points
# ----------------------------------------------------------------------------------------------------------------------
def test_multi_model_equals_separate_common_calls(data, models):
    pair = [models["mlp"], models["formula"]]
    with M.deterministic():
        multi = M.test_nll_multi(pair, data)
        assert multi.shape == (2, len(data)) and multi.dtype == np.float32
        for j, model in enumerate(pair):
            assert np.array_equal(multi[j], C.test_nll(model, data))              # separate calls, same points
        assert np.array_equal(M.test_nll_multi(pair[::-1], data), multi[::-1])    # the order of models is irrelevant
        with torch.no_grad():
            b = next(iter(data.batches(64)))
            res = M.event_nll_multi(pair, b, data.box, 64, gen_at(9))
            for (nll, logP), model in zip(res, pair):
                ref, ref_logP = C.event_nll(model, b, data.box, 64, gen_at(9))
                assert torch.equal(nll, ref) and torch.equal(logP, ref_logP)
            assert M.event_nll_multi([], b, data.box, 64, gen_at(9)) == []
            with pytest.raises(ValueError):
                M.event_nll_multi([pair[0](b.lnp)[:-1]], b, data.box, 64, gen_at(9))   # log P of the wrong batch


# ----------------------------------------------------------------------------------------------------------------------
# (3) decomposition
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{}, {"fold": "stratified"}, {"chunk": 16}, {"bulk": True},
                                {"bulk": M.Bulk(renormalise=False), "chunk": 16}])
def test_parts_sum_back_to_the_nll(data, models, kw):
    b = whole(data)
    with torch.no_grad(), M.deterministic():
        out = [M.event_nll(m, b, data.box, 64, gen_at(7), return_parts=True, **kw) for m in models.values()]
        plain = [M.event_nll(m, b, data.box, 64, gen_at(7), **kw)[0] for m in models.values()]
    recon = []
    for (nll, logP, parts), ref in zip(out, plain):
        assert torch.equal(torch.nan_to_num(nll, nan=-1.0), torch.nan_to_num(ref, nan=-1.0))   # parts do not change the NLL
        assert parts.hit.shape == parts.weight.shape == b.w.shape and parts.log_mass.shape == nll.shape
        r = parts.weight_sum.double() * parts.log_mass.double() - seg_sum(parts.hit, b.ev, len(nll))
        ok = torch.isfinite(nll)
        assert ok.sum() >= len(nll) - 2
        # float32 sums of a few hundred terms against a float64 sum: a few 1e-6 on an NLL of ~20
        assert torch.allclose(r[ok], nll[ok].double(), rtol=0, atol=1e-4)
        recon.append((nll, parts, ok))
    # exact split of a paired difference into a hit term and a normalisation term, on common points
    (na, pa, oka), (nb, pb, okb) = recon
    ok = oka & okb
    assert torch.equal(pa.weight, pb.weight) and torch.equal(pa.weight_sum, pb.weight_sum)
    hit_term = seg_sum(pb.hit.double() - pa.hit.double(), b.ev, len(na))
    norm_term = pa.weight_sum.double() * (pa.log_mass.double() - pb.log_mass.double())
    assert torch.allclose((hit_term + norm_term)[ok], (na.double() - nb.double())[ok], rtol=0, atol=1e-4)
    if "bulk" not in kw:
        assert pa.retained is None and pa.z_limit is None
        assert torch.equal(pa.weight, b.w) and torch.allclose(pa.weight_sum, torch.ones_like(pa.weight_sum), atol=1e-5)


def test_test_nll_multi_parts_align_with_the_data(data, models):
    pair = list(models.values())
    with M.deterministic():
        out = M.test_nll_multi(pair, data, return_parts=True, batch=64)
        assert np.array_equal(out["nll"], M.test_nll_multi(pair, data, batch=64))
    H, N = len(data.xyz), len(data)
    assert out["hit"].shape == (2, H) and out["log_mass"].shape == (2, N) and out["weight_sum"].shape == (N,)
    assert np.array_equal(out["weight"], data.w.cpu().numpy())             # per-hit arrays follow data.xyz / data.w
    owner = np.repeat(np.arange(N), np.diff(data.off.cpu().numpy()))
    for j in range(2):
        r = out["weight_sum"] * out["log_mass"][j].astype(np.float64) - np.bincount(owner, weights=out["hit"][j], minlength=N)
        assert np.abs(r - out["nll"][j]).max() < 1e-4
    single = M.test_nll(pair[0], data, return_parts=True, batch=64, bulk=True)
    assert set(single) == {"nll", "hit", "log_mass", "weight", "weight_sum", "retained", "z_limit"}
    assert single["nll"].shape == (1, N) and single["retained"].shape == (N,)


# ----------------------------------------------------------------------------------------------------------------------
# (4) chunked evaluation
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["mlp", "formula"])
def test_chunked_equals_unchunked(data, models, name):
    """Same points, partial sums combined with logsumexp: only rounding differs (measured 2-4e-6 on NLL ~ 20;
    the per-hit ln g agrees to 1e-5, the per-event NLL to 1e-5)."""
    model = models[name]
    b = whole(data)
    with torch.no_grad(), M.deterministic():
        ref, _, pr = M.event_nll(model, b, data.box, 64, gen_at(11), return_parts=True)
        for chunk in (1, 7, 16, 32, 63):
            g = gen_at(11)
            new, _, pn = M.event_nll(model, b, data.box, 64, g, chunk=chunk, return_parts=True)
            assert float((new - ref).abs().max()) <= 1e-5, chunk
            assert float((pn.hit / b.w - pr.hit / b.w).abs().max()) <= 1e-5, chunk          # ln g per hit
            g0 = gen_at(11)
            C.fold(b, data.box, 64, g0)
            assert torch.equal(torch.rand(3, device=DEV, generator=g), torch.rand(3, device=DEV, generator=g0))
        ref = M.test_nll(model, data)
        assert np.abs(M.test_nll(model, data, chunk=16) - ref).max() <= 1e-5
        bulk = M.test_nll(model, data, bulk=True)
        assert np.nanmax(np.abs(M.test_nll(model, data, bulk=True, chunk=16) - bulk)) <= 1e-5
        assert np.array_equal(np.isnan(M.test_nll(model, data, bulk=True, chunk=16)), np.isnan(bulk))
    with pytest.raises(ValueError):
        M.event_nll(model, b, data.box, 64, gen_at(1), chunk=0)


# ----------------------------------------------------------------------------------------------------------------------
# (5) fold points: i.i.d. is common.fold; stratified is unbiased with a smaller variance
# ----------------------------------------------------------------------------------------------------------------------
def test_iid_draw_is_common_fold(data):
    b = next(iter(data.batches(32)))
    u = M.draw(b, data.box, 16, gen_at(4))
    pts = b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)
    assert torch.equal(pts, C.fold(b, data.box, 16, gen_at(4)))
    half = 0.5 * data.box
    assert bool((u >= -half).all() and (u <= half).all())
    with pytest.raises(ValueError):
        M.draw(b, data.box, 16, gen_at(4), fold="sobol")


def test_strata_shape():
    expect = {1: (1, 1, 1), 2: (1, 1, 2), 4: (2, 2, 1), 8: (2, 2, 2), 16: (4, 4, 1), 32: (4, 4, 2), 64: (4, 4, 4),
              256: (8, 8, 4), 1024: (16, 16, 4), 6: (1, 1, 6), 12: (2, 2, 3), 17: (1, 1, 17)}
    for K, shape in expect.items():
        assert M.strata_shape(K, BOX) == shape, K
    for K in range(1, 200):
        n1, n2, m = M.strata_shape(K, BOX)
        assert n1 == n2 and n1 * n2 * m == K
    assert M.strata_shape(64, (10.0, 10.0, 640.0)) == (1, 1, 64)           # follows the cell shape
    with pytest.raises(ValueError):
        M.strata_shape(0, BOX)


@pytest.mark.parametrize("K", [8, 16, 64, 12])
def test_stratified_points_fill_every_stratum_once(data, K):
    b = next(iter(data.batches(16)))
    u = M.draw(b, data.box, K, gen_at(2), fold="stratified")
    assert u.shape == (len(b.xyz), K, 3)
    shape = torch.tensor(M.strata_shape(K, BOX), device=DEV)
    unit = u / data.box + 0.5                                              # back to the unit cube
    assert bool((unit >= 0).all() and (unit <= 1).all())
    cell = torch.minimum((unit * shape).floor().long(), shape - 1)         # stratum index per axis
    flat = (cell[..., 0] * shape[1] + cell[..., 1]) * shape[2] + cell[..., 2]
    assert torch.equal(flat.sort(1).values, torch.arange(K, device=DEV).expand(len(b.xyz), K))   # one point each
    # the jitter is uniform inside the stratum: mean position 1/2, variance 1/12 (H*K*3 > 1e5 numbers)
    frac = unit * shape - cell
    assert abs(float(frac.mean()) - 0.5) < 0.01 and abs(float(frac.var()) - 1 / 12) < 0.005
    # same consumption of the generator as the i.i.d. scheme
    g1, g2 = gen_at(2), gen_at(2)
    M.draw(b, data.box, K, g1, fold="stratified"), M.draw(b, data.box, K, g2, fold="iid")
    assert torch.equal(torch.rand(3, device=DEV, generator=g1), torch.rand(3, device=DEV, generator=g2))


@pytest.fixture(scope="module")
def ghat_samples():
    """Cell averages ghat_i of a sharp map for 300 seeds, i.i.d. and stratified, at K = 16 and 64, together with a
    high-K reference (24 x 1024 i.i.d. points per hit)."""
    data = make_showers(40, seed=1, hits=(50, 150), tails=False)
    b = whole(data)
    with torch.inference_mode():
        logP = sharp_map(b.lnp)

        def ghat(K, seed, fold, chunk=None):
            _, _, p = M.event_nll(logP, b, data.box, K, gen_at(seed), fold=fold, chunk=chunk, return_parts=True)
            return (p.hit / p.weight).double().exp()

        ref = torch.stack([ghat(1024, 900 + s, "iid", 256) for s in range(24)]).mean(0)
        g = {(K, fold): torch.stack([ghat(K, s, fold) for s in range(300)])
             for K in (16, 64) for fold in ("iid", "stratified")}
    return ref.clone(), {k: v.clone() for k, v in g.items()}


@pytest.mark.parametrize("K", [16, 64])
def test_stratified_fold_is_unbiased_and_has_smaller_variance(ghat_samples, K):
    """Unbiasedness per hit: z_i = (mean over 300 seeds of ghat_i - g_i) / se_i must look standard normal over the
    ~3900 hits (se_i from the seed scatter plus the error of the reference). A weight or volume error in the
    strata would shift the z-scores by many units. Bounds: |mean z| < 0.1 (sd of the mean 1/sqrt(3900) = 0.016,
    plus the small skewness of the seed means; measured |mean| <= 0.03), rms in [0.9, 1.1] (measured 0.99-1.01).
    Variance: the mean relative variance of ghat falls to 0.86 (K = 16, strata 4x4x1) and 0.69 (K = 64, 4x4x4)
    of the i.i.d. value for this map, whose structure sits at the grid scale - the least favourable case. The
    ratio of two such means is known to ~0.5 % here (300 seeds x 3900 hits); required < 0.95 / 0.80.
    Hit by hit the variance estimates from 300 seeds scatter by ~14 %, so "smaller for most hits" is all that
    can be asked: measured 86 % (K = 16) and 97 % (K = 64) of the hits; required 75 % / 90 %."""
    ref, g = ghat_samples
    iid, strat = g[(K, "iid")], g[(K, "stratified")]
    S = iid.shape[0]
    var_ref = K * iid.var(0) / (1024 * 24)                                 # cell variance / points in the reference
    for sample in (iid, strat):
        z = (sample.mean(0) - ref) / torch.sqrt(sample.var(0) / S + var_ref)
        assert abs(float(z.mean())) < 0.1 and 0.9 < float(z.pow(2).mean().sqrt()) < 1.1
        agg = (sample / ref - 1).mean(1)                                   # per-seed mean relative deviation
        assert abs(float(agg.mean())) < 4 * float(agg.std() / math.sqrt(S)) + 4 * float((var_ref.sqrt() / ref).mean()) / math.sqrt(len(ref))
    ratio = float((strat.var(0) / ref ** 2).mean() / (iid.var(0) / ref ** 2).mean())
    assert ratio < (0.95 if K == 16 else 0.80), ratio
    assert float((strat.var(0) < iid.var(0)).double().mean()) > (0.75 if K == 16 else 0.90)


# ----------------------------------------------------------------------------------------------------------------------
# (6) Jensen bias
# ----------------------------------------------------------------------------------------------------------------------
def test_jensen_bias_sign_size_and_scaling(ghat_samples):
    """E[-ln ghat_K] + ln g = Var_cell(rho) / (2 K g^2) + O(K^-2), per hit, averaged over the hits.
    With cv2_i = K Var(ghat_i) / g_i^2 (from the seed scatter) the leading term is mean_i cv2_i / (2K); here
    mean cv2 = 0.86, so the bias is 0.0067 at K = 64 and 0.027 + second order at K = 16 (measured 0.0067, 0.0284).
    The statistical error of these means (300 seeds x 3900 hits) is about 1 %; the bounds leave room for the
    K^-2 term: 10 % at K = 64, and a ratio bias(16)/bias(64) of 4.2 inside [3.6, 4.8] (4 = pure 1/K).
    Stratified points: bias 0.88 (K = 16) and 0.70 (K = 64) of the i.i.d. bias; required < 0.95, and falling with K."""
    ref, g = ghat_samples
    bias = {k: float((-(v.log()).mean(0) + ref.log()).mean()) for k, v in g.items()}
    cv2 = float((64 * g[(64, "iid")].var(0) / ref ** 2).mean())
    assert bias[(16, "iid")] > bias[(64, "iid")] > 0                       # sign: the K-point NLL is biased upwards
    assert bias[(64, "iid")] == pytest.approx(cv2 / (2 * 64), rel=0.10)
    assert 3.6 < bias[(16, "iid")] / bias[(64, "iid")] < 4.8               # ~ 1/K
    for K in (16, 64):                                                     # stratification shrinks the bias
        assert 0 < bias[(K, "stratified")] < 0.95 * bias[(K, "iid")]
    assert bias[(64, "stratified")] / bias[(64, "iid")] < bias[(16, "stratified")] / bias[(16, "iid")]


def test_k_scan_and_jensen_bias_on_events():
    data = make_showers(96, seed=2, hits=(50, 150), tails=False)
    Ks, seeds = (8, 32, 128), (100, 101, 102, 103, 104, 105)
    scan = M.k_scan([sharp_map], data, Ks, seeds)
    assert scan.shape == (1, 3, 6, len(data)) and scan.dtype == np.float64 and np.isfinite(scan).all()
    with M.deterministic():                                                # the (K = 64, seed 12345) slice is the report metric
        official = M.k_scan([sharp_map], data, (16, 64), (12345,))[0, :, 0]
        assert np.array_equal(official[1].astype(np.float32), C.test_nll(sharp_map, data))
        assert np.array_equal(official[0].astype(np.float32), C.test_nll(sharp_map, data, 16))
    fit = M.jensen_fit(scan[0], Ks)
    again = M.jensen_bias(sharp_map, data, Ks, seeds)
    assert again.value_inf == pytest.approx(fit.value_inf, abs=1e-4) and again.coef[0] == pytest.approx(fit.coef[0], abs=1e-2)
    m = scan[0].mean((1, 2))
    assert m[0] > m[1] > m[2] > fit.value_inf                              # NLL falls with K towards the limit
    assert fit.coef[0] > 10 * fit.coef_se[0] > 0                           # c > 0, clearly
    assert fit.bias(8) == pytest.approx(fit.coef[0] / 8) and fit.bias(32) == pytest.approx(m[1] - fit.value_inf, abs=4 * fit.mean_se[1] + 4 * fit.value_inf_se)
    assert fit.dof == 1 and fit.chi2 < 20 and fit.value_inf_se < fit.mean_se[0]
    # the gap between two models has its own, much smaller, K dependence: fit it on paired differences
    two = M.k_scan([sharp_map, lambda lnp: sharp_map(lnp, amp=0.5)], data, Ks, seeds)
    gap = M.jensen_fit(two[0] - two[1], Ks)
    assert gap.coef[0] > 0 and gap.mean_se[0] < fit.mean_se[0] * 1.5       # rougher map: larger bias
    # stratified points: lower NLL at every K
    strat = M.k_scan([sharp_map], data, Ks, seeds, fold="stratified")[0].mean((1, 2))
    assert np.all(strat < m)


def test_jensen_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    Ks, n_seeds, n_ev = (16, 64, 256, 1024), 4, 3000
    base = 18.0 + rng.standard_normal(n_ev)
    v_inf, c1, c2 = float(base.mean()), 0.2, 6.0             # c2/K^2 = 0.023 at K = 16, 20 sd of the fold noise there
    K = np.array(Ks, dtype=float)[:, None, None]
    noise = 0.5 / np.sqrt(K) * rng.standard_normal((4, n_seeds, n_ev))
    fit = M.jensen_fit(base + c1 / K + noise, Ks)
    assert fit.value_inf == pytest.approx(v_inf, abs=4 * fit.value_inf_se) and fit.coef[0] == pytest.approx(c1, abs=4 * fit.coef_se[0])
    assert fit.mean_se == pytest.approx(0.5 / np.sqrt(np.array(Ks) * n_seeds * n_ev), rel=0.05)   # the fold noise only
    assert fit.dof == 2 and fit.chi2 < 14 and fit.bias(64) == pytest.approx(c1 / 64, abs=4 * fit.bias_se(64))
    assert fit.value_inf_se < 0.5 * float(base.std() / np.sqrt(n_ev))      # far below the event-sampling error
    quad = M.jensen_fit(base + c1 / K + c2 / K ** 2 + noise, Ks, powers=(1.0, 2.0))
    assert quad.coef == pytest.approx([c1, c2], abs=4 * float(quad.coef_se.max())) and quad.dof == 1
    assert quad.value_inf == pytest.approx(v_inf, abs=4 * quad.value_inf_se)
    lin = M.jensen_fit(base + c1 / K + c2 / K ** 2 + noise, Ks)            # the 1/K model is rejected when it is wrong
    assert lin.chi2 > 20
    with pytest.raises(ValueError):
        M.jensen_fit((base + c1 / K + noise)[:, :1], Ks)                   # one seed: no fold-noise estimate
    with pytest.raises(ValueError):
        M.jensen_fit(base + c1 / K + noise, Ks[:3])
    with pytest.raises(ValueError):
        M.jensen_fit(np.where(noise > 10, 0.0, np.nan) + base, Ks)


# ----------------------------------------------------------------------------------------------------------------------
# (7) bulk-restricted likelihood
# ----------------------------------------------------------------------------------------------------------------------
def outside_cells(z_limit, drop_rings=1):
    """(B, N_Z, N_R) bool: grid cells entirely outside the bulk region of each event."""
    out = (C.Z_EDGES[:-1][None, :, None] >= z_limit[:, None, None]).expand(len(z_limit), C.N_Z, C.N_R).clone()
    out[:, :, C.N_R - drop_rings:] = True
    return out


@pytest.mark.parametrize("name", ["mlp", "formula"])
def test_bulk_is_invariant_to_the_model_outside_the_region(data, models, name):
    """Move mass beyond the z limit and into the last ring (random factors e^N(0, 2^2) on every cell outside A_e,
    then renormalise the whole map): the bulk NLL does not change, the standard NLL does."""
    b = whole(data)
    region = M.Bulk()
    with torch.no_grad(), M.deterministic():
        logP = models[name](b.lnp)
        out = outside_cells(region.z_limit(b.zb))
        noise = 2.0 * torch.randn(logP.shape, generator=torch.Generator().manual_seed(1)).to(DEV)
        moved = torch.log_softmax((logP + noise * out).reshape(len(logP), -1), 1).reshape(logP.shape)
        assert float((moved - logP)[~out].abs().max()) > 0.05              # the inside changed by a per-event constant
        for spec, kw in ((region, {}), (region, {"fold": "stratified"}), (region, {"chunk": 16}),
                         (M.Bulk(renormalise=False), {}), (M.Bulk(z_cut=1000.0, drop_rings=3), {})):
            out2 = outside_cells(spec.z_limit(b.zb), spec.drop_rings)
            moved2 = torch.log_softmax((logP + noise * out2).reshape(len(logP), -1), 1).reshape(logP.shape)
            a, _, pa = M.event_nll(logP, b, data.box, 64, gen_at(21), bulk=spec, return_parts=True, **kw)
            c, _, pc = M.event_nll(moved2, b, data.box, 64, gen_at(21), bulk=spec, return_parts=True, **kw)
            assert torch.equal(torch.isnan(a), torch.isnan(c))
            ok = ~torch.isnan(a)
            assert ok.sum() >= len(a) - 2 and torch.allclose(a[ok], c[ok], rtol=0, atol=2e-4), (spec, kw)
            assert torch.equal(pa.weight, pc.weight) and torch.equal(pa.retained, pc.retained)
        s1 = M.event_nll(logP, b, data.box, 64, gen_at(21))[0]
        s2 = M.event_nll(moved, b, data.box, 64, gen_at(21))[0]
        assert float((s1 - s2).abs().mean()) > 0.05                        # the report metric does see the change
        mon1, mon2 = M.mass_monitors(logP, b), M.mass_monitors(moved, b)
        assert float((mon1["outside_bulk"] - mon2["outside_bulk"]).abs().mean()) > 1e-3


def test_bulk_reduces_to_the_standard_nll_for_a_model_confined_to_the_region(models):
    """Model with no mass outside A (z-bins from the one containing z_cut, and the last ring, set to zero), back
    faces beyond z_cut, hits whose cells lie inside A: same points, same density, M_A = M_e = 1, weights unchanged."""
    data = make_showers(60, seed=3, zb="deep", z_range=(-150.0, 1080.0), r_max=600.0, tails=False)
    b = whole(data)
    k_cut = int((1150.0 - C.Z_MIN) // C.DZ)                                # 104: the bin that contains z_cut
    with torch.no_grad(), M.deterministic():
        for model in models.values():
            raw = model(b.lnp).clone()
            raw[:, k_cut:, :] = -math.inf
            raw[:, :, -1] = -math.inf
            logP = torch.log_softmax(raw.reshape(len(raw), -1), 1).reshape(raw.shape)
            std, _, ps = M.event_nll(logP, b, data.box, 64, gen_at(5), return_parts=True)
            blk, _, pb = M.event_nll(logP, b, data.box, 64, gen_at(5), bulk=True, return_parts=True)
            part = M.event_nll(logP, b, data.box, 64, gen_at(5), bulk=M.Bulk(renormalise=False))[0]
            assert torch.isfinite(std).all() and torch.isfinite(blk).all()
            assert torch.allclose(blk, std, rtol=0, atol=1e-4) and torch.allclose(part, std, rtol=0, atol=1e-4)
            assert torch.allclose(pb.retained, torch.ones_like(pb.retained), atol=1e-5)
            assert torch.allclose(pb.log_mass, ps.log_mass, atol=1e-5) and float(pb.log_mass.abs().max()) < 1e-5
            assert torch.allclose(pb.hit, ps.hit, rtol=1e-5, atol=1e-7) and bool((pb.z_limit == 1150.0).all())
            mon = M.mass_monitors(logP, b)
            assert float(mon["outside_bulk"].abs().max()) == 0.0 and float(mon["beyond_zb"].abs().max()) == 0.0


def lattice_event(theta, phi, origin, z_hi, margin=40.0):
    """All cells of an unbounded detector lattice (pitch = the cell box, offset `origin`, global frame) that can
    reach the bulk region {Z_MIN <= z' < z_hi, r < R_LAST} of a shower with this direction: a tiling of it."""
    R = C.rotation(np.array([theta]), np.array([phi]))[0].astype(np.float64)       # global -> shower frame
    ax = [np.arange(-math.ceil(1700.0 / s), math.ceil(1700.0 / s) + 1) * s + o for s, o in zip(BOX, origin)]
    loc = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) @ R.T
    keep = (np.hypot(loc[:, 0], loc[:, 1]) < R_LAST + margin) & (loc[:, 2] > C.Z_MIN - margin) & (loc[:, 2] < z_hi + margin)
    return loc[keep].astype(np.float32), R.astype(np.float32)


def test_bulk_likelihood_is_normalised_over_a_tiling():
    """The defining property of the truncated likelihood: summed over detector cells that tile the region, the
    cell probabilities V * gA_i / M_A add up to 1 - for a tilted shower, and for one whose back face (800 mm) lies
    inside the nominal region (short="clip"). gA is unbiased, so the sum is an unbiased Monte-Carlo estimate of 1;
    with stratified K = 64 its sd over seeds is 0.3-0.4 % (measured), the mean of 6 seeds 0.15 %; bound 1 %.
    The same sum for the standard likelihood is not 1 (flat back-face cut, clamped look-ups)."""
    events = [(0.5, 0.7, 1400.0, (7.0, -11.0, 3.0)), (0.3, -2.0, 800.0, (20.0, 5.0, -6.0))]
    xyz, Rs, cnt = [], [], []
    for theta, phi, zb, origin in events:
        x, R = lattice_event(theta, phi, origin, min(zb, 1150.0))
        xyz.append(x), Rs.append(R), cnt.append(len(x))
    ev = {"p": torch.tensor([2.0, 12.0]), "bin": torch.zeros(2, dtype=torch.long),
          "zb": torch.tensor([e[2] for e in events]), "R": torch.tensor(np.stack(Rs)), "evis": torch.ones(2)}
    ev["lnp"] = torch.log(ev["p"])
    w = np.concatenate([np.full(c, 1.0 / c, dtype=np.float32) for c in cnt])
    data = C.Showers(ev, torch.tensor(np.concatenate([[0], np.cumsum(cnt)])), torch.tensor(np.concatenate(xyz)),
                     torch.tensor(w), torch.tensor(BOX, dtype=torch.float32)).to(DEV)
    b = whole(data)
    V = float(np.prod(BOX))
    sums, sums_std = [], []
    with torch.inference_mode():
        logP = sharp_map(b.lnp)
        for seed in range(6):
            _, _, p = M.event_nll(logP, b, data.box, 64, gen_at(seed), fold="stratified", chunk=16,
                                  bulk=M.Bulk(renormalise=False), return_parts=True)
            gA = torch.where(p.weight > 0, (p.hit / p.weight.clamp_min(1e-30)).double().exp(), torch.zeros((), dtype=torch.float64, device=DEV))
            sums.append(seg_sum(gA * V, b.ev, 2) / p.log_mass.double().exp())
            _, _, q = M.event_nll(logP, b, data.box, 64, gen_at(seed), fold="stratified", chunk=16, return_parts=True)
            sums_std.append(seg_sum((q.hit / q.weight).double().exp() * V, b.ev, 2) / q.log_mass.double().exp())
        assert torch.equal(p.z_limit, torch.tensor([1150.0, 800.0], device=DEV))
        mass = M.mass_monitors(logP, b)
    total = torch.stack(sums).mean(0)
    assert torch.allclose(total, torch.ones(2, dtype=torch.float64, device=DEV), atol=0.01), total
    assert float(p.log_mass.exp()[1]) == pytest.approx(float(1 - mass["outside_bulk"][1]), rel=1e-5)
    assert float(torch.stack(sums_std).mean(0)[1]) > 1.02                  # short event, standard metric: over-counted


def test_bulk_events_without_retained_energy_and_short_events():
    """Event 0: every hit beyond z_cut. Event 1: every point in the last ring. Event 2: ordinary. Event 3: back
    face at 600 mm (inside the nominal region) with hits on both sides of it."""
    rng = np.random.default_rng(0)
    n = [40, 40, 80, 80]
    z = np.concatenate([rng.uniform(1250, 1400, 40), rng.uniform(0, 800, 40), rng.uniform(0, 800, 80), rng.uniform(300, 900, 80)])
    r = np.concatenate([rng.uniform(0, 100, 40), rng.uniform(699.0, 699.9, 40), rng.uniform(0, 200, 80), rng.uniform(0, 200, 80)])
    a = rng.uniform(0, 2 * np.pi, len(z))
    owner = np.repeat(np.arange(4), n)
    theta, phi = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, 2.0, 3.0])
    ev = {"p": torch.tensor([1.0, 2.0, 5.0, 9.0]), "bin": torch.arange(4), "zb": torch.tensor([1300.0, 1300.0, 1300.0, 600.0]),
          "R": torch.tensor(C.rotation(theta, phi)), "evis": torch.ones(4)}
    ev["lnp"] = torch.log(ev["p"])
    w = (1.0 / np.array(n))[owner].astype(np.float32)
    data = C.Showers(ev, torch.tensor(np.concatenate([[0], np.cumsum(n)])),
                     torch.tensor(np.stack([r * np.cos(a), r * np.sin(a), z], 1).astype(np.float32)), torch.tensor(w),
                     torch.tensor(BOX, dtype=torch.float32)).to(DEV)
    b = whole(data)
    with torch.no_grad():
        logP = sharp_map(b.lnp)
        nll, _, p = M.event_nll(logP, b, data.box, 64, gen_at(0), bulk=True, return_parts=True)
        assert torch.isnan(nll[:2]).all() and torch.isfinite(nll[2:]).all()        # undefined, not silently 0
        assert bool((p.retained[:2] == 0).all()) and float(p.retained[2]) == pytest.approx(1.0, abs=1e-5)
        assert 0.2 < float(p.retained[3]) < 0.8                                     # only the part in front of zb
        assert torch.equal(p.z_limit, torch.tensor([1150.0, 1150.0, 1150.0, 600.0], device=DEV))
        assert torch.isfinite(p.hit).all() and bool((p.hit[:80] == 0).all()) and bool((p.weight[:80] == 0).all())
        part, _, q = M.event_nll(logP, b, data.box, 64, gen_at(0), bulk=M.Bulk(renormalise=False), return_parts=True)
        assert bool((part[:2] == 0).all()) and torch.isfinite(part).all()
        assert torch.allclose(part[2:], (p.retained * nll)[2:], rtol=1e-5, atol=1e-5)   # bulk part = W_e * bulk NLL
        assert torch.allclose(q.weight_sum, p.retained, atol=1e-6)
        # the hits of event 3 behind its back face carry no weight; M_A is the mass in front of zb in the kept rings
        behind = (b.ev == 3) & (b.xyz[:, 2] > 600 + 40)
        assert behind.any() and bool((p.weight[behind] == 0).all())
        mon = M.mass_monitors(logP, b)
        assert torch.allclose(p.log_mass.double().exp(), 1 - mon["outside_bulk"], rtol=1e-5)
        with pytest.raises(ValueError, match="zb < z_cut"):
            M.event_nll(logP, b, data.box, 64, gen_at(0), bulk=M.Bulk(short="raise"))
        ok = data.subset([0, 1, 2])
        M.event_nll(logP[:3], whole(ok), ok.box, 64, gen_at(0), bulk=M.Bulk(short="raise"))   # no short event: fine
        # test_nll keeps the NaN markers; the caller masks them
        out = M.test_nll(sharp_map, data, bulk=True)
        assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()
    for bad in ({"drop_rings": C.N_R}, {"short": "ignore"}, {"z_cut": 2000.0}):
        with pytest.raises(ValueError):
            M.Bulk(**bad)
    with pytest.raises(TypeError):
        M.event_nll(logP, b, data.box, 64, gen_at(0), bulk=1150.0)


def test_bulk_uses_the_same_points_as_the_standard_metric(data, models):
    """Options never change the random stream: after any variant the generator is where common leaves it."""
    b = next(iter(data.batches(32)))
    with torch.no_grad():
        ref = gen_at(8)
        C.event_nll(models["mlp"], b, data.box, 64, ref)
        expect = torch.rand(5, device=DEV, generator=ref)
        for kw in ({"bulk": True}, {"bulk": True, "chunk": 9}, {"fold": "stratified", "chunk": 9}, {"return_parts": True}):
            g = gen_at(8)
            M.event_nll(models["mlp"], b, data.box, 64, g, **kw)
            assert torch.equal(torch.rand(5, device=DEV, generator=g), expect), kw


# ----------------------------------------------------------------------------------------------------------------------
# Mass monitors
# ----------------------------------------------------------------------------------------------------------------------
def test_mass_monitors_known_values():
    B = 4
    logP = torch.full((B, C.N_Z, C.N_R), -math.log(C.N_Z * C.N_R), device=DEV)        # uniform over the cells
    zb = torch.tensor([C.Z_MIN + 64.5 * C.DZ, 2000.0, C.Z_MIN - 5.0, 1150.0], device=DEV)
    mask = torch.zeros(C.N_Z, C.N_R, dtype=torch.bool, device=DEV)
    mask[3, 5] = mask[100, 31] = True
    mon = M.mass_monitors(logP, zb=zb, masks={"two": mask})
    assert all(v.dtype == torch.float64 and v.shape == (B,) for v in mon.values())
    one = torch.ones(B, dtype=torch.float64, device=DEV)
    assert torch.allclose(mon["total"], one, atol=1e-6)
    assert torch.allclose(mon["last_ring"], one / C.N_R, rtol=1e-5)
    assert torch.allclose(mon["first_z"], one / C.N_Z, rtol=1e-5) and torch.allclose(mon["last_z"], one / C.N_Z, rtol=1e-5)
    assert torch.allclose(mon["mask:two"], 2 * one / (C.N_Z * C.N_R), rtol=1e-5)
    inside = torch.tensor([64.5 / C.N_Z, 1.0, 0.0, (1150.0 - C.Z_MIN) / C.DZ / C.N_Z], dtype=torch.float64, device=DEV)
    assert torch.allclose(mon["inside_zb"], inside, rtol=1e-5, atol=1e-9)
    assert torch.allclose(mon["beyond_zb"], 1 - inside, rtol=1e-5, atol=1e-9)
    assert float(mon["beyond_zb"][1]) == 0.0 and float(mon["inside_zb"][2]) == 0.0
    # bulk region: z' < min(1150, zb), rings 0..30
    frac_r = (C.N_R - 1) / C.N_R
    z_in = torch.tensor([64.5, (1150.0 - C.Z_MIN) / C.DZ, 0.0, (1150.0 - C.Z_MIN) / C.DZ], dtype=torch.float64, device=DEV) / C.N_Z
    assert torch.allclose(mon["outside_bulk"], 1 - frac_r * z_in, rtol=1e-5)
    nozb = M.mass_monitors(logP)
    assert "beyond_zb" not in nozb and torch.allclose(nozb["outside_bulk"], 1 - frac_r * z_in[1:2], rtol=1e-5)
    with pytest.raises(ValueError):
        M.mass_monitors(logP, zb=zb, masks={"bad": mask[:5]})


def test_mass_monitors_resolve_tiny_masses_and_match_the_metric(data, models):
    # 1e-12 of the mass in the last z-bin: reported as such, not lost in 1 - (1 - 1e-12)
    raw = torch.zeros(1, C.N_Z, C.N_R, device=DEV)
    raw[:, -1, :] = math.log(1e-12 * (C.N_Z - 1))
    logP = torch.log_softmax(raw.reshape(1, -1), 1).reshape(raw.shape)
    mon = M.mass_monitors(logP, zb=C.Z_EDGES[-2].reshape(1))
    assert float(mon["beyond_zb"]) == pytest.approx(1e-12, rel=1e-3) and float(mon["last_z"]) == pytest.approx(1e-12, rel=1e-3)
    # inside_zb is the normaliser M_e of the metric
    b = whole(data)
    with torch.no_grad():
        logP = models["formula"](b.lnp)
        _, _, parts = M.event_nll(logP, b, data.box, 16, gen_at(0), return_parts=True)
        mon = M.mass_monitors(logP, b)
    assert torch.allclose(parts.log_mass.double().exp(), mon["inside_zb"].clamp_min(1e-8), rtol=2e-5, atol=1e-9)
    assert torch.allclose(mon["inside_zb"] + mon["beyond_zb"], mon["total"], atol=1e-9)
    assert bool((mon["beyond_zb"] >= 0).all()) and bool((mon["outside_bulk"] >= mon["last_ring"] - 1e-12).all())


# ----------------------------------------------------------------------------------------------------------------------
# (8) real data: validation split only, skipped without the cache
# ----------------------------------------------------------------------------------------------------------------------
TRAINVAL_CACHE = paths.CACHE_DIR / "pion_s1_trainval.pt"


@pytest.mark.skipif(not TRAINVAL_CACHE.exists(), reason=f"validation cache {TRAINVAL_CACHE} not built")
def test_real_data_validation_equals_common_test_nll():
    """Gate-0 check: at default settings sfm.metric returns the numbers of common.test_nll on the validation split,
    bit for bit inside deterministic(), for a random MLP, a random formula and (if the adapter loads) mlp_std and the
    component formula (``legacy.load_component_formula``; its ln p centring comes from the training split, which is
    loaded on the CPU for that and dropped again).
    Opens only the train/val cache file (``load_sets(("val",))``); the test split is never touched."""
    from sfm import data as sfm_data

    sets, _ = sfm_data.load_sets(("val",))
    val = sets["val"]
    torch.manual_seed(0)
    cands = {"mlp_random": C.FieldModel(16, 2).to(DEV).eval(),
             "formula_random": C.Formula("gamma", "core_tail", 1).to(DEV).eval()}
    try:
        from sfm import legacy
        cands["mlp_std"] = legacy.load_mlp_std()
        tr, _ = sfm_data.load_sets(("train",), device="cpu")
        cands["component_formula"] = legacy.load_component_formula(tr["train"])
        del tr
    except Exception as err:                                               # the adapter is optional for this check
        print(f"benchmarks not loaded ({type(err).__name__}: {err}); random models only")
    with M.deterministic():
        ref = {k: C.test_nll(m, val) for k, m in cands.items()}
        for k, m in cands.items():
            new = M.test_nll(m, val)
            assert new.shape == (len(val),) and np.array_equal(new, ref[k]), k
        multi = M.test_nll_multi(list(cands.values()), val)
        for j, k in enumerate(cands):
            assert np.array_equal(multi[j], ref[k]), k
    for k, m in cands.items():                                             # default mode: equal up to index_add_ order
        assert (np.abs(M.test_nll(m, val) - ref[k]) / np.maximum(1.0, np.abs(ref[k]))).max() <= 1e-5, k
    for k in cands:
        print(f"{k}: validation NLL {ref[k].astype(np.float64).mean():.6f} ({len(val)} events)")
    # information for the gate (shown with -s / -rP): how the bulk region meets the data
    zb = val.ev["zb"]
    out = M.test_nll(next(iter(cands.values())), val, bulk=True, return_parts=True)
    print(f"events with zb < 1150 mm: {int((zb < 1150).sum())} of {len(val)} (min zb {float(zb.min()):.1f} mm); "
          f"events without bulk energy (NaN): {int(np.isnan(out['nll'][0]).sum())}; "
          f"retained energy fraction: mean {out['retained'].mean():.4f}, min {out['retained'].min():.4f}")
    assert np.array_equal(np.isnan(out["nll"][0]), out["retained"] == 0)
