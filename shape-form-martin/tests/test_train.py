"""Tests of sfm.train (the resumable fit), the run-config loader and sfm.gpulock.

CPU checks. ``common.py`` builds its grid tensors on ``common.DEVICE`` (CUDA when available), so a model cannot be
trained on the CPU inside a process that sees a GPU. The CPU checks therefore run once in a child process with
``CUDA_VISIBLE_DEVICES=""`` (``_cpu_suite`` below, executed by the module fixture ``cpu_suite``) and the individual
tests assert on its JSON report: loss decreases, resume equivalence (bit for bit), skip counting, EMA, best-val
selection, the done-marker and the config-mismatch guard.

CUDA checks (skipped without a GPU; tiny synthetic data, < 100 MB of GPU memory, seconds): resume equivalence
inside ``deterministic=True`` (bitwise) and outside it (within the accumulation-order noise of ``index_add_``),
``val_nll`` = ``common.test_nll``, and equivalence with ``shapelab.fit_fast``.
"""
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
from sfm import metric, train

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent
BOX = (48.6, 48.6, 20.5)


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic showers and a tiny model
# ----------------------------------------------------------------------------------------------------------------------
def make_showers(n_ev, seed, device, hits=(20, 80)):
    """Random events whose shape depends on ln p (depth grows, width shrinks with p), front-face back planes."""
    rng = np.random.default_rng(seed)
    cnt = rng.integers(hits[0], hits[1], n_ev)
    H = int(cnt.sum())
    theta, phi = rng.uniform(0.05, 0.6, n_ev), rng.uniform(-np.pi, np.pi, n_ev)
    p = np.exp(rng.uniform(np.log(0.3), np.log(20.0), n_ev)).astype(np.float32)
    lnp = np.log(p)
    owner = np.repeat(np.arange(n_ev), cnt)
    z = rng.gamma(2.0, 80.0 + 25.0 * lnp[owner]) - 30.0
    r = rng.exponential(25.0 + 8.0 * (3.0 - lnp[owner]))
    z, r = np.clip(z, -199.0, 1449.0), np.clip(r, 0.0, 699.0)
    a = rng.uniform(0, 2 * np.pi, H)
    xyz = np.stack([r * np.cos(a), r * np.sin(a), z], 1).astype(np.float32)
    e = rng.exponential(1.0, H)
    w = (e / np.bincount(owner, weights=e)[owner]).astype(np.float32)
    ev = {"p": torch.tensor(p), "bin": torch.tensor(np.floor(p).astype(np.int64)),
          "zb": torch.tensor(1221.0 / np.cos(theta), dtype=torch.float32), "R": torch.tensor(C.rotation(theta, phi)),
          "evis": torch.ones(n_ev)}
    ev["lnp"] = torch.log(ev["p"])
    return C.Showers(ev, torch.tensor(np.concatenate([[0], np.cumsum(cnt)])), torch.tensor(xyz), torch.tensor(w),
                     torch.tensor(BOX, dtype=torch.float32)).to(device)


class _NanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * float("nan")


class Tiny(nn.Module):
    """logits = a + b * ln p / 3 per cell (zero-initialised: the uniform map). Training-mode forward calls listed
    in ``bad_loss`` return NaN, those in ``bad_grad`` a finite map with NaN gradients. The call counter is a buffer,
    so it is checkpointed and restored like everything else."""

    def __init__(self, bad_loss=(), bad_grad=()):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(C.N_Z, C.N_R))
        self.b = nn.Parameter(torch.zeros(C.N_Z, C.N_R))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        self.bad_loss, self.bad_grad = set(bad_loss), set(bad_grad)

    def forward(self, lnp):
        z = self.a + self.b * (lnp / 3)[:, None, None]
        if self.training:
            self.calls += 1
            c = int(self.calls)
            if c in self.bad_loss:
                z = z + float("nan")
            if c in self.bad_grad:
                z = _NanGrad.apply(z)
        return F.log_softmax(z.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)

    def n_params(self):
        return self.a.numel() + self.b.numel()


def states_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def no_time(history):
    """History without the wall-clock field."""
    return [{k: v for k, v in h.items() if k != "seconds"} for h in history]


def max_state_diff(a: dict, b: dict) -> float:
    return max(float((a[k].double().cpu() - b[k].double().cpu()).abs().max()) for k in a)


class Interrupt(Exception):
    pass


def interrupt_at(step_to_stop):
    def cb(step, model):
        if step == step_to_stop:
            raise Interrupt(step)
    return cb


def run_pair(tmp, device, cfg_kw, stop, model_fn, deterministic=False):
    """Uninterrupted run vs a run interrupted after step ``stop`` and resumed. Returns (resA, resB, objA, objB)."""
    data_tr, data_va = make_showers(160, 1, device), make_showers(60, 2, device)
    base = dict(cfg_kw, deterministic=deterministic)
    torch.manual_seed(0)
    m0 = model_fn().to(device)
    mA, mB = copy.deepcopy(m0), copy.deepcopy(m0)
    cA = train.FitConfig(**base, name="pair", out_dir=str(tmp / "A"))
    cB = train.FitConfig(**base, name="pair", out_dir=str(tmp / "B"))
    rA = train.fit(mA, data_tr, data_va, cA, log=None)
    with pytest.raises(Interrupt):
        train.fit(mB, data_tr, data_va, cB, log=None, callback=interrupt_at(stop))
    assert (tmp / "B" / "checkpoint.pt").exists() and not (tmp / "B" / "fit_done.json").exists()
    mB2 = copy.deepcopy(m0)                                          # a fresh process would rebuild the model
    rB = train.fit(mB2, data_tr, data_va, cB, log=None)
    oA = torch.load(tmp / "A" / "model.pt", weights_only=False)
    oB = torch.load(tmp / "B" / "model.pt", weights_only=False)
    return rA, rB, oA, oB, (mA, mB2)


# ----------------------------------------------------------------------------------------------------------------------
# The CPU suite (runs in a child process without CUDA)
# ----------------------------------------------------------------------------------------------------------------------
def _cpu_suite(out_json: str, tmp: str) -> None:
    assert C.DEVICE.type == "cpu", "the CPU suite must run without CUDA"
    tmp = Path(tmp)
    dev = torch.device("cpu")
    rep = {}

    # loss decreases ------------------------------------------------------------------------------------------------
    tr, va = make_showers(300, 1, dev), make_showers(80, 2, dev)
    m = Tiny()
    v0 = float(np.mean(train.val_nll(m, va, 64), dtype=np.float64))
    res = train.fit(m, tr, va, train.FitConfig(steps=60, lr=0.05, lr_end=0.005, batch=32, eval_every=20, k_val=64,
                                               name="dec", out_dir=str(tmp / "dec")), log=None)
    rep["decrease"] = {"val0": v0, "history": res.history, "final_val": res.final_val,
                       "val_after": float(np.mean(train.val_nll(m, va, 64), dtype=np.float64)),
                       "training_mode_after": m.training}

    # resume equivalence, with skips and EMA ---------------------------------------------------------------------------
    kw = dict(steps=40, lr=0.05, lr_end=0.005, batch=32, eval_every=10, ckpt_every=10, k_val=16, ema_decay=0.9,
              selection="last")
    rA, rB, oA, oB, (mA, mB) = run_pair(tmp / "resume", dev, kw, stop=27,
                                         model_fn=lambda: Tiny(bad_loss={4, 33}, bad_grad={12}))
    rep["resume"] = {"state_equal": states_equal(oA["state"], oB["state"]),
                     "ema_equal": states_equal(oA["ema_state"], oB["ema_state"]),
                     "model_equal": states_equal(mA.state_dict(), mB.state_dict()),
                     "history_equal": no_time(oA["history"]) == no_time(oB["history"]),
                     "skips_A": [rA.skipped_loss, rA.skipped_grad], "skips_B": [rB.skipped_loss, rB.skipped_grad],
                     "resumed_from": rB.resumed_from, "max_diff": max_state_diff(oA["state"], oB["state"])}

    # skip counting --------------------------------------------------------------------------------------------------
    m = Tiny(bad_loss={2, 5}, bad_grad={8})
    res = train.fit(m, tr, va, train.FitConfig(steps=12, lr=0.05, batch=32, eval_every=6, k_val=16, name="skip",
                                               out_dir=str(tmp / "skip")), log=None)
    done = json.loads((tmp / "skip" / "fit_done.json").read_text())
    logtxt = (tmp / "skip" / "fit.log").read_text()
    rep["skip"] = {"skipped_loss": res.skipped_loss, "skipped_grad": res.skipped_grad,
                   "done": [done["skipped_loss"], done["skipped_grad"]],
                   "history": [[h["skipped_loss"], h["skipped_grad"], h["n_train_steps"]] for h in res.history],
                   "log_mentions": logtxt.count("batch skipped"), "finite": bool(torch.isfinite(m.a).all())}

    # EMA against a manual EMA of the iterates --------------------------------------------------------------------------
    torch.manual_seed(0)
    m = Tiny()
    init = {k: p.detach().clone() for k, p in m.named_parameters()}
    manual = {k: v.clone() for k, v in init.items()}
    d = 0.8

    def track(step, model):
        for k, p in model.named_parameters():
            manual[k].lerp_(p.detach(), 1 - d)

    res = train.fit(m, tr, va, train.FitConfig(steps=15, lr=0.05, batch=32, eval_every=5, k_val=16, ema_decay=d,
                                               name="ema", out_dir=str(tmp / "ema")), log=None, callback=track)
    obj = torch.load(tmp / "ema" / "model.pt", weights_only=False)
    me = Tiny()
    me.load_state_dict(obj["ema_state"])
    rep["ema"] = {"max_diff_manual": max(float((obj["ema_state"][k] - manual[k]).abs().max()) for k in manual),
                  "ema_differs_from_last": not states_equal({k: obj["ema_state"][k] for k in manual},
                                                            {k: obj["state"][k] for k in manual}),
                  "val_ema_hist": res.final_val_ema,
                  "val_ema_recomputed": float(np.mean(train.val_nll(me, va, 16), dtype=np.float64)),
                  "has_ema": res.has_ema}

    # best-val selection ------------------------------------------------------------------------------------------------
    m = Tiny()
    res = train.fit(m, tr, va, train.FitConfig(steps=30, lr=0.3, lr_end=0.3, batch=32, eval_every=5, k_val=16,
                                               selection="best_val", name="best", out_dir=str(tmp / "best")),
                    log=None)
    vals = [h["val"] for h in res.history]
    rep["best"] = {"vals": vals, "selected_step": res.selected_step, "selected_val": res.selected_val,
                   "steps": [h["step"] for h in res.history],
                   "val_loaded": float(np.mean(train.val_nll(m, va, 16), dtype=np.float64))}

    # done marker: a second call loads, does not train; a different config is refused ----------------------------------
    calls = []
    m2 = Tiny()
    res2 = train.fit(m2, tr, va, train.FitConfig(steps=30, lr=0.3, lr_end=0.3, batch=32, eval_every=5, k_val=16,
                                                 selection="best_val", name="best", out_dir=str(tmp / "best")),
                     log=None, callback=lambda s, mm: calls.append(s))
    mismatch = None
    try:
        train.fit(Tiny(), tr, va, train.FitConfig(steps=30, lr=0.2, batch=32, name="best", out_dir=str(tmp / "best")),
                  log=None)
    except train.ConfigMismatch as err:
        mismatch = str(err)
    rep["done"] = {"callback_calls": len(calls), "same_weights": states_equal(m.state_dict(), m2.state_dict()),
                   "history_equal": res2.history == res.history, "mismatch": mismatch}
    Path(out_json).write_text(json.dumps(rep))


@pytest.fixture(scope="module")
def cpu_suite(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cpu_suite")
    out = tmp / "report.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    code = (f"import sys; sys.dont_write_bytecode = True; sys.path[:0] = [{str(MARTIN / 'src')!r}, {str(MARTIN)!r}, "
            f"{str(HERE)!r}]; import test_train as T; T._cpu_suite({str(out)!r}, {str(tmp)!r})")
    proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    return json.loads(out.read_text())


def test_cpu_loss_decreases(cpu_suite):
    r = cpu_suite["decrease"]
    h = r["history"]
    assert r["final_val"] < r["val0"] - 0.05, r                       # validation NLL falls clearly from the uniform map
    assert h[-1]["train_nll"] < h[0]["train_nll"]
    assert r["val_after"] == pytest.approx(r["final_val"], abs=1e-9)  # "last" selection: the model is the last iterate
    assert r["training_mode_after"] is False                          # left in eval mode


def test_cpu_resume_is_bitwise_identical(cpu_suite):
    r = cpu_suite["resume"]
    assert r["resumed_from"] == 20                                    # last checkpoint before the interrupt at step 27
    assert r["state_equal"] and r["ema_equal"] and r["model_equal"], r
    assert r["max_diff"] == 0.0
    assert r["history_equal"]
    assert r["skips_A"] == r["skips_B"] == [2, 1]                    # skip counters survive the resume


def test_cpu_skip_counting(cpu_suite):
    r = cpu_suite["skip"]
    assert (r["skipped_loss"], r["skipped_grad"]) == (2, 1)
    assert r["done"] == [2, 1]
    # cumulative counters per validation line; finite steps per window: 6 - 2 (loss at 2, 5), 6 - 1 (grad at 8)
    assert r["history"] == [[2, 0, 4], [2, 1, 5]]
    assert r["log_mentions"] == 3 and r["finite"]


def test_cpu_ema(cpu_suite):
    r = cpu_suite["ema"]
    assert r["has_ema"] and r["ema_differs_from_last"]
    assert r["max_diff_manual"] < 1e-6, r                            # EMA of the iterates, decay 0.8, from the start
    assert r["val_ema_recomputed"] == pytest.approx(r["val_ema_hist"], abs=1e-9)


def test_cpu_best_val_selection(cpu_suite):
    r = cpu_suite["best"]
    i = int(np.argmin(r["vals"]))
    assert r["selected_step"] == r["steps"][i] and r["selected_val"] == r["vals"][i]
    assert r["val_loaded"] == pytest.approx(r["selected_val"], abs=1e-9)


def test_cpu_done_marker_and_config_guard(cpu_suite):
    r = cpu_suite["done"]
    assert r["callback_calls"] == 0 and r["same_weights"] and r["history_equal"]
    assert r["mismatch"] and "config" in r["mismatch"]


# ----------------------------------------------------------------------------------------------------------------------
# CUDA (tiny)
# ----------------------------------------------------------------------------------------------------------------------
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@cuda
def test_cuda_resume_bitwise_inside_deterministic(tmp_path):
    kw = dict(steps=30, lr=0.05, lr_end=0.005, batch=32, eval_every=10, ckpt_every=10, k_val=16, ema_decay=0.9)
    rA, rB, oA, oB, _ = run_pair(tmp_path, C.DEVICE, kw, stop=17, model_fn=Tiny, deterministic=True)
    assert rB.resumed_from == 10
    assert states_equal(oA["state"], oB["state"]) and states_equal(oA["ema_state"], oB["ema_state"])
    assert no_time(oA["history"]) == no_time(oB["history"])


@cuda
def test_cuda_resume_default_mode_within_index_add_noise(tmp_path):
    """Without deterministic(): index_add_ (forward) and the scatter of the gather's backward accumulate in an
    unspecified order on CUDA, so resumed and uninterrupted runs agree only to float32 rounding amplified by 30
    Adam steps (measured ~1e-6 relative); bound 1e-4 absolute on weights of order 0.1-1."""
    kw = dict(steps=30, lr=0.05, lr_end=0.005, batch=32, eval_every=10, ckpt_every=10, k_val=16)
    rA, rB, oA, oB, _ = run_pair(tmp_path, C.DEVICE, kw, stop=17, model_fn=Tiny)
    d = max_state_diff(oA["state"], oB["state"])
    print(f"max |weight difference| resumed vs uninterrupted, CUDA default mode: {d:.2e}")
    assert d < 1e-4
    assert abs(oA["history"][-1]["val"] - oB["history"][-1]["val"]) < 1e-4


@cuda
def test_val_nll_equals_common_test_nll():
    data = make_showers(70, 5, C.DEVICE)
    torch.manual_seed(1)
    m = Tiny().to(C.DEVICE)
    with torch.no_grad():
        m.a.normal_(0, 0.5)
        m.b.normal_(0, 0.5)
    with metric.deterministic():
        ref64, ref16 = C.test_nll(m, data), C.test_nll(m, data, 16)
    assert np.array_equal(train.val_nll(m, data), ref64)
    assert np.array_equal(train.val_nll(m, data, 16), ref16)
    assert np.array_equal(train.val_nll(m, data, 64, seed=12345, batch=128), metric.test_nll(m, data, deterministic_sum=True))


@cuda
def test_matches_fit_fast(tmp_path):
    """With fit_fast's settings (K_train = K_val = 16, best-val selection, global fold generator, same seed and RNG
    state) fit reaches the same weights as shapelab.fit_fast (inside deterministic(): bit for bit)."""
    from sfm import legacy
    import shapelab

    tr, va = make_showers(150, 1, C.DEVICE), make_showers(50, 2, C.DEVICE)
    CC = dict(legacy.C)
    CC["BATCH"] = 32
    torch.manual_seed(3)
    m0 = Tiny().to(C.DEVICE)
    m_ff, m_us = copy.deepcopy(m0), copy.deepcopy(m0)
    steps, lr, lr_end, every, seed = 24, 0.05, 0.005, 6, 4
    with metric.deterministic():
        train.seed_everything(seed)
        hist = shapelab.fit_fast(CC, {"train": tr, "val": va}, m_ff, steps=steps, lr=lr, lr_end=lr_end,
                                 eval_every=every, seed=seed, log=None)
    cfg = train.FitConfig(steps=steps, lr=lr, lr_end=lr_end, eval_every=every, seed=seed, batch=32, k_train=16,
                          k_val=16, selection="best_val", fold_rng="global", deterministic=True, name="ff",
                          out_dir=str(tmp_path / "ff"))
    res = train.fit(m_us, tr, va, cfg, log=None)
    assert [s for s, _ in hist] == [h["step"] for h in res.history]
    np.testing.assert_allclose([v for _, v in hist], [h["val"] for h in res.history], rtol=0, atol=2e-6)
    assert states_equal(m_ff.state_dict(), m_us.state_dict()), max_state_diff(m_ff.state_dict(), m_us.state_dict())


# ----------------------------------------------------------------------------------------------------------------------
# Config loading, validation, GPU lock
# ----------------------------------------------------------------------------------------------------------------------
def test_fitconfig_validation():
    train.FitConfig().validate()
    for bad in ({"steps": 0}, {"fold": "sobol"}, {"selection": "best"}, {"ema_decay": 1.0}, {"lr": 0.0},
                {"lr_end": 1.0, "lr": 0.1}, {"name": "a/b"}, {"steps": 1.5}, {"fold_rng": "other"}):
        with pytest.raises((ValueError, TypeError)):
            train.FitConfig(**bad).validate()
    with pytest.raises(KeyError):
        train.FitConfig.from_dict({"stepz": 3})
    a, b = train.FitConfig(out_dir="/x", ckpt_every=7), train.FitConfig(out_dir="/y", ckpt_every=9)
    assert a.hash() == b.hash()                                       # out_dir and ckpt_every do not change results
    assert a.hash() != train.FitConfig(lr=0.01).hash()


def test_load_run_config(tmp_path):
    good = tmp_path / "run.toml"
    good.write_text('[run]\nname = "b0_seed1"\nstage = "m2b_mlp"\nseed = 1\nnotes = "x"\n'
                    '[model]\nfactory = "field"\nhidden = 8\nn_layers = 1\n'
                    '[fit]\nsteps = 10\nlr = 1\nlr_end = 0.5\nema_decay = 0.999\n'
                    '[evaluate]\nguards = true\nC = 13\n')
    rc = train.load_run_config(good)
    assert (rc.name, rc.stage, rc.seed, rc.model_factory) == ("b0_seed1", "m2b_mlp", 1, "field")
    assert rc.model_kwargs == {"hidden": 8, "n_layers": 1}
    assert rc.fit.seed == 1 and rc.fit.name == "b0_seed1" and rc.fit.lr == 1.0 and isinstance(rc.fit.lr, float)
    assert Path(rc.fit.out_dir) == rc.out_dir and rc.out_dir.parts[-2:] == ("m2b_mlp", "b0_seed1")
    assert rc.evaluate.guards and rc.evaluate.C == 13.0 and rc.evaluate.registry
    h = rc.hash()
    good.write_text(good.read_text().replace('notes = "x"', 'notes = "y"'))
    assert train.load_run_config(good).hash() == h                   # notes do not change the hash
    for text in ('[run]\nname = "a"\nstage = "s"\n[model]\nfactory = "f"\n[fit]\nseed = 2\n',
                 '[run]\nname = "a"\nstage = "s"\n[model]\nfactory = "f"\n[fit]\nstepz = 2\n',
                 '[run]\nname = "a"\nstage = "s"\n[model]\nfactory = "f"\n[other]\n',
                 '[run]\nname = "a/b"\nstage = "s"\n[model]\nfactory = "f"\n',
                 '[run]\nname = "a"\nstage = "s"\n[model]\nhidden = 3\n',
                 '[run]\nname = "a"\nstage = "s"\n[model]\nfactory = "f"\n[evaluate]\nguard = true\n'):
        bad = tmp_path / "bad.toml"
        bad.write_text(text)
        with pytest.raises((KeyError, ValueError, TypeError)):
            train.load_run_config(bad)


def test_configs_in_repo_load():
    """Every config shipped in configs/ parses and names a registered factory."""
    from sfm import models
    files = sorted((MARTIN / "configs").rglob("*.toml"))
    assert files, "no configs"
    for f in files:
        rc = train.load_run_config(f)
        assert rc.model_factory in models.FACTORIES, f


def test_gpu_lock(tmp_path):
    from sfm import gpulock
    path = tmp_path / "gpu.lock"
    lines = []
    with gpulock.gpu_lock("first job", path=path, log=lines.append):
        assert "first job" in gpulock.holder(path)
        with pytest.raises(TimeoutError):
            with gpulock.gpu_lock("second job", path=path, log=lines.append, timeout=0.3):
                pass
    assert any("waits" in s and "first job" in s for s in lines)
    with gpulock.gpu_lock("second job", path=path, log=lines.append, timeout=0.3):
        assert "second job" in gpulock.holder(path)
    assert gpulock.holder(path) == ""
    with pytest.raises(RuntimeError):                                 # released on exceptions
        with gpulock.gpu_lock("third", path=path, log=None):
            raise RuntimeError("boom")
    with gpulock.gpu_lock("fourth", path=path, log=None, timeout=0.3):
        pass
