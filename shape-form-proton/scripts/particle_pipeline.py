"""Egy részecske teljes, folytatható elemzése (minden lépés fájlba ment; újraindításkor a kész lépéseket kihagyja).

    python particle_pipeline.py <electron|pion|proton|muon>

Lépések (results/particles/<részecske>/):
  repro      a report baseline-ja és MLP-je újraszámolva vs. a report exportja (eseményenként)
  abar       a vágás eseményátlagolt sablonja Ā (tanító események, elöl + hátul)
  base_std   a report baseline-családja × Ā, fokszám-létra n = 1..6 (fit_fast, melegindítás, 1-SE a validáción)
  base_slab  ugyanez a 3D (szelet-) likelihooddal, Ā nélkül
  variants   az új formula változatai n = 2-n (report-mérce × Ā), választás a validáción (1-SE: a legkevesebb paraméter
             a legjobbtól 1 SE-n belül)
  new_std / new_slab   a kiválasztott változat fokszám-létrája mindkét mércén
  mlp_std / mlp_slab   a report MLP-je fit_fast-tal finomhangolva (erősebb háló-referencia)
  test       teszt-NLL csak a végleges modellekre (eseményenként, npz) + páros különbségek
  s2         ugyanez a független második mintán
  audit      normáltság és a szabad paraméterek száma perturbációval
A teszt- és az s2-halmazt semmilyen választás nem használja.
"""
import json
import math
import os
import sys
import time

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import protonlab
import shapelab
from particle_shape import MASS, ParticleShape

particle = sys.argv[1]
OUT = os.path.join(ROOT, "results", "particles", particle)
os.makedirs(OUT, exist_ok=True)
STEPS = 4000
NMAX = 6
T0 = time.time()


def log(msg):
    print(f"[{particle} {time.time() - T0:6.0f}s] {msg}", flush=True)


def jpath(name):
    return os.path.join(OUT, f"{name}.json")


def done(name):
    return os.path.exists(jpath(name))


def save_json(name, obj):
    with open(jpath(name) + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(jpath(name) + ".tmp", jpath(name))


def load_json(name):
    with open(jpath(name)) as fh:
        return json.load(fh)


def paired(a, b):
    return shapelab.paired(a, b)


# ---------------------------------------------------------------- setup (two namespaces: report metric and 3D metric)
C, geo, data = shapelab.setup_particle(particle, out_dir=OUT, slab=False)
CS, _, dataS = shapelab.setup_particle(particle, out_dir=OUT, slab=True)
dev = C["DEVICE"]
c0, s0 = shapelab.lnp_center_scale(data)
RM = shapelab.report_models(C, data, particle)
EX = shapelab.report_exports(particle)
log(f"data {[len(v) for v in data.values()]}, report baseline {RM['baseline_name']} n={RM['baseline_degree']}, MLP {RM['mlp_config']}")

# ---------------------------------------------------------------- repro
if not done("repro"):
    tb, tm = C["test_nll"](RM["baseline"], data["test"]), C["test_nll"](RM["mlp"], data["test"])
    eb, em = EX[RM["baseline_name"]]["test_nll"], EX["mlp"]["test_nll"]
    save_json("repro", {"baseline": RM["baseline_name"], "baseline_degree": RM["baseline_degree"], "mlp": RM["mlp_config"],
                        "baseline_test": float(tb.mean()), "baseline_export": float(eb.mean()), "baseline_max_abs_diff": float(np.abs(tb - eb).max()),
                        "mlp_test": float(tm.mean()), "mlp_export": float(em.mean()), "mlp_max_abs_diff": float(np.abs(tm - em).max()),
                        "report_exports": {k: {"test": float(v["test_nll"].mean()), "n_params": v["n_params"]} for k, v in EX.items()},
                        "n_events": {k: len(v) for k, v in data.items()}})
    log(f"repro: {load_json('repro')}")

# ---------------------------------------------------------------- abar
apath = os.path.join(OUT, "abar.pt")
if os.path.exists(apath):
    ABAR = torch.load(apath, map_location=dev)
else:
    ABAR = shapelab.mean_acceptance(C, data["train"], nsub=4, front=True, back=True)
    torch.save(ABAR, apath)


# ---------------------------------------------------------------- ladders
def ladder(name, CC, dd, make, init, nmax=NMAX):
    """Degree ladder n = 1..nmax with warm start; returns {n: model}, chosen n (1-SE on validation)."""
    if done(name):
        info = load_json(name)
    else:
        info = {"rungs": {}}
    models, prev, val = {}, None, {}
    for n in range(1, nmax + 1):
        core, m = make(n)
        ck = os.path.join(OUT, f"{name}_n{n}.pt")
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state"])
        else:
            init(core)
            if prev is not None:
                core.warm_start(prev)
            t = time.time()
            shapelab.fit_fast(CC, dd, m, steps=STEPS, log=None)
            torch.save({"state": m.state_dict()}, ck)
            log(f"{name} n={n}: {time.time() - t:.0f}s")
        m.eval()
        val[n] = CC["test_nll"](m, dd["val"])
        info["rungs"][str(n)] = {"params": int(m.n_params()), "val": float(val[n].mean())}
        models[n], prev = m, core
    best = min(val, key=lambda k: val[k].mean())
    diff = {n: paired(val[n], val[best]) for n in val}
    chosen = min(n for n in val if n == best or diff[n][0] < diff[n][1])
    info.update({"n_star": best, "chosen": chosen, "diff_vs_best": {str(n): diff[n] for n in diff}})
    save_json(name, info)
    return models, chosen


class BaseWrap(torch.nn.Module):
    """The report's Formula with warm_start / n_params helpers (for the ladder)."""

    def __init__(self, long, n):
        super().__init__()
        self.f = C["Formula"](long, "core_tail", degree=n, lnp0=c0, lnp_s=s0)

    def forward(self, lnp):
        return self.f(lnp)

    def n_params(self):
        return self.f.n_params()

    def warm_start(self, prev):
        with torch.no_grad():
            self.f.coef.zero_()
            self.f.coef[:, :prev.f.coef.shape[1]] = prev.f.coef
            self.f.slope.copy_(prev.f.slope)


LONG = RM["baseline_long"]
H = data["train"].hits()


def base_init(core):
    core.f.init_from(H.z.values, H.r.values, H.w.values)


def make_base(acc):
    def mk(n):
        core = BaseWrap(LONG, n).to(dev)
        return core, (shapelab.WithAcceptance(core, ABAR).to(dev) if acc else core)
    return mk


BASE_STD, nb_std = ladder("base_std", C, data, make_base(True), base_init)
log(f"base_std chosen n={nb_std}")
BASE_SLAB, nb_slab = ladder("base_slab", CS, dataS, make_base(False), base_init)
log(f"base_slab chosen n={nb_slab}")

# ---------------------------------------------------------------- variants (report metric x A_bar, degree 2)
VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
            "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
            "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}


def make_new(kw, n, acc, spkdeg=None):
    degrees = {"logit_tr": 2} if (n == 1 and kw.get("track")) else {}
    if spkdeg:
        degrees.update({k: min(n, v) for k, v in spkdeg.items()})
    core = ParticleShape(C, MASS[particle], lnp0=c0, lnp_s=s0, degree=n, degrees=degrees or None, **kw).to(dev)
    return core, (shapelab.WithAcceptance(core, ABAR).to(dev) if acc else core)


if done("variants"):
    VAR = load_json("variants")
else:
    VAR = {"val": {}, "params": {}}
vals = {}
for name, kw in VARIANTS.items():
    core, m = make_new(kw, 2, True)
    ck = os.path.join(OUT, f"variant_{name}.pt")
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state"])
    else:
        torch.manual_seed(0)
        core.init_from_data(C, data)
        t = time.time()
        shapelab.fit_fast(C, data, m, steps=STEPS, log=None)
        torch.save({"state": m.state_dict()}, ck)
        log(f"variant {name}: {time.time() - t:.0f}s")
    m.eval()
    vals[name] = C["test_nll"](m, data["val"])
    VAR["val"][name], VAR["params"][name] = float(vals[name].mean()), int(m.n_params())
best = min(vals, key=lambda k: vals[k].mean())
dif = {k: paired(vals[k], vals[best]) for k in vals}
ok = [k for k in vals if k == best or dif[k][0] < dif[k][1]]
CHOICE = min(ok, key=lambda k: (VAR["params"][k], VAR["val"][k]))
VAR.update({"best": best, "chosen": CHOICE, "diff_vs_best": dif})
save_json("variants", VAR)
log(f"variants: { {k: round(v, 4) for k, v in VAR['val'].items()} } -> chosen {CHOICE}")
KW = VARIANTS[CHOICE]
ROUND = int(os.environ.get("ROUND", "2" if os.environ.get("V2") else "1"))
TAG, SUF, SPKDEG = "new", "", None          # SPKDEG: extra per-parameter degrees (entry spike, lateral slopes)
if ROUND >= 2:                              # physics extensions chosen on validation (scripts/variants_v2.py, variants_v3.py)
    VJ = json.load(open(jpath(f"variants_v{ROUND}")))
    CHOICE = VJ["chosen"]
    if CHOICE in VJ.get("kwargs", {}):
        KW = {k: v for k, v in VJ["kwargs"][CHOICE].items() if k != "degrees"}
        SPKDEG = VJ["kwargs"][CHOICE].get("degrees") or None
    if "base_name" in VJ:
        CHOICE = VJ["base_name"] + ("" if CHOICE == "base" else "+" + CHOICE)
    TAG, SUF = f"new{ROUND}", f"_v{ROUND}"
    log(f"round {ROUND}: chosen {CHOICE} {KW} extra degrees {SPKDEG}")

NEW_STD, nn_std = ladder(f"{TAG}_std", C, data, lambda n: make_new(KW, n, True, SPKDEG), lambda core: core.init_from_data(C, data))
log(f"{TAG}_std chosen n={nn_std}")
NEW_SLAB, nn_slab = ladder(f"{TAG}_slab", CS, dataS, lambda n: make_new(KW, n, False, SPKDEG), lambda core: core.init_from_data(C, data))
log(f"{TAG}_slab chosen n={nn_slab}")


# ---------------------------------------------------------------- MLP references (fine-tuned from the report weights)
def mlp_ft(name, CC, dd):
    ck = os.path.join(OUT, f"{name}.pt")
    m = C["FieldModel"](RM["mlp"].mlp[0].out_features, (len(RM["mlp"].mlp) - 1) // 2).to(dev)
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state"])
    else:
        m.load_state_dict(RM["mlp"].state_dict())
        t = time.time()
        shapelab.fit_fast(CC, dd, m, steps=6000, lr=1e-3, lr_end=1e-4, log=None)
        torch.save({"state": m.state_dict()}, ck)
        log(f"{name}: {time.time() - t:.0f}s")
    m.eval()
    m.n_params = lambda: sum(p.numel() for p in m.parameters())
    if not done(name):
        save_json(name, {"val": float(CC["test_nll"](m, dd["val"]).mean()),
                         "val_report_weights": float(CC["test_nll"](RM["mlp"], dd["val"]).mean())})
    return m


MLP_STD = mlp_ft("mlp_std", C, data)
MLP_SLAB = mlp_ft("mlp_slab", CS, dataS)

# ---------------------------------------------------------------- final models: test and s2 (evaluated once)
FINAL = {"std": {"report_baseline": RM["baseline"], "report_mlp": RM["mlp"], "base_acc": BASE_STD[nb_std],
                 "new_n2": NEW_STD[2], "new": NEW_STD[nn_std], "mlp_ft": MLP_STD},
         "slab": {"base": BASE_SLAB[nb_slab], "new_n2": NEW_SLAB[2], "new": NEW_SLAB[nn_slab], "mlp_ft": MLP_SLAB}}
PAR = {"report_baseline": EX[RM["baseline_name"]]["n_params"], "report_mlp": EX["mlp"]["n_params"],
       "base_acc": int(BASE_STD[nb_std].n_params()), "base": int(BASE_SLAB[nb_slab].n_params()),
       "new_n2": int(NEW_STD[2].n_params()), "new": int(NEW_STD[nn_std].n_params()), "new_slab": int(NEW_SLAB[nn_slab].n_params()),
       "mlp_ft": int(sum(p.numel() for p in MLP_STD.parameters()))}


def evaluate(tag, sets):
    path = os.path.join(OUT, f"{tag}{SUF}_nll.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    out = {}
    for metric, CC in (("std", C), ("slab", CS)):
        for k, m in FINAL[metric].items():
            out[f"{metric}:{k}"] = np.concatenate([CC["test_nll"](m, sh) for sh in sets])
    np.savez(path, **out)
    return out


def summarise(tag, nll):
    ref = {"std": "std:base_acc", "slab": "slab:base"}
    mlp = {"std": "std:mlp_ft", "slab": "slab:mlp_ft"}
    res = {"n_events": int(len(next(iter(nll.values())))), "nll": {k: float(v.mean()) for k, v in nll.items()}, "delta_vs_base": {}, "gap_closed": {}}
    for k, v in nll.items():
        metric = k.split(":")[0]
        res["delta_vs_base"][k] = paired(v, nll[ref[metric]])
    for metric in ("std", "slab"):
        g, m_ = nll[ref[metric]].mean(), nll[mlp[metric]].mean()
        for k in ("new", "new_n2"):
            res["gap_closed"][f"{metric}:{k}"] = float((g - nll[f"{metric}:{k}"].mean()) / (g - m_))
        res["new_minus_mlp"] = res.get("new_minus_mlp", {})
        res["new_minus_mlp"][metric] = paired(nll[f"{metric}:new"], nll[mlp[metric]])
    if "std:report_mlp" in nll:
        res["new_minus_report_mlp_std"] = paired(nll["std:new"], nll["std:report_mlp"])
        res["base_acc_minus_report_baseline_std"] = paired(nll["std:base_acc"], nll["std:report_baseline"])
    return res


if not done("test" + SUF):
    nll = evaluate("test", [data["test"]])
    s = summarise("test", nll)
    b = data["test"].ev["bin"].cpu().numpy()
    d = nll["slab:new"] - nll["slab:base"]
    s["per_bin_new_minus_base_slab"] = [float(d[b == k].mean()) for k in range(20)]
    d = nll["std:new"] - nll["std:base_acc"]
    s["per_bin_new_minus_base_std"] = [float(d[b == k].mean()) for k in range(20)]
    s.update({"params": PAR, "variant": CHOICE, "variant_kwargs": {**KW, "spike_degrees": SPKDEG},
              "degrees": {"base_std": nb_std, "base_slab": nb_slab, "new_std": nn_std, "new_slab": nn_slab}})
    save_json("test" + SUF, s)
    log(f"test: { {k: round(v, 4) for k, v in s['nll'].items()} }")

if not done("s2" + SUF):
    C2 = protonlab.load_common_for(particle, "s2", OUT_DIR=OUT)
    _, d2 = protonlab.load_particle(C2, particle, "s2")
    nll = evaluate("s2", list(d2.values()))
    save_json("s2" + SUF, summarise("s2", nll))
    log("s2 done")

# ---------------------------------------------------------------- audit (normalisation, effective parameters)
if not done("audit" + SUF):
    res = {}
    lnp = torch.linspace(math.log(0.25), math.log(20.0), 48, device=dev)
    for tag, m in (("new_std", FINAL["std"]["new"]), ("new_slab", FINAL["slab"]["new"])):
        with torch.no_grad():
            lp = m(lnp)
            eff = 0
            for p in m.parameters():
                flat = p.data.view(-1)
                for i in range(flat.numel()):
                    old = float(flat[i]); flat[i] = old + 0.05
                    eff += float((m(lnp) - lp).abs().max()) > 1e-7
                    flat[i] = old
        res[tag] = {"max_abs_sum_minus_1": float((lp.exp().sum((1, 2)) - 1).abs().max()), "finite": bool(torch.isfinite(lp).all()),
                    "effective_params": int(eff), "claimed_params": int(m.n_params())}
    save_json("audit" + SUF, res)
    log(f"audit: {res}")
log("all steps done")
