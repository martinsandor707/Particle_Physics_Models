"""shapelab – kísérleti keret a proton új alakjaihoz.

Minden a `00_common.ipynb` VÁLTOZATLAN definícióira épül (rács, detektor-konvolúció, `event_nll`, `test_nll`,
`fit`, `export`), így az itt mért NLL közvetlenül összevethető a report számaival.

Fő elemek
---------
setup(batch, out_dir, slab)   -> (C, geo, data)      a közös névtér + a proton train/val/test
baseline_nll(C, data, tag)    -> {modell: {"val","test","n_params"}}   a report mentett modelljeinek eseményenkénti NLL-je
PolyFormula                   -> ősosztály új zárt formulákhoz (paraméter = ln p polinom, paraméterenként állítható fokszám)
fit_fast(...)                 -> gyors illesztés (Adam + koszinuszos lr), a legjobb validációs állapottal
evaluate(...)                 -> val/teszt NLL, páros különbség a baseline-hoz, JSON + export

`slab=True`: a likelihood normálása a ferde homlok- és hátlappal levágott detektor-szeletre történik (3D-s levágás),
a találatonkénti (cellára átlagolt) sűrűség változatlan. Lásd SLAB_SRC.
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import protonlab

HERE = protonlab.HERE
RESULTS = os.path.join(HERE, "results")

# A 3D-s (szeletre normált) likelihood: ugyanaz, mint a 00_common `event_nll`-je, csak a normálás más.
# Az esemény találatai mind a detektor-szeletben vannak:  -layer/2 <= z_trans = z' cos(theta) - x' sin(theta) <= D,
# ahol D = z_back - z0 (a hátlap távolsága az első találati rétegtől, a detektor tengelye mentén).
# A (z', r) gyűrű azimutális akceptanciája zárt alakú:  A = [arccos(lo) - arccos(hi)] / pi,
#   lo = (z' cos(theta) - D) / (r sin(theta)),  hi = (z' cos(theta) + layer/2) / (r sin(theta)),  [-1, 1]-re vágva.
SLAB_GEOM_SRC = r'''
def slab_accept(cos_t, sin_t, D, half, nsub=2, front=True, back=True):
    # Azimuthal acceptance A (B, N_Z, N_R): fraction of the (z', r) ring inside the detector slab of each event.
    k = (torch.arange(nsub, device=cos_t.device, dtype=torch.float32) + 0.5) / nsub
    zs = Z_EDGES[:-1, None] + DZ * k                                                                      # (N_Z, nsub)
    rs = torch.sqrt(R_EDGES[:-1, None] ** 2 + (R_EDGES[1:, None] ** 2 - R_EDGES[:-1, None] ** 2) * k)     # (N_R, nsub)
    zc = zs[None, :, :, None, None] * cos_t[:, None, None, None, None]
    den = (rs[None, None, None, :, :] * sin_t[:, None, None, None, None]).clamp_min(1e-9)
    lo = ((zc - D[:, None, None, None, None]) / den).clamp(-1, 1) if back else -torch.ones_like(zc / den)
    hi = ((zc + half) / den).clamp(-1, 1) if front else torch.ones_like(zc / den)
    return ((torch.acos(lo) - torch.acos(hi)) / math.pi).mean((2, 4))


def event_slab(b, box, nsub=2, front=True, back=True):
    cos_t = b.R[:, 2, 2]
    sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
    return slab_accept(cos_t, sin_t, b.zb * cos_t, float(box[2]) / 2, nsub, front, back)
'''

SLAB_NLL_SRC = r'''
def event_nll(model, b, box, K=N_FOLD, gen=None):
    # Per-event NLL, detector-folded, normalised over the detector slab (tilted front and back faces).
    logP = model(b.lnp)
    logrho = logP - LOG_VOL
    pts = fold(b, box, K, gen)
    iz, ir = locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(K)
    mass = (logP.exp() * event_slab(b, box)).sum((1, 2))
    ll = logg - torch.log(mass.clamp_min(1e-8))[b.ev]
    return -torch.zeros(len(b.lnp), device=ll.device).index_add_(0, b.ev, b.w * ll), logP
'''


def setup(batch=256, out_dir=None, slab=False):
    """A közös névtér (C), a geometria és a proton-adat. `slab=True`: 3D-s (szeletre normált) likelihood."""
    C = protonlab.load_common(BATCH=batch, OUT_DIR=out_dir or RESULTS)
    geo, data = protonlab.load_proton(C)
    exec(SLAB_GEOM_SRC, C)                 # geometria: slab_accept, event_slab (a likelihoodot nem változtatja)
    if slab:
        exec(SLAB_NLL_SRC, C)              # 3D-s likelihood: az event_nll normálása a detektor-szeletre
    C["SLAB"] = bool(slab)
    return C, geo, data


def lnp_center_scale(data):
    """A select_degree-vel azonos centrálás: L = (ln p - c) / s, a tanítóhalmaz ln p tartományának közepe és félszélessége."""
    lnp = data["train"].ev["lnp"].cpu().numpy()
    return float((lnp.max() + lnp.min()) / 2), float((lnp.max() - lnp.min()) / 2)


def _conv_model_class(C):
    """A CNN osztálya a 06_cnn_proton.ipynb 6. cellájából (változatlanul)."""
    if "ConvModel" not in C:
        with open("/home/user/sofiya/shape-form/06_cnn_proton.ipynb") as fh:
            nb = json.load(fh)
        src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "class ConvModel" in "".join(c["source"]))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src, C)
    return C["ConvModel"]


def baseline_models(C, data):
    """A report mentett proton-modelljei: GFlash n=1..6 (n=6 a baseline), MLP L3xH16, CNN és a két desztillált formula."""
    B, dev, Formula = protonlab.BASELINE_DIR, C["DEVICE"], C["Formula"]
    c, s = lnp_center_scale(data)
    ld = lambda f: torch.load(os.path.join(B, f), map_location=dev, weights_only=False)
    out = {}
    for n in range(1, 7):
        m = Formula("gamma", "core_tail", degree=n, lnp0=c, lnp_s=s).to(dev)
        m.load_state_dict(ld(f"gflash_n{n}.pt")["model"])
        out[f"gflash_n{n}"] = m.eval()
    m = C["FieldModel"](16, 3).to(dev)
    m.load_state_dict(ld("mlp_L3xH16.pt")["model"])
    out["mlp"] = m.eval()
    m = _conv_model_class(C)().to(dev)
    m.load_state_dict(ld("cnn.pt")["model"])
    out["cnn"] = m.eval()
    for k in ("mlp_formula", "cnn_formula"):
        ck = ld(f"{k}.pt")
        m = Formula(ck["long"], ck["lat"], ck["degree"]).to(dev)
        m.load_state_dict(ck["state"])
        out[k] = m.eval()
    return out


def baseline_nll(C, data):
    """Eseményenkénti val/teszt NLL a report modelljeire, a C aktuális likelihoodjával (standard vagy szelet); gyorsítótárazva."""
    path = os.path.join(HERE, "cache", f"baseline_nll_{'slab' if C.get('SLAB') else 'std'}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    out = {}
    for k, m in baseline_models(C, data).items():
        out[k] = {"val": C["test_nll"](m, data["val"]), "test": C["test_nll"](m, data["test"]),
                  "n_params": sum(p.numel() for p in m.parameters())}
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Új formulák ősosztálya
# ----------------------------------------------------------------------------------------------------------------------
class PolyFormula(nn.Module):
    """Zárt formula a (z', r) rácson. Minden nyers paraméter az L = (ln p - lnp0)/lnp_s változó polinomja;
    a fokszám paraméterenként adható meg (`degrees`), így a p-független paraméter csak 1 együttható.

    A leszármazott a `log_density(th, Z, R)` függvényt adja meg: th = {név: (B,1,1) tenzor}, Z, R = rácsközéppontok (1, N_Z, N_R);
    visszaad: nem normált log-sűrűség (B, N_Z, N_R)  [1/mm^3 skálán, azaz dE/dV].  A normálást a rácson a softmax végzi.
    """

    SPEC = []            # [(név, kezdőérték), ...]

    def __init__(self, C, degree=1, degrees=None, lnp0=0.0, lnp_s=1.0):
        super().__init__()
        self.names = [n for n, _ in self.SPEC]
        self.deg = {n: degree for n in self.names}
        self.deg.update(degrees or {})
        self.degree = max(self.deg.values())
        self.register_buffer("lnp0", torch.tensor(float(lnp0)))
        self.register_buffer("lnp_s", torch.tensor(float(lnp_s)))
        coef = torch.zeros(len(self.names), self.degree + 1)
        coef[:, 0] = torch.tensor([float(v) for _, v in self.SPEC])
        mask = torch.zeros_like(coef)
        for i, n in enumerate(self.names):
            mask[i, :self.deg[n] + 1] = 1.0
        self.coef = nn.Parameter(coef)
        self.register_buffer("mask", mask)
        self.register_buffer("ZZ", C["ZZ"].clone())
        self.register_buffer("RR", C["RR"].clone())
        self.register_buffer("LOG_VOL", C["LOG_VOL"].clone())
        self.N_Z, self.N_R = C["N_Z"], C["N_R"]

    def params(self, lnp):
        x = (lnp - self.lnp0) / self.lnp_s
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ (self.coef * self.mask).T      # (B, n_names)

    def theta(self, lnp):
        th = self.params(lnp)[:, :, None, None]
        return {n: th[:, i] for i, n in enumerate(self.names)}

    def log_density(self, th, Z, R):
        raise NotImplementedError

    def forward(self, lnp):
        logd = self.log_density(self.theta(lnp), self.ZZ[None], self.RR[None]) + self.LOG_VOL
        return F.log_softmax(logd.reshape(len(lnp), -1), 1).reshape(len(lnp), self.N_Z, self.N_R)

    def n_params(self):
        """A ténylegesen szabad együtthatók száma (a maszkolt, mindig nulla együtthatók nem számítanak)."""
        return int(self.mask.sum().item()) + sum(p.numel() for n, p in self.named_parameters() if n != "coef")

    def warm_start(self, prev):
        """Azonos függvény, mint `prev` (ugyanaz a család alacsonyabb fokszámmal): az új együtthatók nullák."""
        with torch.no_grad():
            self.coef.zero_()
            self.coef[:, :prev.coef.shape[1]] = prev.coef * prev.mask
            for (n, p), (_, q) in zip(self.named_parameters(), prev.named_parameters()):
                if n != "coef":
                    p.copy_(q)
        return self

    def describe(self, title=""):
        rows = []
        for i, n in enumerate(self.names):
            terms = " ".join(f"{float(v):+.4g}" + ("" if k == 0 else r"\,L" if k == 1 else rf"\,L^{k}")
                             for k, v in enumerate((self.coef * self.mask)[i].detach()) if k <= self.deg[n])
            rows.append(f"| `{n}` | ${terms}$ |")
        c, s = float(self.lnp0), float(self.lnp_s)
        var = "$L=\\ln(p/\\mathrm{GeV})$" if (c == 0 and s == 1) else f"$L=(\\ln(p/\\mathrm{{GeV}}) {-c:+.4g})/{s:.4g}$"
        return "\n".join([f"### {title}", "", (self.__doc__ or "").strip(), "", f"| parameter | value, {var} |", "|---|---|"] + rows
                         + ["", f"Number of parameters: **{self.n_params()}**"])


# ----------------------------------------------------------------------------------------------------------------------
# Illesztés és kiértékelés
# ----------------------------------------------------------------------------------------------------------------------
def fit_fast(C, data, model, steps=3000, lr=0.03, lr_end=0.002, eval_every=250, seed=0, log=print, clip=5.0):
    """Gyors illesztés: Adam, koszinuszos lr-csökkentés, a legjobb validációs állapot megtartása.

    Ugyanazt a veszteséget minimalizálja (`event_nll`, C["BATCH"] méretű kötegek), mint a közös `fit`."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_end)
    gen = torch.Generator().manual_seed(1000 + seed)
    best, best_state, hist, step, t0 = float("inf"), None, [], 0, time.time()
    event_nll, test_nll, tr = C["event_nll"], C["test_nll"], data["train"]
    model.train()
    while step < steps:
        for b in tr.batches(C["BATCH"], shuffle=True, gen=gen):
            nll, _ = event_nll(model, b, tr.box)
            loss = nll.mean()
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                if all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                model.eval()
                val = float(test_nll(model, data["val"], C["N_FOLD"]).mean())
                model.train()
                hist.append((step, val))
                if val < best:
                    best, best_state = val, copy.deepcopy(model.state_dict())
                if log:
                    log(f"  step {step:5d}  val {val:.4f}  best {best:.4f}  ({time.time() - t0:.0f}s)")
            if step >= steps:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def paired(a, b):
    """Páros különbség átlaga és standard hibája (a − b)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def evaluate(C, data, model, name, base=None, formula_md=None, extra=None, save=True):
    """Val/teszt NLL a közös `test_nll`-lel + páros különbség a report modelljeihez; JSON (és export) mentése."""
    model.eval()
    base = base if base is not None else baseline_nll(C, data)
    val, test = C["test_nll"](model, data["val"]), C["test_nll"](model, data["test"])
    n_params = model.n_params() if hasattr(model, "n_params") else sum(p.numel() for p in model.parameters())
    res = {"name": name, "likelihood": "slab" if C.get("SLAB") else "std", "n_params": int(n_params),
           "val_nll": float(val.mean()), "test_nll": float(test.mean()), "extra": extra or {}}
    for k in ("gflash_n6", "mlp"):
        if k in base:
            res[f"val_vs_{k}"] = paired(val, base[k]["val"])
            res[f"test_vs_{k}"] = paired(test, base[k]["test"])
    if save:
        d = os.path.join(C["OUT_DIR"], "proton")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        torch.save({"state": model.state_dict(), "val_nll": val, "test_nll": test, "n_params": int(n_params),
                    "formula": formula_md, "class": type(model).__name__}, os.path.join(d, f"{name}.pt"))
    return res, val, test


# ----------------------------------------------------------------------------------------------------------------------
# Eseményátlagolt geometriai akceptancia: tisztán (z', r | p) modellhez (a standard metrikában is használható)
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def mean_acceptance(C, sh, nsub=4, front=True, back=False, lnp_bins=None):
    """Ā(z', r): a detektor-szelet azimutális akceptanciájának átlaga az `sh` (tanító) eseményeken – paramétermentes geometria.

    front/back: melyik síkot vegye figyelembe (a standard likelihood a hátlapot már kezeli a z_b szerinti normálással)."""
    tot, n = 0.0, 0
    for b in sh.batches(256):
        A = C["event_slab"](b, sh.box, nsub, front, back)
        tot, n = tot + A.sum(0), n + len(A)
    return tot / n


class WithAcceptance(nn.Module):
    """rho(z', r | p) ∝ alapmodell(z', r | p) · Ā(z', r): forgástest, amelyet a (ferde) detektorsík átlagos akceptanciája vág le."""

    def __init__(self, base, A, floor=1e-6):
        super().__init__()
        self.base = base
        self.register_buffer("logA", torch.log(A.clamp_min(floor)))

    def forward(self, lnp):
        lp = self.base(lnp) + self.logA
        return F.log_softmax(lp.reshape(len(lnp), -1), 1).reshape(lp.shape)

    def n_params(self):
        return self.base.n_params() if hasattr(self.base, "n_params") else sum(p.numel() for p in self.base.parameters())


# ----------------------------------------------------------------------------------------------------------------------
# Mind a négy részecske: beállítás és a report mentett modelljei
# ----------------------------------------------------------------------------------------------------------------------
REPORT_BASE = {"electron": ("gflash", "gamma", 4), "pion": ("gflash", "gamma", 3), "proton": ("gflash", "gamma", 6),
               "muon": ("cylinder", "plateau", 6)}               # the report's baseline family and its chosen degree (1-SE)


def setup_particle(particle, batch=256, out_dir=None, slab=False, sample="s1"):
    """(C, geo, data) a részecskére; slab=True: 3D (szeletre normált) likelihood."""
    C = protonlab.load_common_for(particle, sample, BATCH=batch, OUT_DIR=out_dir or RESULTS)
    geo, data = protonlab.load_particle(C, particle, sample)
    exec(SLAB_GEOM_SRC, C)
    if slab:
        exec(SLAB_NLL_SRC, C)
    C["SLAB"] = bool(slab)
    return C, geo, data


def report_models(C, data, particle):
    """A report mentett modelljei a részecskére: a baseline (GFlash / henger, a report fokszámával) és a kiválasztott MLP."""
    import pandas as pd
    R, dev = os.path.join(protonlab.REPORT_RESULTS, particle), C["DEVICE"]
    name, long, n = REPORT_BASE[particle]
    c, s = lnp_center_scale(data)
    base = C["Formula"](long, "core_tail", degree=n, lnp0=c, lnp_s=s).to(dev)
    base.load_state_dict(torch.load(os.path.join(R, f"{name}_n{n}.pt"), map_location=dev, weights_only=False)["model"])
    t = pd.read_csv(os.path.join(R, "mlp_architecture.csv")).drop_duplicates("config", keep="last")
    row = t[t["chosen"] == "◀"].iloc[-1]
    mlp = C["FieldModel"](int(row["hidden"]), int(row["n_layers"])).to(dev)
    mlp.load_state_dict(torch.load(os.path.join(R, f"mlp_{row['config']}.pt"), map_location=dev, weights_only=False)["model"])
    return {"baseline": base.eval(), "mlp": mlp.eval(), "baseline_name": name, "baseline_long": long, "baseline_degree": n,
            "mlp_config": row["config"]}


def report_exports(particle):
    """A report exportjai (eseményenkénti teszt-NLL és paraméterszám) a részecskére."""
    R = os.path.join(protonlab.REPORT_RESULTS, particle)
    out = {}
    for f in sorted(os.listdir(R)):
        if f.startswith("export_") and f.endswith(".pt"):
            e = torch.load(os.path.join(R, f), weights_only=False)
            out[f[7:-3]] = {"test_nll": np.asarray(e["test_nll"], dtype=np.float64), "n_params": int(e["n_params"])}
    return out



def final_round(D):
    """Highest finished round of the pipeline in results/particles/<p>: (round, json/npz suffix, checkpoint tag).
    Round 1 = base variants; rounds 2-3 = physics extensions chosen on validation (scripts/variants_v2.py, variants_v3.py)."""
    import os as _os
    for k in (3, 2):
        if _os.path.exists(_os.path.join(D, f"test_v{k}.json")):
            return k, f"_v{k}", f"new{k}"
    return 1, "", "new"



ROUND1_VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
                   "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
                   "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}


def final_kwargs(D, n=None, metric="std"):
    """(constructor kwargs, per-parameter degrees, degree n, checkpoint tag, test json, round) of the final formula of a particle."""
    k, suf, tag = final_round(D)
    t = json.load(open(os.path.join(D, f"test{suf}.json")))
    n = t["degrees"]["new_std" if metric == "std" else "new_slab"] if n is None else n
    if k >= 2:
        kw = dict(t["variant_kwargs"]); extra = kw.pop("spike_degrees", None)   # extra per-parameter degrees (spike, slopes)
    else:
        kw, extra = dict(ROUND1_VARIANTS[t["variant"]]), None
    degrees = {"logit_tr": 2} if (n == 1 and kw.get("track")) else {}
    if extra:
        degrees.update({q: min(n, v) for q, v in extra.items()})
    return kw, degrees, n, tag, t, k


def build_final(C, particle, D, c0, s0, metric="std", n=None):
    """The final formula of a particle from its pipeline checkpoints. metric 'std': report metric (wrapped with the template A_bar),
    'slab': 3D metric. Returns (model, core formula, test json, round)."""
    from particle_shape import MASS, ParticleShape
    kw, degrees, n, tag, t, k = final_kwargs(D, n, metric)
    dev = C["DEVICE"]
    core = ParticleShape(C, MASS[particle], lnp0=c0, lnp_s=s0, degree=n, degrees=degrees or None, **kw).to(dev)
    m = WithAcceptance(core, torch.load(os.path.join(D, "abar.pt"), map_location=dev)).to(dev) if metric == "std" else core
    m.load_state_dict(torch.load(os.path.join(D, f"{tag}_{metric}_n{n}.pt"), map_location=dev, weights_only=False)["state"])
    return m.eval(), core, t, k
