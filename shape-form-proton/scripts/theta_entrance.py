"""Beesési szögtől függő belépés (3D-mérce): a végleges formula belépéshez tartozó paraméterei (csúcs: súly, hely, szélességek,
a csúcs–nyom kötés; a zápor kezdete; a nyom kezdete) egy u = 1/cosθ − 1,1 szerinti lineáris taggal változhatnak:
    P_k(x, u) = Σ c_k,j x^j + β_k · u      (β_k: részecskénként 6–8 új paraméter, kezdetben 0, azaz a végleges formula).
Összevetés ugyanazzal a finomhangolással θ-tag nélkül (kontroll), és a végleges formulával; teszt és független s2 minta, páros
különbség. A belépés előtti (z' < 0) energia θ-harmadonként (s2, p ≥ 2 GeV), mint a front_region.py-ban.
    python theta_entrance.py <particle>          kimenet: results/particles/<p>/theta_entrance.json (+ checkpointok)"""
import copy
import json
import math
import os
import sys
import time

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import shapelab
from particle_shape import MASS, ParticleShape

particle = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "joint"      # "beta": only the new angle terms are trained, the formula stays fixed
SUF = {"joint": "", "beta": "_beta", "two": "_two", "front": "_two"}[MODE]   # "two": angle terms first, then all together (small lr);
# "front": no fitting, only the front-region shares of the saved two-stage models are recomputed
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=D, slab=True)
dev = C["DEVICE"]
ENTRANCE = ["logit_sp", "sp_z", "log_sp_sz", "log_sp_sr", "log_d1", "zs", "log_w1", "z0", "zt0"]
U0 = 1.1


class ThetaShape(ParticleShape):
    """The final formula; selected raw parameters get + beta * u, u = 1/cos(theta) - U0 (set per batch through `_u`)."""

    def setup_theta(self, names):
        self.tnames = [n for n in names if n in self.names]
        self.beta = torch.nn.Parameter(torch.zeros(len(self.tnames), device=self.coef.device))
        self._u = None
        return self

    def theta(self, lnp):
        th = super().theta(lnp)
        if self._u is not None:
            th = dict(th)
            for i, n in enumerate(self.tnames):
                th[n] = th[n] + self.beta[i] * self._u[:, None, None]
        return th


class Ctx(torch.nn.Module):
    """Calls the core with the incidence angle of the current batch (the common NLL code only passes ln p)."""

    def __init__(self, core, u):
        super().__init__()
        self.core, self.u = core, u

    def forward(self, lnp):
        self.core._u = self.u
        try:
            return self.core(lnp)
        finally:
            self.core._u = None


def u_of(b):
    return 1.0 / b.R[:, 2, 2] - U0


def build(theta_terms):
    kw, degrees, n, tag, t, k = shapelab.final_kwargs(D, metric="slab")
    core = ThetaShape(C, MASS[particle], lnp0=0.0, lnp_s=1.0, degree=n, degrees=degrees or None, **kw).to(dev)
    core.load_state_dict(torch.load(os.path.join(D, f"{tag}_slab_n{n}.pt"), map_location=dev, weights_only=False)["state"])
    core.setup_theta(ENTRANCE if theta_terms else [])
    return core


@torch.no_grad()
def nll(core, sh, K=None):
    """Exactly the common test_nll (seed 12345, batches of 128, K = N_FOLD_EVAL), with the angle passed per batch."""
    K = K or C["N_FOLD_EVAL"]
    gen = torch.Generator(device=dev).manual_seed(12345)
    core.eval()
    out = torch.cat([C["event_nll"](Ctx(core, u_of(b)), b, sh.box, K, gen)[0] for b in sh.batches(128)]).cpu().numpy()
    return out


def fit(core, steps=3000, lr=3e-3, lr_end=3e-4, eval_every=250, seed=0, only_beta=False):
    """fit_fast with the angle context (same loss, Adam + cosine lr, best validation state kept)."""
    params = [core.beta] if only_beta else [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_end)
    gen = torch.Generator().manual_seed(1000 + seed)
    tr = data["train"]
    best, best_state, step = float(nll(core, data["val"], C["N_FOLD"]).mean()), copy.deepcopy(core.state_dict()), 0
    core.train()
    while step < steps:
        for b in tr.batches(C["BATCH"], shuffle=True, gen=gen):
            loss, _ = C["event_nll"](Ctx(core, u_of(b)), b, tr.box)
            loss = loss.mean()
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
            sched.step(); step += 1
            if step % eval_every == 0 or step == steps:
                v = float(nll(core, data["val"], C["N_FOLD"]).mean())
                core.train()
                if v < best:
                    best, best_state = v, copy.deepcopy(core.state_dict())
            if step >= steps:
                break
    core.load_state_dict(best_state)
    return core.eval()


@torch.no_grad()
def front_share(core, sh, M=256, seed=7):
    """Per event: share of the model's in-slab deposits (rho x cell x cell) with z' < 0 (as front_region.py)."""
    ZE, RE, DZ, N_R = C["Z_EDGES"], C["R_EDGES"], C["DZ"], C["N_R"]
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for b in sh.batches(128):
        n = len(b.lnp)
        A = C["event_slab"](b, sh.box, 4)
        P = Ctx(core, u_of(b))(b.lnp).exp()
        cell = torch.multinomial((P * A).reshape(n, -1) + 1e-30, M, True, generator=gen)
        iz, ir = cell // N_R, cell % N_R
        u = torch.rand(n, M, 3, device=dev, generator=gen)
        z = ZE[iz] + u[..., 0] * DZ
        r = torch.sqrt(RE[ir] ** 2 + u[..., 1] * (RE[ir + 1] ** 2 - RE[ir] ** 2))
        pts = torch.stack([r * torch.cos(2 * math.pi * u[..., 2]), r * torch.sin(2 * math.pi * u[..., 2]), z], -1)
        ct = b.R[:, 2, 2][:, None]
        zt = pts[..., 2] * ct - pts[..., 0] * torch.sqrt(1 - ct ** 2)
        keep = (zt >= -float(sh.box[2]) / 2) & (zt <= b.zb[:, None] * ct)
        box = (torch.rand(n, M, 3, device=dev, generator=gen) - 0.5) * sh.box + (torch.rand(n, M, 3, device=dev, generator=gen) - 0.5) * sh.box
        pts = pts + torch.einsum("bij,bmj->bmi", b.R, box)
        a = A[torch.arange(n, device=dev)[:, None], iz, ir].clamp_min(1e-6)      # 1/A weight: kept deposits follow P*A, not P*A^2
        wk = keep.float() / a
        out.append((wk * (pts[..., 2] < 0)).sum(1) / wk.sum(1).clamp_min(1e-30))
    return torch.cat(out).cpu().numpy()


res = {"particle": particle, "u": "1/cos(theta) - 1.1", "entrance_params": None}
t0 = time.time()
final = build(False).eval()
if MODE == "front":
    path = os.path.join(D, "theta_entrance_two.json")
    res = json.load(open(path))
    theta = build(True)
    ck = torch.load(os.path.join(D, "theta_entrance_two.pt"), map_location=dev, weights_only=False)
    theta.load_state_dict(ck["state"]); theta.eval()
    ctrl = build(False)
    ctrl.load_state_dict(torch.load(os.path.join(D, "theta_control_two.pt"), map_location=dev, weights_only=False)["state"]); ctrl.eval()
    C2, _, d2 = shapelab.setup_particle(particle, out_dir=D, slab=True, sample="s2")
    fr = json.load(open(os.path.join(D, "front_region.json")))
    sets = [sh.subset((sh.ev["p"] >= 2).nonzero().squeeze(1)) for sh in d2.values()]
    th = np.concatenate([torch.acos(sh.ev["R"][:, 2, 2]).cpu().numpy() for sh in sets])
    shares = {name: np.concatenate([front_share(m, sh) for sh in sets]) for name, m in (("final", final), ("control", ctrl), ("theta", theta))}
    q = np.quantile(th, [0, 1 / 3, 2 / 3, 1])
    res["front"] = []
    for i in range(3):
        sel = (th >= q[i]) & (th <= q[i + 1])
        res["front"].append({"theta_mean": float(th[sel].mean()), "data": fr["bins"][i]["data"],
                             **{name: float(v[sel].mean()) for name, v in shares.items()}})
    res["front_note"] = "recomputed with the corrected sampler (1/A weights)"
    json.dump(res, open(path, "w"), indent=1)
    print(particle, "front", json.dumps(res["front"]), flush=True)
    sys.exit(0)
if MODE == "beta":
    ctrl = final                                           # nothing else is retrained: the comparison is directly with the final formula
    theta = fit(build(True), lr=1e-2, lr_end=1e-3, only_beta=True)
elif MODE == "two":
    ctrl = fit(build(False), lr=1e-3, lr_end=1e-4)          # the same joint fine-tuning without the angle terms
    theta = fit(fit(build(True), lr=1e-2, lr_end=1e-3, only_beta=True), lr=1e-3, lr_end=1e-4)
else:
    ctrl = fit(build(False))
    theta = fit(build(True))
res["entrance_params"] = theta.tnames
res["beta"] = {n: float(v) for n, v in zip(theta.tnames, theta.beta.detach().cpu())}
res["n_params"] = {"final": int(final.n_params()), "control": int(ctrl.n_params()), "theta": int(ctrl.n_params() + len(theta.tnames))}
torch.save({"state": theta.state_dict(), "tnames": theta.tnames}, os.path.join(D, f"theta_entrance{SUF}.pt"))
if MODE != "beta":
    torch.save({"state": ctrl.state_dict()}, os.path.join(D, f"theta_control{SUF}.pt"))
print(particle, "fits done", f"{time.time() - t0:.0f}s", res["beta"], flush=True)

ev = {}
for name, m in (("final", final), ("control", ctrl), ("theta", theta)):
    ev[("val", name)] = nll(m, data["val"])
    ev[("test", name)] = nll(m, data["test"])
C2, _, d2 = shapelab.setup_particle(particle, out_dir=D, slab=True, sample="s2")
s2 = list(d2.values())
for name, m in (("final", final), ("control", ctrl), ("theta", theta)):
    ev[("s2", name)] = np.concatenate([nll(m, sh) for sh in s2])
for split in ("val", "test", "s2"):
    res[split] = {name: float(ev[(split, name)].mean()) for name in ("final", "control", "theta")}
    res[split]["theta_minus_control"] = shapelab.paired(ev[(split, "theta")], ev[(split, "control")])
    res[split]["control_minus_final"] = shapelab.paired(ev[(split, "control")], ev[(split, "final")])
    res[split]["theta_minus_final"] = shapelab.paired(ev[(split, "theta")], ev[(split, "final")])
print(particle, json.dumps({k: res[k] for k in ("val", "test", "s2")}), flush=True)

# front region by theta tercile on s2, p >= 2 GeV (data share from front_region.json)
fr = json.load(open(os.path.join(D, "front_region.json"))) if os.path.exists(os.path.join(D, "front_region.json")) else None
sets = [sh.subset((sh.ev["p"] >= 2).nonzero().squeeze(1)) for sh in s2]
th = np.concatenate([torch.acos(sh.ev["R"][:, 2, 2]).cpu().numpy() for sh in sets])
shares = {name: np.concatenate([front_share(m, sh) for sh in sets]) for name, m in (("control", ctrl), ("theta", theta))}
q = np.quantile(th, [0, 1 / 3, 2 / 3, 1])
res["front"] = []
for i in range(3):
    sel = (th >= q[i]) & (th <= q[i + 1])
    row = {"theta_mean": float(th[sel].mean()), "control": float(shares["control"][sel].mean()), "theta": float(shares["theta"][sel].mean())}
    if fr:
        row["data"] = fr["bins"][i]["data"]
    res["front"].append(row)
res["mode"] = MODE
json.dump(res, open(os.path.join(D, f"theta_entrance{SUF}.json"), "w"), indent=1)
print(particle, "front", json.dumps(res["front"]), f"total {time.time() - t0:.0f}s", flush=True)
