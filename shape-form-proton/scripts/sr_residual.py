"""Szimbolikus regresszió (PySR) a formula maradékára:  g(z', r, x) ≈ ln P_MLP − ln P_formula   (report-mérce, a 2. körös formula × Ā
és a finomhangolt MLP térképein, az MLP sűrűségével súlyozott pontokon; x = (ln T − c)/s a formula változója).
Utána a formula kiegészítése:  ln P' = log_softmax(ln P_formula + a · g),  a és a formula paraméterei a likelihooddal finomhangolva
(fit_fast, a 2. körös checkpointból), validáción összevetve a kiindulással.  A tesztet nem használja.
    python sr_residual.py <particle>          kimenet: results/particles/<p>/sr_residual.json (+ checkpoint)"""

import json
import os
import sys
import time

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import shapelab
import sympy
import torch
import torch.nn.functional as F
from particle_shape import MASS, ParticleShape

particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=D)
dev = C["DEVICE"]
c0, s0 = shapelab.lnp_center_scale(data)
t2 = json.load(open(os.path.join(D, "test_v2.json")))
n = t2["degrees"]["new_std"]
kw = dict(t2["variant_kwargs"])
spk = kw.pop("spike_degrees", None)
degrees = {"logit_tr": 2} if (n == 1 and kw.get("track")) else {}
if spk:
    degrees.update({k: min(n, v) for k, v in spk.items()})
ABAR = torch.load(os.path.join(D, "abar.pt"), map_location=dev)


def build_formula():
    core = ParticleShape(
        C, MASS[particle], lnp0=c0, lnp_s=s0, degree=n, degrees=degrees or None, **kw
    ).to(dev)
    m = shapelab.WithAcceptance(core, ABAR).to(dev)
    m.load_state_dict(
        torch.load(
            os.path.join(D, f"new2_std_n{n}.pt"), map_location=dev, weights_only=False
        )["state"]
    )
    return core, m.eval()


core, FORM = build_formula()
rm = shapelab.report_models(C, data, particle)
MLP = C["FieldModel"](rm["mlp"].mlp[0].out_features, (len(rm["mlp"].mlp) - 1) // 2).to(
    dev
)
MLP.load_state_dict(
    torch.load(os.path.join(D, "mlp_std.pt"), map_location=dev, weights_only=False)[
        "state"
    ]
)
MLP.eval()

# ---------------- data set for the regression
lnp_tr = data["train"].ev["lnp"]
grid = torch.linspace(
    float(torch.quantile(lnp_tr, 0.01)),
    float(torch.quantile(lnp_tr, 0.99)),
    16,
    device=dev,
)
with torch.no_grad():
    lf, lm = FORM(grid), MLP(grid)
xv = (core.lnT(grid) - core.v0) / core.vs  # (16,)
Z, R = C["ZZ"][None].expand(len(grid), -1, -1), C["RR"][None].expand(len(grid), -1, -1)
X = torch.stack([Z / 1000, R / 1000, xv[:, None, None].expand_as(Z)], -1).reshape(-1, 3)
res = (lm - lf).reshape(-1)
w = lm.exp().reshape(-1) / len(grid)
keep = (w > 1e-7) & torch.isfinite(res)
X, res, w = X[keep], res[keep], w[keep]
g = torch.Generator(device=dev).manual_seed(0)
idx = torch.multinomial(
    w / w.sum(), 6000, replacement=True, generator=g
)  # sampled proportionally to the network's density
Xs, ys = (
    X[idx].cpu().numpy().astype(np.float64),
    res[idx].cpu().numpy().astype(np.float64),
)
print(
    particle,
    "residual (weighted) mean",
    float((w * res).sum() / w.sum()),
    "rms",
    float(torch.sqrt((w * res**2).sum() / w.sum())),
    flush=True,
)

# ---------------- PySR
from pysr import PySRRegressor

t = time.time()
sr = PySRRegressor(
    niterations=80,
    populations=24,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "square"],
    maxsize=18,
    parsimony=1e-4,
    verbosity=0,
    progress=False,
    temp_equation_file=True,
    random_state=0,
    deterministic=True,
    parallelism="serial",
    model_selection="best",
)
sr.fit(Xs, ys, variable_names=["z", "r", "x"])
eqs = sr.equations_[["complexity", "loss", "equation"]].copy()
print(eqs.to_string(), flush=True)
print(f"PySR {time.time() - t:.0f}s", flush=True)

# ---------------- add the best few expressions to the formula and refit with the likelihood
z_, r_, x_ = sympy.symbols("z r x")
TORCH = {
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "square": lambda a: a * a,
    "Abs": torch.abs,
}


class Corrected(torch.nn.Module):
    def __init__(self, base, expr):
        super().__init__()
        self.base = base
        self.f = sympy.lambdify((z_, r_, x_), expr, modules=[TORCH, "math"])
        self.a = torch.nn.Parameter(torch.tensor(1.0))
        self.expr = str(expr)

    def forward(self, lnp):
        lp = self.base(lnp)
        x = ((self.base.base.lnT(lnp) - self.base.base.v0) / self.base.base.vs)[
            :, None, None
        ]
        corr = self.f(C["ZZ"][None] / 1000, C["RR"][None] / 1000, x) * torch.ones_like(
            lp
        )
        corr = torch.nan_to_num(corr, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
        out = lp + self.a * corr
        return F.log_softmax(out.reshape(len(lnp), -1), 1).reshape(lp.shape)

    def n_params(self):
        return self.base.n_params() + 1


base_val = C["test_nll"](FORM, data["val"])
out = {"base_val": float(base_val.mean()), "candidates": []}
cands = sr.equations_.sort_values("loss")
picked = []
for cx in (5, 8, 11, 14, 18):  # best equation up to a given complexity
    sub = cands[cands.complexity <= cx]
    if len(sub):
        e = sub.sort_values("loss").iloc[0]
        if str(e["sympy_format"]) not in [str(p_["sympy_format"]) for p_ in picked]:
            picked.append(e)
for e in picked:
    _, base = build_formula()
    m = Corrected(base, e["sympy_format"]).to(dev)
    t = time.time()
    shapelab.fit_fast(C, data, m, steps=3000, lr=3e-3, lr_end=3e-4, log=None)
    v = C["test_nll"](m.eval(), data["val"])
    d = shapelab.paired(v, base_val)
    out["candidates"].append(
        {
            "equation": str(e["sympy_format"]),
            "complexity": int(e["complexity"]),
            "sr_loss": float(e["loss"]),
            "val": float(v.mean()),
            "delta_val_vs_base": d,
            "amplitude": float(m.a),
            "n_params": int(m.n_params()),
            "sec": round(time.time() - t),
        }
    )
    torch.save(
        {"state": m.state_dict(), "expr": str(e["sympy_format"])},
        os.path.join(D, f"sr_c{int(e['complexity'])}.pt"),
    )
    print(particle, out["candidates"][-1], flush=True)
json.dump(out, open(os.path.join(D, "sr_residual.json"), "w"), indent=1)
