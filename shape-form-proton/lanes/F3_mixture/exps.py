"""Experiment registry of lane F3: name -> kwargs of fam.Mixture (plus optional 'warm', 'steps')."""
import math

ln = math.log
E = {}

# ---- initial values at clearly different scales ----
CORE = {"z0": -20.0, "ltmax": ln(80), "lam1": ln(1.5)}
BODY = {"z0": -20.0, "ltmax": ln(220), "lam1": ln(2.0)}
HALO = {"z0": -60.0, "ltmax": ln(250), "lam1": ln(1.0)}


def init(*parts, **w):
    out = {}
    for k, d in enumerate(parts):
        out.update({f"c{k}_{n}": v for n, v in d.items()})
    out.update(w)
    return out


# ============ batch 1: K = 2, core + halo/body with separate longitudinal profiles, all parameters degree 2 ============
for lat0, w0 in (("cauchy", ("lR", ln(8))), ("gauss", ("lsg", ln(10))), ("expr", ("llam", ln(20))), ("exp", ("llam", ln(12)))):
    for lat1, w1 in (("cauchy", ("lR", ln(60))), ("expr", ("llam", ln(95))), ("exp", ("llam", ln(70))), ("k0", ("llam", ln(80)))):
        E[f"K2_{lat0}_{lat1}"] = dict(comps=[f"gamma/{lat0}", f"gamma/{lat1}"], degree=2,
                                      init=init({**CORE, w0[0]: w0[1]}, {**BODY, w1[0]: w1[1]}, w1=-0.5))
# report-style gamma (with the 1e-3 floor) for both components
E["K2g0_cauchy_cauchy"] = dict(comps=["gamma0/cauchy", "gamma0/cauchy"], degree=2,
                               init=init({**CORE, "lR": ln(8)}, {**BODY, "lR": ln(60)}, w1=-0.5))
E["K2g0_cauchy_expr"] = dict(comps=["gamma0/cauchy", "gamma0/expr"], degree=2,
                             init=init({**CORE, "lR": ln(8)}, {**BODY, "llam": ln(95)}, w1=-0.5))
