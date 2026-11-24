"""Q3: energy variable for the plain reference family (GCTVar), degrees 1..3 (+ degree 6 in ln p), slab likelihood."""
import os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/F4_lowp")
import torch
import lane, fam, shapelab

C, geo, data, base, c, s = lane.setup()
ref, _ = lane.ref_val(C, data, base, c, s)
jobs = [(v, d) for d in (1, 2, 3) for v in ("lnp", "lnT", "lnE", "beta", "lnTsat", "asinhT", "lnR")] + [("lnp", 6), ("lnT", 6), ("lnT", 4)]
if len(sys.argv) > 1:
    jobs = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
for var, deg in jobs:
    torch.manual_seed(0)
    m = lane.init_gamma(C, data, fam.GCTVar(C, degree=deg, lnp0=c, lnp_s=s, var=var).to(C["DEVICE"]))
    try:
        lane.fit_eval(C, data, base, ref, m, f"q3_gct_{var}_n{deg}", steps=3000, extra=f"GCTVar var={var} degree={deg}")
    except Exception as e:
        print("FAILED", var, deg, repr(e), flush=True)
