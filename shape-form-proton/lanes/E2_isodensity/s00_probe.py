"""Probe: timings, deposit statistics, saved envelope/drop reproduction for one bin."""
import sys, time, math
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab, envlab
LANE = "/home/user/tamas/shape-form-proton/lanes/E2_isodensity"
C, geo, data = shapelab.setup(batch=256, out_dir=LANE + "/results")
print("geo", geo, "box", data["train"].box)
ENV = torch.load("/home/user/sofiya/shape-form/results/proton/envelope.pt", weights_only=False)
DROP = torch.load("/home/user/sofiya/shape-form/results/proton/drop.pt", weights_only=False)
print("env keys", list(ENV[10].keys())); print("drop keys", list(DROP[10].keys()))
for b in (0, 10):
    tr, te = envlab.bin_sets(C, data, b)
    t0 = time.time(); P = envlab.deposits(C, tr, xy=True); torch.cuda.synchronize(); t1 = time.time()
    print(f"bin {b}: n_train {len(tr)}, points {len(P['z'])/1e6:.2f}M, deposits {t1-t0:.2f}s")
    out = ((P["z"] < C["Z_MIN"]) | (P["z"] > C["Z_MAX"]) | (P["r"] >= C["R_MAX"]))
    print("  weight fraction of points outside the domain:", float((P["w"] * out).sum() / P["w"].sum()),
          " r>=700:", float((P["w"] * (P["r"] >= 700)).sum() / P["w"].sum()),
          " z>1450:", float((P["w"] * (P["z"] > 1450)).sum() / P["w"].sum()), " z<-200:", float((P["w"] * (P["z"] < -200)).sum() / P["w"].sum()))
    env, drop = envlab.Envelope(), envlab.BetaDrop(fixed=(0.5, 1.5))
    e, d = ENV[b], DROP[b]
    xe = [math.log(e["a"]), math.log(e["z_env"]), e["z0"]]
    xd = [math.log(d["L"]), d["z0"]]
    t0 = time.time()
    for _ in range(5):
        ce = envlab.threshold(P, env.score(P, xe))
    torch.cuda.synchronize(); t1 = time.time()
    cd = envlab.threshold(P, drop.score(P, xd))
    print(f"  threshold time {(t1-t0)/5*1000:.0f} ms; env A saved {e['A']:.3f} recomputed {math.exp(ce):.3f}; drop A saved {d['A']:.3f} recomputed {math.exp(cd):.3f}")
    Pte = envlab.deposits(C, te)
    print("  test cov env", envlab.coverage(Pte, env.score(Pte, xe), math.log(e["A"])), "saved", e["cov_test"],
          "| drop", envlab.coverage(Pte, drop.score(Pte, xd), math.log(d["A"])), "saved", d["cov_test"])
    # slab geometry of the events
    print("  theta range", float(torch.acos(P["cos_t"]).min()), float(torch.acos(P["cos_t"]).max()), "D quantiles", np.quantile(P["D"].cpu().numpy(), [0, .01, .05, .5, .95, 1]))
    zt = P["z"] * P["cos_t"][P["ev"]] - P["x"] * P["sin_t"][P["ev"]]
    print("  z_trans of points: min", float(zt.min()), "max minus D:", float((zt - P["D"][P["ev"]]).max()), "half", P["half"])
