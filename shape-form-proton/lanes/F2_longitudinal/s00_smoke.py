"""Smoke test of fam.LongMix (shapes, normalisation, gradients, timing) + reference fit."""
import time, torch, numpy as np
import lab, fam, shapelab
C, geo, data, c, s = lab.setup()
dev = C["DEVICE"]
t0 = time.time()
ref, ref_val = lab.ref_as_init(C, data, c, s)
print(f"reference GammaCoreTail deg 2: val {ref_val.mean():.5f}  n={ref.n_params()}  ({time.time()-t0:.0f}s)")
lnp = data["train"].ev["lnp"][:256]
for kw in (dict(long="gamma"), dict(long="gamma2"), dict(long="conv"), dict(long="conv2"), dict(long="conv2", lat2=True),
           dict(long="gamma", spike="gauss"), dict(long="gamma", spike="gg", track="gg"), dict(long="conv", spike="gg", track="exp", track_range=True)):
    m = fam.LongMix(C, degree=2, lnp0=c, lnp_s=s, **kw).to(dev)
    t0 = time.time()
    lp = m(lnp)
    b = next(iter(data["train"].batches(256)))
    nll, _ = C["event_nll"](m, b, data["train"].box)
    nll.mean().backward()
    ok = bool(torch.isfinite(m.coef.grad).all())
    print(kw, "shape", tuple(lp.shape), "sum", float(lp.exp().sum((1, 2)).mean()), "n_params", m.n_params(), "nll", float(nll.mean()), "grad finite", ok,
          f"{time.time()-t0:.2f}s")
