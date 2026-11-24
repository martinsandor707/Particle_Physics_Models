"""Smoke test of fam.LatMix: construct every block kind, check finiteness, normalisation, gradients; build the references."""
import time
import torch
import lab, fam

t0 = time.time()
C, geo, data, c, s = lab.setup(slab=True)
dev = C["DEVICE"]
print("setup", round(time.time() - t0, 1), "s", flush=True)
lnp = torch.linspace(-1.4, 3.0, 7, device=dev)
tr = data["train"]
b = next(iter(tr.batches(256)))
for comps in [("cauchy", "cauchy"), ("cauchy", "expr"), ("gauss", "expr"), ("cauchy", "exp"), ("cauchy", "k0"), ("student",),
              ("cauchy", "student"), ("cauchy", "cauchy", "expr"), ("cauchy", "ggamma"), ("cauchy", "sexp"), ("cauchy", "sexpr"),
              ("ggamma",)]:
    for depth, frac in (("lin", "logit"), ({"log_w0": ("sqrt",), "log_w1": ("sat",), "lf1": ("pos", "neg")}, "logit"), ("lin", "expdecay")):
        if len(comps) == 1 and (isinstance(depth, dict) or frac == "expdecay"):
            continue
        m = fam.LatMix(C, degree=2, comps=comps, depth=depth, frac=frac, lnp0=c, lnp_s=s).to(dev)
        lp = m(lnp)
        nll, _ = C["event_nll"](m, b, tr.box)
        loss = nll.mean()
        loss.backward()
        g = m.coef.grad
        print(f"{str(comps):38s} {str(frac):9s} n={m.n_params():3d} finite={bool(torch.isfinite(lp).all())} "
              f"sum-1={float((lp.exp().sum((1, 2)) - 1).abs().max()):.1e} loss={float(loss):.4f} grad_finite={bool(torch.isfinite(g).all())} "
              f"zero-grad coefs among free={int(((g == 0) & (m.mask > 0)).sum())}", flush=True)
for d in (1, 2):
    lab.reference(C, data, c, s, d)
print("done", round(time.time() - t0, 1), "s")
