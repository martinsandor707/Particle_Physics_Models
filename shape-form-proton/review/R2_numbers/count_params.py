"""R2_numbers: free-parameter count by perturbation + Jacobian rank, and normalisation of the output maps."""
import json, math, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
import r2lib, shapelab

C, geo, data = r2lib.setup()
dev = C["DEVICE"]
A_bar = shapelab.mean_acceptance(C, data["train"], front=True, back=True)
M = r2lib.build_models(C, data, A_bar, rungs=True, extra_mlp_acc=False)
res = {}
lnp = torch.linspace(math.log(0.24), math.log(20.0), 64, device=dev, dtype=torch.float64)
lnp_wide = torch.linspace(math.log(0.15), math.log(30.0), 64, device=dev)


@torch.no_grad()
def effect_count(m, delta):
    m = m.double()
    base = m(lnp)
    eff, tot, per_tensor = 0, 0, {}
    for name, p in m.named_parameters():
        flat = p.data.view(-1)
        e = 0
        for i in range(flat.numel()):
            old = float(flat[i])
            flat[i] = old + delta
            d = float((m(lnp) - base).abs().max())
            flat[i] = old
            e += d > 1e-9
        per_tensor[name] = [int(e), int(flat.numel())]
        eff += e; tot += flat.numel()
    return int(eff), int(tot), per_tensor


def jac_rank(m):
    """Numerical rank of d logP / d(free coefficients) over the p grid (float64, autograd)."""
    m = m.double()
    params = [p for p in m.parameters()]
    # restrict to entries that change the output (mask for PolyFormula)
    core = m.base if hasattr(m, "base") else m
    if hasattr(core, "mask"):
        sel = core.mask.bool().reshape(-1)
    else:
        sel = None
    flat0 = torch.cat([p.detach().reshape(-1) for p in params])
    shapes = [p.shape for p in params]
    names = [n for n, _ in m.named_parameters()]

    def f(v):
        i = 0
        out = {}
        for n, s in zip(names, shapes):
            k = int(np.prod(s))
            out[n] = v[i:i + k].reshape(s)
            i += k
        return torch.func.functional_call(m, out, (lnp,)).reshape(-1)

    J = torch.autograd.functional.jacobian(f, flat0, vectorize=True, strategy="forward-mode")   # (64*4096, n_raw)
    if sel is not None and len(names) == 1:
        J = J[:, sel]
    s = torch.linalg.svdvals(J)
    s = (s / s[0]).cpu().numpy()
    return {"n_columns": int(J.shape[1]), "rank(tol 1e-10)": int((s > 1e-10).sum()), "rank(tol 1e-7)": int((s > 1e-7).sum()),
            "smallest 5 sigma/sigma_max": [float(x) for x in s[-5:]]}


for name in ("PS_slab_chosen", "PS_stdacc_chosen", "PS_slab_n1", "PS_slab_n2", "PS_stdacc_n2", "report_gflash_n6", "ref_slab_gflash_n6",
             "ref_std_gflash_acc_n6"):
    m = M[name]
    claimed = int(m.n_params()) if hasattr(m, "n_params") else sum(p.numel() for p in m.parameters())
    r = {"n_params() claimed": claimed}
    for delta in (0.05, -0.05, 1e-3):
        eff, tot, per = effect_count(m, delta)
        r[f"effective (delta={delta})"] = eff
    r["raw entries"] = tot
    r["per tensor (effective, raw)"] = per
    try:
        r["jacobian"] = jac_rank(m)
    except Exception as e:
        r["jacobian"] = f"failed: {e}"
    m.float()
    with torch.no_grad():
        for tag, g in (("train range p 0.24-20", lnp.float()), ("wide p 0.15-30", lnp_wide)):
            lp = m(g)
            r[f"normalisation {tag}: max |logsumexp|"] = float(torch.logsumexp(lp.reshape(len(g), -1), 1).abs().max())
            r[f"normalisation {tag}: finite"] = bool(torch.isfinite(lp).all())
    res[name] = r
    print(name, json.dumps({k: v for k, v in r.items() if k != "per tensor (effective, raw)"}), flush=True)

# ProtonShape: which named parameters carry how many coefficients (chosen n=6)
core = M["PS_slab_chosen"]
res["PS names x degree (chosen)"] = {n: int(core.deg[n]) + 1 for n in core.names}
json.dump(res, open(f"{r2lib.OUT}/count_params.json", "w"), indent=1)
