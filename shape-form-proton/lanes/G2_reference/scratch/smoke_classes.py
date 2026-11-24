"""Smoke test of the fam.py classes and the acceptance maps (no training)."""
import sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/G2_reference")
import g2lib, numpy as np, torch, shapelab, fam

t0 = time.time()
C, geo, data, use = g2lib.setup_both(batch=256)
c, s = shapelab.lnp_center_scale(data)
dev = C["DEVICE"]
print(f"setup {time.time()-t0:.1f}s  c,s = {c:.6f} {s:.6f}  geo {geo}  box {data['train'].box.tolist()}")
tr = data["train"]
print("sizes", {k: len(v) for k, v in data.items()}, "hits train", len(tr.w))
# theta vs p independence
cos_t = tr.ev["R"][:, 2, 2]; th = torch.acos(cos_t.clamp(-1, 1)).cpu().numpy(); lnp = tr.ev["lnp"].cpu().numpy()
print("theta range", th.min(), th.max(), "corr(theta, lnp)", np.corrcoef(th, lnp)[0, 1], "corr(theta,p)", np.corrcoef(th, np.exp(lnp))[0, 1])
zbc = (tr.ev["zb"] * cos_t).cpu().numpy()
print("D = zb*cos range", zbc.min(), zbc.max(), "corr(D, lnp)", np.corrcoef(zbc, lnp)[0, 1], "corr(D, theta)", np.corrcoef(zbc, th)[0, 1])
bins = tr.ev["bin"].cpu().numpy()
for b in (0, 1, 2, 5, 10, 19):
    m = bins == b
    print(f"  bin {b:2d}: n {m.sum():5d}  <theta> {th[m].mean():.4f} sd {th[m].std():.4f}  <D> {zbc[m].mean():.1f} sd {zbc[m].std():.1f}")

lnp_t = torch.tensor([np.log(0.3), np.log(1.0), np.log(5.0), np.log(19.0)], dtype=torch.float32, device=dev)
for kind in ("none", "front", "both", "frontP", "bothP", "margF", "margB"):
    t1 = time.time()
    kw = g2lib.acceptance(C, data, kind)
    m = g2lib.build(C, kind, 2, c, s, kw).init_from_hits(tr.hits()) if kind == "none" else g2lib.build(C, kind, 2, c, s, kw)
    lp = m(lnp_t)
    A = kw.get("A")
    print(f"{kind:7s} A {None if A is None else tuple(A.shape)} min {None if A is None else float(A.min()):} max {None if A is None else float(A.max())} "
          f"logP {tuple(lp.shape)} sum {lp.exp().sum((1,2)).tolist()} finite {bool(torch.isfinite(lp).all())} n_params {m.n_params()} ({time.time()-t1:.1f}s)")
    if kind in ("frontP", "bothP"):
        print("   edges", [round(float(e), 3) for e in kw["edges"]], "bucket idx", torch.bucketize(lnp_t, kw["edges"][1:-1].contiguous()).tolist())
# difference between binned and global mean acceptance
A0 = g2lib.acceptance(C, data, "front")["A"]; AP, edges, cnt = g2lib.mean_acceptance_binned(C, tr, front=True, back=False)
print("counts per bin", cnt.long().tolist())
print("max |A_bin - A_all| per bin", [round(float((AP[k] - A0).abs().max()), 4) for k in range(len(AP))])
# timing of test_nll
m = g2lib.build(C, "none", 2, c, s, {})
for lik in ("std", "slab"):
    use(lik); t1 = time.time(); v = C["test_nll"](m, data["val"]); torch.cuda.synchronize(); print(lik, "test_nll val", float(v.mean()), f"{time.time()-t1:.1f}s")
print("GPU mem MB", torch.cuda.max_memory_allocated() / 2**20)
