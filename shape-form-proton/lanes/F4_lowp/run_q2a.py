"""Q2, first batch: shower + spike + stopping track (StopMix), free vs tied weights."""
import os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/F4_lowp")
import torch
import lane, fam, shapelab

C, geo, data, base, c, s = lane.setup()
ref, _ = lane.ref_val(C, data, base, c, s)
dev = C["DEVICE"]
VARIANTS = {
    "mix_sp_lnT_n2":        dict(degree=2, spike=True, track=False, var="lnT"),
    "mix_sp_tr_lnT_n2":     dict(degree=2, spike=True, track=True, var="lnT"),
    "mix_sp_tr_lnp_n2":     dict(degree=2, spike=True, track=True, var="lnp"),
    "mix_tie_lnT_n2":       dict(degree=2, spike=True, track=True, tie=True, var="lnT"),
    "mix_tie_rg_lnT_n2":    dict(degree=2, spike=True, track=True, tie=True, gate="range", var="lnT"),
    "mix_tie_lnT_n1":       dict(degree=1, spike=True, track=True, tie=True, var="lnT", degrees={"logit_tr": 2}),
    "mix_sp_tr_lnT_n3w":    dict(degree=2, spike=True, track=True, var="lnT", degrees={"logit_sp": 3, "logit_tr": 3}),
}
names = sys.argv[1:] or list(VARIANTS)
for name in names:
    kw = VARIANTS[name]
    torch.manual_seed(0)
    m = lane.init_gamma(C, data, fam.StopMix(C, lnp0=c, lnp_s=s, **kw).to(dev))
    try:
        row, val = lane.fit_eval(C, data, base, ref, m, name, steps=3000, extra=f"StopMix {kw}")
        bb = lane.by_bin(data, val, ref)
        print("   per-bin d vs ref:", " ".join(f"{k}:{bb[k][1]:+.3f}" for k in range(6)), flush=True)
        with torch.no_grad():
            lnp = torch.log(torch.tensor([0.65, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0], device=dev))
            lw, comp, th = m.components(lnp)
            print("   p      :", " ".join(f"{float(v):7.2f}" for v in lnp.exp()))
            for nm, w in zip(("w_sh", "w_tube", "w_spike"), lw):
                if w is not None:
                    print(f"   {nm:7s}:", " ".join(f"{float(v):7.3f}" for v in w.exp().flatten()))
            if m.use_track:
                print("   R [mm] :", " ".join(f"{float(v):7.1f}" for v in m.range_mm(th).flatten()))
            cst = {n: float(m.coef[i, 0]) for i, n in enumerate(m.names) if m.deg[n] == 0}
            print("   const  :", {k: round(v, 4) for k, v in cst.items()}, flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("FAILED", name, repr(e), flush=True)
