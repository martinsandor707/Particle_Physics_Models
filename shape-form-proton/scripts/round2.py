"""2. kör: a ProtonShape-változatok illesztése és összevetése (validáción), szelet- vagy standard likelihooddal.

Használat: python scripts/round2.py <slab|std|stdacc> <config név> [<config név> ...]
  slab   – 3D-s (szeletre normált) likelihood
  std    – a report standard likelihoodja, sima modell
  stdacc – a report standard likelihoodja, a modell × Ā (eseményátlagolt akceptancia, elöl+hátul)
"""
import json, os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab, families
from proton_shape import ProtonShape

CONFIGS = {
    "cc_n1":   dict(degree=1, lateral="cc", degrees={"logit_tr": 2}),
    "ce_n1":   dict(degree=1, lateral="ce", degrees={"logit_tr": 2}),
    "cc_n2":   dict(degree=2, lateral="cc"),
    "ce_n2":   dict(degree=2, lateral="ce"),
    "ce_n1b":  dict(degree=1, lateral="ce", degrees={"logit_tr": 2, "logit_sp": 2}),
    "ce_n1c":  dict(degree=1, lateral="ce", degrees={"logit_tr": 3, "logit_sp": 3}),
    "ce_n3":   dict(degree=3, lateral="ce"),
    "cc_n3":   dict(degree=3, lateral="cc"),
    "ce_n1_lnp": dict(degree=1, lateral="ce", degrees={"logit_tr": 2}, var="lnp"),
    "ce_n1_notie": dict(degree=1, lateral="ce", degrees={"logit_tr": 2}, tie=False),
    "ce_n1_nospike": dict(degree=1, lateral="ce", degrees={"logit_tr": 2}, spike=False),
    "ce_n1_notrack": dict(degree=1, lateral="ce", track=False),
    "ce_n1_plain": dict(degree=1, lateral="ce", track=False, spike=False),
    "ce_n2_plain": dict(degree=2, lateral="ce", track=False, spike=False),
    "cc_n2_plain": dict(degree=2, lateral="cc", track=False, spike=False),
    "gce_n1": dict(degree=1, lateral="gce", degrees={"logit_tr": 2}),
    "gce_n2": dict(degree=2, lateral="gce"),
    "cce_n1": dict(degree=1, lateral="cce", degrees={"logit_tr": 2}),
    "cce_n2": dict(degree=2, lateral="cce"),
    "gce_n2_plain": dict(degree=2, lateral="gce", track=False, spike=False),
    "cc_n3t": dict(degree=3, lateral="cc"),
}
mode, names = sys.argv[1], sys.argv[2:]
OUT = "/home/user/tamas/shape-form-proton/results/round2"
C, geo, data = shapelab.setup(batch=256, out_dir=OUT, slab=(mode == "slab"))
base = shapelab.baseline_nll(C, data)
c, s = shapelab.lnp_center_scale(data)
dev = C["DEVICE"]
A = shapelab.mean_acceptance(C, data["train"], front=True, back=True) if mode == "stdacc" else None
bins = {k: data[k].ev["bin"].cpu().numpy() for k in ("val", "test")}
for name in names:
    t0 = time.time()
    torch.manual_seed(0)
    core = ProtonShape(C, lnp0=c, lnp_s=s, **CONFIGS[name]).to(dev).init_from_data(C, data)
    m = shapelab.WithAcceptance(core, A).to(dev) if A is not None else core
    steps = int(os.environ.get("STEPS", 4000))
    shapelab.fit_fast(C, data, m, steps=steps, log=None)
    tag = f"{mode}_{name}"
    res, val, test = shapelab.evaluate(C, data, m, tag, base)
    d = val - base["mlp"]["val"]
    row = {"tag": tag, "mode": mode, "name": name, "n_params": res["n_params"], "val": res["val_nll"], "test": res["test_nll"],
           "val_vs_gflash_n6": res["val_vs_gflash_n6"], "val_vs_mlp": res["val_vs_mlp"],
           "val_vs_mlp_p<1": float(d[bins["val"] == 0].mean()), "val_vs_mlp_1-2": float(d[bins["val"] == 1].mean()),
           "val_vs_mlp_2-5": float(d[(bins["val"] >= 2) & (bins["val"] < 5)].mean()), "val_vs_mlp_p>=5": float(d[bins["val"] >= 5].mean()),
           "steps": steps, "sec": round(time.time() - t0)}
    with open(os.path.join(OUT, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{tag:22s} n={row['n_params']:3d} val {row['val']:.4f} test {row['test']:.4f} | vs MLP(same metric, report weights): all {row['val_vs_mlp'][0]:+.4f} "
          f"p<1 {row['val_vs_mlp_p<1']:+.3f} 1-2 {row['val_vs_mlp_1-2']:+.3f} 2-5 {row['val_vs_mlp_2-5']:+.4f} >=5 {row['val_vs_mlp_p>=5']:+.4f}  ({row['sec']}s)", flush=True)
