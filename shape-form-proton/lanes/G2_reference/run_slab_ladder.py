"""Task 1 - slab likelihood, the report's protocol: select_degree('gamma','core_tail'), n = 1..6, batch 32, epochs 150, lr 0.03.

Exactly the call of notebook 01 (cell 10), only the likelihood normalisation is the 3D slab (shapelab.setup(slab=True)).
Checkpoints of the report's `fit`: results_slab/proton/gflash_n<n>.pt (finished ones are only loaded, so the script is resumable).
Afterwards every degree is evaluated with shapelab.evaluate (val + test, K = 64, fixed seed) under the name ref_slab_gflash_n<n>.
"""
import json
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/G2_reference")
import numpy as np
import torch

import shapelab
import fam

LANE = "/home/user/tamas/shape-form-proton/lanes/G2_reference"
OUT = LANE + "/results_slab"

t0 = time.time()
C, geo, data = shapelab.setup(batch=32, out_dir=OUT, slab=True)
assert C["BATCH"] == 32 and C["SLAB"]
base = shapelab.baseline_nll(C, data)
c, s = shapelab.lnp_center_scale(data)
print(f"setup {time.time() - t0:.0f}s  lnp centre/scale {c:.6f} {s:.6f}", flush=True)

model, tab, models = C["select_degree"]("gamma", "core_tail", data, "proton", "gflash", degrees=range(1, 7), epochs=150, lr=0.03)
print(tab.round(5).to_string(), flush=True)
tab.to_csv(OUT + "/proton/gflash_degree_table.csv")
n_sel = int(model.degree)
print(f"chosen degree (1-SE rule, validation): {n_sel}   total {time.time() - t0:.0f}s", flush=True)

rows = []
for n, m in models.items():
    w = fam.GFlashRef(C, degree=n, lnp0=c, lnp_s=s).to(C["DEVICE"])
    w.base.load_state_dict(m.state_dict())
    ck = torch.load(f"{OUT}/proton/gflash_n{n}.pt", map_location="cpu", weights_only=False)
    res, val, test = shapelab.evaluate(C, data, w, f"ref_slab_gflash_n{n}", base,
                                       extra={"protocol": "report select_degree: fit(epochs=150, lr=0.03, patience=15), batch 32, warm start from n-1",
                                              "epochs_run": len(ck["history"]["val"]), "chosen_1SE": bool(n == n_sel)})
    rows.append(res)
    print(json.dumps(res), flush=True)
json.dump({"chosen_degree": n_sel, "rows": rows, "table": json.loads(tab.to_json(orient="index"))},
          open(OUT + "/proton/ref_slab_gflash_ladder.json", "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
