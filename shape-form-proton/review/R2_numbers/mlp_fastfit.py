"""R2_numbers: is the 'MLP upper bound' under-trained? FieldModel(16,3) (881 params, the report's MLP) trained with fit_fast
(the optimiser used for the formulas) under the slab likelihood. usage: mlp_fastfit.py <steps> <lr> <lr_end> [init] [slab|std]
init = 'report' starts from the report's MLP weights; default = fresh init (torch.manual_seed(0))."""
import json, os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab, protonlab
import r2lib

steps, lr, lr_end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
init = sys.argv[4] if len(sys.argv) > 4 else "fresh"
metric = sys.argv[5] if len(sys.argv) > 5 else "slab"
OUTD = os.path.join(r2lib.OUT, "mlp_fastfit")
os.makedirs(OUTD, exist_ok=True)
C, geo, data = shapelab.setup(batch=256, out_dir=OUTD, slab=(metric == "slab"))
dev = C["DEVICE"]
torch.manual_seed(0)
m = C["FieldModel"](16, 3).to(dev)
if init == "report":
    m.load_state_dict(torch.load(os.path.join(protonlab.BASELINE_DIR, "mlp_L3xH16.pt"), map_location=dev, weights_only=False)["model"])
t0 = time.time()
hist = shapelab.fit_fast(C, data, m, steps=steps, lr=lr, lr_end=lr_end, eval_every=500, log=None)
m.eval()
out = {"metric": metric, "steps": steps, "lr": lr, "lr_end": lr_end, "init": init, "sec": round(time.time() - t0),
       "best_step": int(min(hist, key=lambda x: x[1])[0]), "hist": hist}
for sp in ("val", "test"):
    parts = r2lib.eval_parts(C, m, data[sp], nsub=8, want=("slab_mine", "slab_sl2"))
    out[f"{sp}_slab8"] = float(parts["slab_mine"].mean())
    out[f"{sp}_slab2"] = float(parts["slab_sl2"].mean())
    out[f"{sp}_testnll_{metric}"] = float(C["test_nll"](m, data[sp]).mean())
tag = f"mlp_slab_s{steps}_lr{lr}_{init}" if metric == "slab" else f"mlp_{metric}_s{steps}_lr{lr}_{init}"
torch.save({"state": m.state_dict()}, os.path.join(OUTD, tag + ".pt"))
json.dump(out, open(os.path.join(OUTD, tag + ".json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "hist"}))
