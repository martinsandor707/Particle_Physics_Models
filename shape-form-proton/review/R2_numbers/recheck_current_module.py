"""Re-check that the CURRENT proton_shape.py (modified 06:30, new lat_norm option) still reproduces the chosen checkpoints."""
import sys, json, importlib
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
import r2lib, shapelab
C, geo, data = r2lib.setup()
A_bar = shapelab.mean_acceptance(C, data["train"], front=True, back=True)
M = r2lib.build_models(C, data, A_bar, rungs=False, extra_mlp_acc=False)
Z = np.load(f"{r2lib.OUT}/per_event_s1.npz")
out = {}
for name, mm in (("PS_slab_chosen", "slab2"), ("PS_stdacc_chosen", "std")):
    parts = r2lib.eval_parts(C, M[name], data["val"], nsub=8)
    v = parts["slab_sl2"] if mm == "slab2" else C["test_nll"](M[name], data["val"])
    out[name] = {"val now": float(v.mean()), "val earlier": float(Z[f"{name}|{mm}|val"].mean()), "max |diff| per event": float(np.abs(v - Z[f"{name}|{mm}|val"]).max())}
import proton_shape, inspect
out["module mtime-check: has lat_norm"] = "lat_norm" in inspect.signature(proton_shape.ProtonShape.__init__).parameters
print(json.dumps(out, indent=1))
json.dump(out, open(f"{r2lib.OUT}/recheck_current_module.json", "w"), indent=1)
