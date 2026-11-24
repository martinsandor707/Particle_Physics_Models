"""Task 2 - the report's MLP (FieldModel) trained with the report's `fit` (epochs=60, lr=3e-3, smooth=1e-4, patience=15),
under the slab likelihood (default) or the standard one (--slab 0).

usage: run_slab_mlp.py --hidden 16 --layers 3 --batch 32 [--slab 1] [--acc none|front|both]
Checkpoint of `fit`: <out>/proton/mlp_L<l>xH<h>[_b<batch>].pt (resumable); evaluation: ref_<lik>_mlp_L<l>xH<h>[_b<batch>].json/.pt
"""
import argparse
import copy
import json
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/G2_reference")
import torch

import shapelab
import fam

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--slab", type=int, default=1)
ap.add_argument("--acc", default="none")
ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args()

LANE = "/home/user/tamas/shape-form-proton/lanes/G2_reference"
lik = "slab" if a.slab else "std"
OUT = f"{LANE}/results_{lik}"
t0 = time.time()
C, geo, data = shapelab.setup(batch=a.batch, out_dir=OUT, slab=bool(a.slab))
assert C["BATCH"] == a.batch
base = shapelab.baseline_nll(C, data)
dev = C["DEVICE"]
tag = f"L{a.layers}xH{a.hidden}" + ("" if a.batch == 32 else f"_b{a.batch}") + ("" if a.acc == "none" else f"_acc{a.acc}")

if a.acc == "none":
    model = fam.MLPRef(C, a.hidden, a.layers).to(dev)
else:
    A = shapelab.mean_acceptance(C, data["train"], front=True, back=(a.acc == "both"))
    model = fam.MLPAcc(C, a.hidden, a.layers, A=A).to(dev)
print(f"setup {time.time() - t0:.0f}s  {tag}  params {model.n_params()}  likelihood {lik}  batch {a.batch}", flush=True)

# timing probe on a throw-away copy (the real training starts from the untouched initialisation)
probe = copy.deepcopy(model)
opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
tr = data["train"]
t1, k = time.time(), 0
for b in tr.batches(a.batch, shuffle=True, gen=torch.Generator().manual_seed(0)):
    nll, logP = C["event_nll"](probe, b, tr.box)
    loss = nll.mean() + 1e-4 * ((logP[:, 2:] - 2 * logP[:, 1:-1] + logP[:, :-2]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    k += 1
    if k == 60:
        break
torch.cuda.synchronize()
per = (time.time() - t1) / k
print(f"timing probe: {per * 1e3:.1f} ms/step -> {per * (len(tr) / a.batch):.0f} s/epoch", flush=True)
del probe, opt

path = f"{OUT}/proton/mlp_{tag}.pt"
t1 = time.time()
hist = C["fit"](model, data, path, epochs=a.epochs, lr=3e-3, smooth=1e-4)
print(f"fit {time.time() - t1:.0f}s  epochs run {len(hist['val'])}  best val (K=16) {min(hist['val']):.4f}", flush=True)
res, val, test = shapelab.evaluate(C, data, model, f"ref_{lik}_mlp_{tag}", base,
                                   extra={"protocol": f"report fit(epochs={a.epochs}, lr=3e-3, smooth=1e-4, patience=15), batch {a.batch}",
                                          "epochs_run": len(hist["val"]), "fit_seconds": time.time() - t1, "acceptance": a.acc})
print(json.dumps(res, indent=1), flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
