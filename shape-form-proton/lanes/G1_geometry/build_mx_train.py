"""G1: per-event first azimuthal moments (mx, mx2) of the full acceptance for the TRAIN split (val/test were built by build_acc.py)."""
import os, time, torch
import g1lib
t0 = time.time()
C, geo, data, G = g1lib.load()
det = g1lib.Detector(G, C["DEVICE"])
out = g1lib.build_acceptance(C, det, data, G, "train", keys=("mx", "mx2"), log=lambda s: print(s, flush=True))
for k, v in out.items():
    torch.save(v, os.path.join(g1lib.ACC_DIR, f"acc_{k}_train.pt"))
print(f"done ({time.time() - t0:.0f}s)", flush=True)
