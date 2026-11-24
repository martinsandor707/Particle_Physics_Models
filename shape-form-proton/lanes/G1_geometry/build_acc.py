"""G1: build the transverse raster and the per-event acceptances (full = slab AND smooth outline; cells = exact cell union)."""
import os, sys, time
import numpy as np, torch
import g1lib

t0 = time.time()
C, geo, data, G = g1lib.load()
dev = C["DEVICE"]
det = g1lib.Detector(G, dev)
print(f"raster {det.idx_np.shape}, tower pixels {(det.idx_np >= 0).mean():.4f}, smooth {(det.smooth_np).mean():.4f}, "
      f"fill factor boxes/smooth = {(det.idx_np >= 0).sum() / det.smooth_np.sum():.5f}  ({time.time() - t0:.0f}s)", flush=True)
# consistency check: the numerical acceptance with an everywhere-inside raster must equal shapelab's closed form
sh = data["val"]
R, zb, org = g1lib.geometry_tensors(data, G, "val", dev)
a = g1lib.acceptance(C, det, R[:64], zb[:64], org[:64], sh.box)
b = sh.batch(torch.arange(64, device=dev))
ref = C["event_slab"](b, sh.box)
print("max |A_slab(g1lib) - A_slab(shapelab)| =", float((a["slab"] - ref).abs().max()),
      "| max (A_full - A_slab) =", float((a["full"] - a["slab"]).max()), "| mean A_full/A_slab =", float(a["full"].sum() / a["slab"].sum()),
      "| mean A_cells/A_slab =", float(a["cells"].sum() / a["slab"].sum()), flush=True)
for split in ("val", "test", "train"):
    keys = ("full", "cells", "mx", "mx2") if split != "train" else ("full", "cells")
    out = g1lib.build_acceptance(C, det, data, G, split, keys=keys, log=lambda s: print(s, flush=True))
    for k, v in out.items():
        torch.save(v, os.path.join(g1lib.ACC_DIR, f"acc_{k}_{split}.pt"))
    print(f"saved {split} ({time.time() - t0:.0f}s)", flush=True)
print("done", flush=True)
