"""Független ellenőrző minta: a proton s2 szimuláció (másik véletlenmag) ugyanazzal a `load_data`-val, külön gyorsítótárba."""
import os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import torch
import protonlab
t0 = time.time()
C = protonlab.load_common(FILE_PATTERN="{particle}_standalone_N100000_M0_20GeV_reg_balanced_s2_v3.csv",
                          OUT_DIR="/home/user/tamas/shape-form-proton/results")
df, geo, sets = C["load_data"]("proton")
del df
out = os.path.join(protonlab.HERE, "cache", "proton_s2_sets.pt")
torch.save({"geo": geo, "sets": {k: {"ev": {a: b.cpu() for a, b in s.ev.items()}, "off": s.off.cpu(), "xyz": s.xyz.cpu(),
                                     "w": s.w.cpu(), "box": s.box.cpu()} for k, s in sets.items()}}, out)
print("geo", geo, {k: len(s) for k, s in sets.items()}, f"done in {time.time() - t0:.0f}s")
