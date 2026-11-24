"""Burok-testek egy részecskére (a 02/03 kritériuma: a záporok F_EV hányadában az energia >= F_E része a testben, legkisebb térfogat).

    python particle_bodies.py <particle> <bins, pl. 0,1,2,3,4> [F_E F_EV]

Családok: a report cseppje (újraillesztve), tojás (két fél ellipszoid) és szuper-tojás (két fél szuper-ellipszoid, a tojásból indítva);
az illesztő és a testek az E1 szál kódja (lanes/E1_bodies: fam.py, fitlib.py, bodylab.py). Minden binre JSON:
results/particles/<részecske>/bodies_<krit>/<család>_binXX.json, benne a térfogat az adattartományban, a detektor-szeletben
(a bin tanító eseményeinek átlagos szelet-elfogadásával) és a teszt-lefedés.
"""
import json
import math
import os
import sys
import time

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lanes/E1_bodies"))
import numpy as np
import torch

import shapelab
import bodylab as bl
import fitlib as fl
import fam

particle, bins = sys.argv[1], [int(b) for b in sys.argv[2].split(",")]
F_E, F_EV = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.95, 0.98)
FAMS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["drop", "egg", "superegg"]
TAG = fl.crit_tag(F_E, F_EV)
OUT = os.path.join(ROOT, "results", "particles", particle, f"bodies_{TAG}")
os.makedirs(OUT, exist_ok=True)
REG = fam.registry()
L2 = math.log(2.0)
C, geo, data = shapelab.setup_particle(particle, out_dir=os.path.join(ROOT, "results", "particles", particle))
VOL = torch.exp(C["LOG_VOL"])
ZE, RE = C["Z_EDGES"].cpu().numpy(), C["R_EDGES"].cpu().numpy()


def cell_fraction(radius_fn, nz=8, nr=8):
    kz, kr = (np.arange(nz) + 0.5) / nz, (np.arange(nr) + 0.5) / nr
    z = (ZE[:-1, None] + np.diff(ZE)[:, None] * kz[None, :]).ravel()
    rz = radius_fn(z).reshape(len(ZE) - 1, nz)
    r = np.sqrt(RE[:-1, None] ** 2 + (RE[1:, None] ** 2 - RE[:-1, None] ** 2) * kr[None, :])
    return torch.tensor((rz[:, None, :, None] >= r[None, :, None, :]).mean((2, 3)), dtype=torch.float32, device=C["DEVICE"])


for b in bins:
    sets = {k: bl.bin_P(C, data, k, b)[0] for k in ("train", "val", "test")}
    tr = data["train"].subset(bl.bin_idx(data, "train", b))
    with torch.no_grad():
        abar = sum(C["event_slab"](bb, tr.box, 4, True, True).sum(0) for bb in tr.batches(256)) / len(tr)
    fits = {}
    for f in FAMS:
        path = os.path.join(OUT, f"{f}_bin{b:02d}.json")
        if os.path.exists(path):
            fits[f] = json.load(open(path))
            continue
        body = REG[f][0]
        obj = bl.objective_factory(body, P=sets["train"], f_e=F_E, f_ev=F_EV)
        extra = [fits["egg"]["x"] + [L2, L2]] if (f == "superegg" and "egg" in fits) else []
        t = time.time()
        x, lv, nfev = fl.fit_body(body, obj, x_extra=[np.asarray(e, float) for e in extra if body.valid(e)])
        res = fl.metrics(C, body, x, sets, F_E, F_EV)
        frac = cell_fraction(lambda z: body.radius(z, x, res["A"]))
        res.update({"family": f, "bin": b, "V_slab_L": float((VOL * frac * abar).sum()) / 1e6,
                    "V_grid_cells_L": float((VOL * frac).sum()) / 1e6, "nfev": int(nfev), "sec": round(time.time() - t, 1)})
        json.dump(res, open(path, "w"), indent=1)
        fits[f] = res
        print(f"{particle} {TAG} bin {b:2d} {f:9s} V {res['V_L']:8.1f} V_in {res['V_in_L']:8.1f} V_slab {res['V_slab_L']:8.1f} "
              f"cov_test {res['cov_test']:.3f} ({res['sec']}s)", flush=True)
print("done", particle, bins, flush=True)
