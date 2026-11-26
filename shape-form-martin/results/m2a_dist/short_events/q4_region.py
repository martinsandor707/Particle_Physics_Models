"""Question 4/5 supplement: model mass in the regions that carry the short-event hit term (z' 300-1221 mm, r >= 200 mm,
last ring), averaged over the short events' momenta and over the others' momenta; the minimum component weights over
momentum. CPU only, read-only.   CUDA_VISIBLE_DEVICES= uv run python -B .../q4_region.py"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import se_common as S  # noqa: E402
assert not torch.cuda.is_available()
from sfm import models  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

dfac.set_template_dir(S.STAGE_DIR / "templates")
ctx = models.BuildContext(device="cpu", results_dir=S.RUNS, skip_init=True, log=lambda *a, **k: None)
V = S.val_info()
sh = S.short_mask()
Z_MIN, DZ, N_R = -200.0, 1650.0 / 128, 32
zc = Z_MIN + DZ * (np.arange(128) + 0.5)
RE = 700.0 * (np.arange(N_R + 1) / N_R) ** 1.7
deep = (zc >= 300) & (zc < 1221)
wide = RE[:-1] >= 200
out = {}
for s in (0, 1):
    for n in [f"A2s_s{s}", f"A2c_s{s}", f"A7_s{s}", f"A7s_s{s}", f"A8_s{s}", f"A9_s{s}", f"A8c_s{s}", f"A7l_s{s}",
              f"A12_s{s}", f"A9r_s{s}"] + (["A2s_s0_x2"] if s == 0 else []):
        m = models.load_run(f"m2a_dist/{n}", ctx).eval()
        res = {}
        for key, msk in (("short", sh), ("other", ~sh)):
            lnp = torch.tensor(np.log(V["p"][msk][:: (1 if key == "short" else 20)]), dtype=torch.float32)
            with torch.inference_mode():
                P = m(lnp).double().exp().numpy()
            res[key] = {"deep_wide": float(P[:, deep][:, :, wide].sum((1, 2)).mean()),
                        "deep_last_ring": float(P[:, deep, -1].sum(1).mean()),
                        "front_z<0": float(P[:, zc < 0].sum((1, 2)).mean()),
                        "front_z<0_wide": float(P[:, zc < 0][:, :, wide].sum((1, 2)).mean())}
        out[n] = res
base = {s: out[f"A2c_s{s}"] for s in (0, 1)}
print(f"{'run':11s} {'deep-wide short':>16s} {'rel A2c':>8s} {'last ring':>10s} {'rel A2c':>8s} {'front':>9s} "
      f"{'deep-wide other':>16s} {'rel A2c':>8s}")
for n, r in out.items():
    b = base[1 if "_s1" in n else 0]
    print(f"{n:11s} {r['short']['deep_wide']:16.5f} {r['short']['deep_wide'] / b['short']['deep_wide'] - 1:+8.4f} "
          f"{r['short']['deep_last_ring']:10.6f} {r['short']['deep_last_ring'] / b['short']['deep_last_ring'] - 1:+8.4f} "
          f"{r['short']['front_z<0']:9.5f} {r['other']['deep_wide']:16.5f} "
          f"{r['other']['deep_wide'] / b['other']['deep_wide'] - 1:+8.4f}")
S.write_json(out, "q4_region.json")
