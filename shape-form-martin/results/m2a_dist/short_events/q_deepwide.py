"""Supplement to 3b/4: the hit term of the deep-wide hits (hit centre z' >= 300 mm and r >= 200 mm) against the rest,
for every recomputed comparison on the short events, and the far-halo / detector-halo share of the model density in
that region (short events' momenta). CPU only, read-only.
    CUDA_VISIBLE_DEVICES= uv run python -B .../q_deepwide.py <decomp.npz>"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import se_common as S  # noqa: E402
from sfm import stats  # noqa: E402
assert not torch.cuda.is_available()
from sfm import models  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

Z = np.load(sys.argv[1])
runs = [str(x) for x in Z["runs"]]
R = {n: i for i, n in enumerate(runs)}
hit = Z["hit"].astype(np.float64)
logm = Z["log_mass"].astype(np.float64)
wsum = Z["weight_sum"].astype(np.float64)
V = S.val_info()
val = V["_sets"]["val"]
off = val.off.numpy()
N = len(val)
ev = np.repeat(np.arange(N), np.diff(off))
xyz = val.xyz.numpy().astype(np.float64)
zp, r = xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
sh = S.short_mask()
dw = (zp >= 300) & (r >= 200)
out = {"region": "hit centre z' >= 300 mm and r >= 200 mm", "pairs": {}}
for a, b in [("A9_s0", "A2c_s0"), ("A9_s1", "A2c_s1"), ("A7_s0", "A2c_s0"), ("A7s_s0", "A2c_s0"), ("A8_s0", "A2c_s0"),
             ("A2s_s0", "A2c_s0"), ("A2s_s1", "A2c_s1"), ("A12_s0", "A2c_s0"), ("A9r_s0", "A2c_s0"),
             ("A2s_s0_x2", "A2c_s0"), ("A9_s0", "A7_s0"), ("A9_s0", "A7s_s0")]:
    d = -(hit[R[a]] - hit[R[b]])
    x_dw = np.bincount(ev, weights=d * dw, minlength=N)
    x_rest = np.bincount(ev, weights=d * ~dw, minlength=N)
    x_n = wsum * (logm[R[a]] - logm[R[b]])
    row = {k: S.pd(stats.paired(v[m], np.zeros(int(m.sum())))) for k, v, m in
           (("short_deepwide_hits", x_dw, sh), ("short_other_hits", x_rest, sh), ("short_norm", x_n, sh),
            ("other_deepwide_hits", x_dw, ~sh), ("other_other_hits", x_rest, ~sh))}
    out["pairs"][f"{a} - {b}"] = row
    print(f"{a:10s}-{b:7s} short: deep-wide hits {S.fmt(row['short_deepwide_hits'])}  other hits "
          f"{S.fmt(row['short_other_hits'])}  norm {S.fmt(row['short_norm'])} | others: deep-wide "
          f"{S.fmt(row['other_deepwide_hits'])}")
dfac.set_template_dir(S.STAGE_DIR / "templates")
ctx = models.BuildContext(device="cpu", results_dir=S.RUNS, skip_init=True, log=lambda *a, **k: None)
Z_MIN, DZ = -200.0, 1650.0 / 128
zc = Z_MIN + DZ * (np.arange(128) + 0.5)
RE = 700.0 * (np.arange(33) / 32) ** 1.7
deep, wide = (zc >= 300) & (zc < 1221), RE[:-1] >= 200
lnp = torch.tensor(np.log(V["p"][sh]), dtype=torch.float32)
shares = {}
for n in ("A2c_s0", "A2s_s0", "A7_s0", "A9_s0", "A9r_s0", "A2c_s1", "A9_s1"):
    m = models.load_run(f"m2a_dist/{n}", ctx).eval()
    with torch.inference_mode():
        th = m.theta(lnp)
        lw, comp = m.components(lnp, th)
        core = m.core_map(lnp, th)
        lse = torch.logsumexp((core + m.logA).reshape(len(lnp), -1), 1)[:, None, None]
        tot = m(lnp).double().exp().numpy()[:, deep][:, :, wide].sum((1, 2))
        res = {}
        for k in comp:
            ld = lw[k][:, None, None] + comp[k] + m.logA - lse                 # component k after the acceptance
            if m.det_halo is not None:
                ld = ld + F.logsigmoid(-th["logit_dh"])[:, None, None]
            res[k] = float((torch.exp(ld).double().numpy()[:, deep][:, :, wide].sum((1, 2)) / tot).mean())
        if m.det_halo is not None:
            dh = (F.logsigmoid(th["logit_dh"])[:, None, None] + m.det_halo()[None]).exp().double().numpy()
            res["det_halo"] = float((dh[:, deep][:, :, wide].sum((1, 2)) / tot).mean())
    shares[n] = {"deep_wide_mass": float(tot.mean()), "component_share": res}
    print(n, f"deep-wide mass {tot.mean():.5f}", {k: round(v, 4) for k, v in res.items()})
out["component_shares_short_momenta"] = shares
S.write_json(out, "q_deepwide.json")
