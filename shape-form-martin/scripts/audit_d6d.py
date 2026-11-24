"""M1 audit D6d - border clamping of the fold points (validation, K = 64, seed 12345, batch 128).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d6d.py

The fold points of the report metric are regenerated exactly as sfm.metric.test_nll_multi draws them (one
``sfm.metric.draw`` call per batch of 128 events from a CUDA generator seeded with 12345), rotated into the shower
frame, and classified (model-independent):
    outside the grid: z' < -200 mm, z' > 1450 mm (z' >= 1450 is clamped into z-bin 127), r >= 700 mm
    landing in the last ring (ring 31, incl. clamped), in the first z-bin (incl. clamped), in the last z-bin
Energy-weighted shares: event mean of sum_i w_i (1/K) sum_k 1[...]; all events and six momentum groups.
Extra (labelled): the part of the D2 hit term (w ln g_MLP - w ln g_F) carried by hits with >= 1 clamped point or
>= 1 point in a border cell. Writes d6d.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

common = L.common
K = 64
FLAGS = ("z_lt_-200", "z_gt_1450", "r_ge_700", "any_outside", "last_ring", "first_z_bin", "last_z_bin", "any_border")


@torch.inference_mode()
def point_flags(sh):
    """Per hit: fraction of its K points with each flag (H, n_flags) float64, in data order."""
    gen = torch.Generator(device=L.DEV).manual_seed(12345)
    out = []
    for b in sh.batches(128):
        u = L.metric.draw(b, sh.box, K, gen)                                       # the only consumer of gen
        pts = b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)
        z = pts[..., 2]
        r = torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
        iz, ir = common.locate(z, r)
        f = [z < common.Z_MIN, z >= common.Z_MAX, r >= common.R_MAX]
        f.append(f[0] | f[1] | f[2])
        f += [ir == common.N_R - 1, iz == 0, iz == common.N_Z - 1]
        f.append(f[4] | f[5] | f[6])
        out.append(torch.stack([x.double().mean(1) for x in f], 1).cpu().numpy())
    return np.concatenate(out)


def main():
    run = L.Run("d6d")
    sets, extras, _, _ = L.load()
    val = sets["val"]
    N = len(val)
    fl = point_flags(val)
    ev = L.hit_event(val)
    w = val.w.cpu().numpy().astype(np.float64)
    per_ev = np.stack([np.bincount(ev, weights=w * fl[:, j], minlength=N) for j in range(len(FLAGS))], 1)
    grp = L.p_group(val.ev["p"].cpu().numpy())
    parts = L.parts_val()
    ht = parts["hit"][0].astype(np.float64) - parts["hit"][1].astype(np.float64)
    out = {"K": K, "seed": 12345, "batch": 128, "flags": FLAGS,
           "definitions": {"z_gt_1450": "z' >= 1450 mm (outside the last z-bin's upper edge; clamped)",
                           "r_ge_700": "r >= 700 mm (clamped into ring 31)",
                           "last_ring": "ring 31 (r >= 641.9 mm), including clamped points",
                           "first_z_bin": "z-bin 0 (z' < -187.1 mm), including clamped points",
                           "last_z_bin": "z-bin 127 (z' >= 1437.1 mm), including clamped points"},
           "groups": {}}
    for gname, m in [("all", np.ones(N, bool))] + [(lab, grp == g) for g, lab in enumerate(L.P_GROUP_LABELS)]:
        row = {"n_events": int(m.sum())}
        for j, f in enumerate(FLAGS):
            row[f] = L.mean_se(per_ev[m, j])
            row[f]["n_events_with_any"] = int((per_ev[m, j] > 0).sum())
        out["groups"][gname] = row
    any_out = fl[:, 3] > 0
    any_border = fl[:, 7] > 0
    out["extra_hit_term"] = {
        "hits_total": int(len(w)), "hits_with_clamped_point": int(any_out.sum()),
        "hits_with_border_point": int(any_border.sum()),
        "hit_term_total": float(ht.sum() / N),
        "hit_term_from_hits_with_clamped_point": float(ht[any_out].sum() / N),
        "hit_term_from_hits_with_border_point": float(ht[any_border].sum() / N),
        "note": "hit term = w ln g_MLP - w ln g_F per hit (D2 convention), summed and divided by the number of events"}
    run.write(out)
    for g, row in out["groups"].items():
        print(g, {f: f"{row[f]['mean']:.3e}" for f in FLAGS})
    print(out["extra_hit_term"])


if __name__ == "__main__":
    main()
