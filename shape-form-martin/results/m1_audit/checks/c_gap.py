"""M1-d Part C helper - where the formula-MLP gap sits, by the same cell partition as the D7 score test, from the stored
per-hit parts of the audit (results/m1_audit/parts_val_k64.npz; validation, K = 64, seed 12345). CPU only.

    CUDA_VISIBLE_DEVICES="" uv run python -B shape-form-martin/results/m1_audit/checks/c_gap.py

gap = sum over hits of [w ln g_MLP - w ln g_F] (hit term, assigned to the region of the HIT CENTRE) + normalisation term.
Per-event sums give paired SEs. Also: strata of the event (zb, axis exit depth from c_side.npz if present).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402


def main():
    sets, extras = chk.load_cpu(("val",))
    sh = sets["val"]
    N = len(sh)
    f = np.load(chk.AUDIT / "parts_val_k64.npz")
    hit = f["hit"].astype(np.float64)                                  # (2, H): mlp, formula
    lm = f["log_mass"].astype(np.float64)
    ws = f["weight_sum"].astype(np.float64)
    nll = f["nll"].astype(np.float64)
    ev = chk.hit_event(sh)
    xyz = sh.xyz.numpy().astype(np.float64)
    z, r = xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
    dh = hit[0] - hit[1]                                                # > 0: formula worse
    norm = ws * (lm[1] - lm[0])
    d = nll[1] - nll[0]
    r31 = 700.0 * (31 / 32) ** 1.7
    r5 = 700.0 * (5 / 32) ** 1.7
    zlast, zfirst = 1450.0 - 1650.0 / 128, -200.0 + 1650.0 / 128
    reg = np.full(len(z), "bulk (60<=z'<1221, 30<=r<663)", dtype=object)
    reg[(z >= 60) & (z < 1221) & (r < r5)] = "core r<30 mm (60<=z'<1221)"
    reg[(z >= -20) & (z < 60)] = "entry (-20<=z'<60)"
    reg[z < -20] = "backward (z'<-20)"
    reg[z >= 1221] = "beyond typical back face (z'>=1221)"
    reg[r >= r31] = "border: last ring (r>=663)"
    reg[(z < zfirst) | (z >= zlast)] = "border: first/last z-bin"
    out = {"gap": chk.mean_se(d), "hit_term": float(dh.sum() / N), "norm_term": chk.mean_se(norm), "regions": {}}
    for name in sorted(set(reg)):
        m = reg == name
        pe = np.bincount(ev, weights=dh * m, minlength=N)
        wsh = np.bincount(ev, weights=sh.w.numpy().astype(np.float64) * m, minlength=N)
        out["regions"][name] = {"hit_term": chk.mean_se(pe), "share_of_gap": float(pe.mean() / d.mean()),
                                "energy_share": float(wsh.mean())}
    zb = sh.ev["zb"].numpy().astype(np.float64)
    full = zb >= 1221
    # proper-score core: hits in entry+core+bulk of events with zb in [1221, 1450] vs the rest
    inner = np.isin(reg, ["bulk (60<=z'<1221, 30<=r<663)", "core r<30 mm (60<=z'<1221)", "entry (-20<=z'<60)", "backward (z'<-20)"])
    pe_inner = np.bincount(ev, weights=dh * inner, minlength=N)
    out["hit_term_inner_regions_all_events"] = chk.mean_se(pe_inner)
    out["hit_term_inner_regions_zb_ge_1221"] = {"mean_over_all_events": float(pe_inner[full].sum() / N), "per_event_in_stratum": chk.mean_se(pe_inner[full])}
    out["norm_term_by_zb"] = {"zb<1221": float(norm[~full].sum() / N), "1221<=zb<=1450": float(norm[full & (zb <= 1450)].sum() / N),
                              "zb>1450": float(norm[zb > 1450].sum() / N)}
    side = chk.HERE / "c_side.npz"
    if side.exists():
        s = np.load(side)
        zex = s["z_exit"]
        edges = [0, 400, 700, 1000, 1450, np.inf]
        labs = ["<400", "400-700", "700-1000", "1000-1450", "none (>=1450)"]
        cls = np.clip(np.searchsorted(edges, zex, side="right") - 1, 0, len(labs) - 1)
        out["by_axis_exit_depth"] = {lab: {"n": int((cls == i).sum()), "d_e": chk.mean_se(d[cls == i]),
                                           "contribution": float(d[cls == i].sum() / N)} for i, lab in enumerate(labs) if (cls == i).sum() > 1}
        cont = (zex >= 1450) & full & (zb <= 1450)
        out["gap_contained_full_depth_events"] = {"n": int(cont.sum()), "d_e": chk.mean_se(d[cont]), "contribution": float(d[cont].sum() / N),
                                                  "hit_inner": float(pe_inner[cont].sum() / N)}
        # D6(c) per-event arrays by exit depth
        for tag in ("", "_valK1024"):
            fn = chk.HERE / f"b2_lookup{tag}.npz"
            if fn.exists():
                b = np.load(fn)
                keys = [k for k in b.files if k.startswith("K") and k.endswith("_m4")]
                for k in keys:
                    dd = b[k] - b[k.replace("_m4", "_m1")]
                    out[f"lookup_m4_minus_m1_{k.split('_')[0]}{tag}_by_exit"] = {
                        "side-leaking (z_exit<1450)": chk.mean_se(dd[zex < 1450]), "contained": chk.mean_se(dd[zex >= 1450]),
                        "zb>1450": chk.mean_se(dd[zb > 1450]), "1221<=zb<=1450": chk.mean_se(dd[(zb >= 1221) & (zb <= 1450)])}
    chk.write("c_gap.json", out)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
