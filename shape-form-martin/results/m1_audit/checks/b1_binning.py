"""M1-d diagnostic D6(b) - binning error of the component formula: cell-centre rule vs exact cell integration at the
SAME (legacy) coefficients, on validation.

    uv run python -B shape-form-martin/results/m1_audit/checks/b1_binning.py

Models (``sfm.families.build_variant``, legacy coefficients, stored acceptance template):
    legacy formula   ``sfm.legacy.load_component_formula``        (the benchmark of record)
    legacy_centre    same forms, cell-centre evaluation            (must equal the benchmark)
    legacy_exact     same forms and coefficients, exact cell masses
    survival_exact   exact cell masses, regulariser-free halos (a slightly different function)
Scores: unmodified ``common.test_nll`` per model (K = 64, seed 12345) inside ``deterministic()``; the localisation
uses ``sfm.metric.test_nll_multi(return_parts=True)`` on the same points (exact split into hit and normalisation
terms). No training.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm.families import build_variant, load_legacy_core  # noqa: E402

common, metric, legacy = chk.common, chk.metric, chk.legacy

Z_EDGES = np.array([-200, -20, 12, 45, 100, 400, 1450], float)
R_EDGES = np.array([0, 12.5, 30, 100, 700], float)
P_FINE = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0 + 1e-6]
P_FINE_LAB = ["<0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20"]


def main():
    t0 = time.time()
    sets, extras = chk.load_cpu(("train", "val"))
    out = {}
    with chk.gpu_lock("m1d D6b binning"):
        val = sets["val"].to(chk.DEV)
        formula = legacy.load_component_formula(sets["train"])
        core, logA = load_legacy_core(chk.DEV)
        models = {"formula": formula,
                  "legacy_centre": build_variant("legacy_centre", core, logA).eval(),
                  "legacy_exact": build_variant("legacy_exact", core, logA).eval(),
                  "survival_exact": build_variant("survival_exact", core, logA).eval()}
        nll = {}
        with metric.deterministic():
            for k, m in models.items():
                nll[k] = common.test_nll(m, val).astype(np.float64)
        parts = metric.test_nll_multi([models["legacy_centre"], models["legacy_exact"], models["survival_exact"]], val,
                                      return_parts=True, deterministic_sum=True)
        # map-level KL(centre || exact) at a few momenta
        lnp = torch.log(torch.tensor([0.2, 0.5, 1.0, 3.0, 10.0, 20.0], device=chk.DEV))
        with torch.no_grad():
            lc, le = models["legacy_centre"](lnp).double(), models["legacy_exact"](lnp).double()
            kl = (lc.exp() * (lc - le)).sum((1, 2)).cpu().numpy()
            # where in the map: |P_c - P_e| mass by z' band
            dP = (lc.exp() - le.exp()).abs().cpu().numpy()
        zc = common.ZC.cpu().numpy()
        out["map_kl_centre_exact"] = {"p_GeV": [0.2, 0.5, 1.0, 3.0, 10.0, 20.0], "kl": kl,
                                      "share_of_L1_diff_at_z_lt_45mm": (dP[:, zc < 45].sum((1, 2)) / dP.sum((1, 2)))}
    p = sets["val"].ev["p"].numpy().astype(np.float64)
    out["means"] = {k: chk.mean_se(v) for k, v in nll.items()}
    out["centre_vs_formula"] = {"max_abs": float(np.abs(nll["legacy_centre"] - nll["formula"]).max()),
                                "mean": float((nll["legacy_centre"] - nll["formula"]).mean())}
    d = nll["legacy_exact"] - nll["legacy_centre"]
    ds = nll["survival_exact"] - nll["legacy_centre"]
    out["exact_minus_centre"] = chk.mean_se(d)
    out["survival_minus_centre"] = chk.mean_se(ds)
    out["survival_minus_exact"] = chk.mean_se(ds - d)
    g = np.clip(np.searchsorted(P_FINE, p, side="right") - 1, 0, len(P_FINE_LAB) - 1)
    out["by_p"] = {lab: {"n": int((g == i).sum()), "exact_minus_centre": chk.mean_se(d[g == i]),
                         "contribution": float(d[g == i].sum() / len(d)),
                         "survival_minus_centre": chk.mean_se(ds[g == i])} for i, lab in enumerate(P_FINE_LAB)}
    # ---- localisation: hit term by region of the hit centre, and the normalisation term ---------------------------------
    ev = chk.hit_event(sets["val"])
    xyz = sets["val"].xyz.numpy().astype(np.float64)
    z, r = xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
    N = len(p)
    hit_diff = (parts["hit"][0].astype(np.float64) - parts["hit"][1].astype(np.float64))        # centre - exact = +: exact worse
    norm_diff = parts["weight_sum"].astype(np.float64) * (parts["log_mass"][1].astype(np.float64) - parts["log_mass"][0].astype(np.float64))
    out["split"] = {"hit_term": float(hit_diff.sum() / N), "norm_term": chk.mean_se(norm_diff),
                    "closure": float(hit_diff.sum() / N + norm_diff.mean() - (parts["nll"][1].astype(np.float64) - parts["nll"][0].astype(np.float64)).mean())}
    iz = np.clip(np.searchsorted(Z_EDGES, z, side="right") - 1, 0, len(Z_EDGES) - 2)
    ir = np.clip(np.searchsorted(R_EDGES, r, side="right") - 1, 0, len(R_EDGES) - 2)
    reg = np.zeros((len(Z_EDGES) - 1, len(R_EDGES) - 1))
    np.add.at(reg, (iz, ir), hit_diff)
    out["hit_term_by_region_1e-4"] = {"z_edges": Z_EDGES, "r_edges": R_EDGES, "matrix": reg / N * 1e4}
    # per-event sums of the hit term in z' < 45 mm, for an SE
    per_ev_front = np.bincount(ev, weights=np.where(z < 45, hit_diff, 0.0), minlength=N)
    per_ev_rest = np.bincount(ev, weights=np.where(z >= 45, hit_diff, 0.0), minlength=N)
    out["hit_term_z_lt_45"] = chk.mean_se(per_ev_front)
    out["hit_term_z_ge_45"] = chk.mean_se(per_ev_rest)
    out["hit_term_z_lt_45_by_p"] = {lab: chk.mean_se(per_ev_front[g == i]) for i, lab in enumerate(P_FINE_LAB)}
    chk.write("b1_binning.json", out, t0)
    for k in ("means", "centre_vs_formula", "exact_minus_centre", "survival_minus_centre", "survival_minus_exact", "by_p",
              "split", "hit_term_z_lt_45", "hit_term_z_ge_45", "map_kl_centre_exact"):
        print(k, out[k])
    print(np.round(out["hit_term_by_region_1e-4"]["matrix"], 3))


if __name__ == "__main__":
    main()
