"""Question 4/5: which parameters of the base moved between A2s and A2c versus between A2s and A7 / A8 / A9 / A7s /
A12, evaluated as raw-parameter functions of momentum (provider output by name) on the momenta of the short events and
on a momentum grid; the component weights over momentum (negative / exploding / saturated weights); the training
histories. CPU only, read-only.

    CUDA_VISIBLE_DEVICES= uv run python -B shape-form-martin/results/m2a_dist/short_events/q4_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import se_common as S  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("run with CUDA hidden")

from sfm import models  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402


def raw_params(model, lnp):
    model = getattr(model, "base", model)
    with torch.inference_mode():
        th = model.theta(lnp)
    return {k: v.double().numpy() for k, v in th.items() if v.ndim == 1}


def main():
    dfac.set_template_dir(S.STAGE_DIR / "templates")
    ctx = models.BuildContext(device="cpu", results_dir=S.RUNS, skip_init=True, log=lambda *a, **k: None)
    V = S.val_info()
    sh = S.short_mask()
    lnp_s = torch.tensor(np.log(V["p"][sh]), dtype=torch.float32)
    lnp_o = torch.tensor(np.log(V["p"][~sh][::10]), dtype=torch.float32)
    grid = torch.linspace(np.log(0.2), np.log(20.0), 49)
    out = {"seeds": {}}
    for s in (0, 1):
        names = [f"A2s_s{s}", f"A2c_s{s}", f"A7_s{s}", f"A7s_s{s}", f"A8_s{s}", f"A9_s{s}", f"A12_s{s}", f"A8c_s{s}",
                 f"A7l_s{s}", f"A9r_s{s}"]
        ms = {n: models.load_run(f"m2a_dist/{n}", ctx).eval() for n in names}
        base = f"A2s_s{s}"
        P = {n: {"short": raw_params(m, lnp_s), "other": raw_params(m, lnp_o), "grid": raw_params(m, grid)}
             for n, m in ms.items()}
        common_names = sorted(set.intersection(*[set(P[n]["short"]) for n in names]) - {"lnT"})
        tab = {}
        for k in common_names:
            row = {}
            for n in names[1:]:
                d_s = P[n]["short"][k] - P[base]["short"][k]
                d_o = P[n]["other"][k] - P[base]["other"][k]
                d_g = P[n]["grid"][k] - P[base]["grid"][k]
                row[n] = {"mean_short": float(d_s.mean()), "mean_other": float(d_o.mean()),
                          "max_abs_grid": float(np.abs(d_g).max())}
            tab[k] = row
        # how similar is each rung's move (from A2s) to A2c's move, over the shared parameters (short-event momenta)
        vec = {n: np.concatenate([P[n]["short"][k] - P[base]["short"][k] for k in common_names]) for n in names[1:]}
        sim = {}
        for n in names[1:]:
            a, c = vec[n], vec[f"A2c_s{s}"]
            sim[n] = {"norm": float(np.linalg.norm(a)), "cos_with_A2c_move": float(a @ c / (np.linalg.norm(a) *
                                                                                          np.linalg.norm(c) + 1e-300)),
                      "cos_with_A7_move": float(a @ vec[f"A7_s{s}"] / (np.linalg.norm(a) *
                                                                       np.linalg.norm(vec[f"A7_s{s}"]) + 1e-300))}
        # parameters whose move separates the cluster {A7, A7s, A8, A9} from {A2c, A8c, A7l}
        clusterA = [f"A7_s{s}", f"A7s_s{s}", f"A8_s{s}", f"A9_s{s}"]
        clusterB = [f"A2c_s{s}", f"A8c_s{s}", f"A7l_s{s}"]
        sep = []
        for k in common_names:
            a = np.array([tab[k][n]["mean_short"] for n in clusterA])
            b = np.array([tab[k][n]["mean_short"] for n in clusterB])
            spread = max(a.std(), b.std(), 1e-12)
            sep.append((k, float(a.mean() - b.mean()), float(a.mean()), float(b.mean()), float(spread),
                        float(abs(a.mean() - b.mean()) / spread)))
        sep.sort(key=lambda x: -abs(x[1]))
        print(f"== seed {s}: moves from {base} (mean over the short events' momenta of raw(p) - raw_A2s(p))")
        print("similarity of the moves:", json.dumps(sim, indent=0))
        print(f"{'param':12s} {'A(A7,A7s,A8,A9)':>16s} {'B(A2c,A8c,A7l)':>16s} {'A-B':>10s} {'sep':>7s}  "
              + " ".join(f"{n.split('_')[0]:>8s}" for n in names[1:]))
        for k, d, am, bm, spr, z in sep[:20]:
            print(f"{k:12s} {am:+16.4e} {bm:+16.4e} {d:+10.3e} {z:7.1f}  "
                  + " ".join(f"{tab[k][n]['mean_short']:+8.4f}" for n in names[1:]))
        # component weights over momentum (Q5): ln weights, A9's det-halo weight, far halo
        wts = {}
        for n, m in ms.items():
            mm = getattr(m, "base", m)
            if not hasattr(mm, "log_weights"):
                continue
            with torch.inference_mode():
                th = mm.theta(grid)
                Rr = mm.range_mm(th) if mm.use_track else None
                lw = {k: v.double().numpy() for k, v in mm.log_weights(th, Rr).items()}
                if "logit_dh" in th:
                    lw["det_halo"] = torch.nn.functional.logsigmoid(th["logit_dh"]).double().numpy()
            wts[n] = {k: {"min_w": float(np.exp(v).min()), "max_w": float(np.exp(v).max()),
                          "finite": bool(np.isfinite(v).all())} for k, v in lw.items()}
        # A9 detector-halo parameters
        dh = {}
        for n in (f"A9_s{s}", f"A9r_s{s}"):
            m = ms[n]
            with torch.inference_mode():
                th = m.theta(grid)
                L = float(m.det_halo.L) if hasattr(m.det_halo, "L") else None
            st = {k: float(v) for k, v in m.state_dict().items() if k.startswith("det_halo.raw")}
            dh[n] = {"logit_dh_range": [float(th["logit_dh"].min()), float(th["logit_dh"].max())],
                     "w_dh": float(torch.sigmoid(th["logit_dh"]).mean()), "raw": st, "L": L}
        print("det halo:", dh)
        a12 = {n: float(ms[n].log_f_last) for n in (f"A12_s{s}",)}
        print("A12 last-ring log factor:", a12)
        dh["A12_log_f_last"] = a12
        out["seeds"][s] = {"param_moves": tab, "similarity": sim, "separating": sep[:25], "weights": wts,
                           "det_halo": dh}
    # training histories (validation mean every 500 steps, selection "last")
    hist = {}
    for n in ("A2s_s0", "A2c_s0", "A7_s0", "A7s_s0", "A8_s0", "A9_s0", "A8c_s0", "A7l_s0", "A12_s0", "A9r_s0",
              "A2s_s1", "A2c_s1", "A7_s1", "A8_s1", "A9_s1", "A2s_s0_x2", "A2s_s1_x2"):
        d = json.loads((S.STAGE_DIR / n / "done.json").read_text())
        hist[n] = {"history": [(h["step"], h["val"]) for h in d["history"]], "selection": d["fit"]["selection"],
                   "selected_step": d["fit"]["selected_step"], "skipped_loss": d["fit"]["skipped_loss"],
                   "skipped_grad": d["fit"]["skipped_grad"], "kl_map": d.get("kl_map")}
        h = hist[n]["history"]
        print(f"{n:11s} sel {hist[n]['selection']}@{hist[n]['selected_step']} skipped {hist[n]['skipped_loss']}/"
              f"{hist[n]['skipped_grad']} kl_map {hist[n]['kl_map']}  "
              + " ".join(f"{v:.5f}" for _, v in h[-8:]))
    out["histories"] = hist
    S.write_json(out, "q4_params.json")


if __name__ == "__main__":
    main()
