"""M1 audit D8 - zero-training ensemble probe (validation).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d8.py

Members: mlp_std (4x32, sfm.legacy) and shape-form-original/results/pion/mlp_L3xH{16,32,64}.pt
(common.FieldModel(hidden=H, n_layers=3), weights under "model"; read only). mlp_L3xH128.pt is skipped: its checkpoint
says done=False (verified and recorded here). Uniform probability averages log P = logsumexp_m(log P_m) - ln M:
all four; mlp_std + each local one; the three local ones. Every model is scored on identical fold points
(K=64, seed 12345, batch 128, deterministic sums); paired deltas against mlp_std (and the formula for reference).
Map level: symmetric KL between members (and against the formula) at the 12 momenta of D3. Writes d8.json.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    run = L.Run("d8")
    sets, extras, mlp, formula = L.load()
    val, train = sets["val"], sets["train"]
    ck128 = torch.load(L.paths.ORIGINAL_DIR / "results" / "pion" / "mlp_L3xH128.pt", map_location="cpu",
                       weights_only=False)
    h128 = {"done": bool(ck128.get("done")), "epoch": int(ck128["epoch"]), "skipped": not bool(ck128.get("done"))}
    del ck128
    members = {"mlp_std": mlp}
    meta = {"mlp_std": {"file": "shape-form-proton/results/particles/pion/mlp_std.pt", "n_params":
                        int(sum(p.numel() for p in mlp.parameters())), "arch": "FieldModel(32, 4)"}}
    for H in (16, 32, 64):
        m, mt = L.load_local_mlp(H)
        assert mt["done"], f"mlp_L3xH{H}.pt is not finished"
        members[f"L3xH{H}"] = m
        meta[f"L3xH{H}"] = mt | {"arch": f"FieldModel({H}, 3)"}
    local = ["L3xH16", "L3xH32", "L3xH64"]
    ens = {"ens: all four": L.LogMix([members[k] for k in ["mlp_std"] + local])}
    for k in local:
        ens[f"ens: mlp_std + {k}"] = L.LogMix([members["mlp_std"], members[k]])
    ens["ens: three local"] = L.LogMix([members[k] for k in local])
    models = {**members, **ens, "formula": formula}
    names = list(models)
    nll = L.metric.test_nll_multi(list(models.values()), val, deterministic_sum=True).astype(np.float64)
    parts = L.parts_val()
    check = {"mlp_std": float(np.abs(nll[0] - parts["nll"][0]).max()),
             "formula": float(np.abs(nll[names.index("formula")] - parts["nll"][1]).max())}
    j0, jf = 0, names.index("formula")
    composition = {"ens: all four": ["mlp_std"] + local, "ens: three local": local,
                   **{f"ens: mlp_std + {k}": ["mlp_std", k] for k in local}}
    rows = []
    for j, n in enumerate(names):
        extra = {}
        if n in composition:
            best = min(composition[n], key=lambda k: nll[names.index(k)].mean())
            extra = {"members": composition[n], "best_member": best,
                     "minus_best_member": L.paired(nll[j], nll[names.index(best)])}
        rows.append({"model": n, **extra, "nll": L.mean_se(nll[j]), "minus_mlp_std": L.paired(nll[j], nll[j0]),
                     "minus_formula": L.paired(nll[j], nll[jf]),
                     "n_params": meta[n]["n_params"] if n in meta else
                     (int(sum(sum(p.numel() for p in m.parameters()) for m in models[n].models)) if n.startswith("ens")
                      else int(L.legacy.n_params(formula)))})
    # map-level symmetric KL at the 12 momenta of D3
    p12 = L.momenta12(train)
    lp = {n: L.logP_at(m, p12) for n, m in {**members, "formula": formula}.items()}
    skl = {}
    for a, b in itertools.combinations(list(lp), 2):
        v = L.kl_map(lp[a], lp[b]) + L.kl_map(lp[b], lp[a])
        skl[f"{a} | {b}"] = {"per_momentum": v, "mean": float(v.mean())}
    out = {"members": meta, "mlp_L3xH128": h128, "scores": rows, "parts_file_check_max_abs": check,
           "momenta_GeV": p12, "symmetric_kl": skl,
           "notes": {"ensemble weights": "uniform", "settings": "K=64, seed 12345, batch 128, deterministic sums",
                     "local checkpoints": "trained by common.fit (early stopping on K=16 validation NLL, best state "
                                          "kept); best_val_K16 is the checkpoint's own record"}}
    run.write(out)
    for r in rows:
        print(f"{r['model']:24s} {r['nll']['mean']:.6f}  vs std {r['minus_mlp_std']['mean']:+.6f} ± {r['minus_mlp_std']['se']:.6f}")
    for k, v in skl.items():
        print(k, round(v["mean"], 5))


if __name__ == "__main__":
    main()
