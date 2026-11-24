"""M1-d Part A - independent recomputation of the D3 hybrids, the D6a gap at K = 64 / 1024 and D8 ensembles.

    uv run python -B shape-form-martin/results/m1_audit/checks/a_verify.py

Code path (deliberately not the audit's): every score is a separate call of the UNMODIFIED ``common.test_nll`` /
``common.event_nll`` (the audit used ``sfm.metric.test_nll_multi``); hybrids and ensembles are rebuilt here in float64
probability space (the audit worked in float32 log space). Validation only; no training.
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

common, metric, legacy = chk.common, chk.metric, chk.legacy


class Prob64(nn.Module):
    """Wraps fn(list of float64 probability maps) -> float64 probability map; returns float32 log P."""

    def __init__(self, models, fn):
        super().__init__()
        self.models, self.fn = nn.ModuleList(models), fn

    def forward(self, lnp):
        P = self.fn([m(lnp).double().exp() for m in self.models])
        P = P / P.sum((1, 2), keepdim=True)
        return torch.log(P).float()


def hybrid(long_model, lat_model):
    def fn(Ps):
        Pl, Pt = Ps
        return Pl.sum(2, keepdim=True) * (Pt / Pt.sum(2, keepdim=True))
    return Prob64([long_model, lat_model], fn)


def average(models):
    return Prob64(models, lambda Ps: torch.stack(Ps, 0).mean(0))


def score(model, val):
    with metric.deterministic():
        return common.test_nll(model, val).astype(np.float64)


@torch.no_grad()
def score_K(model, val, K, seed, batch):
    """common.event_nll in a loop like common.test_nll, but with a chosen K, seed and batch size."""
    gen = torch.Generator(device=chk.DEV).manual_seed(seed)
    with metric.deterministic():
        return torch.cat([common.event_nll(model, b, val.box, K, gen)[0] for b in val.batches(batch)]).cpu().numpy().astype(np.float64)


def local_mlp(H):
    ck = torch.load(chk.paths.ORIGINAL_DIR / "results" / "pion" / f"mlp_L3xH{H}.pt", map_location=chk.DEV, weights_only=False)
    m = common.FieldModel(hidden=H, n_layers=3).to(chk.DEV)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def sym_kl(a, b, p12):
    lnp = torch.log(torch.as_tensor(p12, dtype=torch.float32, device=chk.DEV))
    la, lb = a(lnp).double(), b(lnp).double()
    return float((((la.exp() - lb.exp()) * (la - lb)).sum((1, 2))).mean())


def main():
    t0 = time.time()
    sets, extras = chk.load_cpu(("train", "val"))
    p_train = sets["train"].ev["p"].numpy().astype(np.float64)
    p12 = np.quantile(p_train, (np.arange(12) + 0.5) / 12)
    audit = {k: chk.json.loads((chk.AUDIT / f"{k}.json").read_text()) for k in ("d3", "d6a", "d8")}
    out = {}
    with chk.gpu_lock("m1d part A verify"):
        val = sets["val"].to(chk.DEV)
        mlp = legacy.load_mlp_std()
        formula = legacy.load_component_formula(sets["train"])
        nm, nf = score(mlp, val), score(formula, val)
        rec_m, rec_f = np.load(chk.M0 / "mlp_val_nll.npy").astype(np.float64), np.load(chk.M0 / "formula_val_nll.npy").astype(np.float64)
        out["benchmarks"] = {"mlp": chk.mean_se(nm), "formula": chk.mean_se(nf), "gap": chk.mean_se(nf - nm),
                             "max_abs_vs_m0_parity": [float(np.abs(nm - rec_m).max()), float(np.abs(nf - rec_f).max())],
                             "mean_diff_vs_m0_parity": [float((nm - rec_m).mean()), float((nf - rec_f).mean())]}
        # ---- D3 hybrids ------------------------------------------------------------------------------------------------
        h1, h2 = score(hybrid(mlp, formula), val), score(hybrid(formula, mlp), val)
        grp = chk.p_group(val.ev["p"].cpu().numpy())
        d3 = {"H1 - MLP": chk.mean_se(h1 - nm), "H2 - MLP": chk.mean_se(h2 - nm), "H1 - formula": chk.mean_se(h1 - nf),
              "H2 - formula": chk.mean_se(h2 - nf), "interaction": chk.mean_se((h1 - nm) + (h2 - nm) - (nf - nm)),
              "H1 mean": float(h1.mean()), "H2 mean": float(h2.mean()),
              "by_p": {lab: {"H1 - MLP": chk.mean_se((h1 - nm)[grp == g]), "H2 - MLP": chk.mean_se((h2 - nm)[grp == g])}
                       for g, lab in enumerate(chk.P_LABELS)}}
        d3["audit"] = {k: [audit["d3"]["paired"][k]["mean"], audit["d3"]["paired"][k]["se"]]
                       for k in ("H1 - MLP", "H2 - MLP", "H1 - formula", "H2 - formula")}
        out["d3"] = d3
        # ---- D6a gap at K = 64 and K = 1024 ------------------------------------------------------------------------------
        d6 = {}
        for K, batch in ((64, 128), (1024, 32)):
            per_seed = {}
            stack = []
            for seed in (12345, 1, 2):
                a, b = score_K(mlp, val, K, seed, batch), score_K(formula, val, K, seed, batch)
                per_seed[str(seed)] = {"mlp": float(a.mean()), "formula": float(b.mean()), "gap": chk.mean_se(b - a)}
                stack.append(np.stack([a, b]))
            st = np.stack(stack)                                   # (seed, model, event)
            d6[str(K)] = {"per_seed": per_seed, "mlp_mean3": float(st[:, 0].mean()), "formula_mean3": float(st[:, 1].mean()),
                          "gap_seedavg": chk.mean_se((st[:, 1] - st[:, 0]).mean(0)), "_arr": st}
        k64, k1024 = d6["64"].pop("_arr"), d6["1024"].pop("_arr")
        d6["bias_64_minus_1024"] = {"mlp": chk.mean_se((k64[:, 0] - k1024[:, 0]).mean(0)),
                                    "formula": chk.mean_se((k64[:, 1] - k1024[:, 1]).mean(0)),
                                    "gap": chk.mean_se(((k64[:, 1] - k64[:, 0]) - (k1024[:, 1] - k1024[:, 0])).mean(0))}
        d6["audit_per_K"] = audit["d6a"]["per_K"]
        out["d6a"] = d6
        # ---- D8 ensembles ----------------------------------------------------------------------------------------------
        loc = {H: local_mlp(H) for H in (16, 32, 64)}
        nl = {H: score(m, val) for H, m in loc.items()}
        members = {"mlp_std": (mlp, nm), **{f"L3xH{H}": (loc[H], nl[H]) for H in loc}}
        combos = [("L3xH16", "L3xH32", "L3xH64"), ("mlp_std", "L3xH32"), ("mlp_std", "L3xH16"), ("mlp_std", "L3xH64"),
                  ("L3xH16", "L3xH32"), ("L3xH16", "L3xH64"), ("L3xH32", "L3xH64"), ("mlp_std", "L3xH16", "L3xH32", "L3xH64")]
        d8 = {"members_minus_mlp_std": {k: chk.mean_se(v[1] - nm) for k, v in members.items()}, "ensembles": {}}
        for c in combos:
            e = score(average([members[k][0] for k in c]), val)
            mem = np.stack([members[k][1] for k in c])
            best = int(np.argmin(mem.mean(1)))
            pairs = [(a, b) for i, a in enumerate(c) for b in c[i + 1:]]
            skl = float(np.mean([sym_kl(members[a][0], members[b][0], p12) for a, b in pairs]))
            M = len(c)
            gain_mean = chk.mean_se(e - mem.mean(0))
            d8["ensembles"][" + ".join(c)] = {
                "mean": float(e.mean()), "minus_mlp_std": chk.mean_se(e - nm), "minus_best_member": chk.mean_se(e - mem[best]),
                "best_member": c[best], "minus_mean_member": gain_mean, "mean_pair_symKL_map": skl,
                "ratio_r": -gain_mean[0] / ((1 - 1 / M) * skl / 4)}
        d8["audit_scores"] = {k: v for k, v in audit["d8"]["scores"].items()} if isinstance(audit["d8"]["scores"], dict) else audit["d8"]["scores"]
        out["d8"] = d8
    chk.write("a_verify.json", out, t0)
    b = out["benchmarks"]
    print("benchmarks", b)
    print("D3", {k: v for k, v in out["d3"].items() if k not in ("by_p",)})
    for K in ("64", "1024"):
        print("D6a K", K, out["d6a"][K])
    print("D6a bias", out["d6a"]["bias_64_minus_1024"])
    for k, v in out["d8"]["ensembles"].items():
        print("D8", k, {kk: vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
