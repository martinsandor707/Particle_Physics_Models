"""M1-d diagnostic D6(c) - piecewise-constant lookup on the 128 x 32 grid vs lookup on refined grids, for the FIXED
component formula (legacy coefficients, exact cell masses from ``sfm.families``), on validation.

    uv run python -B shape-form-martin/results/m1_audit/checks/b2_lookup.py [--m 1 2 4 8] [--events N] [--dtype float64]

What is compared
    The same closed-form density (``build_variant("legacy_exact")``: forms and coefficients of the legacy formula,
    cell masses by exact integration) is tabulated on grids in which every z cell and every ring is split into m
    parts (z edges Z_MIN + j DZ/m; ring edges 700 (i/(32 m))^1.7 mm, which contain the 33 standard edges, so the
    refined grid is nested in the standard one). The acceptance template is upsampled piecewise-constantly. Because
    the cell masses are exact integrals and the template is constant inside a standard cell, the refined map summed
    over the m x m sub-cells of a standard cell IS the standard exact map (checked below: "projected" rows): the model
    is the same, only the look-up resolution of the density changes.

    refined metric (this file; NOT the report metric)
        x_ik    the fold points of ``common.fold`` - same generator, seed 12345, batches of 128, K points per hit
        g_i     (1/K) sum_k rho_m(x_ik),  rho_m = P_m / V_m piecewise constant on the refined grid; points outside
                the grid are clamped to the refined border cells (as ``common.locate`` does on the standard grid)
        M_e     mass of the refined longitudinal profile with z' < zb_e, linear inside the refined z-bin of zb_e
        NLL_e   - sum_i w_i [ln g_i - ln M_e]
    At m = 1 this is ``common.event_nll`` operation by operation (validated: <= 1e-6 on the mean).

Outputs: b2_lookup.json (+ per-event arrays b2_lookup.npz).
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm.families import Grid, build_variant, load_legacy_core  # noqa: E402

common, metric = chk.common, chk.metric
DEV = chk.DEV


class Refined:
    """Refined grid (factor m in z and in the ring index) with the look-up and the back-face normaliser."""

    def __init__(self, m: int):
        self.m = int(m)
        self.NZ, self.NR = common.N_Z * self.m, common.N_R * self.m
        self.dz = common.DZ / self.m
        if self.m == 1:
            re = common.R_EDGES.clone()
        else:
            re = (common.R_MAX * (torch.arange(self.NR + 1, dtype=torch.float64) / self.NR) ** 1.7).float().to(DEV)
            re[:: self.m] = common.R_EDGES                           # nested exactly in the float32 standard edges
        self.r_edges = re
        self.log_vol = torch.log(self.dz * math.pi * (re[1:] ** 2 - re[:-1] ** 2))[None, :]          # (1, NR)
        if self.m == 1:
            self.log_vol = common.LOG_VOL[:1]

    def locate(self, z, r):
        if self.m == 1:
            return common.locate(z, r)
        iz = ((z - common.Z_MIN) / self.dz).floor().long().clamp(0, self.NZ - 1)
        ir = torch.bucketize(r.contiguous(), self.r_edges[1:-1].contiguous())
        return iz, ir

    def mass_below(self, Pz, zb):
        cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
        t = ((zb - common.Z_MIN) / self.dz).clamp(0, self.NZ)
        k = t.floor().long().clamp(max=self.NZ - 1)
        ar = torch.arange(len(Pz), device=Pz.device)
        return cdf[ar, k] + Pz[ar, k] * (t - k)

    def event_nll(self, logP, b, pts, chunk=64):
        """logP (B, NZ, NR) float32; pts (H, K, 3) shower-frame fold points."""
        K = pts.shape[1]
        logrho = logP - self.log_vol
        parts = []
        for a in range(0, K, chunk):
            p = pts[:, a:a + chunk]
            iz, ir = self.locate(p[..., 2], torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2))
            parts.append(torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1))
        logg = (parts[0] if len(parts) == 1 else torch.logsumexp(torch.stack(parts, 1), 1)) - math.log(K)
        Pz = logP.exp().sum(2)
        mass = self.mass_below(Pz, b.zb)
        ll = logg - torch.log(mass.clamp_min(1e-8))[b.ev]
        return -torch.zeros(len(b.lnp), device=ll.device).index_add_(0, b.ev, b.w * ll)


def build(m, core, logA, dtype):
    if m == 1:
        return build_variant("legacy_exact", core, logA, dtype=dtype).eval()             # Grid.from_common
    grid = Grid(n_z=5, n_r=4, N_Z=common.N_Z * m, N_R=common.N_R * m)
    la = logA.repeat_interleave(m, 0).repeat_interleave(m, 1)
    return build_variant("legacy_exact", core, la, grid=grid, dtype=dtype).eval()


@torch.no_grad()
def forward(model, lnp, sub, dtype):
    out = []
    for a in range(0, len(lnp), sub):
        out.append(model(lnp[a:a + sub].to(dtype)).float())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--events", type=int, default=0, help="first N validation events only (timing / smoke)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--K2", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    t0 = time.time()
    sets, extras = chk.load_cpu((a.split,))
    val_cpu = sets[a.split]
    if a.events:
        val_cpu = val_cpu.subset(np.arange(a.events))
    N = len(val_cpu)
    ms = list(a.m)
    res = {f"K64_m{m}": [] for m in ms}
    res.update({f"K{a.K2}_m{m}": [] for m in ms})
    res.update({f"proj_m{m}": [] for m in ms if m > 1})
    res["std_exact"] = []
    timing = {}
    with chk.gpu_lock(f"m1d D6c lookup m={ms}"):
        val = val_cpu.to(DEV)
        core, logA = load_legacy_core(DEV)
        models = {m: build(m, core, logA, dtype) for m in ms}
        grids = {m: Refined(m) for m in ms}
        std_model = models[1] if 1 in models else build(1, core, logA, dtype)
        sub = {1: 128, 2: 128, 4: 32, 8: 8, 16: 2}
        gen64 = torch.Generator(device=DEV).manual_seed(12345)
        gen2 = torch.Generator(device=DEV).manual_seed(12345)
        with metric.deterministic(), torch.no_grad():
            for nb, b in enumerate(val.batches(128)):
                pts64 = common.fold(b, val.box, 64, gen64)
                pts2 = common.fold(b, val.box, a.K2, gen2)
                for m in ms:
                    tt = time.time()
                    logP = forward(models[m], b.lnp, sub[m] if dtype == torch.float32 else max(1, sub[m] // 2), dtype)
                    torch.cuda.synchronize()
                    timing[m] = timing.get(m, 0.0) + time.time() - tt
                    g = grids[m]
                    res[f"K64_m{m}"].append(g.event_nll(logP, b, pts64))
                    res[f"K{a.K2}_m{m}"].append(g.event_nll(logP, b, pts2))
                    if m > 1:                                                # same map, summed back onto the standard grid
                        B = len(b.lnp)
                        Pc = logP.double().exp().reshape(B, common.N_Z, m, common.N_R, m).sum((2, 4))
                        res[f"proj_m{m}"].append(grids[ms[0]].event_nll(torch.log(Pc).float(), b, pts64) if ms[0] == 1
                                                 else Refined(1).event_nll(torch.log(Pc).float(), b, pts64))
                    del logP
                if nb % 20 == 0:
                    print(f"batch {nb}  t = {time.time() - t0:.0f} s", flush=True)
            res = {k: torch.cat(v).cpu().numpy().astype(np.float64) for k, v in res.items() if v}
            # validation of the refined metric at m = 1 against the unmodified common.test_nll (same model)
            ref = common.test_nll(std_model if dtype == torch.float32 else (lambda lnp: std_model(lnp.to(dtype)).float()), val).astype(np.float64)
    out = {"split": a.split, "n_events": N, "dtype": a.dtype, "m": ms, "K2": a.K2, "model_seconds": timing}
    if 1 in ms:
        out["validation_m1_vs_common"] = {"mean_diff": float((res["K64_m1"] - ref).mean()),
                                          "max_abs": float(np.abs(res["K64_m1"] - ref).max()),
                                          "mean_common": float(ref.mean()), "mean_refined_code_m1": float(res["K64_m1"].mean())}
    p = val_cpu.ev["p"].numpy().astype(np.float64)
    zb = val_cpu.ev["zb"].numpy().astype(np.float64)
    grp = chk.p_group(p)
    base64, base2 = res.get("K64_m1"), res.get(f"K{a.K2}_m1")
    rows = {}
    for m in ms:
        if m == 1 or base64 is None:
            continue
        d64, d2 = res[f"K64_m{m}"] - base64, res[f"K{a.K2}_m{m}"] - base2
        dinf = d2 - (d64 - d2) / (a.K2 / 64 - 1)                    # per event; exact for a pure c/K bias (different points: noisy)
        rows[str(m)] = {"K64": chk.mean_se(d64), f"K{a.K2}": chk.mean_se(d2), "K_inf_extrap": chk.mean_se(dinf),
                        "projected_to_standard_grid_minus_m1": chk.mean_se(res[f"proj_m{m}"] - base64),
                        "by_p_K64": {lab: chk.mean_se(d64[grp == g]) for g, lab in enumerate(chk.P_LABELS)},
                        f"by_p_K{a.K2}": {lab: chk.mean_se(d2[grp == g]) for g, lab in enumerate(chk.P_LABELS)},
                        "by_zb": {"zb<1221": chk.mean_se(d64[zb < 1221]), "1221<=zb<=1450": chk.mean_se(d64[(zb >= 1221) & (zb <= 1450)]),
                                  "zb>1450": chk.mean_se(d64[zb > 1450])},
                        "mean_K64": float(res[f"K64_m{m}"].mean()), f"mean_K{a.K2}": float(res[f"K{a.K2}_m{m}"].mean())}
    out["refined_minus_standard_lookup"] = rows
    if base64 is not None:
        out["m1"] = {"mean_K64": float(base64.mean()), f"mean_K{a.K2}": float(base2.mean())}
    # continuous-limit extrapolation from the last three refinements (geometric model d(m) = d_inf - c m^-q)
    seq = [m for m in ms if m > 1]
    if len(seq) >= 3:
        for key in ("K64", f"K{a.K2}"):
            d = {m: (res[f"{key}_m{m}"] - res[f"{key}_m1"]) for m in seq}
            m1, m2, m3 = seq[-3:]
            a1, a2 = (d[m2] - d[m1]).mean(), (d[m3] - d[m2]).mean()
            ratio = a1 / a2 if a2 != 0 else float("nan")
            q = math.log(abs(ratio), m2 / m1) if ratio > 0 else float("nan")
            # per-event extrapolation with the fitted ratio (linear in the per-event values, so an SE is available)
            ext = d[m3] + (d[m3] - d[m2]) / (ratio - 1) if ratio > 1 else None
            out[f"extrapolation_{key}"] = {"increments": [float(a1), float(a2)], "ratio": float(ratio), "order_q": q,
                                           "continuous_limit": None if ext is None else chk.mean_se(ext),
                                           "by_p": None if ext is None else {lab: chk.mean_se(ext[grp == g]) for g, lab in enumerate(chk.P_LABELS)}}
    tag = a.tag or ((f"_{a.split}" if a.split != "val" else "") + (f"_n{a.events}" if a.events else ""))
    chk.write(f"b2_lookup{tag}.json", out, t0)
    np.savez_compressed(chk.HERE / f"b2_lookup{tag}.npz", **res, ref=ref)
    for k, v in out.items():
        if k != "refined_minus_standard_lookup":
            print(k, v)
    for m, r in rows.items():
        print("m =", m, {k: v for k, v in r.items()})


if __name__ == "__main__":
    main()
