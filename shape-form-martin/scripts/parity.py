"""Gate-0 parity of the two existing pion benchmarks (no training, nothing saved but results).

    uv run python -B shape-form-martin/scripts/parity.py

Evaluates `mlp_std.pt` (4x32 MLP) and the final 101-parameter component formula (`new3_std_n3.pt` x `abar.pt`)
with the UNMODIFIED `common.test_nll` (K=64, CUDA generator seed 12345) on validation and test, and writes
results/m0_parity/parity.json plus the per-event arrays results/m0_parity/{mlp,formula}_{val,test}_nll.npy.
The test cache is opened exactly once, through sfm.data.load_sets(..., allow_test=True, reason=...).
Also: split integrity, low-momentum boundary, finiteness at extreme momenta, the acceptance template Ā,
a units/ranges table and a GPU timing probe (200 Adam steps per K on a throwaway FieldModel(32, 4)).
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data, legacy, paths  # noqa: E402

common = legacy.common
DEV = common.DEVICE
OUT = paths.M0_DIR
TEST_REASON = "gate0 parity of existing benchmarks"

REF = {"mlp": {"test": 18.120842, "val": 18.118084, "npz_key": "std:mlp_ft"},
       "formula": {"test": 18.123459, "val": 18.120407, "npz_key": "std:new"}}
REF_DIFF_TEST = (0.002616, 0.000292)          # formula - MLP on test (test_v3.json "new_minus_mlp" std)
TOL_MEAN, TOL_EVENT, TOL_CORR = 5e-5, 1e-4, 0.9999


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def paired(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": int(len(d))}


def compare_arrays(new, ref):
    new, ref = np.asarray(new, np.float64), np.asarray(ref, np.float64)
    d = new - ref
    ad = np.abs(d)
    return {"n": int(len(d)), "max_abs_diff": float(ad.max()), "mean_diff": float(d.mean()),
            "median_abs_diff": float(np.median(ad)), "rms_diff": float(np.sqrt((d ** 2).mean())),
            "frac_within_1e-4": float((ad <= TOL_EVENT).mean()), "n_above_1e-4": int((ad > TOL_EVENT).sum()),
            "pearson_r": float(np.corrcoef(new, ref)[0, 1]),
            "mean_new": float(new.mean()), "mean_ref": float(ref.mean())}


def timed_test_nll(model, ds):
    sync()
    t = time.time()
    out = common.test_nll(model, ds)
    sync()
    return out, time.time() - t


@torch.no_grad()
def nll_with_seed(model, ds, seed, K=common.N_FOLD_EVAL):
    """common.test_nll's loop with another generator seed (diagnostic of Monte-Carlo noise; validation only)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.cat([common.event_nll(model, b, ds.box, K, gen)[0] for b in ds.batches(128)]).cpu().numpy()


def git_head():
    try:
        return subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def low_momentum(sets, extras, core):
    v0, vs = float(core.v0), float(core.vs)
    out = {"formula_v0": v0, "formula_vs": vs, "m_pion_GeV": data.M_PION}
    for s, ds in sets.items():
        p = ds.ev["p"].double().cpu().numpy()
        i = int(np.argmin(p))
        T = data.kinetic(p)
        lnp_min = ds.ev["lnp"][i:i + 1]
        with torch.no_grad():
            x_model = float((core.lnT(lnp_min) - core.v0) / core.vs)      # the formula's own float32 path
        low = np.flatnonzero(p < 0.1)
        out[s] = {"p_min_GeV": float(p[i]), "T_min_GeV": float(T.min()),
                  "x_min": float((np.log(T.min()) - v0) / vs), "x_min_formula_float32": x_model,
                  "lnp_min": float(lnp_min), "n_p_below_0.1GeV": int(len(low)),
                  "events_p_below_0.1GeV": [{"event_number": int(extras[s]["event_number"][j]), "p_GeV": float(p[j])}
                                            for j in low]}
    p5 = 0.0005
    T5 = float(data.kinetic(p5))
    out["p_0.0005GeV"] = {"T_GeV": T5, "x": (np.log(T5) - v0) / vs}
    return out


@torch.no_grad()
def finiteness(models, lnp_values):
    res = {}
    for label, lnp in lnp_values.items():
        t = torch.tensor([lnp], dtype=torch.float32, device=DEV)
        res[label] = {"p_GeV": float(np.exp(lnp))}
        for name, m in models.items():
            lp = m(t)
            s = float(lp.double().exp().sum())
            res[label][name] = {"all_logP_finite": bool(torch.isfinite(lp).all()), "sum_exp": s,
                                "abs_sum_minus_1": abs(s - 1.0), "logP_max": float(lp.max()),
                                "logP_min": float(lp.min())}
    return res


def clamp_audit():
    """Every clamp / floor / softplus line in the formula code (read only)."""
    out = {}
    for f in ("particle_shape.py", "proton_shape.py", "families.py"):
        lines = (paths.PROTON_DIR / f).read_text().splitlines()
        out[f] = [f"{i + 1}: {ln.strip()}" for i, ln in enumerate(lines)
                  if any(k in ln for k in ("clamp", "softplus", "floor")) and not ln.strip().startswith("#")]
    return out


def timing_probe(train, K, steps=200, warmup=10, batch=256):
    torch.manual_seed(0)
    net = common.FieldModel(32, 4).to(DEV).train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    it = train.batches(batch, shuffle=True, gen=gen)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = None
    for step in range(warmup + steps):
        if step == warmup:
            sync()
            t0 = time.time()
        b = next(it)
        nll, _ = common.event_nll(net, b, train.box, K=K)
        loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    sync()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEV.type == "cuda" else None
    del net, opt
    return {"K": K, "batch": batch, "steps_timed": steps, "warmup_steps": warmup,
            "seconds_per_step": dt / steps, "peak_gpu_mem_GiB": peak, "final_loss": float(loss)}


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = data.load_meta()
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head": git_head(),
           "device": str(DEV), "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
           "versions": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
           "cache_files": meta["files"], "metric": "common.test_nll (unmodified), K=64, CUDA generator seed 12345",
           "tolerances": {"mean": TOL_MEAN, "per_event": TOL_EVENT, "fallback_corr": TOL_CORR}}

    # ---- data -------------------------------------------------------------------------------------------------------
    sets, extras = data.load_sets(("train", "val"), device=DEV)
    tsets, textras = data.load_sets(("test",), device=DEV, allow_test=True, reason=TEST_REASON)
    sets.update(tsets)
    extras.update(textras)
    res["test_access"] = {"reason": TEST_REASON, "log": paths.rel_to(paths.TEST_ACCESS_LOG, paths.REPO_ROOT)}

    counts = {s: len(sets[s]) for s in data.SPLITS}
    en = {s: set(extras[s]["event_number"].tolist()) for s in data.SPLITS}
    overlaps = {f"{a}&{b}": len(en[a] & en[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    bins = {s: sets[s].ev["bin"].cpu().numpy() for s in data.SPLITS}
    per_bin = {}
    for b in sorted(set(np.concatenate(list(bins.values())).tolist())):
        n = {s: int((bins[s] == b).sum()) for s in data.SPLITS}
        tot = sum(n.values())
        per_bin[int(b)] = {"n": tot, **{f"frac_{s}": n[s] / tot for s in data.SPLITS}}
    fr = {s: [v[f"frac_{s}"] for v in per_bin.values()] for s in data.SPLITS}
    res["splits"] = {"counts": counts, "counts_expected": data.EXPECTED_COUNTS,
                     "counts_ok": counts == data.EXPECTED_COUNTS,
                     "event_number_overlaps": overlaps, "overlaps_ok": all(v == 0 for v in overlaps.values()),
                     "events_total": sum(counts.values()), "unique_event_numbers": len(set().union(*en.values())),
                     "per_bin_fraction_range": {s: [min(fr[s]), max(fr[s])] for s in data.SPLITS},
                     "per_bin": per_bin}

    # ---- models -----------------------------------------------------------------------------------------------------
    mlp = legacy.load_mlp_std()
    formula = legacy.load_component_formula(sets["train"])
    core = formula.base
    models = {"mlp": mlp, "formula": formula}
    res["models"] = {"mlp": {"file": "shape-form-proton/results/particles/pion/mlp_std.pt",
                             "class": "common.FieldModel(32, 4)", "n_params": legacy.n_params(mlp)},
                     "formula": {"file": "shape-form-proton/results/particles/pion/new3_std_n3.pt x abar.pt",
                                 "class": f"{type(formula).__name__}({type(core).__name__})",
                                 "n_params": legacy.n_params(formula),
                                 "buffers": {k: float(getattr(core, k)) for k in ("lnp0", "lnp_s", "v0", "vs")}}}

    # ---- NLL parity -------------------------------------------------------------------------------------------------
    ref_npz = np.load(paths.PION_RESULTS_DIR / "test_v3_nll.npz")
    nll, times = {}, {}
    for name, m in models.items():
        for s in ("val", "test"):
            nll[(name, s)], times[(name, s)] = timed_test_nll(m, sets[s])
            np.save(OUT / f"{name}_{s}_nll.npy", nll[(name, s)])
    for s in ("val", "test"):
        np.save(OUT / f"{s}_event_number.npy", extras[s]["event_number"])
    par = {}
    for name in models:
        r = REF[name]
        e = {}
        for s in ("val", "test"):
            mean = float(np.mean(nll[(name, s)], dtype=np.float64))
            e[s] = {"mean": mean, "reference": r[s], "diff": mean - r[s], "within_5e-5": abs(mean - r[s]) <= TOL_MEAN}
        cmp = compare_arrays(nll[(name, "test")], ref_npz[r["npz_key"]])
        cmp["npz_key"] = r["npz_key"]
        cmp["within_1e-4_all_events"] = cmp["max_abs_diff"] <= TOL_EVENT
        cmp["fallback_corr_ok"] = cmp["pearson_r"] > TOL_CORR
        e["test_per_event_vs_npz"] = cmp
        par[name] = e
    d_test = paired(nll[("formula", "test")], nll[("mlp", "test")])
    d_test.update({"reference_mean": REF_DIFF_TEST[0], "reference_se": REF_DIFF_TEST[1]})
    d_ref = paired(ref_npz["std:new"], ref_npz["std:mlp_ft"])
    par["formula_minus_mlp"] = {"test": d_test, "val": paired(nll[("formula", "val")], nll[("mlp", "val")]),
                                "test_recomputed_from_npz": d_ref}
    res["parity"] = par

    # ---- Monte-Carlo noise of the fold points (validation only, 8 other generator seeds) ----------------------------
    seeds = list(range(1, 9))
    mc = {"seeds": seeds, "K": common.N_FOLD_EVAL, "split": "val"}
    alt = {n: [nll_with_seed(m, sets["val"], sd) for sd in seeds] for n, m in models.items()}
    for n in models:
        means = np.array([a.mean(dtype=np.float64) for a in alt[n]])
        ev_sd = float(np.std(np.stack(alt[n]).astype(np.float64), axis=0, ddof=1).mean())
        mc[n] = {"means": means.tolist(), "sd_of_mean_across_seeds": float(means.std(ddof=1)),
                 "mean_of_means": float(means.mean()),
                 "seed12345_mean": par[n]["val"]["mean"],
                 "mean_per_event_sd_across_seeds": ev_sd,
                 "rms_per_event_diff_seed1_vs_seed2": float(np.sqrt(np.mean((alt[n][0].astype(np.float64)
                                                                             - alt[n][1]) ** 2)))}
    dm = np.array([(a.astype(np.float64) - b).mean() for a, b in zip(alt["formula"], alt["mlp"])])
    mc["formula_minus_mlp"] = {"means": dm.tolist(), "sd_across_seeds": float(dm.std(ddof=1))}
    # run-to-run reproducibility of common.test_nll itself (same seed, same process): CUDA index_add_ uses atomics
    rerun = common.test_nll(mlp, sets["val"]).astype(np.float64)
    dd = rerun - nll[("mlp", "val")]
    mc["test_nll_rerun_same_seed_mlp_val"] = {"max_abs_diff": float(np.abs(dd).max()), "mean_diff": float(dd.mean()),
                                              "n_events_differing": int((dd != 0).sum())}
    res["mc_noise_val"] = mc

    # ---- low-momentum boundary and finiteness -----------------------------------------------------------------------
    res["low_momentum"] = low_momentum(sets, extras, core)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(paths.parquet_path()))
    raw = pf.read(columns=["event_number", "incoming_momentum"])
    ev_raw, p_raw = raw.column(0).to_numpy(), raw.column(1).to_numpy().astype(np.float64)
    del raw
    order = np.argsort(ev_raw, kind="stable")
    e_s, p_s = ev_raw[order], p_raw[order]
    del ev_raw, p_raw, order
    starts = np.r_[0, np.flatnonzero(np.diff(e_s)) + 1]
    p_ev = p_s[starts]
    res["low_momentum"]["raw_file_before_cuts"] = {
        "events": int(len(starts)), "p_min_GeV": float(p_s.min()), "n_events_p_below_0.1GeV": int((p_ev < 0.1).sum()),
        "p_constant_within_event": bool(np.array_equal(np.minimum.reduceat(p_s, starts),
                                                       np.maximum.reduceat(p_s, starts)))}
    del e_s, p_s
    lnp_vals = {"lowest_val_p": float(sets["val"].ev["lnp"].min()),
                "lowest_train_p": float(sets["train"].ev["lnp"].min()),
                "p_0.0005GeV": float(np.log(np.float32(0.0005)))}
    res["finiteness"] = finiteness(models, lnp_vals)
    res["floors_in_formula_code"] = {
        "statement": ("No floor or clamp on T or x: particle_shape.kinetic(lnp, m) = p^2/(sqrt(p^2+m^2)+m) with p = exp(lnp), "
                      "lnT = log(kinetic), x = (lnT - v0)/vs in ParticleShape.params (and proton_shape.VARS/params) are "
                      "unclamped. The only lower bounds in this code act on other quantities: range_mm = softplus(raw/5)*5 + 1 mm "
                      "(R(T) >= 1 mm), (Z/Rr).clamp(-1, 1.5) in the track depth term, Lt.clamp_min(1e-12) in the tied weights, "
                      "u.clamp(0,1) (bragg), Z.clamp_min(1.0) (ms), Z.clamp_min(0) (att/far), long_gamma's softplus + 1e-3; "
                      "outside: WithAcceptance logA = log(Abar.clamp_min(1e-6)) and common.event_nll mass.clamp_min(1e-8). "
                      "The MLP input lnp/3 is unbounded."),
        "lines": clamp_audit()}

    # ---- acceptance template ----------------------------------------------------------------------------------------
    abar = legacy.load_abar().to(DEV)
    a_fb = legacy.mean_acceptance(sets["train"], front=True, back=True, nsub=4)
    a_f = legacy.mean_acceptance(sets["train"], front=True, back=False, nsub=4)
    torch.save(a_f.cpu(), OUT / "abar_front.pt")
    torch.save(a_fb.cpu(), OUT / "abar_front_back.pt")
    dfb = (a_fb - abar).abs()
    df_ = (a_f - abar).abs()
    res["abar"] = {"stored": "shape-form-proton/results/particles/pion/abar.pt", "shape": list(abar.shape),
                   "recomputed_front_back_vs_stored": {"max_abs_diff": float(dfb.max()), "mean_abs_diff": float(dfb.mean())},
                   "front_only_vs_stored": {"max_abs_diff": float(df_.max()), "mean_abs_diff": float(df_.mean())},
                   "front_only_saved": "results/m0_parity/abar_front.pt",
                   "front_back_saved": "results/m0_parity/abar_front_back.pt",
                   "stored_min": float(abar.min()), "stored_max": float(abar.max()),
                   "n_cells_stored_le_0.3": int((abar <= 0.3).sum()), "n_cells_stored_le_0.01": int((abar <= 0.01).sum())}

    # ---- units / ranges ---------------------------------------------------------------------------------------------
    keys = ("events", "hits", "p_min_GeV", "p_max_GeV", "theta_min_rad", "theta_max_rad", "zb_min_mm", "zb_max_mm",
            "n_zb_beyond_grid", "hits_per_event_min", "hits_per_event_median", "hits_per_event_max", "z_local_min_mm",
            "z_local_max_mm", "r_max_mm", "evis_min", "evis_max")
    res["units_ranges"] = {"units": meta["units"], "geometry_mm": meta["geometry"], "cuts": meta["cuts"],
                           "box_mm": sets["train"].box.cpu().tolist(),
                           "grid": {"z_mm": [common.Z_MIN, common.Z_MAX, common.N_Z], "r_mm": [0.0, common.R_MAX, common.N_R]},
                           "per_split": {s: {k: meta["splits"][s][k] for k in keys} for s in data.SPLITS}}

    # ---- timing -----------------------------------------------------------------------------------------------------
    tim = {"test_nll_seconds": {f"{n}_{s}": times[(n, s)] for (n, s) in times}}
    for K in (16, 64):
        tim[f"train_step_K{K}"] = timing_probe(sets["train"], K)
    if DEV.type == "cuda":
        tim["peak_gpu_mem_GiB_whole_run"] = torch.cuda.max_memory_allocated() / 2 ** 30
        tim["gpu_total_GiB"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    res["timing"] = tim
    res["seconds_total"] = time.time() - t_start

    # ---- gate-0 summary flags (the gate reviewer decides) -----------------------------------------------------------
    res["flags"] = {
        "counts_ok": res["splits"]["counts_ok"], "overlaps_ok": res["splits"]["overlaps_ok"],
        "means_within_5e-5": all(par[n][s]["within_5e-5"] for n in models for s in ("val", "test")),
        "per_event_within_1e-4": all(par[n]["test_per_event_vs_npz"]["within_1e-4_all_events"] for n in models),
        "per_event_corr_gt_0.9999": all(par[n]["test_per_event_vs_npz"]["fallback_corr_ok"] for n in models),
        "formula_n_params_101": res["models"]["formula"]["n_params"] == 101,
        "maps_finite": all(v[n]["all_logP_finite"] for v in res["finiteness"].values() for n in models),
    }
    (OUT / "parity.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("splits", "parity", "flags")}, indent=1, default=float)[:6000])
    print(f"wrote {OUT / 'parity.json'}  ({res['seconds_total']:.0f} s)")


if __name__ == "__main__":
    main()
