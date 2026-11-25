"""Gate-0 parity of the two existing pion benchmarks (no training, nothing saved but results).

    uv run python -B shape-form-martin/scripts/parity.py [--overwrite]
    uv run python -B shape-form-martin/scripts/parity.py --reference-stream      # item C1 only (see below)

Default mode (the Gate 0 run; it has been run once, 24 Nov 2026 - re-running it needs --overwrite, because it opens the
test split again and replaces the pairing partners results/m0_parity/*_nll.npy):
Evaluates `mlp_std.pt` (4x32 MLP) and the final 101-parameter component formula (`new3_std_n3.pt` x `abar.pt`)
with the UNMODIFIED `common.test_nll` (K=64, CUDA generator seed 12345) on validation and test, and writes
results/m0_parity/parity.json plus the per-event arrays results/m0_parity/{mlp,formula}_{val,test}_nll.npy.
The test cache is opened exactly once, through sfm.data.load_sets(..., allow_test=True, reason=...).
Also: split integrity, low-momentum boundary, finiteness at extreme momenta, the acceptance template Ā,
a units/ranges table and a GPU timing probe (200 Adam steps per K on a throwaway FieldModel(32, 4)).
The summary flags follow the AMENDED criteria of PLAN.md section 13 (A1-A3; the unamended 5e-5 / 0.9999 numbers
are still reported, labelled as such): validation means on the emulated reference stream within 4e-6 (A1a), the
stream-independent identities (A1b), sfm.metric bit-identical to common.test_nll inside deterministic() and within
1e-4 per event / 1e-6 on the mean outside it (A2), r >= 0.9995 and the rms of the per-event test difference within
+-15 % of the validation seed-to-seed analogue (A3). A1e (exact per-event test parity on the reference stream) is
item C1 and is only evaluated with --reference-stream.

--reference-stream (item C1, an explicit flag is required; it is the user's decision whether and when to run it):
evaluates the two benchmarks on TEST with sfm.refstream.test_nll_ref(grid=1056) - the fold points of the reference
machine - and compares per event with test_v3_nll.npz (std:mlp_ft, std:new). Pass: max |Δ| <= 1e-4 per event and
|Δ mean| <= 4e-6 for both. The test access is logged with the reason "gate0 parity on reference stream (C1)". Writes
only new files (results/m0_parity/c1_reference_stream.json and {mlp,formula}_{val,test}_nll_ref1056.npy); the Gate 0
outputs are not touched. Refuses to run a second time unless --overwrite is given as well.
"""
import argparse
import itertools
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

from sfm import data, legacy, metric, paths, refstream  # noqa: E402

common = legacy.common
DEV = common.DEVICE
OUT = paths.M0_DIR
TEST_REASON = "gate0 parity of existing benchmarks"
C1_REASON = "gate0 parity on reference stream (C1)"
C1_JSON = "c1_reference_stream.json"

# Stored by the earlier study (reference machine, other fold points). `val_f32` = the exact float32 means of
# mlp_std.json["val"] and new3_std.json["rungs"]["3"]["val"] (read again at run time and checked).
REF = {"mlp": {"test": 18.120842, "val": 18.118084, "val_f32": 18.118083953857422, "npz_key": "std:mlp_ft"},
       "formula": {"test": 18.123459, "val": 18.120407, "val_f32": 18.120407104492188, "npz_key": "std:new"}}
REF_DIFF_TEST = (0.002616, 0.000292)          # formula - MLP on test (test_v3.json "new_minus_mlp" std)

# Amended criteria (PLAN.md section 13; Gate 0 verdict PASS WITH AMENDMENTS)
TOL_REF_MEAN = 4e-6          # A1(a), A1(e)/C1: means on the emulated reference stream vs the stored values
TOL_REF_PAIRED = 1e-6        # A1(a): paired differences on the reference stream vs stored float64 differences
TOL_EVENT = 1e-4             # A1(e)/C1: per event on the reference stream; A2: per event outside deterministic()
TOL_A2_MEAN = 1e-6           # A2: |mean difference| sfm.metric vs common.test_nll outside deterministic()
MIN_CORR = 0.9995            # A3: Pearson r between the local and the stored per-event test arrays
RMS_REL_TOL = 0.15           # A3: rms of the per-event difference within +-15 % of the validation seed-to-seed rms
TOL_MEAN_UNAMENDED, TOL_CORR_UNAMENDED = 5e-5, 0.9999    # original section-3 criteria (reported, informational)


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


def stored_val_means():
    """Exact stored validation means (float32 values written by the earlier study), checked against REF."""
    j_mlp = json.loads((paths.PION_RESULTS_DIR / "mlp_std.json").read_text())
    j_new = json.loads((paths.PION_RESULTS_DIR / "new3_std.json").read_text())
    out = {"mlp": float(j_mlp["val"]), "formula": float(j_new["rungs"]["3"]["val"])}
    for n, v in out.items():
        if abs(v - REF[n]["val_f32"]) > 0.0:
            raise RuntimeError(f"stored validation mean of {n} changed: {v} != {REF[n]['val_f32']}")
    return out


def reference_stream_val(models, val):
    """A1(a): validation on the emulated reference stream (grid 1056, inside deterministic()) vs the stored means."""
    stored = stored_val_means()
    arr = {}
    with metric.deterministic():
        for n, m in models.items():
            arr[n] = refstream.test_nll_ref(m, val, grid=refstream.REFERENCE_GRID)
    out = {"grid": refstream.REFERENCE_GRID, "tolerance_mean": TOL_REF_MEAN}
    for n, a in arr.items():
        m32 = float(np.float32(a.mean()))                     # the stored values are float32 means
        out[n] = {"mean_f32": m32, "mean_f64": float(a.astype(np.float64).mean()), "stored": stored[n],
                  "diff": m32 - stored[n], "within_4e-6": abs(m32 - stored[n]) <= TOL_REF_MEAN}
    d = a_minus_b(arr["formula"], arr["mlp"])
    out["formula_minus_mlp"] = {**d, "from_stored_means": stored["formula"] - stored["mlp"],
                                "note": ("the stored float32 means resolve the difference only to ~2e-6; the 1e-6 paired "
                                         "criterion of A1(a) was verified in Gate 0 (c04) on the stored float64 ladder "
                                         "differences")}
    return out, arr


def a_minus_b(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": int(len(d))}


def a2_metric_equality(models, val, common_val):
    """A2: sfm.metric.test_nll vs common.test_nll on validation - bitwise inside deterministic(); outside it within
    1e-4 per event and 1e-6 on the mean (against the default-mode arrays of this run)."""
    out = {}
    for n, m in models.items():
        with metric.deterministic():
            c = common.test_nll(m, val)
            s = metric.test_nll(m, val)
        s_def = metric.test_nll(m, val)
        dd = s_def.astype(np.float64) - common_val[n].astype(np.float64)
        out[n] = {"bitwise_in_deterministic": bool(np.array_equal(c, s)),
                  "default_max_abs_diff": float(np.abs(dd).max()), "default_mean_diff": float(dd.mean()),
                  "default_ok": bool(np.abs(dd).max() <= TOL_EVENT and abs(dd.mean()) <= TOL_A2_MEAN)}
    return out


def seed_pair_rms(arrays):
    """Mean over all seed pairs of the rms per-event difference (the validation seed-to-seed analogue of A3)."""
    vals = [float(np.sqrt(np.mean((a.astype(np.float64) - b) ** 2))) for a, b in itertools.combinations(arrays, 2)]
    return float(np.mean(vals)), vals


def amended_flags(res, names):
    """Gate-0 summary flags under the amended criteria A1-A3 (PLAN.md section 13). None = not evaluated here."""
    par, mc = res["parity"], res["mc_noise_val"]
    rv, a2 = res.get("reference_stream_val"), res.get("a2_metric_equality")
    a3_rms = {}
    for n in names:
        rms_test = par[n]["test_per_event_vs_npz"]["rms_diff"]
        rms_val = mc[n]["rms_per_event_diff_seed_pairs_mean"]
        a3_rms[n] = abs(rms_test / rms_val - 1.0) <= RMS_REL_TOL
    return {
        "counts_ok": res["splits"]["counts_ok"], "overlaps_ok": res["splits"]["overlaps_ok"],
        "A1a_val_means_ref_stream_within_4e-6": (all(rv[n]["within_4e-6"] for n in names) if rv else None),
        "A1b_abar_recomputed_bit_identical": res["abar"]["recomputed_front_back_vs_stored"]["max_abs_diff"] == 0.0,
        "A1b_formula_centring_identical": res["models"]["formula"].get("centring_identical"),
        "A1e_test_per_event_ref_stream": "open (C1): run with --reference-stream",
        "A2_sfm_metric_bitwise_in_deterministic": (all(a2[n]["bitwise_in_deterministic"] for n in names)
                                                   if a2 else None),
        "A2_default_mode_within_1e-4_event_1e-6_mean": (all(a2[n]["default_ok"] for n in names) if a2 else None),
        "A3_corr_ge_0.9995": all(par[n]["test_per_event_vs_npz"]["pearson_r"] >= MIN_CORR for n in names),
        "A3_rms_within_15pct_of_val_seed_analogue": all(a3_rms.values()),
        "formula_n_params_101": res["models"]["formula"]["n_params"] == 101,
        "maps_finite": all(v[n]["all_logP_finite"] for v in res["finiteness"].values() for n in names),
        "informational_unamended_means_within_5e-5": all(par[n][s]["within_5e-5"] for n in names
                                                         for s in ("val", "test")),
        "informational_unamended_corr_gt_0.9999": all(par[n]["test_per_event_vs_npz"]["pearson_r"] > TOL_CORR_UNAMENDED
                                                      for n in names),
    }


def c1_compare(arr, npz, test_event_number, saved_event_number):
    """C1 verdict from the reference-stream test arrays and the stored arrays (pure function of its inputs)."""
    out = {"criteria": {"max_abs_diff_per_event": TOL_EVENT, "abs_mean_diff": TOL_REF_MEAN},
           "test_event_numbers_equal_gate0_run": bool(np.array_equal(test_event_number, saved_event_number))}
    ok = out["test_event_numbers_equal_gate0_run"]
    for n, a in arr.items():
        cmp = compare_arrays(a, npz[REF[n]["npz_key"]])
        cmp["npz_key"] = REF[n]["npz_key"]
        cmp["mean_new_f32"] = float(np.float32(np.asarray(a, np.float32).mean()))
        cmp["pass_per_event"] = cmp["max_abs_diff"] <= TOL_EVENT
        cmp["pass_mean"] = abs(cmp["mean_diff"]) <= TOL_REF_MEAN
        out[n] = cmp
        ok = ok and cmp["pass_per_event"] and cmp["pass_mean"]
    d_new = a_minus_b(arr["formula"], arr["mlp"])
    d_ref = a_minus_b(npz[REF["formula"]["npz_key"]], npz[REF["mlp"]["npz_key"]])
    out["formula_minus_mlp"] = {"reference_stream": d_new, "stored": d_ref, "abs_diff": abs(d_new["mean"] - d_ref["mean"]),
                                "within_1e-6_informational": abs(d_new["mean"] - d_ref["mean"]) <= TOL_REF_PAIRED}
    out["C1"] = "PASS" if ok else "FAIL"
    return out


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


def run_reference_stream(overwrite=False):
    """Item C1: the two benchmarks on TEST on the emulated reference stream vs test_v3_nll.npz. Writes new files only."""
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / C1_JSON
    if out_json.exists() and not overwrite:
        print(f"{out_json} exists: C1 has already been evaluated. Running it again is a further logged test access; "
              "pass --overwrite together with --reference-stream to do so.")
        return 2
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head": git_head(),
           "device": str(DEV), "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
           "versions": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
           "metric": (f"common.event_nll (unmodified) on the fold points of launch grid {refstream.REFERENCE_GRID} "
                      "(sfm.refstream.test_nll_ref: K=64, seed 12345, batch 128), inside sfm.metric.deterministic()"),
           "local_grid": refstream.local_grid() if DEV.type == "cuda" else None}
    sets, _ = data.load_sets(("train", "val"), device=DEV)
    models = {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(sets["train"])}
    res["reference_stream_val"], val_arr = reference_stream_val(models, sets["val"])
    for n, a in val_arr.items():
        np.save(OUT / f"{n}_val_nll_ref1056.npy", a)
    del sets
    saved_en = np.load(OUT / "test_event_number.npy")
    npz = np.load(paths.PION_RESULTS_DIR / "test_v3_nll.npz")
    tsets, textras = data.load_sets(("test",), device=DEV, allow_test=True, reason=C1_REASON)
    res["test_access"] = {"reason": C1_REASON, "log": paths.rel_to(paths.TEST_ACCESS_LOG, paths.REPO_ROOT)}
    arr = {}
    with metric.deterministic():
        for n, m in models.items():
            arr[n] = refstream.test_nll_ref(m, tsets["test"], grid=refstream.REFERENCE_GRID)
            np.save(OUT / f"{n}_test_nll_ref1056.npy", arr[n])
    res["c1"] = c1_compare(arr, npz, textras["test"]["event_number"], saved_en)
    res["seconds_total"] = time.time() - t_start
    out_json.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("reference_stream_val", "c1")}, indent=1, default=float)[:6000])
    print(f"C1: {res['c1']['C1']}  -> {out_json}  ({res['seconds_total']:.0f} s)")
    return 0 if res["c1"]["C1"] == "PASS" else 1



def _refuse_record_writer(what: str) -> None:
    """transfer/REVIEW_STATUS.md section 4, point 15: this script overwrites a tracked laptop record (or opens test).
    Blocked unless SFM_ALLOW_RECORD_WRITER=1 is set explicitly, and always blocked with an output root
    (SFM_OUT_ROOT: the H200 / pristine-clone setting)."""
    import os as _os
    if _os.environ.get("SFM_OUT_ROOT") or _os.environ.get("SFM_ALLOW_RECORD_WRITER") != "1":
        raise SystemExit(f"{what}: blocked - it overwrites tracked laptop records (spec point 15). Never run it on "
                         "the H200; on the laptop only with SFM_ALLOW_RECORD_WRITER=1 and the user's go-ahead.")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-stream", action="store_true",
                    help="item C1 only: benchmarks on TEST on the emulated reference stream (logged test access)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow a run whose outputs already exist (default mode: parity.json; C1: its JSON)")
    args = ap.parse_args(argv)
    _refuse_record_writer("parity.py (opens the test split; rewrites parity.json and the pairing arrays)")
    if args.reference_stream:
        return run_reference_stream(args.overwrite)
    if (OUT / "parity.json").exists() and not args.overwrite:
        print(f"{OUT / 'parity.json'} exists: the Gate 0 parity run has been done. Re-running it opens the test split "
              "again and replaces the pairing partners results/m0_parity/*_nll.npy; pass --overwrite to do so. "
              "(Item C1 is --reference-stream.)")
        return 2
    return run_gate0()


def run_gate0():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = data.load_meta()
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head": git_head(),
           "device": str(DEV), "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
           "versions": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
           "cache_files": meta["files"], "metric": "common.test_nll (unmodified), K=64, CUDA generator seed 12345",
           "tolerances": {"amended (PLAN.md section 13)": {
                              "A1a_ref_stream_mean": TOL_REF_MEAN, "A1a_ref_stream_paired": TOL_REF_PAIRED,
                              "A1e_C1_per_event": TOL_EVENT, "A1e_C1_mean": TOL_REF_MEAN,
                              "A2_per_event": TOL_EVENT, "A2_mean": TOL_A2_MEAN,
                              "A3_min_corr": MIN_CORR, "A3_rms_rel": RMS_REL_TOL},
                          "unamended (section 3, informational)": {"mean": TOL_MEAN_UNAMENDED, "per_event": TOL_EVENT,
                                                                  "fallback_corr": TOL_CORR_UNAMENDED}}}

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
    c0, s0 = legacy.shapelab.lnp_center_scale({"train": sets["train"]})
    res["models"]["formula"]["centring_recomputed"] = {"lnp0": c0, "lnp_s": s0}
    res["models"]["formula"]["centring_identical"] = bool(np.float32(c0) == np.float32(float(core.lnp0))
                                                          and np.float32(s0) == np.float32(float(core.lnp_s)))

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
            e[s] = {"mean": mean, "reference": r[s], "diff": mean - r[s],
                    "within_5e-5": abs(mean - r[s]) <= TOL_MEAN_UNAMENDED}       # unamended, informational
        cmp = compare_arrays(nll[(name, "test")], ref_npz[r["npz_key"]])
        cmp["npz_key"] = r["npz_key"]
        cmp["within_1e-4_all_events"] = cmp["max_abs_diff"] <= TOL_EVENT          # unamended (other stream)
        cmp["fallback_corr_ok"] = cmp["pearson_r"] > TOL_CORR_UNAMENDED            # unamended, informational
        cmp["corr_ge_0.9995"] = cmp["pearson_r"] >= MIN_CORR                       # A3
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
        mc[n]["rms_per_event_diff_seed_pairs_mean"], mc[n]["rms_per_event_diff_seed_pairs"] = seed_pair_rms(alt[n])
        rms_test = par[n]["test_per_event_vs_npz"]["rms_diff"]                    # A3: local vs stored, test
        par[n]["test_per_event_vs_npz"]["rms_ratio_to_val_seed_analogue"] = rms_test / mc[n][
            "rms_per_event_diff_seed_pairs_mean"]
    dm = np.array([(a.astype(np.float64) - b).mean() for a, b in zip(alt["formula"], alt["mlp"])])
    mc["formula_minus_mlp"] = {"means": dm.tolist(), "sd_across_seeds": float(dm.std(ddof=1))}
    # run-to-run reproducibility of common.test_nll itself (same seed, same process): CUDA index_add_ uses atomics
    rerun = common.test_nll(mlp, sets["val"]).astype(np.float64)
    dd = rerun - nll[("mlp", "val")]
    mc["test_nll_rerun_same_seed_mlp_val"] = {"max_abs_diff": float(np.abs(dd).max()), "mean_diff": float(dd.mean()),
                                              "n_events_differing": int((dd != 0).sum())}
    res["mc_noise_val"] = mc

    # ---- amended criteria on validation: A1(a) reference stream, A2 sfm.metric vs common ----------------------------
    res["reference_stream_val"], val_ref = reference_stream_val(models, sets["val"])
    for n, a in val_ref.items():
        np.save(OUT / f"{n}_val_nll_ref1056.npy", a)
    res["a2_metric_equality"] = a2_metric_equality(models, sets["val"], {n: nll[(n, "val")] for n in models})

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

    # ---- gate-0 summary flags under the amended criteria A1-A3 (the gate reviewer decides) ---------------------------
    res["flags"] = amended_flags(res, list(models))
    (OUT / "parity.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("splits", "parity", "flags")}, indent=1, default=float)[:6000])
    print(f"wrote {OUT / 'parity.json'}  ({res['seconds_total']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
