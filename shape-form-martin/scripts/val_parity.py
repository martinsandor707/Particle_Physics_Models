"""Validation-only benchmark parity, machine identity and the reference set of the live tag (run-book steps 6-7;
rows R0, R1, R1b (benchmarks), R2, R3 of PLAN.md section 16.4). NEVER opens the test split.

    .venv/bin/python -B shape-form-martin/scripts/val_parity.py                     # checks only, writes a report
    .venv/bin/python -B shape-form-martin/scripts/val_parity.py --create-refset     # + the reference set, once

What it does (one process, under ``sfm.gpulock.gpu_lock``)
  R0  E10 start-up (hard failure without CUDA, TF32 off), the machine tag (GPU, ``refstream.local_grid()``, torch /
      CUDA build, Python, TF32 flags, deterministic flag) and the generator check (``EmulGen(seed, local_grid())``
      equals ``torch.rand`` bit for bit). Stored once as ``results/machines/<machine id>/identity.json``.
  R1  Per-event validation NLL of both benchmarks (``mlp_std.pt``, component formula) with the UNMODIFIED
      ``common.test_nll`` inside ``sfm.metric.deterministic()``, computed twice: the two runs must be bit-identical.
      Compared - as TOLERANCE CHECKS, never as deltas - with the laptop's emulated arrays
      ``results/m0_parity/{mlp,formula}_val_nll_ref1056.npy`` (laptop records; on the H200 in the quarantine folder
      ``cache/laptop_records/``) and with the stored reference means. Grid 1056: float32 means 18.118083953857422 /
      18.120407104492188 within 4e-6, every event within 1e-4, means within 1e-6, formula - MLP = +23.24 +- 2.94e-4.
      The emulated grid-1056 arrays are recomputed here as well (on a grid-1056 GPU they must equal the native ones
      bit for bit; on any other GPU they are the fallback's targets).
      P3: if any event differs from the laptop's emulated array by more than 5e-5, a banner says so and
      ``checks/val_parity/outliers_<benchmark>.npz/.json`` keep what a reviewer needs to decide whether the outliers
      are 1-ULP bin-edge flips: event index and number, both values, batch, Philox offset of the batch, the
      emulated fold points of every hit of the event, their grid cells and their distance to the nearest bin edge
      in float32 ULPs.
  R1b The laptop stream (emulated grid 144) for both benchmarks, against the laptop's arrays (tolerance check).
      (The eight networks: ``scripts/rescore_registered.py --emulate-laptop``.)
  R3  Run-to-run floor outside ``deterministic()``: two runs within 1e-4 per event and 1e-6 on the mean (A2).
  R2  With ``--create-refset``: the reference set of the live tag (K = 64 arrays of run 1, guard references
      K = 1024 and bulk on the stratum zb >= 1221 mm), created ONCE by this process, made read-only, hashes in the
      tracked ``refset.json``; then the guard targets (grid 1056: K = 1024 means 18.1155556 / 18.1177364, bulk
      18.0429497 / 18.0448778 within 1e-6; 14,784 events in the stratum). A second invocation recomputes and
      compares bit for bit; nothing is overwritten.
  Fallback (grid != 1056, PLAN.md section 16.4): R1 and R2 are evaluated on the EMULATED grid 1056 (benchmarks K = 64
      through ``refstream.test_nll_ref``; guards through ``refstream.patched_draw``, with ``--emulated-guards`` or
      automatically) against the same targets, and the native arrays are compared with the emulated ones on A3
      (correlation >= 0.9995; rms of the difference reported next to the laptop's validation values 0.033 / 0.036).

Outputs, all under this machine's directory R/machines/<id>/ (R = $SFM_OUT_ROOT/shape-form-martin/results; on the
laptop by default the repository's results/): identity.json, refsets/<points id>/ (arrays + refset.json),
checks/val_parity/report_<utc>.json and outliers_* (JSON + npz). Nothing is written inside the clone in pristine
mode. Exit code 0 if every applicable acceptance check passed, 1 otherwise.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import machine, metric, paths, provenance, refstream, registry, stats  # noqa: E402

common = registry.C
BENCH_REL = {"mlp": "results/m0_parity/mlp_val_nll{}.npy", "formula": "results/m0_parity/formula_val_nll{}.npy"}
GUARD_REL = {("mlp", "k1024"): "results/registry_refs/mlp_val_k1024.npy",
             ("formula", "k1024"): "results/registry_refs/formula_val_k1024.npy",
             ("mlp", "bulk_zb1221"): "results/registry_refs/mlp_val_bulk_stratum.npy",
             ("formula", "bulk_zb1221"): "results/registry_refs/formula_val_bulk_stratum.npy"}
TARGETS = {   # transfer/REVIEW_STATUS.md, measurements session 1 and 2 (validation, K = 64, seed 12345)
    1056: {"float32_mean": {"mlp": 18.118083953857422, "formula": 18.120407104492188}, "float32_tol": 4e-6,
           "mean64": {"mlp": 18.1180829, "formula": 18.1204074},
           "formula_minus_mlp": (0.002324, 0.000294),
           "k1024_mean": {"mlp": 18.1155556, "formula": 18.1177364},
           "bulk_mean": {"mlp": 18.0429497, "formula": 18.0448778},
           "guard_formula_minus_mlp": {"k1024": (0.002181, 0.000283), "bulk_zb1221": (0.001928, 0.000275)}},
    144: {"mean64": {"mlp": 18.117857, "formula": 18.120061}, "formula_minus_mlp": (0.002204, 0.000293),
          "k1024_mean": {"mlp": 18.1155965, "formula": 18.1177640},
          "bulk_mean": {"mlp": 18.0423351, "formula": 18.0442180},
          "guard_formula_minus_mlp": {"k1024": (0.002167, 0.000285), "bulk_zb1221": (0.001883, 0.000275)}},
}
N_STRATUM, N_SHORT = 14784, 166
A3_RMS = {"mlp": 0.033, "formula": 0.036}
GEN_SHAPES = [(1000,), (36_863,), (147_457,), (35_000, 64, 3), (7, 16, 3), (41_237, 64, 3), (5,)]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generator_check(grid: int) -> dict:
    """R0: EmulGen(seed, grid) against torch.rand on this GPU, bit for bit (consecutive calls and fresh seeds)."""
    out = {"grid": grid, "shapes": [], "seeds": []}
    g = torch.Generator(device="cuda").manual_seed(12345)
    e = refstream.EmulGen(12345, grid)
    for shp in GEN_SHAPES:
        a = torch.rand(*shp, device="cuda", generator=g)
        b = e.rand(*shp, device="cuda")
        out["shapes"].append({"shape": list(shp), "equal": bool(torch.equal(a, b))})
    for seed in (0, 1, 12345, 2 ** 40 + 7):
        g = torch.Generator(device="cuda").manual_seed(seed)
        out["seeds"].append({"seed": seed, "equal": bool(torch.equal(torch.rand(300_001, 3, device="cuda", generator=g),
                                                                   refstream.EmulGen(seed, grid).rand(300_001, 3, device="cuda")))})
    out["pass"] = all(x["equal"] for x in out["shapes"] + out["seeds"])
    return out


def within(x, target, tol) -> dict:
    return {"value": x, "target": target, "tol": tol, "diff": x - target, "pass": bool(abs(x - target) <= tol)}


def fold_points_for_events(sh, events, grid: int, K: int = 64, seed: int = 12345, batch: int = 128) -> dict:
    """Re-derive the fold points of the given validation events on the stream of ``grid`` (EmulGen), batch by batch
    in storage order as common.test_nll draws them, with the expression of ``common.fold`` evaluated on the whole
    batch on ``common.DEVICE`` (the arithmetic of the evaluation). For every event: batch index, Philox offset at the
    start of the batch, H of the batch, the event's hit range inside the batch, offsets u (H_e, K, 3), z', r, grid
    cells and the distance of z' and r to the nearest bin edge in float32 ULPs. Runs on the GPU: call it inside the
    GPU lock."""
    events = sorted(set(int(e) for e in events))
    if not events:
        return {}
    off = sh.off.detach().cpu().numpy()
    gen = refstream.EmulGen(seed, grid)
    zed = common.Z_EDGES.detach().cpu().numpy().astype(np.float32)
    red = common.R_EDGES.detach().cpu().numpy().astype(np.float32)
    out = {}
    n = len(sh)
    with torch.inference_mode():
        for bi in range(events[-1] // batch + 1):
            s, e_ = bi * batch, min((bi + 1) * batch, n)
            H = int(off[e_] - off[s])
            offset0 = gen.offset
            u = gen.rand(H, K, 3, device=common.DEVICE)                     # advances the generator like the GPU
            want = [ev for ev in events if s <= ev < e_]
            if not want:
                continue
            b = sh.batch(torch.arange(s, e_, device=sh.off.device))
            uo = (u - 0.5) * sh.box                                         # common.fold, line 1
            pts = b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], uo)   # common.fold, line 2
            z = pts[..., 2]
            r = torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
            iz, ir = common.locate(z, r)
            for ev in want:
                lo, hi = int(off[ev] - off[s]), int(off[ev + 1] - off[s])
                zn, rn = z[lo:hi].cpu().numpy(), r[lo:hi].cpu().numpy()
                dz = np.min(np.abs(zn[..., None] - zed), -1) / np.spacing(np.abs(zn).astype(np.float32))
                dr = np.min(np.abs(rn[..., None] - red), -1) / np.spacing(np.abs(rn).astype(np.float32)
                                                                          + np.float32(1e-30))
                out[ev] = {"batch": bi, "batch_first_event": s, "philox_offset_at_batch": int(offset0), "H_batch": H,
                           "hit_lo_in_batch": lo, "hit_hi_in_batch": hi, "u": uo[lo:hi].cpu().numpy(), "z": zn,
                           "r": rn, "iz": iz[lo:hi].cpu().numpy(), "ir": ir[lo:hi].cpu().numpy(), "z_edge_ulp": dz,
                           "r_edge_ulp": dr, "w": b.w[lo:hi].cpu().numpy()}
    return out


def save_outliers(cdir: Path, bench: str, event_number, native, target, grid: int, label: str, fp: dict) -> dict:
    ad = np.abs(native.astype(np.float64) - target.astype(np.float64))
    idx = np.flatnonzero(ad > 5e-5)
    summary = {"benchmark": bench, "compared_with": label, "grid_of_fold_points": grid,
               "n_events_above_5e-5": int(len(idx)),
               "events": [{"index": int(i), "event_number": int(event_number[i]), "native": float(native[i]),
                           "target": float(target[i]), "abs_diff": float(ad[i]),
                           "batch": fp[int(i)]["batch"], "philox_offset_at_batch": fp[int(i)]["philox_offset_at_batch"],
                           "H_batch": fp[int(i)]["H_batch"], "hits": [fp[int(i)]["hit_lo_in_batch"],
                                                                     fp[int(i)]["hit_hi_in_batch"]],
                           "min_z_edge_ulp": float(fp[int(i)]["z_edge_ulp"].min()),
                           "min_r_edge_ulp": float(fp[int(i)]["r_edge_ulp"].min()),
                           "n_points_within_2ulp_of_an_edge": int(((fp[int(i)]["z_edge_ulp"] <= 2) |
                                                                   (fp[int(i)]["r_edge_ulp"] <= 2)).sum())}
                          for i in idx if int(i) in fp],
               "how_to_rederive": "EmulGen(12345, grid): draw rand(H_batch, 64, 3) for every batch of 128 validation "
                                  "events in storage order (consecutive calls on one generator); the event's hits are "
                                  "rows [hits[0], hits[1]) of its batch; offsets u = (rand - 0.5) * box; points = "
                                  "xyz + R u, as common.fold (scripts/val_parity.py fold_points_for_events). On the "
                                  "laptop: refstream.EmulGen(12345, grid) reproduces this GPU's draw for that grid."}
    cdir.mkdir(parents=True, exist_ok=True)
    jf = cdir / f"outliers_{bench}_{utc()}.json"
    jf.write_text(paths.scrub_str(json.dumps(summary, indent=1)) + "\n")
    arrays = {}
    for i, d in fp.items():
        for k in ("u", "z", "r", "iz", "ir", "z_edge_ulp", "r_edge_ulp", "w"):
            arrays[f"ev{i}_{k}"] = d[k]
    if arrays:
        np.savez(jf.with_suffix(".npz"), **arrays)
    summary["files"] = [str(jf), str(jf.with_suffix(".npz")) if arrays else None]
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create-refset", action="store_true", help="create the reference set of the live tag (once)")
    ap.add_argument("--no-guards", action="store_true", help="reference set without guard arrays (tests only)")
    ap.add_argument("--emulated-guards", action="store_true",
                    help="also compute the guard arrays on the emulated grid 1056 (automatic if the grid is not 1056)")
    ap.add_argument("--skip-r3", action="store_true", help="skip the run-to-run floor outside deterministic()")
    args = ap.parse_args(argv)

    batch = lib.start("val_parity")
    tag = machine.machine_tag()
    grid = tag.fields["launch_grid"]
    env = registry.RegistryEnv()
    mdir = env.machine_dir
    cdir = provenance.checks_dir(env.results_dir, tag.id, "val_parity")
    report = {"created_utc": utc(), "machine": tag.to_dict(), "batch": batch, "checks": {}, "notes": []}
    rec_hashes = provenance.laptop_record_hashes()

    def record(rel):
        r = provenance.laptop_record(rel, hashes=rec_hashes)
        r_out = {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
        return (np.load(r["path"]) if r["path"] is not None else None), r_out

    ref1056 = {b: record(BENCH_REL[b].format("_ref1056")) for b in registry.BENCHMARKS}

    def job():
        out = {}
        out["generator"] = generator_check(grid)
        print(f"[R0] generator check (EmulGen vs torch.rand, grid {grid}): {'PASS' if out['generator']['pass'] else 'FAIL'}")
        val = registry.load_validation()
        ref = registry.validation_reference(results_dir=env.results_dir)
        registry.check_validation(val, ref)
        models = registry._default_bench_models()
        for b in registry.BENCHMARKS:
            models[b] = models[b].to(common.DEVICE).eval()
        runs = {}
        for b in registry.BENCHMARKS:
            arrs = []
            for _ in range(2):
                with metric.deterministic():
                    arrs.append(common.test_nll(models[b], val.showers))
            runs[b] = arrs
        out["k64"] = {b: runs[b][0] for b in registry.BENCHMARKS}
        out["k64_second_run_bit_identical"] = {b: bool(np.array_equal(runs[b][0], runs[b][1])) for b in runs}
        if not args.skip_r3:
            out["r3"] = {b: provenance.tolerance_check(common.test_nll(models[b], val.showers),
                                                       common.test_nll(models[b], val.showers), relaxed=False, top=5)
                         for b in registry.BENCHMARKS}
        out["emul"] = {}
        for g in sorted({refstream.REFERENCE_GRID, refstream.LAPTOP_GRID}):
            with metric.deterministic():
                out["emul"][g] = {b: refstream.test_nll_ref(models[b], val.showers, grid=g) for b in registry.BENCHMARKS}
        if args.emulated_guards or grid != refstream.REFERENCE_GRID:
            zb = val.showers.ev["zb"].detach().cpu().numpy()
            idx = np.flatnonzero(zb >= registry.ZB_STRATUM_MM)
            eg = {}
            for b in registry.BENCHMARKS:
                eg[b] = {}
                with refstream.patched_draw(refstream.REFERENCE_GRID):
                    with torch.inference_mode():
                        eg[b]["k1024"] = metric.test_nll(models[b], val.showers, K=registry.K_GUARD,
                                                         chunk=registry.GUARD_CHUNK, deterministic_sum=True)
                with refstream.patched_draw(refstream.REFERENCE_GRID):
                    with torch.inference_mode():
                        sub = val.showers.subset(torch.as_tensor(idx, device=val.showers.off.device))
                        eg[b]["bulk_zb1221"] = metric.test_nll(models[b], sub, bulk=metric.Bulk(short="raise"),
                                                               deterministic_sum=True)
            out["emul_guards_1056"] = eg
            out["n_stratum"] = int(len(idx))
        if args.create_refset:
            out["refset"] = registry.create_reference_set(val, env=env, reference=ref, bench_models=lambda: models,
                                                          guards=not args.no_guards, k64_arrays=out["k64"])
        # P3: events above 5e-5 against the laptop's emulated arrays -> fold points of those events (on this GPU)
        out["fold_outliers"] = {}
        for b in registry.BENCHMARKS:
            rec = ref1056[b][0]
            if rec is None:
                continue
            cmp_arr, g_fp = ((out["k64"][b], grid) if grid == refstream.REFERENCE_GRID
                             else (out["emul"][refstream.REFERENCE_GRID][b], refstream.REFERENCE_GRID))
            idx = np.flatnonzero(np.abs(cmp_arr.astype(np.float64) - rec.astype(np.float64)) > 5e-5)
            if len(idx):
                out["fold_outliers"][b] = fold_points_for_events(val.showers, idx, g_fp)
        out["val"] = val
        return out

    res = lib.run_on_gpu("val_parity (exclusive)", job, min_free_mb=3000)
    val = res.pop("val")
    ok = []

    # ---- R0 ---------------------------------------------------------------------------------------------------------
    ident = machine.identity_record(generator_check=res["generator"])
    mdir.mkdir(parents=True, exist_ok=True)
    idf = mdir / "identity.json"
    if idf.exists():
        old = json.loads(idf.read_text())
        same = old.get("machine") == ident["machine"]
        report["notes"].append(f"identity.json exists (same machine tag: {same}); kept, not rewritten")
        if not same:
            (mdir / f"identity_{utc()}.json").write_text(paths.scrub_str(json.dumps(ident, indent=1, default=str)) + "\n")
    else:
        idf.write_text(paths.scrub_str(json.dumps(ident, indent=1, default=str)) + "\n")
    report["checks"]["R0"] = {"identity_file": paths.portable(idf), "generator": res["generator"], "launch_grid": grid,
                              "pass": res["generator"]["pass"]}
    ok.append(res["generator"]["pass"])

    # ---- R1 ---------------------------------------------------------------------------------------------------------
    k64 = res["k64"]
    r1 = {"second_run_bit_identical": res["k64_second_run_bit_identical"], "means": {}, "vs_laptop_ref1056": {},
          "emulated_1056_vs_native": {}, "emulated_1056_vs_laptop_ref1056": {}}
    ok.append(all(res["k64_second_run_bit_identical"].values()))
    for b in registry.BENCHMARKS:
        a = k64[b]
        r1["means"][b] = {"float32_mean": float(np.float32(a.mean())), "float64_mean": float(a.astype(np.float64).mean())}
    stream_pts = registry._points(registry.validation_reference(results_dir=env.results_dir).fingerprint, grid, "k64")
    ta = {b: provenance.TaggedArray.make(k64[b], stream_pts, tag, label=f"native {b}") for b in registry.BENCHMARKS}
    fm = provenance.paired_delta(ta["formula"], ta["mlp"])                 # same tag: a delta of this machine
    r1["formula_minus_mlp"] = {"mean": fm.mean, "se": fm.se}
    p3_events = 0
    for b in registry.BENCHMARKS:
        rec, info = ref1056[b]
        emu = res["emul"][refstream.REFERENCE_GRID][b]
        r1["emulated_1056_vs_native"][b] = provenance.tolerance_check(k64[b], emu, top=5)
        if rec is None:
            r1["vs_laptop_ref1056"][b] = {"record": info, "comparable": False}
            r1["emulated_1056_vs_laptop_ref1056"][b] = {"record": info, "comparable": False}
            continue
        r1["emulated_1056_vs_laptop_ref1056"][b] = {"record": info, **provenance.tolerance_check(emu, rec, top=5)}
        if grid == refstream.REFERENCE_GRID:
            tc = provenance.tolerance_check(k64[b], rec)
            r1["vs_laptop_ref1056"][b] = {"record": info, **tc}
            p3_events += tc["p3_flag_events_above_5e-5"]
            if tc["p3_flag_events_above_5e-5"]:
                r1.setdefault("p3_outliers", {})[b] = save_outliers(cdir, b, val.event_number, k64[b], rec, grid,
                                                                    info["path"], res["fold_outliers"].get(b, {}))
        else:   # fallback: the emulated grid-1056 arrays take the role of the native ones for the targets
            tc = r1["emulated_1056_vs_laptop_ref1056"][b]
            p3_events += tc.get("p3_flag_events_above_5e-5", 0)
            if tc.get("p3_flag_events_above_5e-5"):
                r1.setdefault("p3_outliers", {})[b] = save_outliers(cdir, b, val.event_number, emu, rec,
                                                                    refstream.REFERENCE_GRID, info["path"],
                                                                    res["fold_outliers"].get(b, {}))
    t = TARGETS[1056]
    if grid == refstream.REFERENCE_GRID:
        src = {b: k64[b] for b in registry.BENCHMARKS}
        r1["targets_on"] = "native arrays (grid 1056)"
    else:
        src = res["emul"][refstream.REFERENCE_GRID]
        r1["targets_on"] = f"EMULATED grid 1056 arrays (native grid is {grid}: fallback of PLAN.md section 16.4)"
        r1["A3_native_vs_emulated"] = {}
        for b in registry.BENCHMARKS:
            d = k64[b].astype(np.float64) - src[b].astype(np.float64)
            r1["A3_native_vs_emulated"][b] = {
                "correlation": float(np.corrcoef(k64[b].astype(np.float64), src[b].astype(np.float64))[0, 1]),
                "rms_diff": float(np.sqrt(np.mean(d * d))), "laptop_validation_rms": A3_RMS[b],
                "pass": bool(np.corrcoef(k64[b], src[b])[0, 1] >= 0.9995 and
                             abs(np.sqrt(np.mean(d * d)) / A3_RMS[b] - 1) <= 0.15)}
    r1["targets"] = {}
    for b in registry.BENCHMARKS:
        r1["targets"][f"{b}_float32_mean"] = within(float(np.float32(src[b].mean())), t["float32_mean"][b],
                                                    t["float32_tol"])
    tfm = provenance.paired_delta(*(provenance.TaggedArray.make(src[b], stream_pts, tag) for b in ("formula", "mlp")))
    r1["targets"]["formula_minus_mlp_mean"] = within(tfm.mean, t["formula_minus_mlp"][0], 1.5e-6)
    r1["targets"]["formula_minus_mlp_se"] = within(tfm.se, t["formula_minus_mlp"][1], 1.5e-6)
    for b in registry.BENCHMARKS:
        v = r1["vs_laptop_ref1056"].get(b) if grid == refstream.REFERENCE_GRID else r1["emulated_1056_vs_laptop_ref1056"].get(b)
        if v and v.get("comparable"):
            r1["targets"][f"{b}_per_event_vs_laptop_emulated"] = {"strict_pass": v["strict_pass"],
                                                                  "relaxed_pass": v["relaxed_pass"],
                                                                  "max_abs": v["max_abs"], "mean_diff": v["mean_diff"],
                                                                  "pass": v["strict_pass"]}
        else:
            r1["targets"][f"{b}_per_event_vs_laptop_emulated"] = {"pass": False, "reason": "laptop record not found"}
    if grid == refstream.REFERENCE_GRID:
        r1["targets"]["emulated_equals_native"] = {
            "pass": all(r1["emulated_1056_vs_native"][b]["bit_identical"] for b in registry.BENCHMARKS)}
    r1["pass"] = all(x["pass"] for x in r1["targets"].values()) and all(r1["second_run_bit_identical"].values())
    r1["p3_events_above_5e-5_total"] = p3_events
    report["checks"]["R1"] = r1
    ok.append(r1["pass"])
    if grid == refstream.LAPTOP_GRID:          # the laptop itself: its own numbers of 24 Nov (A2 tolerances)
        lap = {}
        for b in registry.BENCHMARKS:
            rec, info = record(BENCH_REL[b].format(""))
            lap[b] = {"record": info, **(provenance.tolerance_check(k64[b], rec, top=5) if rec is not None else {})}
            lap[b]["stored_mean"] = within(float(k64[b].astype(np.float64).mean()), TARGETS[144]["mean64"][b], 1.5e-6)
        report["checks"]["laptop_native_vs_laptop_records"] = lap

    # ---- R1b (benchmarks) -------------------------------------------------------------------------------------------
    r1b = {}
    for b in registry.BENCHMARKS:
        rec, info = record(BENCH_REL[b].format(""))
        emu144 = res["emul"][refstream.LAPTOP_GRID][b]
        r1b[b] = ({"record": info, **provenance.tolerance_check(emu144, rec, top=5)} if rec is not None
                  else {"record": info, "comparable": False, "strict_pass": False})
    r1b["note"] = ("laptop arrays m0_parity/*_val_nll.npy were written outside deterministic() (up to 2.7e-5 of own "
                   "noise); the clean partners are the eight registered networks (rescore_registered.py "
                   "--emulate-laptop)")
    r1b["pass"] = all(r1b[b].get("strict_pass", False) for b in registry.BENCHMARKS)
    report["checks"]["R1b_benchmarks"] = r1b

    # ---- R3 ---------------------------------------------------------------------------------------------------------
    if "r3" in res:
        r3 = {b: res["r3"][b] for b in registry.BENCHMARKS}
        r3["pass"] = all(r3[b]["strict_pass"] for b in registry.BENCHMARKS)
        report["checks"]["R3"] = r3
        ok.append(r3["pass"])

    # ---- R2 ---------------------------------------------------------------------------------------------------------
    if "refset" in res:
        rs_info = res["refset"]
        r2 = {"refset": {k: v for k, v in rs_info.items() if k != "manifest"}}
        rs = registry.reference_set(env, need_guards=not args.no_guards)
        tg = TARGETS.get(grid)
        if rs.guards:
            tag_pts = {k: rs.guards["mlp"][k] for k in ("k1024", "bulk_zb1221")}
            r2["means"] = {f"{b}_{k}": float(np.nanmean(rs.guards[b][k].values.astype(np.float64)))
                           for b in registry.BENCHMARKS for k in ("k1024", "bulk_zb1221")}
            r2["n_stratum"] = int(rs.guards["mlp"]["bulk_zb1221"].values.size)
            r2["n_bulk_undefined"] = {b: int((~np.isfinite(rs.guards[b]["bulk_zb1221"].values)).sum())
                                      for b in registry.BENCHMARKS}
            r2["formula_minus_mlp"] = {k: {"mean": p.mean, "se": p.se} for k, p in
                                       ((k, provenance.paired_delta(rs.guards["formula"][k], rs.guards["mlp"][k]))
                                        for k in ("k1024", "bulk_zb1221"))}
            r2["targets"] = {"n_stratum": {"value": r2["n_stratum"], "target": N_STRATUM,
                                           "pass": r2["n_stratum"] == N_STRATUM},
                             "no_undefined_bulk": {"pass": not any(r2["n_bulk_undefined"].values())}}
            if tg:
                for b in registry.BENCHMARKS:
                    r2["targets"][f"{b}_k1024_mean"] = within(r2["means"][f"{b}_k1024"], tg["k1024_mean"][b], 1e-6)
                    r2["targets"][f"{b}_bulk_mean"] = within(r2["means"][f"{b}_bulk_zb1221"], tg["bulk_mean"][b], 1e-6)
                for k, (mu, se) in tg["guard_formula_minus_mlp"].items():
                    r2["targets"][f"formula_minus_mlp_{k}"] = within(r2["formula_minus_mlp"][k]["mean"], mu, 1.5e-6)
            if grid == refstream.LAPTOP_GRID:
                r2["vs_laptop_registry_refs"] = {}
                for (b, k), rel in GUARD_REL.items():
                    rec, info = record(rel)
                    if rec is not None:
                        r2["vs_laptop_registry_refs"][f"{b}_{k}"] = {
                            "record": info, **provenance.tolerance_check(rs.guards[b][k].values, rec, top=3)}
            r2["pass"] = all(v["pass"] for v in r2["targets"].values())
            del tag_pts
        else:
            r2["pass"] = None
            report["notes"].append("reference set without guard arrays (--no-guards)")
        report["checks"]["R2"] = r2
        if r2["pass"] is not None:
            ok.append(r2["pass"])
    if "emul_guards_1056" in res:
        eg, t = res["emul_guards_1056"], TARGETS[1056]
        r2e = {"targets_on": "EMULATED grid 1056 guard arrays (refstream.patched_draw)", "n_stratum": res["n_stratum"]}
        for b in registry.BENCHMARKS:
            r2e[f"{b}_k1024_mean"] = within(float(eg[b]["k1024"].astype(np.float64).mean()), t["k1024_mean"][b], 1e-6)
            r2e[f"{b}_bulk_mean"] = within(float(np.nanmean(eg[b]["bulk_zb1221"].astype(np.float64))),
                                           t["bulk_mean"][b], 1e-6)
        r2e["pass"] = all(v["pass"] for k, v in r2e.items() if isinstance(v, dict)) and res["n_stratum"] == N_STRATUM
        report["checks"]["R2_emulated_1056"] = r2e
        if grid != refstream.REFERENCE_GRID:
            ok.append(r2e["pass"])

    report["all_applicable_checks_pass"] = bool(all(ok))
    cdir.mkdir(parents=True, exist_ok=True)
    rf = cdir / f"report_{report['created_utc']}.json"

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return o
    rf.write_text(paths.scrub_str(json.dumps(_clean(report), indent=1, default=str)) + "\n")

    # ---- summary ----------------------------------------------------------------------------------------------------
    print("\n================ val_parity summary (validation only; tolerance checks are not deltas) ================")
    print(f"machine {tag.id}; launch grid {grid}{'' if grid == 1056 else '  <-- NOT 1056: fallback of PLAN.md 16.4'}")
    print(f"R0  generator check: {'PASS' if res['generator']['pass'] else 'FAIL'}; identity: {idf}")
    print(f"R1  second run bit-identical: {res['k64_second_run_bit_identical']}; targets on {r1['targets_on']}")
    for b in registry.BENCHMARKS:
        print(f"    {b:<8} float32 mean {r1['means'][b]['float32_mean']:.9f}  float64 {r1['means'][b]['float64_mean']:.7f}")
    print(f"    formula - mlp (native, this machine): {fm.mean * 1e4:+.2f} +- {fm.se * 1e4:.2f} e-4")
    for k, v in r1["targets"].items():
        print(f"    {k:<40} {'PASS' if v['pass'] else 'FAIL'}  {({kk: v[kk] for kk in v if kk != 'pass'})}")
    if p3_events:
        print("\n" + "!" * 110)
        print(f"!!! P3: {p3_events} validation event(s) differ from the laptop's emulated arrays by MORE THAN 5e-5.")
        print("!!! The relaxed per-event clause of the final run may be activated ONLY after Fable has confirmed, on")
        print("!!! validation, that these outliers are 1-ULP bin-edge flips. Material for the reviewer:")
        for b, s in r1.get("p3_outliers", {}).items():
            print(f"!!!   {b}: {s['n_events_above_5e-5']} event(s): {s['files']}")
        print("!" * 110 + "\n")
    else:
        print("    P3: no validation event differs from the laptop's emulated arrays by more than 5e-5")
    if "A3_native_vs_emulated" in r1:
        print(f"    A3 native vs emulated 1056: {r1['A3_native_vs_emulated']}")
    print(f"R1b benchmarks, emulated grid 144 vs laptop arrays: {'PASS' if r1b['pass'] else 'FAIL'} "
          f"(max |diff| {[r1b[b].get('max_abs') for b in registry.BENCHMARKS]})")
    if "R3" in report["checks"]:
        r3 = report["checks"]["R3"]
        print(f"R3  run-to-run outside deterministic(): {'PASS' if r3['pass'] else 'FAIL'} "
              f"(max |diff| {[r3[b]['max_abs'] for b in registry.BENCHMARKS]})")
    if "R2" in report["checks"]:
        r2 = report["checks"]["R2"]
        print(f"R2  reference set: {r2['refset']['status']} at {r2['refset']['dir']}; guard targets: "
              f"{'PASS' if r2['pass'] else ('n/a' if r2['pass'] is None else 'FAIL')}")
        for k, v in (r2.get("targets") or {}).items():
            print(f"    {k:<40} {'PASS' if v['pass'] else 'FAIL'}  {({kk: v[kk] for kk in v if kk != 'pass'})}")
    else:
        print("R2  reference set not requested (--create-refset)")
    if "R2_emulated_1056" in report["checks"]:
        print(f"R2e guards on the emulated grid 1056: {'PASS' if report['checks']['R2_emulated_1056']['pass'] else 'FAIL'}")
    print(f"report: {rf}")
    print(f"ALL APPLICABLE CHECKS: {'PASS' if report['all_applicable_checks_pass'] else 'FAIL'}")
    return 0 if report["all_applicable_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
