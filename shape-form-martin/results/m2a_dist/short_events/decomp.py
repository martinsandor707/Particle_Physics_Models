"""Question 3b/3c/5: per-hit decomposition of the K = 64 metric of record for registered runs of the machine of record,
recomputed on the laptop CPU on the SAME fold points (the H200's CUDA random stream, launch grid 1056, emulated by
sfm.refstream). Read-only: the runs are rebuilt with sfm.models.load_run from their model.pt (BuildContext pointed at
the copied machine directory; skip_init), the Track A templates read from that directory (set_template_dir).

    CUDA_VISIBLE_DEVICES= uv run python -B shape-form-martin/results/m2a_dist/short_events/decomp.py \
        --out <scratch>/decomp.npz [--subset] [--max-batches N] RUN [RUN ...]

--subset   evaluate only the batches that contain a short event or one of 1,500 randomly chosen other events (the
           generator offset is advanced exactly for the skipped batches, so the points stay those of the full pass)
Runs entirely on the CPU (no GPU used; CUDA hidden): metric.event_nll_multi(return_parts=True) on the emulated points.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import se_common as S  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("run with CUDA hidden (CUDA_VISIBLE_DEVICES=): this diagnostic is CPU only")

from sfm import metric, models, refstream  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

C = metric.C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--n-other", type=int, default=1500)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dfac.set_template_dir(S.STAGE_DIR / "templates")
    ctx = models.BuildContext(device="cpu", results_dir=S.RUNS, skip_init=True, log=lambda *a, **k: None)
    t0 = time.time()
    ms = [models.load_run(f"m2a_dist/{n}", ctx).eval() for n in args.runs]
    print(f"loaded {len(ms)} runs in {time.time() - t0:.1f} s", flush=True)
    V = S.val_info()
    val = V["_sets"]["val"]
    N = len(val)
    sh = S.short_mask()
    sel = sh.copy()
    if args.subset:
        rng = np.random.default_rng(1500)
        sel[rng.choice(np.flatnonzero(~sh), size=args.n_other, replace=False)] = True
    else:
        sel[:] = True
    off = val.off.numpy()
    K, B = C.N_FOLD_EVAL, 128
    M = len(ms)
    nll = np.full((M, N), np.nan, np.float32)
    logm = np.full((M, N), np.nan, np.float32)
    wsum = np.full(N, np.nan, np.float32)
    done_ev = np.zeros(N, bool)
    hit = np.full((M, int(off[-1])), np.nan, np.float32)
    pz_short = {}                      # event index -> (M, N_Z) longitudinal profile (short events only)
    t0 = time.time()
    with torch.inference_mode(), refstream.patched_draw(refstream.REFERENCE_GRID, 12345) as gen:
        for j, b in enumerate(val.batches(B)):
            lo, hi = j * B, min((j + 1) * B, N)
            H = len(b.xyz)
            if args.max_batches is not None and j >= args.max_batches:
                break
            if not sel[lo:hi].any():                         # advance the stream exactly as a draw would
                numel = H * K * 3
                g = min(gen.grid, (numel + refstream.THREADS_PER_BLOCK - 1) // refstream.THREADS_PER_BLOCK)
                T = refstream.THREADS_PER_BLOCK * g
                gen.offset += ((numel - 1) // (T * 4) + 1) * 4
                continue
            res = metric.event_nll_multi(ms, b, val.box, K, None, return_parts=True)
            for m, (e_nll, logP, parts) in enumerate(res):
                nll[m, lo:hi] = e_nll.numpy()
                logm[m, lo:hi] = parts.log_mass.numpy()
                hit[m, off[lo]:off[hi]] = parts.hit.numpy()
            wsum[lo:hi] = res[0][2].weight_sum.numpy()
            done_ev[lo:hi] = True
            for e in np.flatnonzero(sh[lo:hi]):
                pz_short[lo + e] = np.stack([r[1][e].exp().double().sum(1).numpy() for r in res])
            if j % 10 == 0:
                print(f"batch {j} ({hi} events) {time.time() - t0:.1f} s", flush=True)
    print(f"pass finished in {time.time() - t0:.1f} s; events evaluated {int(done_ev.sum())}", flush=True)
    ks = sorted(pz_short)
    np.savez(args.out, runs=np.array(args.runs), nll=nll, log_mass=logm, weight_sum=wsum, done=done_ev, hit=hit,
             pz_short_idx=np.array(ks, dtype=np.int64),
             pz_short=np.stack([pz_short[k] for k in ks]) if ks else np.zeros((0, M, C.N_Z)), subset=sel)
    # agreement with the registered arrays (tolerance check, not a delta)
    from sfm import provenance
    for m, n in enumerate(args.runs):
        reg = S.tagged(n).values
        tc = provenance.tolerance_check(nll[m, done_ev], reg[done_ev], top=3)
        print(f"{n}: recomputed vs registered val_nll on {int(done_ev.sum())} events: max |d| {tc['max_abs']:.2e}, "
              f"rms {tc['rms']:.2e}, mean diff {tc['mean_diff']:.2e}, n>1e-4 {tc['n_above']['0.0001']}, "
              f"n>1e-5 {tc['n_above']['1e-05']}", flush=True)


if __name__ == "__main__":
    main()
