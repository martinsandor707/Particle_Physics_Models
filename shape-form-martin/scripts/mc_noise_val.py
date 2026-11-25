"""Monte-Carlo noise of common.test_nll's fold points, on VALIDATION only (no test access).

    uv run python -B shape-form-martin/scripts/mc_noise_val.py

Gives the reviewer of Gate 0 a same-machine baseline for the parity comparison: the per-event agreement
(Pearson r, RMS / median / max |Δ|) and the spread of means between two independent fold-point streams
(generator seeds 1..8, K=64, otherwise exactly common.test_nll's loop), for both benchmarks.
Also checks that the formula's stored ln p centring equals the one recomputed from the cached training set.
Writes results/m0_parity/mc_noise_val.json.
"""
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data, legacy, paths  # noqa: E402

common = legacy.common


@torch.no_grad()
def nll_with_seed(model, ds, seed, K=common.N_FOLD_EVAL):
    gen = torch.Generator(device=common.DEVICE).manual_seed(seed)
    return torch.cat([common.event_nll(model, b, ds.box, K, gen)[0] for b in ds.batches(128)]).cpu().numpy()



def _refuse_record_writer(what: str) -> None:
    """transfer/REVIEW_STATUS.md section 4, point 15: this script overwrites a tracked laptop record (or opens test).
    Blocked unless SFM_ALLOW_RECORD_WRITER=1 is set explicitly, and always blocked with an output root
    (SFM_OUT_ROOT: the H200 / pristine-clone setting)."""
    import os as _os
    if _os.environ.get("SFM_OUT_ROOT") or _os.environ.get("SFM_ALLOW_RECORD_WRITER") != "1":
        raise SystemExit(f"{what}: blocked - it overwrites tracked laptop records (spec point 15). Never run it on "
                         "the H200; on the laptop only with SFM_ALLOW_RECORD_WRITER=1 and the user's go-ahead.")

def main():
    _refuse_record_writer("mc_noise_val.py (writes results/m0_parity/mc_noise_val.json)")
    sets, _ = data.load_sets(("train", "val"))
    models = {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(sets["train"])}
    seeds = list(range(1, 9))
    out = {"split": "val", "K": common.N_FOLD_EVAL, "seeds": seeds}
    for name, m in models.items():
        arr = {s: nll_with_seed(m, sets["val"], s).astype(np.float64) for s in seeds}
        pairs = []
        for a, b in itertools.combinations(seeds, 2):
            d = arr[a] - arr[b]
            pairs.append({"pearson_r": float(np.corrcoef(arr[a], arr[b])[0, 1]), "rms": float(np.sqrt((d ** 2).mean())),
                          "median_abs": float(np.median(np.abs(d))), "max_abs": float(np.abs(d).max()),
                          "mean_diff": float(d.mean())})
        out[name] = {k: {"mean": float(np.mean([p[k] for p in pairs])), "min": float(np.min([p[k] for p in pairs])),
                         "max": float(np.max([p[k] for p in pairs]))}
                     for k in ("pearson_r", "rms", "median_abs", "max_abs", "mean_diff")}
        out[name]["n_pairs"] = len(pairs)
        out[name]["sd_of_mean_diff_between_two_streams"] = float(np.std([p["mean_diff"] for p in pairs], ddof=1))
    c0, s0 = legacy.shapelab.lnp_center_scale({"train": sets["train"]})
    core = models["formula"].base
    out["formula_lnp_centring"] = {"recomputed_from_cache": [c0, s0],
                                   "stored_in_checkpoint": [float(core.lnp0), float(core.lnp_s)],
                                   "equal_float32": bool(np.float32(c0) == np.float32(float(core.lnp0))
                                                         and np.float32(s0) == np.float32(float(core.lnp_s)))}
    dest = paths.M0_DIR / "mc_noise_val.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
