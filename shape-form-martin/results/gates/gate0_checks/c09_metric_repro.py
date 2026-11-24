"""Gate 0 check 4: run-to-run reproducibility of common.test_nll on CUDA and equality with sfm.metric (validation).

  * 8 consecutive common.test_nll runs per benchmark: per-event spread, events that differ, spread of the mean;
  * the same under torch.use_deterministic_algorithms(True, warn_only=True): bit-reproducible?
  * which step is non-reproducible: per-hit terms (before index_add_) compared between two runs;
  * sfm.metric.test_nll / test_nll_multi vs common.test_nll: bitwise under deterministic(), and in default mode;
  * chunked evaluation and inference_mode vs no_grad.
Writes c09_metric_repro.json.
"""
import json
import math

from g0common import *  # noqa: F401,F403
from sfm import metric as M

sets, _ = load_trainval()
val, train = sets["val"], sets["train"]
models = {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(train)}
out = {"device": str(DEV), "torch": torch.__version__}
N = 8
for k, m in models.items():
    runs = np.stack([common.test_nll(m, val).astype(np.float64) for _ in range(N)])
    spread = runs.max(0) - runs.min(0)
    means = runs.mean(1)
    r = {"runs": N, "per_event_spread_max": float(spread.max()), "per_event_spread_mean": float(spread.mean()),
         "events_with_any_difference": int((spread > 0).sum()), "n_events": int(runs.shape[1]),
         "events_differing_run0_vs_run1": int((runs[0] != runs[1]).sum()),
         "max_abs_diff_run0_vs_run1": float(np.abs(runs[0] - runs[1]).max()),
         "mean_min": float(means.min()), "mean_max": float(means.max()), "mean_spread": float(means.max() - means.min()),
         "float32_mean_values": sorted({float(np.float32(x).mean()) for x in runs.astype(np.float32)}),
         "relative_per_event_spread_max": float((spread / np.abs(runs[0])).max())}
    with M.deterministic():
        det = np.stack([common.test_nll(m, val) for _ in range(4)])
        r["deterministic_runs_bit_identical"] = bool(all(np.array_equal(det[0], d) for d in det[1:]))
        new = M.test_nll(m, val)
        r["sfm_metric_equals_common_bitwise_deterministic"] = bool(np.array_equal(new, det[0]) and new.dtype == det[0].dtype)
        r["sfm_metric_no_grad_equals_bitwise"] = bool(np.array_equal(M.test_nll(m, val, inference=False), det[0]))
        r["sfm_metric_K16_equals_common_bitwise"] = bool(np.array_equal(M.test_nll(m, val, 16), common.test_nll(m, val, 16)))
        ch = M.test_nll(m, val, chunk=16).astype(np.float64)
        r["chunk16_vs_unchunked_max_abs_diff_deterministic"] = float(np.abs(ch - det[0]).max())
        r["deterministic_vs_default_max_abs_diff"] = float(np.abs(det[0].astype(np.float64) - runs[0]).max())
        r["deterministic_mean"] = float(det[0].astype(np.float64).mean())
    d = M.test_nll(m, val).astype(np.float64) - runs[0]
    r["sfm_metric_default_mode_vs_common"] = {"max_abs_diff": float(np.abs(d).max()), "mean_diff": float(d.mean()),
                                              "max_rel": float((np.abs(d) / np.abs(runs[0])).max())}
    out[k] = r
with M.deterministic():
    multi = M.test_nll_multi(list(models.values()), val)
    out["test_nll_multi_equals_separate_common_calls_bitwise"] = bool(all(np.array_equal(multi[j], common.test_nll(m, val)) for j, m in enumerate(models.values())))

# where does the difference arise? per-hit terms of the first batch, two runs
b = next(iter(val.batches(128)))
with torch.no_grad():
    res = []
    for _ in range(2):
        gen = torch.Generator(device=DEV).manual_seed(12345)
        nll, logP, parts = M.event_nll(models["mlp"], b, val.box, 64, gen, return_parts=True)
        res.append((nll.clone(), parts.hit.clone(), parts.log_mass.clone()))
    out["first_batch_two_runs"] = {"per_hit_terms_identical": bool(torch.equal(res[0][1], res[1][1])),
                                   "log_mass_identical": bool(torch.equal(res[0][2], res[1][2])),
                                   "nll_identical": bool(torch.equal(res[0][0], res[1][0])),
                                   "nll_max_abs_diff": float((res[0][0] - res[1][0]).abs().max()),
                                   "hits_in_batch": int(len(b.w)), "ulp_at_18": float(np.spacing(np.float32(18.0)))}
    # float64 reference sum of the identical per-hit terms: how far is each float32 index_add_ result from it?
    ll = (res[0][1].double() - (b.w.double() * res[0][2].double()[b.ev]))
    exact = -torch.zeros(len(b.lnp), dtype=torch.float64, device=DEV).index_add_(0, b.ev, ll)
    out["first_batch_two_runs"]["max_abs(float32 index_add - float64 sum of the same float32 terms)"] = float((res[0][0].double() - exact).abs().max())
(HERE / "c09_metric_repro.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
