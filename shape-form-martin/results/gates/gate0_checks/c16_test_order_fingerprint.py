"""Gate 0 check 3 (stream-independent, test side): is the reference machine's test set the same event sequence as ours?

test_v3.json stores per-momentum-bin means of (std:new - std:base_acc) and (slab:new - slab:base), computed on the
reference machine with ITS data["test"].ev["bin"]. Recomputing them from the stored per-event arrays (test_v3_nll.npz)
with OUR per-event bins (derived from the Parquet file by c05, in the order of results/m0_parity/test_event_number.npy)
reproduces the stored values only if the bin label of every position agrees - a fingerprint of the split and its order.
Uses stored reference arrays and event metadata only: no test cache, no model evaluation.
Writes c16_test_order_fingerprint.json.
"""
import json

from g0common import *  # noqa: F401,F403

tab = np.load(HERE / "events_table.npz")
ten = np.load(M0 / "test_event_number.npy")
m = tab["split"] == 2
assert np.array_equal(tab["event_number"][m], ten)
b = tab["bin"][m]
npz = np.load(PION / "test_v3_nll.npz")
J = jload(PION / "test_v3.json")
rng = np.random.default_rng(0)
out = {}
for key, a, c in (("per_bin_new_minus_base_std", "std:new", "std:base_acc"), ("per_bin_new_minus_base_slab", "slab:new", "slab:base")):
    d = npz[a] - npz[c]                                               # float32, as in particle_pipeline.py
    mine = np.array([float(d[b == k].mean()) for k in range(20)])
    stored = np.array(J[key])
    # controls: bins shifted by one position, bins of a random permutation, one swapped pair of neighbouring events
    shifted = np.array([float(d[np.roll(b, 1) == k].mean()) for k in range(20)])
    perm = rng.permutation(b)
    permuted = np.array([float(d[perm == k].mean()) for k in range(20)])
    b2 = b.copy()
    i = int(np.flatnonzero(b[:-1] != b[1:])[100])
    b2[i], b2[i + 1] = b2[i + 1], b2[i]
    swapped = np.array([float(d[b2 == k].mean()) for k in range(20)])
    out[key] = {"max_abs_diff_ours_vs_stored": float(np.abs(mine - stored).max()),
                "identical_to_float_repr": bool(np.array_equal(mine, stored)),
                "control_shift_by_one_max_abs_diff": float(np.abs(shifted - stored).max()),
                "control_random_permutation_max_abs_diff": float(np.abs(permuted - stored).max()),
                "control_one_swapped_neighbour_pair_max_abs_diff": float(np.abs(swapped - stored).max()),
                "stored_first3": stored[:3].tolist(), "ours_first3": mine[:3].tolist()}
out["n_events"] = {"npz": int(len(npz["std:new"])), "ours": int(m.sum()), "stored_json": J["n_events"]}
out["bin_counts_ours"] = np.bincount(b, minlength=20).tolist()
(HERE / "c16_test_order_fingerprint.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
