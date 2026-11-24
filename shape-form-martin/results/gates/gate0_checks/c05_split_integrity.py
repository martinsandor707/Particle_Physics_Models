"""Gate 0 check 1: split integrity and leakage, re-derived independently from the Parquet file.

Does NOT use sfm.data.build_showers / split_indices and does NOT load the test cache. The cuts are re-implemented
with plain numpy masks; the split is produced by exec'ing the loop of common.load_data itself (source lines taken
from the module with inspect) on the per-event momentum bins. The result is compared with the cache extras (train,
val) and with results/m0_parity/test_event_number.npy (test).
Writes c05_split_integrity.json and events_table.npz (per-event metadata of all 99,689 events: event_number, split,
bin, p, theta, phi, n_hits after cuts, raw rows; no hit data, no model output).
"""
import ast
import inspect
import json
import textwrap

import pyarrow.parquet as pq

from g0common import *  # noqa: F401,F403

pf = pq.ParquetFile(str(paths.parquet_path()))
col = lambda name: pf.read(columns=[name]).column(0).to_numpy()          # noqa: E731
out = {"parquet": paths.parquet_path().name, "rows": int(pf.metadata.num_rows), "row_groups": int(pf.metadata.num_row_groups),
       "schema": {f.name: str(f.type) for f in pf.schema_arrow}}

# ---- cuts (set-wise identical to from_dataframe: time < 100 ns, then -200 <= z' <= 1450 and r < 700)
evn = col("event_number")
keep = col("earliest_contribution_time") < common.TIME_CUT_NS
out["rows_after_time_cut"] = int(keep.sum())
r = np.hypot(col("x_local"), col("y_local"))
keep &= r < common.R_MAX
del r
zl = col("z_local")
keep &= (zl >= common.Z_MIN) & (zl <= common.Z_MAX)
del zl
out["rows_after_all_cuts"] = int(keep.sum())

# ---- events
order = np.argsort(evn, kind="stable")
evs = evn[order]
starts = np.r_[0, np.flatnonzero(np.diff(evs)) + 1]
raw_ev = evs[starts]
raw_cnt = np.diff(np.r_[starts, len(evs)])
out["events_raw"] = int(len(raw_ev))
kept_cnt = np.add.reduceat(keep[order].astype(np.int64), starts)
surv = kept_cnt > 0
uniq, cnt = raw_ev[surv].astype(np.int64), kept_cnt[surv]
out["events_after_cuts"] = int(len(uniq))
out["events_removed_by_cuts"] = int((~surv).sum())


def per_event_const(name):
    """(first value per raw event in sorted order, True if constant within every event)"""
    v = col(name)[order]
    lo, hi = np.minimum.reduceat(v, starts), np.maximum.reduceat(v, starts)
    return v[starts], bool(np.array_equal(lo, hi))


ev_const = {}
cols = {}
for name in ("incoming_momentum_bin", "incoming_momentum", "incoming_theta", "incoming_phi"):
    cols[name], ev_const[name] = per_event_const(name)
out["event_level_columns_constant_within_event"] = ev_const
# shower origin z0 = z - z_trans: spread within an event (float32 z_trans)
z0 = (col("z") - col("z_trans"))[order]
spread = np.maximum.reduceat(z0, starts) - np.minimum.reduceat(z0, starts)
out["z0_spread_within_event_mm"] = {"max": float(spread.max()), "median": float(np.median(spread))}
del z0
bins = cols["incoming_momentum_bin"][surv].astype(np.int64)
p_ev = cols["incoming_momentum"][surv].astype(np.float64)
# is the bin the integer GeV of p?
out["bin_equals_floor_p"] = float((np.floor(p_ev).astype(np.int64) == bins).mean())

# ---- the split: exec the loop of common.load_data verbatim
src = textwrap.dedent(inspect.getsource(common.load_data)).splitlines()
i0 = next(i for i, l in enumerate(src) if l.strip().startswith("rng = np.random.default_rng"))
i1 = next(i for i, l in enumerate(src) if l.strip().startswith("sets = "))
loop = textwrap.dedent("\n".join(src[i0:i1]))
out["load_data_split_source"] = loop.splitlines()


class _Stub:                                     # stands in for the Showers object: only .ev["bin"] is read by the loop
    ev = {"bin": torch.tensor(bins)}


ns = {"np": np, "sh": _Stub, "SEED": common.SEED, "fracs": inspect.signature(common.load_data).parameters["fracs"].default}
exec(loop, ns)
parts = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in ns["parts"].items()}

# sfm.data.split_indices: same statements as the loop of load_data?
sp = textwrap.dedent(inspect.getsource(data.split_indices))
norm = lambda s: ast.dump(ast.parse(s))          # noqa: E731
loop_body_ref = norm(textwrap.dedent("\n".join(src[i0 + 1:i1])).replace("bins = sh.ev[\"bin\"].numpy()\n", ""))
tree = ast.parse(sp).body[0]
body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
mine = ast.dump(ast.Module(body=body[1:-1], type_ignores=[]))      # drop rng line and the return
out["split_indices_loop_ast_equals_load_data_loop"] = bool(mine == loop_body_ref)
out["split_indices_rng_line"] = ast.unparse(body[0])
out["split_indices_return_line"] = ast.unparse(body[-1])

# ---- compare with the cache (train/val) and the saved test event numbers
sets, extras = data.load_sets(("train", "val"), device="cpu")
saved = {"train": extras["train"]["event_number"], "val": extras["val"]["event_number"],
         "test": np.load(M0 / "test_event_number.npy")}
cmp = {}
for k in ("train", "val", "test"):
    mine_ev = uniq[parts[k]]
    cmp[k] = {"n": int(len(mine_ev)), "n_saved": int(len(saved[k])),
              "event_numbers_identical_and_same_order": bool(np.array_equal(mine_ev, saved[k])),
              "ascending": bool(np.all(np.diff(saved[k]) > 0)),
              "hits_from_parquet": int(cnt[parts[k]].sum())}
for k in ("train", "val"):
    cmp[k]["index_extra_equals_split_positions"] = bool(np.array_equal(extras[k]["index"], parts[k]))
    cmp[k]["hits_per_event_equal_cache_off"] = bool(np.array_equal(np.diff(sets[k].off.numpy()), cnt[parts[k]]))
    cmp[k]["hits_in_cache"] = int(sets[k].off[-1])
    cmp[k]["bin_equal"] = bool(np.array_equal(sets[k].ev["bin"].numpy(), bins[parts[k]]))
    cmp[k]["p_equal_float32"] = bool(np.array_equal(sets[k].ev["p"].numpy(), p_ev[parts[k]].astype(np.float32)))
meta = data.load_meta()
cmp["test"]["hits_meta"] = meta["splits"]["test"]["hits"]
cmp["test"]["hits_equal_meta"] = cmp["test"]["hits_from_parquet"] == meta["splits"]["test"]["hits"]
cmp["test"]["bins_equal_meta"] = {int(b): int(n) for b, n in zip(*np.unique(bins[parts["test"]], return_counts=True))} == \
    {int(b): n for b, n in meta["splits"]["test"]["bins"].items()}
s = {k: set(saved[k].tolist()) for k in saved}
out["overlap"] = {"train&val": len(s["train"] & s["val"]), "train&test": len(s["train"] & s["test"]), "val&test": len(s["val"] & s["test"])}
out["union_equals_all_surviving_events"] = (s["train"] | s["val"] | s["test"]) == set(uniq.tolist())
out["counts"] = {k: cmp[k]["n_saved"] for k in cmp}
out["counts_expected"] = data.EXPECTED_COUNTS
out["hits_sum"] = {"train+val+test": sum(cmp[k]["hits_from_parquet"] for k in cmp), "rows_after_all_cuts": out["rows_after_all_cuts"],
                   "meta_hits_total": meta["hits_total"]}
out["compare"] = cmp

# ---- per-bin fractions
pb = {}
for b in np.unique(bins):
    n = {k: int((bins[parts[k]] == b).sum()) for k in parts}
    tot = sum(n.values())
    exp = {"train": round(0.70 * tot), "val": round(0.85 * tot) - round(0.70 * tot), "test": tot - round(0.85 * tot)}
    pb[int(b)] = {"n": tot, **n, "equals_rounding_rule": n == exp,
                  "frac": {k: n[k] / tot for k in n}}
out["per_bin"] = pb
out["per_bin_all_follow_rounding_rule"] = all(v["equals_rounding_rule"] for v in pb.values())
fr = {k: [v["frac"][k] for v in pb.values()] for k in parts}
out["per_bin_fraction_range"] = {k: [min(v), max(v)] for k, v in fr.items()}

# ---- low momentum, per split, after cuts (test from the Parquet-derived table = meta)
T = data.kinetic(p_ev)
out["low_momentum"] = {k: {"p_min": float(p_ev[parts[k]].min()), "T_min": float(T[parts[k]].min()),
                           "n_p_below_0.1": int((p_ev[parts[k]] < 0.1).sum()),
                           "events_below_0.1": [[int(uniq[i]), float(p_ev[i]), int(cnt[i])] for i in parts[k] if p_ev[i] < 0.1]}
                       for k in parts}
praw = cols["incoming_momentum"].astype(np.float64)
out["low_momentum"]["raw"] = {"events": int(len(praw)), "p_min": float(praw.min()), "n_below_0.1": int((praw < 0.1).sum()),
                              "n_below_0.1_removed_by_cuts": int(((praw < 0.1) & ~surv).sum()),
                              "p_of_removed_events_max": float(praw[~surv].max()) if (~surv).any() else None,
                              "n_removed_events": int((~surv).sum())}

split = np.full(len(uniq), -1, dtype=np.int8)
for i, k in enumerate(("train", "val", "test")):
    split[parts[k]] = i
np.savez(HERE / "events_table.npz", event_number=uniq, split=split, bin=bins, p=p_ev,
         theta=cols["incoming_theta"][surv], phi=cols["incoming_phi"][surv], n_hits=cnt, raw_rows=raw_cnt[surv])
out["raw_rows_per_event"] = {"max": int(raw_cnt.max()), "n_at_max": int((raw_cnt == raw_cnt.max()).sum()),
                             "top_values": {int(v): int(c) for v, c in zip(*np.unique(raw_cnt[raw_cnt >= raw_cnt.max() - 5], return_counts=True))},
                             "median": float(np.median(raw_cnt)), "n_ge_1000": int((raw_cnt >= 1000).sum())}
(HERE / "c05_split_integrity.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
show = {k: v for k, v in out.items() if k not in ("per_bin", "schema")}
print(json.dumps(show, indent=1, default=float))
