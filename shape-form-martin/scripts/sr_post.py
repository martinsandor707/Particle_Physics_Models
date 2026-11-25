"""Post-processing of a chunked PySR search (PLAN.md §7, Gate 2-pilot checklist), run in its OWN process.

    uv run python -B shape-form-martin/scripts/sr_post.py SPEC [--seeds 0 1] [--float32]
    uv run python -B shape-form-martin/scripts/sr_post.py --seed-dir results/m3_sr/<step>/<target>/seed<k>

This process may import torch (the translator's round trip needs it); the search process never does.  It reads only
files the runner wrote atomically (hall_of_fame.json, predictions/*.npz, hof_snapshots/*.csv, spec_snapshot.json) and
the target .npz, so it can run while the search is still going.  Per seed it does, on the latest chunk:

  1. translate every hall-of-fame member (free form: translate.parse; template: translate.parse_template with the
     exported sub-expressions);
  2. scan it: operators.violations (vocabulary, nesting, argument sizes, "exponent of ppow a constant"); for
     templates each sub-expression separately (PySR applies the constraints per sub-expression).  Violating members
     are DROPPED from the front and counted;
  3. check the complexity: operators.tree_complexity with the search's own weights (templates: sum over the
     sub-expressions) = PySR's column;
  4. recompute the loss in NumPy from the parsed string (gauge: operators.gauge_loss_numpy, weighted L2:
     operators.wl2_loss_numpy, PySR default: weighted mean squared error) and compare with the CSV loss;
  5. round trip: translate.round_trip(expr, X, names, reference=<the search's own prediction>) on the exported rows,
     rule max |torch - Julia| <= 1e-5 (float64; --float32 adds a float32 torch pass, for information);
  6. gauge de-duplication (gauge loss only): audit.drop_gauge_terms removes top-level terms in the slice variable
     alone; members whose remainders coincide are one member; the effective complexity is reduced by the dropped
     terms' size (free form);
  7. the front after 2 and 6 (dominating members only), its best loss per tier, and the loss part of the plateau rule
     recomputed over all chunk snapshots with that front -> post/plateau_gauge.json (search.plateau_status prefers it
     while it is current).

Output: <seed_dir>/post/post_chunkNNN.json (per member and summary), post/plateau_gauge.json.  Exit status 1 if a
round trip or a loss recomputation fails, or a complexity disagrees; scan violations are reported, not failures.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sfm.sr import audit as A  # noqa: E402
from sfm.sr import operators as O  # noqa: E402
from sfm.sr import search as S  # noqa: E402
from sfm.sr import translate as T  # noqa: E402

LOSS_RTOL = 1e-6          # recomputed loss vs the CSV loss (relative; the CSV prints the loss with ~17 digits)


# --------------------------------------------------------------------------------------------------------------------
# what was searched (from the seed's spec snapshot)
# --------------------------------------------------------------------------------------------------------------------
def loss_kind(loss: dict) -> tuple[str, int | None]:
    """('gauge', x_index) | ('wl2', None) | ('l2', None) | ('unknown', None) from the spec's loss strings."""
    if not loss:
        return "l2", None
    src = " ".join(str(v) for v in loss.values())
    if "sfm_gauge_loss" in src:
        m = re.search(r"X\[(\d+),\s*i\]", src)
        return "gauge", int(m.group(1)) if m else 3
    if "sfm_wl2" in src:
        return "wl2", None
    return "unknown", None


def vocabulary_of(spec_raw: dict, ops: dict) -> O.Vocabulary:
    sc = spec_raw.get("search_config_resolved") or {}
    stage = (sc.get("kwargs") or {}).get("stage")
    if stage in O.VOCABULARIES:
        return O.VOCABULARIES[stage]
    names_b = [S.operator_name(s) for s in ops.get("binary_operators", [])]
    names_u = [S.operator_name(s) for s in ops.get("unary_operators", [])]
    cons = {k: (tuple(v) if isinstance(v, (list, tuple)) else int(v)) for k, v in (ops.get("constraints") or {}).items()}
    return O.Vocabulary(name="spec", label="from the spec's [operators]", binary=tuple(names_b), unary=tuple(names_u),
                        constraints=cons, nested_constraints=dict(ops.get("nested_constraints") or {}),
                        pole_free=False)


def template_of(tmpl: dict | None) -> O.TemplateSpec | None:
    if tmpl is None:
        return None
    if tmpl.get("name") in O.TEMPLATES:
        return O.TEMPLATES[tmpl["name"]]
    exprs = tuple(tmpl["expressions"])
    args = {}
    for name, inner in re.findall(r"\b([A-Za-z_]\w*)\(([^()]*)\)", tmpl["combine"]):
        if name in exprs:
            args[name] = tuple(a.strip() for a in inner.split(","))
    return O.TemplateSpec(name="spec", label="from the spec", expressions=exprs, arguments=args,
                          combine64=tmpl["combine"], variable_names=tuple(tmpl.get("variable_names") or ()))


# --------------------------------------------------------------------------------------------------------------------
# one member
# --------------------------------------------------------------------------------------------------------------------
def translate_member(eq: str, names, tspec, subexpressions: dict | None):
    """(tree, {sub: tree} or None)."""
    if tspec is None:
        return T.parse(eq, names), None
    parts = subexpressions or T.split_template_string(eq)
    tree, sub = T.parse_template(tspec, {n: parts[n] for n in tspec.expressions if n in parts}, keep_parts=True)
    return tree, sub


def scan_member(tree, sub, vocab) -> list[str]:
    if sub is None:
        return O.violations(tree, vocab)
    out = []
    for n, e in sub.items():
        out += [f"{n}: {m}" for m in O.violations(e, vocab)]
    return out


def complexity_weights(ops: dict, pysr: dict) -> dict:
    """The search's own complexity weights: the spec's complexity_of_operators (PySR's default 1 for every other
    operator) and complexity_of_constants / complexity_of_variables."""
    return {"complexity_of_operators": dict(ops.get("complexity_of_operators") or {}),
            "complexity_of_constants": pysr.get("complexity_of_constants", 1),
            "complexity_of_variables": pysr.get("complexity_of_variables", 1)}


def complexity_of(tree, sub, cw: dict) -> int:
    if sub is None:
        return O.tree_complexity(tree, **cw)
    return sum(O.tree_complexity(e, **cw) for e in sub.values())


def recompute_loss(kind: str, x_index, pred, y, w, X) -> float | None:
    if w is None:
        w = np.ones_like(y)
    if not np.all(np.isfinite(pred)):
        return math.inf
    if kind == "gauge":
        return O.gauge_loss_numpy(pred, y, w, X[:, x_index - 1])
    if kind in ("wl2", "l2"):
        return O.wl2_loss_numpy(pred, y, w)
    return None


def _rel(a, b) -> float:
    if a is None or b is None:
        return math.nan
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


# --------------------------------------------------------------------------------------------------------------------
# fronts
# --------------------------------------------------------------------------------------------------------------------
def reduced_front(members: list[dict]) -> list[dict]:
    """Compliant members, de-duplicated by their gauge-reduced form (lowest effective complexity, then loss), then
    only the dominating ones (strictly better than every simpler member), as PySR lists its front."""
    best: dict[str, dict] = {}
    for m in members:
        if m["violations"] or m["loss"] is None or not math.isfinite(m["loss"]):
            continue
        k = m["reduced_key"]
        cur = best.get(k)
        if cur is None or (m["effective_complexity"], m["loss"]) < (cur["effective_complexity"], cur["loss"]):
            best[k] = m
    front, best_loss = [], math.inf
    for m in sorted(best.values(), key=lambda d: (d["effective_complexity"], d["loss"])):
        if m["loss"] < best_loss:
            front.append(m)
            best_loss = m["loss"]
    return front


def analyse_rows(rows: list[dict], names, tspec, vocab, kind, gauge_var, cw: dict) -> list[dict]:
    """Translate, scan and gauge-reduce CSV-like rows {complexity, loss, equation[, subexpressions]}."""
    out = []
    for r in rows:
        m = {"complexity": int(r["complexity"]), "loss": r["loss"], "equation": r["equation"]}
        try:
            tree, sub = translate_member(r["equation"], names, tspec, r.get("subexpressions"))
        except Exception as e:  # noqa: BLE001
            m.update(parse_error=f"{type(e).__name__}: {e}", violations=["unparsed"], reduced_key=r["equation"],
                     effective_complexity=m["complexity"])
            out.append(m)
            continue
        m["tree"], m["sub"] = tree, sub
        m["violations"] = scan_member(tree, sub, vocab)
        m["complexity_translated"] = complexity_of(tree, sub, cw)
        m["dropped_gauge_terms"] = []
        eff = m["complexity"]
        key = T.expr_hash(tree)
        if kind == "gauge":
            red, dropped = A.drop_gauge_terms(tree, (gauge_var,))
            m["dropped_gauge_terms"] = dropped
            if dropped:
                if sub is None:
                    eff = m["complexity"] - (O.tree_complexity(tree, **cw)
                                             - (O.tree_complexity(red, **cw) if red is not None else 0))
                key = T.expr_hash(red) if red is not None else "<pure gauge>"
        m["effective_complexity"] = int(eff)
        m["reduced_key"] = key
        out.append(m)
    return out


# --------------------------------------------------------------------------------------------------------------------
# a seed directory
# --------------------------------------------------------------------------------------------------------------------
def post_seed(seed_dir: Path, float32: bool = False, log=print) -> dict:
    seed_dir = Path(seed_dir)
    snap = S.read_json(seed_dir / S.SPEC_SNAPSHOT)
    hof = S.read_json(seed_dir / S.HOF_JSON)
    prog = S.read_json(seed_dir / S.PROGRESS_FILE) or {}
    if snap is None or hof is None:
        raise FileNotFoundError(f"{seed_dir}: spec_snapshot.json or hall_of_fame.json missing (no completed chunk)")
    spec_raw = snap["spec"]
    ident = snap["identity"]
    ops, loss, tmpl = ident["operators"], ident["loss"], ident["template"]
    data_path = Path(spec_raw["data"])
    if not data_path.is_absolute():
        data_path = (Path(snap["spec_path"]).parent / data_path).resolve()
    tgt = S.load_target(data_path)
    if tgt.sha256 != ident["data_sha256"]:
        raise RuntimeError(f"{data_path} changed since the search started (sha256 differs)")
    names = list(tgt.variable_names)
    tspec = template_of(tmpl)
    vocab = vocabulary_of(spec_raw, ops)
    kind, x_index = loss_kind(loss)
    gauge_var = names[x_index - 1] if kind == "gauge" else None
    tiers = list(prog.get("tiers") or spec_raw.get("tiers") or S.DEFAULT_TIERS)
    cw = complexity_weights(ops, ident.get("pysr") or {})

    preds = None
    if hof.get("predictions"):
        with np.load(seed_dir / hof["predictions"], allow_pickle=False) as z:
            preds = {k: z[k] for k in z.files}
    rows_idx = preds["rows"] if preds is not None else None
    torch_dtypes = [None]
    if float32:
        import torch
        torch_dtypes.append(torch.float32)

    out_doc = {"seed_dir": str(seed_dir), "chunks_completed": hof["chunks_completed"], "loss_kind": kind,
               "gauge_variable": gauge_var, "vocabulary": vocab.name, "template": tspec.name if tspec else None,
               "predictions": hof.get("predictions"), "round_trip_tol": T.ROUND_TRIP_TOL, "outputs": []}
    failures = []
    for o, outp in enumerate(hof["outputs"]):
        y = tgt.y if tgt.nout == 1 else tgt.y[:, o]
        w = None if tgt.weights is None else (tgt.weights if tgt.nout == 1 else tgt.weights[:, o])
        members = analyse_rows(outp["equations"], names, tspec, vocab, kind, gauge_var, cw)
        if preds is not None:
            eq_exported = [str(s) for s in preds[f"equation_o{o}"]]
            if eq_exported != [m["equation"] for m in members]:
                raise RuntimeError(f"output {o}: predictions file and hall_of_fame.json list different members")
        env_all = {n: tgt.X[:, j] for j, n in enumerate(names)}
        recs = []
        for i, m in enumerate(members):
            rec = {k: m.get(k) for k in ("complexity", "loss", "equation", "violations", "complexity_translated",
                                         "dropped_gauge_terms", "effective_complexity", "parse_error")}
            if "tree" in m:
                tree = m["tree"]
                rec["k_constants"] = T.n_constants(tree)
                rec["complexity_ok"] = (m["complexity_translated"] == m["complexity"])
                if not rec["complexity_ok"]:
                    failures.append(f"output {o} C={m['complexity']}: translated complexity "
                                    f"{m['complexity_translated']}")
                with np.errstate(all="ignore"):
                    p_all = np.asarray(T.evaluate(tree, env_all), dtype=np.float64) + np.zeros(len(tgt.X))
                lr = recompute_loss(kind, x_index, p_all, y, w, tgt.X)
                rec["loss_recomputed"] = lr
                rec["loss_rel_diff"] = _rel(lr, m["loss"])
                if lr is not None and not (rec["loss_rel_diff"] <= LOSS_RTOL):
                    failures.append(f"output {o} C={m['complexity']}: loss {lr!r} vs CSV {m['loss']!r}")
                if preds is not None:
                    ref = preds[f"pred_o{o}"][i]
                    rec["julia_eval_ok"] = bool(preds[f"ok_o{o}"][i])
                    rec["loss_state"] = float(preds[f"loss_state_o{o}"][i])
                    for dt in torch_dtypes:
                        with np.errstate(all="ignore"):
                            rt = T.round_trip(tree, tgt.X[rows_idx], names, reference=ref, dtype=dt)
                        key = "round_trip" if dt is None else "round_trip_float32"
                        rec[key] = rt
                        if dt is None and not rt["passed"]:
                            failures.append(f"output {o} C={m['complexity']}: round trip {rt}")
                if tspec is not None and m.get("sub"):
                    rec["subexpressions"] = {n: T.to_string(e) for n, e in m["sub"].items()}
            recs.append(rec)
        front = reduced_front(members)
        n_dup = sum(1 for m in members if not m["violations"]) - len({m["reduced_key"] for m in members
                                                                       if not m["violations"]})
        rt_diffs = [r["round_trip"]["reference_max_diff"] for r in recs if r.get("round_trip")]
        summ = {"n_members": len(members), "n_violating": sum(1 for m in members if m["violations"]),
                "violating": [{"complexity": m["complexity"], "equation": m["equation"], "violations": m["violations"]}
                              for m in members if m["violations"]],
                "n_gauge_duplicates": n_dup,
                "members_with_gauge_terms": sum(1 for m in members if m.get("dropped_gauge_terms")),
                "n_front_after_scan_and_dedup": len(front),
                "front_after_scan_and_dedup": [{"effective_complexity": m["effective_complexity"],
                                                "complexity": m["complexity"], "loss": m["loss"],
                                                "equation": m["equation"]} for m in front],
                "tier_best_raw": S.tier_bests(outp["equations"], tiers),
                "tier_best_after_scan_and_dedup": S.tier_bests(
                    [{"complexity": m["effective_complexity"], "loss": m["loss"]} for m in front], tiers),
                "round_trip_max_diff": max(rt_diffs) if rt_diffs else None,
                "round_trip_all_passed": (all(r["round_trip"]["passed"] for r in recs if r.get("round_trip"))
                                          if rt_diffs else None),
                "loss_max_rel_diff": max((r["loss_rel_diff"] for r in recs if r.get("loss_rel_diff") is not None
                                          and not math.isnan(r["loss_rel_diff"])), default=None),
                "complexity_all_ok": all(r.get("complexity_ok", False) for r in recs)}
        if float32 and rt_diffs:
            summ["round_trip_float32_max_diff"] = max(r["round_trip_float32"]["reference_max_diff"] for r in recs
                                                      if r.get("round_trip_float32"))
        out_doc["outputs"].append({"output": o, "summary": summ, "members": recs})

    # plateau over the chunk snapshots, with the front after the scan and the gauge de-duplication
    history = []
    for idx in range(int(prog.get("chunks_completed", 0))):
        per_out = []
        for o, name in enumerate(S.hof_csv_names(tgt.nout)):
            cands = sorted((seed_dir / S.SNAPSHOT_DIR).glob(f"chunk{idx:03d}_*_{name}"))
            if not cands:
                per_out = None
                break
            rows = S.read_hof_csv(cands[-1])
            mem = analyse_rows(rows, names, tspec, vocab, kind, gauge_var, cw)
            fr = reduced_front(mem)
            per_out.append(S.tier_bests([{"complexity": m["effective_complexity"], "loss": m["loss"]} for m in fr],
                                        tiers))
        if per_out is not None:
            history.append({"index": idx, "tier_best": per_out})
    plateau_cfg = spec_raw.get("plateau") or {}
    pl = S.plateau_check(history, tiers, int(plateau_cfg.get("window", 3)), float(plateau_cfg.get("rel_tol", 0.01)))
    plateau_doc = {"at": S.utc_now(), "chunks_completed": prog.get("chunks_completed"), "plateau": pl,
                   "history": history, "note": "loss part of the plateau rule on the front after the scan "
                                               "(violating members dropped) and the gauge de-duplication"}
    out_doc["plateau"] = pl
    out_doc["failures"] = failures
    out_doc["passed"] = not failures
    post_dir = seed_dir / S.POST_DIR
    post_dir.mkdir(parents=True, exist_ok=True)
    S.atomic_write_json(post_dir / f"post_chunk{int(hof['chunks_completed']) - 1:03d}.json", out_doc)
    S.atomic_write_json(post_dir / S.POST_PLATEAU_FILE, plateau_doc)
    for oo in out_doc["outputs"]:
        s = oo["summary"]
        log(f"[sr_post] {seed_dir.name} output {oo['output']}: {s['n_members']} members, {s['n_violating']} "
            f"violating (dropped), {s['n_gauge_duplicates']} gauge duplicates, front after scan+dedup "
            f"{s['n_front_after_scan_and_dedup']}; round trip max |torch - Julia| = {s['round_trip_max_diff']} "
            f"(all <= {T.ROUND_TRIP_TOL}: {s['round_trip_all_passed']}); loss max rel diff {s['loss_max_rel_diff']}; "
            f"complexity ok {s['complexity_all_ok']}")
    log(f"[sr_post] plateau (after scan + gauge dedup): {pl.get('plateau')} ({pl.get('reason', '')})")
    log(f"[sr_post] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures[:5])}")
    return out_doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?", help="run spec (.toml or .json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--seed-dir", type=Path, action="append", default=None, help="a seed directory (repeatable)")
    ap.add_argument("--float32", action="store_true", help="also run the torch side in float32 (information)")
    a = ap.parse_args(argv)
    dirs = list(a.seed_dir or [])
    if a.spec:
        raw = S.apply_overrides(S.load_spec_file(a.spec), seeds=a.seeds)
        spec = S.parse_spec(S.resolve_search_config(raw))
        dirs += [spec.seed_dir(s) for s in spec.seeds]
    if not dirs:
        ap.error("give a spec or --seed-dir")
    ok = True
    for d in dirs:
        doc = post_seed(d, float32=a.float32)
        ok = ok and doc["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
