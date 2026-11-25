"""Lean pion data cache: the same events, cuts, ordering, dtypes and split as ``common.load_data``,
built column by column with pyarrow + numpy instead of one large pandas frame, plus per-event
``event_number``, ``theta`` and ``phi``.

Reference code reproduced here (shape-form-martin/common.py): ``detector_geometry``,
``Showers.from_dataframe`` and the split loop of ``load_data``. The builder takes a *column getter*
``get(name) -> np.ndarray`` (all rows, file order), so the same code runs on the full Parquet file and on
small in-memory subsets for the bit-exactness checks.

Files written to the cache directory (default ``shape-form-martin/cache/``, env ``SFM_CACHE_DIR``):
    pion_s1_trainval.pt   train and val sets
    pion_s1_test.pt       test set (separate file; only opened through ``load_sets(..., allow_test=True, reason=...)``)
    pion_s1_meta.json     geometry, counts, ranges

Every test-cache access appends one line (UTC time, script, reason, host id, GPU, launch grid, machine id) to
``results/test_access.log``; it is allowed only on the machine named in ``results/final/LOCK.json`` and never with a
path override (``check_test_access_allowed``).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common  # noqa: E402  (shape-form-martin/common.py, the metric of record)

CACHE_STEM = "pion_s1"
FRACS = (0.70, 0.15, 0.15)
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 69783, "val": 14950, "test": 14956}
EXPECTED_GEOMETRY = {"layer": 20.5, "cell": 48.6, "z_back": 4882.15}
M_PION = 0.139570  # GeV

# Columns read by the builder (same as Showers.from_dataframe uses) and by the geometry.
HIT_COLUMNS = ("event_number", "earliest_contribution_time", "x_local", "y_local", "z_local", "energy")
EVENT_COLUMNS = ("incoming_theta", "incoming_phi", "z", "z_trans", "incoming_momentum", "incoming_momentum_bin")
GEOMETRY_COLUMNS = ("x", "y", "z")
ALL_COLUMNS = tuple(dict.fromkeys(HIT_COLUMNS + EVENT_COLUMNS + GEOMETRY_COLUMNS))


# ----------------------------------------------------------------------------------------------------------------------
# Column access
# ----------------------------------------------------------------------------------------------------------------------
def parquet_getter(path: Path | str | None = None):
    """get(name) -> the full column as a numpy array (file order), read with pyarrow one column at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path or paths.parquet_path()))

    def get(name: str) -> np.ndarray:
        col = pf.read(columns=[name], use_threads=True).column(0)
        if col.null_count:
            raise ValueError(f"column {name} has {col.null_count} nulls; the builder assumes none")
        return col.to_numpy()

    get.parquet_file = pf
    return get


def table_getter(table):
    """get(name) for an in-memory pyarrow Table (used by the exactness checks)."""
    def get(name: str) -> np.ndarray:
        col = table.column(name)
        if col.null_count:
            raise ValueError(f"column {name} has nulls")
        return col.to_numpy()
    return get


# ----------------------------------------------------------------------------------------------------------------------
# Geometry and event building (exact re-implementations of common.detector_geometry / Showers.from_dataframe)
# ----------------------------------------------------------------------------------------------------------------------
def detector_geometry(get) -> dict:
    """common.detector_geometry on columns (pandas Series.round(2) == numpy round(…, 2))."""
    z = np.unique(np.round(get("z"), 2))
    dz = np.diff(z)
    d = np.concatenate([np.diff(np.unique(np.round(get(c), 2))) for c in ("x", "y")])
    return {"layer": float(np.median(dz[dz > 1])), "cell": float(np.median(d[(d > 20) & (d < 80)])),
            "z_back": float(z.max() + np.median(dz[dz > 1]) / 2)}


def _first_per_event(col_k: np.ndarray, codes: np.ndarray, n_ev: int, starts: np.ndarray) -> np.ndarray:
    """groupby(...).first(): first non-null (non-NaN) value per event, rows already sorted by event.
    `starts` = first row of every event (all rows of an event are contiguous)."""
    if not np.issubdtype(col_k.dtype, np.floating):
        return col_k[starts]
    nan = np.isnan(col_k)
    if not nan.any():
        return col_k[starts]
    pos = np.flatnonzero(~nan)                                        # rare path: skip NaNs like pandas does
    u, fi = np.unique(codes[pos], return_index=True)
    if len(u) == n_ev:
        return col_k[pos[fi]]
    out = np.full(n_ev, np.nan, dtype=np.float64 if not np.issubdtype(col_k.dtype, np.floating) else col_k.dtype)
    out[u] = col_k[pos[fi]]
    return out


def build_showers(get, geo: dict):
    """Lean ``common.Showers.from_dataframe``. Returns (Showers on CPU, extras) with
    extras = {"event_number": int64, "theta": float64, "phi": float64} per event (ascending event_number)."""
    evn = get("event_number")
    t = get("earliest_contribution_time")
    idx = np.flatnonzero(t < common.TIME_CUT_NS)                      # df[df.earliest_contribution_time < TIME_CUT_NS]
    del t
    idx = idx[np.argsort(evn[idx], kind="stable")]                    # .sort_values("event_number", kind="stable")
    xl, yl, zl = get("x_local")[idx], get("y_local")[idx], get("z_local")[idx]
    r = np.hypot(xl, yl)
    keep = (zl >= common.Z_MIN) & (zl <= common.Z_MAX) & (r < common.R_MAX)
    del r
    idx, xl, yl, zl = idx[keep], xl[keep], yl[keep], zl[keep]
    del keep
    uniq, codes = np.unique(evn[idx], return_inverse=True)            # pd.factorize(df.event_number, sort=True)
    del evn
    codes = codes.astype(np.intp, copy=False)
    n_ev = len(uniq)
    cnt = np.bincount(codes)
    energy = get("energy")[idx]
    esum = np.bincount(codes, weights=energy)
    w = energy / esum[codes]
    del energy
    starts = np.concatenate([[0], np.flatnonzero(np.diff(codes)) + 1])
    first = {c: _first_per_event(get(c)[idx], codes, n_ev, starts) for c in EVENT_COLUMNS}
    theta, phi = first["incoming_theta"], first["incoming_phi"]
    z0 = first["z"] - first["z_trans"]                                # global z of the shower origin
    ev = {"p": torch.tensor(first["incoming_momentum"], dtype=torch.float32),
          "bin": torch.tensor(first["incoming_momentum_bin"].astype(np.int64)),
          "zb": torch.tensor((geo["z_back"] - z0) / np.cos(theta), dtype=torch.float32),
          "R": torch.tensor(common.rotation(theta, phi)),
          "evis": torch.tensor(esum, dtype=torch.float32)}
    ev["lnp"] = torch.log(ev["p"])
    xyz = np.empty((len(idx), 3), dtype=np.float32)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = xl, yl, zl                      # df[[...]].to_numpy(np.float32)
    del xl, yl, zl
    sh = common.Showers(ev, torch.tensor(np.concatenate([[0], np.cumsum(cnt)])), torch.from_numpy(xyz),
                        torch.tensor(w, dtype=torch.float32),
                        torch.tensor([geo["cell"], geo["cell"], geo["layer"]], dtype=torch.float32))
    extras = {"event_number": uniq.astype(np.int64), "theta": np.asarray(theta, dtype=np.float64),
              "phi": np.asarray(phi, dtype=np.float64)}
    return sh, extras


def split_indices(bins: np.ndarray, fracs=FRACS) -> dict:
    """The split loop of common.load_data, verbatim (rng = default_rng(common.SEED), per momentum bin)."""
    rng = np.random.default_rng(common.SEED)
    parts = {"train": [], "val": [], "test": []}
    for b in np.unique(bins):
        idx = rng.permutation(np.where(bins == b)[0])
        n1, n2 = round(fracs[0] * len(idx)), round((fracs[0] + fracs[1]) * len(idx))
        parts["train"] += idx[:n1].tolist()
        parts["val"] += idx[n1:n2].tolist()
        parts["test"] += idx[n2:].tolist()
    return {k: np.sort(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Cache files
# ----------------------------------------------------------------------------------------------------------------------
def cache_files(cache_dir: Path | None = None) -> dict:
    d = Path(cache_dir or paths.CACHE_DIR)
    return {"trainval": d / f"{CACHE_STEM}_trainval.pt", "test": d / f"{CACHE_STEM}_test.pt",
            "meta": d / f"{CACHE_STEM}_meta.json"}


def _pack(sh, extras, idx) -> dict:
    s = sh.subset(idx)  # same gather as load_data (subset on CPU; .to(DEVICE) happens at load time)
    return {"ev": s.ev, "off": s.off, "xyz": s.xyz, "w": s.w, "box": s.box,
            "event_number": torch.from_numpy(extras["event_number"][idx]),
            "theta": torch.from_numpy(extras["theta"][idx]), "phi": torch.from_numpy(extras["phi"][idx]),
            "index": torch.from_numpy(np.asarray(idx, dtype=np.int64))}


def kinetic(p):
    """T = p^2 / (sqrt(p^2 + m^2) + m) (GeV), the numerically stable kinetic energy of a pion."""
    p = np.asarray(p, dtype=np.float64)
    return p * p / (np.sqrt(p * p + M_PION ** 2) + M_PION)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def split_summary(pack: dict) -> dict:
    p = pack["ev"]["p"].numpy().astype(np.float64)
    th = pack["theta"].numpy()
    zb = pack["ev"]["zb"].numpy().astype(np.float64)
    cnt = np.diff(pack["off"].numpy())
    xyz = pack["xyz"]
    return {"events": int(len(p)), "hits": int(pack["off"][-1]),
            "p_min_GeV": float(p.min()), "p_max_GeV": float(p.max()),
            "n_p_below_0.1GeV": int((p < 0.1).sum()),
            "theta_min_rad": float(th.min()), "theta_max_rad": float(th.max()),
            "zb_min_mm": float(zb.min()), "zb_max_mm": float(zb.max()),
            "n_zb_beyond_grid": int((zb > common.Z_MAX).sum()),
            "hits_per_event_min": int(cnt.min()), "hits_per_event_median": float(np.median(cnt)),
            "hits_per_event_max": int(cnt.max()),
            "z_local_min_mm": float(xyz[:, 2].min()), "z_local_max_mm": float(xyz[:, 2].max()),
            "r_max_mm": float(torch.sqrt(xyz[:, 0].double() ** 2 + xyz[:, 1].double() ** 2).max()),
            "evis_min": float(pack["ev"]["evis"].min()), "evis_max": float(pack["ev"]["evis"].max()),
            "bins": {int(b): int(n) for b, n in zip(*np.unique(pack["ev"]["bin"].numpy(), return_counts=True))}}


def build_cache(parquet: Path | str | None = None, cache_dir: Path | None = None, verbose=True):
    """Build the lean cache from the Parquet file. Returns (full Showers on CPU, extras, split idx, meta)."""
    src = Path(parquet or paths.parquet_path())
    get = parquet_getter(src)
    geo = detector_geometry(get)                                       # all rows, before any cut
    if verbose:
        print("geometry", geo)
    sh, extras = build_showers(get, geo)
    parts = split_indices(sh.ev["bin"].numpy())
    files = cache_files(cache_dir)
    files["trainval"].parent.mkdir(parents=True, exist_ok=True)
    packs = {k: _pack(sh, extras, v) for k, v in parts.items()}
    torch.save({"train": packs["train"], "val": packs["val"], "geometry": geo}, files["trainval"])
    torch.save({"test": packs["test"], "geometry": geo}, files["test"])
    st = src.stat()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"file": src.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                   "rows": int(get.parquet_file.metadata.num_rows)},
        "geometry": geo, "geometry_expected": EXPECTED_GEOMETRY,
        "cuts": {"time_ns_lt": common.TIME_CUT_NS, "z_local_mm": [common.Z_MIN, common.Z_MAX],
                 "r_mm_lt": common.R_MAX},
        "events_total": len(sh), "hits_total": int(sh.off[-1]),
        "counts": {k: int(len(v)) for k, v in parts.items()}, "counts_expected": EXPECTED_COUNTS,
        "fracs": list(FRACS), "seed": common.SEED,
        "splits": {k: split_summary(packs[k]) for k in SPLITS},
        "units": {"p": "GeV", "theta/phi": "rad", "lengths (xyz, zb, box)": "mm", "time cut": "ns",
                  "m_pion_GeV": M_PION},
        "files": {k: {"name": f.name, "sha256": _sha256(f), "bytes": f.stat().st_size}
                  for k, f in files.items() if k != "meta"},
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    files["meta"].write_text(json.dumps(meta, indent=1) + "\n")
    if verbose:
        print(f"{len(sh)} events, {int(sh.off[-1])} hits, split " +
              ", ".join(f"{k} {len(v)}" for k, v in parts.items()))
    return sh, extras, parts, meta


OVERRIDE_ENV = ("SFM_RESULTS_DIR", "SFM_CACHE_DIR")


def lock_file() -> Path:
    """``results/final/LOCK.json`` of the repository (never relocated: overrides are refused for test access)."""
    return paths.MARTIN_DIR / "results" / "final" / "LOCK.json"


def check_test_access_allowed(lock_path: Path | None = None) -> dict:
    """Raise ``PermissionError`` unless a test access is allowed on this machine (spec points 6 and 14 of
    ``transfer/REVIEW_STATUS.md`` section 4): no ``SFM_RESULTS_DIR`` / ``SFM_CACHE_DIR`` override (the log would be
    written outside the tracked file), ``results/final/LOCK.json`` exists and names this machine as the machine of
    record (key ``machine_of_record.machine_id`` = ``sfm.machine.machine_tag().id``; if it also holds ``host_id``,
    that must equal ``sfm.machine.host_id()``). CUDA is required (E10). Returns the LOCK content. Opens nothing else."""
    bad = [k for k in OVERRIDE_ENV if os.environ.get(k)]
    if bad:
        raise PermissionError(f"test access refused: path override(s) set: {bad}; the test-access log and the "
                              "caches must be the repository's own")
    lock_path = Path(lock_path or lock_file())
    if not lock_path.exists():
        raise PermissionError(f"test access refused: {lock_path} does not exist (the test split is opened only by the "
                              "final run on the machine of record, after Gate 2c)")
    from . import machine
    machine.startup("test access")
    lock = json.loads(lock_path.read_text())
    mor = lock.get("machine_of_record") or {}
    live = machine.machine_tag()
    if mor.get("machine_id") != live.id:
        raise PermissionError(f"test access refused: LOCK.json names machine {mor.get('machine_id')!r}, this machine "
                              f"is {live.id!r}")
    if mor.get("host_id") and mor["host_id"] != machine.host_id():
        raise PermissionError(f"test access refused: LOCK.json names host_id {mor['host_id']!r}, this is {machine.host_id()!r}")
    return lock


def test_access_line(reason: str, script: str | None = None) -> str:
    """One line of results/test_access.log: UTC time, script, reason, host id (a hash; no host name), GPU, launch
    grid, machine id."""
    from . import machine
    script = script or (Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<interactive>")
    return (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{script}\t{reason}\t"
            f"{machine.short_label()}")


test_access_line.__test__ = False


def _log_test_access(reason: str):
    log = paths.TEST_ACCESS_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    line = test_access_line(reason)
    with open(log, "a") as f:
        f.write(line + "\n")


def _unpack(pack: dict, device) -> tuple:
    sh = common.Showers(pack["ev"], pack["off"], pack["xyz"], pack["w"], pack["box"]).to(device)
    extras = {"event_number": pack["event_number"].numpy(), "theta": pack["theta"].numpy(),
              "phi": pack["phi"].numpy(), "index": pack["index"].numpy()}
    return sh, extras


def load_sets(splits=("train", "val"), device=common.DEVICE, allow_test=False, reason=None, cache_dir=None):
    """Load cached splits as ``common.Showers`` on `device` plus per-event extras.

    Returns (sets, extras): sets[split] is a Showers object, extras[split] = {"event_number", "theta", "phi",
    "index"} (numpy, CPU; "index" is the event's position in the full ascending event list).
    Loading "test" requires allow_test=True, a non-empty `reason`, no SFM_RESULTS_DIR / SFM_CACHE_DIR override and
    ``results/final/LOCK.json`` naming this machine (``check_test_access_allowed``); each such call is logged with
    host id, GPU and launch grid before the file is opened.
    """
    splits = (splits,) if isinstance(splits, str) else tuple(splits)
    bad = set(splits) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown split(s) {bad}")
    files = cache_files(cache_dir)
    sets, extras = {}, {}
    if "test" in splits:
        if not allow_test:
            raise PermissionError("the test split is locked: pass allow_test=True and a reason")
        if not reason or not str(reason).strip():
            raise ValueError("loading the test split requires a non-empty reason")
        check_test_access_allowed()                    # overrides, LOCK.json naming this machine (before any write)
        _log_test_access(str(reason).strip())
        obj = torch.load(files["test"], map_location="cpu", weights_only=False)
        sets["test"], extras["test"] = _unpack(obj["test"], device)
        del obj
    tv = [s for s in splits if s != "test"]
    if tv:
        obj = torch.load(files["trainval"], map_location="cpu", weights_only=False)
        for s in tv:
            sets[s], extras[s] = _unpack(obj[s], device)
        del obj
    return {s: sets[s] for s in splits}, {s: extras[s] for s in splits}


def load_meta(cache_dir=None) -> dict:
    return json.loads(cache_files(cache_dir)["meta"].read_text())


# ----------------------------------------------------------------------------------------------------------------------
# Bit-exactness checks against the reference pandas code
# ----------------------------------------------------------------------------------------------------------------------
def compare_showers(ref, new) -> dict:
    """Bit-exact comparison (values and dtypes) of two Showers objects."""
    out = {}
    for name in ("off", "xyz", "w", "box"):
        a, b = getattr(ref, name), getattr(new, name)
        out[name] = bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    for k in ref.ev:
        a, b = ref.ev[k], new.ev.get(k)
        out[f"ev.{k}"] = bool(b is not None and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    out["ev_keys_equal"] = sorted(ref.ev) == sorted(new.ev)
    out["all_equal"] = all(out.values())
    return out


def exactness_check_rowgroups(geo: dict, n_row_groups=4, parquet=None) -> dict:
    """Reference ``common.Showers.from_dataframe`` vs ``build_showers`` on the rows of the first
    `n_row_groups` Parquet row groups (the event of the last row is dropped), same `geo`."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet or paths.parquet_path()))
    tbl = pf.read_row_groups(list(range(n_row_groups)), columns=list(ALL_COLUMNS))
    last_ev = tbl.column("event_number")[-1].as_py()
    tbl = tbl.filter(pc.not_equal(tbl.column("event_number"), pa.scalar(last_ev, tbl.schema.field("event_number").type)))
    df = tbl.to_pandas()
    ref = common.Showers.from_dataframe(df, geo)
    new, extras = build_showers(table_getter(tbl), geo)
    res = compare_showers(ref, new)
    rx = _reference_extras(df)
    for k in ("event_number", "theta", "phi"):
        res[f"extras.{k}"] = bool(rx[k].shape == extras[k].shape and np.array_equal(rx[k], extras[k]))
    g_ref, g_new = common.detector_geometry(df), detector_geometry(table_getter(tbl))
    res["geometry_on_subset_equal"] = g_ref == g_new
    res["all_equal"] = all(v for k, v in res.items() if k != "all_equal")
    res["geometry_on_subset"] = g_ref
    res.update({"rows": int(tbl.num_rows), "row_groups": n_row_groups, "dropped_event": int(last_ev),
                "events": len(ref), "hits_after_cuts": int(ref.off[-1]),
                "df_dtypes": {c: str(t) for c, t in df.dtypes.items()}})
    return res


def _reference_extras(df) -> dict:
    """Event numbers, theta, phi of the events that survive the cuts, computed with pandas as in
    from_dataframe (independent of build_showers)."""
    df = df[df.earliest_contribution_time < common.TIME_CUT_NS].sort_values("event_number", kind="stable")
    r = np.hypot(df.x_local, df.y_local)
    df = df[(df.z_local >= common.Z_MIN) & (df.z_local <= common.Z_MAX) & (r < common.R_MAX)]
    first = df.groupby("event_number").first()
    return {"event_number": first.index.to_numpy().astype(np.int64),
            "theta": first.incoming_theta.to_numpy(), "phi": first.incoming_phi.to_numpy()}


def exactness_check_full(sh, extras: dict, geo: dict, n_chunks=12, parquet=None, verbose=True) -> dict:
    """Reference ``from_dataframe`` on complete events, chunk by chunk (event-number ranges), against the
    corresponding events of the full lean build `sh` (and its extras). The reference operations act row-wise
    or per event, and a chunk keeps every row of its events in file order, so the concatenated chunk results
    equal a single full-frame run of ``from_dataframe`` with the same `geo`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet or paths.parquet_path()))
    cols = list(dict.fromkeys(HIT_COLUMNS + EVENT_COLUMNS))
    evn_all = pf.read(columns=["event_number"]).column(0).to_numpy()
    edges = np.unique(np.quantile(np.unique(evn_all), np.linspace(0, 1, n_chunks + 1)).astype(np.int64))
    edges[-1] = evn_all.max() + 1
    del evn_all
    rg_rows = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    results, ev_pos, all_ok = [], 0, True
    for lo, hi in zip(edges[:-1], edges[1:]):
        parts = []
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=cols)
            e = t.column("event_number").to_numpy()
            parts.append(t.filter(pa.array((e >= lo) & (e < hi))))
        tbl = pa.concat_tables(parts)
        del parts
        df = tbl.to_pandas()
        del tbl
        ref = common.Showers.from_dataframe(df, geo)
        rx = _reference_extras(df)
        del df
        n = len(ref)
        piece = sh.subset(np.arange(ev_pos, ev_pos + n))
        res = compare_showers(ref, piece)
        for k in ("event_number", "theta", "phi"):
            a, b = rx[k], extras[k][ev_pos:ev_pos + n]
            res[f"extras.{k}"] = bool(a.shape == b.shape and np.array_equal(a, b))
        res["all_equal"] = all(v for k, v in res.items() if k != "all_equal")
        res.update({"event_range": [int(lo), int(hi)], "events": n, "hits": int(ref.off[-1])})
        all_ok &= res["all_equal"]
        results.append(res)
        if verbose:
            print(f"  chunk [{lo},{hi}): {n} events, {int(ref.off[-1])} hits, equal={res['all_equal']}")
        ev_pos += n
        del ref, piece
    return {"chunks": results, "events_compared": ev_pos, "events_in_cache": len(sh),
            "all_equal": bool(all_ok and ev_pos == len(sh)), "row_groups": int(len(rg_rows) - 1)}
