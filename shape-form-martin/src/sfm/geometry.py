"""sfm.geometry - absolute shower origins, the instrumented volume of the LFHCal and per-event 3-D acceptance maps.

Everything here is *parameter-free geometry*: nothing is fitted. It serves the zero-training geometry acceptance arm
(PLAN section 14) and its audit (``results/m2c_theta/geometry/GEOMETRY.md``). Numbers produced with it are never
mixed with theta-blind deltas: the inputs of a geometry-aware map are (p, theta, phi, entry position).

Coordinates
    global (X, Y, Z) [mm]   detector frame of the simulation (columns ``x, y, z`` of the hit table)
    shower (x', y', z')     ``x_local, y_local, z_local``; ``R = common.rotation(theta, phi)`` has the shower axes
                            as rows, so
                                global = origin + R^T . local                      (exact for every training hit)
    origin                  energy-weighted centroid of the lowest-z hit layer of the event, over ALL rows of the
                            event (also deposits later than 100 ns). It is NOT the point where the straight line
                            from the global origin along (theta, phi) meets that layer: the two differ by more
                            than 100 mm in 9 % of the training events.

Instrumented volume (from training-split hits only; GEOMETRY.md section 1)
    60 layers, centres 3662.4 + 20.5 k mm; front face 3652.15 mm, back face 4882.15 mm.
    Transverse outline: a stepped octagon, mirror-symmetric about Y = 0 and about X = +2.5 mm. Module rows are
    101 mm high (two cell rows each). The 11 innermost module rows per side are full width; from there every
    module row is one module (two cells, 100.944 mm) narrower on each side. A rectangular beam hole, off-centre
    (X from -406 to +209 mm, |Y| < 303 mm), is cut out of the three innermost module rows per side.
    ``Volume("stepped")`` is that envelope (module gaps are filled: a nearly uniform loss of about 8 % of the area, which cancels
    in a normalised map); ``Volume("octagon")`` a straight-edged octagon with a rectangular hole (9 numbers);
    ``Volume("box")`` the square |X|, |Y| <= 2624.4 mm of the Milestone 1 audit, without a hole.

Acceptance of an event
    A_e(z', r) = fraction of the ring (z', r, all azimuths) of the shower frame that lies inside the chosen faces,
    averaged over the grid cell: ``nsub`` sub-points in z' (midpoints), ``nsub`` equal-area sub-rings in r - the
    sub-points of ``shapelab.slab_accept`` - and ``nphi`` equidistant azimuth nodes.
        front   Z >= Z_origin - layer/2    (the plane half a layer in front of the first HIT layer, as
                                            ``event_slab``; no hit of the event can lie in front of it by the
                                            definition of the origin. ``front_plane="detector"`` uses 3652.15 mm.)
        side    (X, Y) inside the transverse outline and outside the beam hole
        back    Z <= 4882.15 mm            (tilted in the shower frame; the metric's ``zb`` is its depth on the axis)

Contents
    origins_from_parquet, save_origins, load_origins     absolute origin per event (train / val only)
    Volume, columns_inside                                the volume models and an independent point test
    acceptance, event_acceptance, axis_exit               per-event acceptance maps, exit depth of the shower axis
    mean_acceptance                                       training-mean template Abar of the ratio form
    apply_acceptance, score_variants                      log P + ln A [- ln Abar] (renormalised) and its evaluation
                                                          on the fold points of ``common.test_nll``
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import metric, paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402  (shape-form-martin/common.py; read only)

__all__ = ["LAYER", "N_LAYERS", "Z_LAYER0", "Z_FRONT", "Z_BACK", "HALF_CELL", "X_CENTRE", "MODULE_PITCH_Y", "X_LAST",
           "Y_MAX", "HOLE_LEFT", "HOLE_RIGHT", "HOLE_Y", "BOX_HALF", "GEOMETRY_STEM", "FACES", "VARIANTS",
           "origins_from_parquet", "save_origins", "load_origins", "straight_line_origin", "to_global",
           "Volume", "columns_inside", "ring_nodes", "acceptance", "event_acceptance", "mean_acceptance", "axis_exit",
           "apply_acceptance", "score_variants"]

# ----------------------------------------------------------------------------------------------------------------------
# Detector constants [mm], read off the training-split hits (GEOMETRY.md section 1; checked in tests/test_geometry.py
# against results/m2c_theta/geometry/explore/train_columns.npz)
# ----------------------------------------------------------------------------------------------------------------------
LAYER = 20.5
N_LAYERS = 60
Z_LAYER0 = 3662.4                                    # centre of the first layer
Z_FRONT = Z_LAYER0 - LAYER / 2                       # 3652.15
Z_BACK = Z_LAYER0 + (N_LAYERS - 0.5) * LAYER         # 4882.15 (= cache geometry "z_back")
HALF_CELL = 24.3                                     # half of the metric's fold box (48.6 mm)
X_CENTRE = 2.5                                       # the lattice is mirror-symmetric about X = +2.5 mm
MODULE_PITCH_Y = 101.0                               # module row a covers 101 a <= |Y| < 101 (a + 1)
# Outermost cell CENTRE (X > 0 side) of module row a = 0 ... 25; the other side is 2 * X_CENTRE - X_LAST[a].
X_LAST = (2599.461, 2601.035, 2599.461, 2601.035, 2599.461, 2601.035, 2599.461, 2601.035, 2599.461, 2601.035, 2599.461,
          2498.516, 2397.572, 2296.627, 2195.683, 2094.738, 1993.794, 1892.849, 1791.905, 1690.961, 1590.016, 1489.072,
          1388.127, 1287.183, 1186.238, 1085.294)
Y_MAX = 2599.8 + HALF_CELL                           # outer edge of the last cell row (2624.1)
# Beam hole: module rows a = 0, 1, 2 (|Y| < 303 mm). Cell centres next to the hole on the -X and +X side.
HOLE_LEFT = (-430.762, -429.465, -430.762)
HOLE_RIGHT = (233.873, 231.815, 233.873)
HOLE_Y = 3 * MODULE_PITCH_Y                          # 303 mm, midway between the cell rows at 276.8 and 329.2
BOX_HALF = 54 * 48.6                                 # 2624.4, the square of the Milestone 1 audit (c_side.py)

GEOMETRY_STEM = "pion_s1_geometry_trainval"
FACES = ("front", "side", "back")
VARIANTS = {"front": ("front",), "front_side": ("front", "side"), "all": ("front", "side", "back")}
PARQUET_COLUMNS = ("event_number", "x", "y", "z", "x_local", "y_local", "z_local", "incoming_theta", "incoming_phi")


# ----------------------------------------------------------------------------------------------------------------------
# Absolute origins
# ----------------------------------------------------------------------------------------------------------------------
def to_global(local, theta, phi):
    """R^T . local for NumPy arrays: ``local`` (..., 3) in the shower frame, angles broadcastable to ``local[..., 0]``.
    Rows of R = common.rotation: e1 = (ct cp, ct sp, -st), e2 = (-sp, cp, 0), e3 = (st cp, st sp, ct)."""
    xl, yl, zl = local[..., 0], local[..., 1], local[..., 2]
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    return np.stack([ct * cp * xl - sp * yl + st * cp * zl, ct * sp * xl + cp * yl + st * sp * zl,
                     -st * xl + ct * zl], -1)


def straight_line_origin(theta, phi, zb, z_back: float = Z_BACK):
    """The origin assumed by the Milestone 1 audit (``c_side.py``): the particle leaves the global origin along
    (theta, phi) and the frame origin sits on that line at the depth of the first hit layer,
    z0 = z_back - zb cos(theta). Returns (n, 3). Kept as the comparison object for question (a)."""
    theta, phi, zb = (np.asarray(v, dtype=np.float64) for v in (theta, phi, zb))
    z0 = z_back - zb * np.cos(theta)
    rho = z0 * np.tan(theta)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z0], 1)


def origins_from_parquet(event_numbers, path=None, *, tol: float = 1e-6, batch_rows: int = 1 << 20,
                         row_hook: Optional[Callable] = None, extra_columns: Sequence[str] = ()) -> dict:
    """Absolute shower origin of the requested events: ``origin = (x, y, z) - R^T (x_local, y_local, z_local)``,
    evaluated for every row of the event (no time or fiducial cut) and required to be the same for all of them.

    The file is read through a pushed-down filter on ``event_number`` (``pyarrow.dataset``), so rows of other events
    - in particular of the locked test events - are never returned to this process' NumPy arrays; every returned
    row is checked again to belong to ``event_numbers``. Call it with train and/or val event numbers only.

    Returns a dict of arrays in the order of ``event_numbers``:
        event_number (n,) int64, origin (n, 3) float64 [mm] (midpoint of the per-event min and max),
        spread (n, 3) float64 (max - min over the rows of the event), rows (n,) int64, theta, phi (n,) float64.
    Raises if an event has no row, if theta / phi vary inside an event, or if a spread exceeds ``tol`` mm.
    ``row_hook(index, cols)`` is called per record batch with the event index of every row (into
    ``event_numbers``) and the dict of column arrays - for audits that need the global hit coordinates;
    ``extra_columns`` are read in addition for the hook (e.g. "energy", "earliest_contribution_time").
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    evn = np.asarray(event_numbers, dtype=np.int64)
    if evn.ndim != 1 or len(evn) == 0:
        raise ValueError("event_numbers must be a non-empty 1-D array")
    order = np.argsort(evn, kind="stable")
    srt = evn[order]
    if (np.diff(srt) == 0).any():
        raise ValueError("event_numbers contains duplicates")
    n = len(srt)
    dataset = ds.dataset(str(path or paths.parquet_path()), format="parquet")
    etype = dataset.schema.field("event_number").type
    columns = list(dict.fromkeys(tuple(PARQUET_COLUMNS) + tuple(extra_columns)))
    scanner = dataset.scanner(columns=columns, filter=ds.field("event_number").isin(pa.array(srt, type=etype)),
                              batch_size=int(batch_rows), use_threads=True)
    mn, mx = np.full((n, 3), np.inf), np.full((n, 3), -np.inf)
    amn, amx = np.full((n, 2), np.inf), np.full((n, 2), -np.inf)
    rows = np.zeros(n, dtype=np.int64)
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        cols = {name: rb.column(i).to_numpy(zero_copy_only=False) for i, name in enumerate(rb.schema.names)}
        e = cols["event_number"].astype(np.int64)
        i = np.searchsorted(srt, e)
        if (i >= n).any() or (srt[np.minimum(i, n - 1)] != e).any():
            raise RuntimeError("the Parquet filter returned a row of an event that was not requested")
        th, ph = cols["incoming_theta"].astype(np.float64), cols["incoming_phi"].astype(np.float64)
        loc = np.stack([cols["x_local"], cols["y_local"], cols["z_local"]], 1).astype(np.float64)
        o = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64) - to_global(loc, th, ph)
        if not np.isfinite(o).all():
            raise ValueError("non-finite origin (NaN in a coordinate or angle column)")
        for k in range(3):
            np.minimum.at(mn[:, k], i, o[:, k])
            np.maximum.at(mx[:, k], i, o[:, k])
        for k, a in enumerate((th, ph)):
            np.minimum.at(amn[:, k], i, a)
            np.maximum.at(amx[:, k], i, a)
        rows += np.bincount(i, minlength=n)
        if row_hook is not None:
            row_hook(order[i], cols)
    if (rows == 0).any():
        raise ValueError(f"{int((rows == 0).sum())} requested event(s) have no row in the file")
    if (amx != amn).any():
        raise ValueError("incoming_theta / incoming_phi vary inside an event")
    spread = mx - mn
    if float(spread.max()) > tol:
        raise ValueError(f"origin not unique inside an event: max spread {spread.max():.3e} mm > {tol:g} mm")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)                                  # position of event_numbers[j] in the sorted list
    return {"event_number": evn.copy(), "origin": (0.5 * (mn + mx))[inv], "spread": spread[inv], "rows": rows[inv],
            "theta": amn[inv, 0], "phi": amn[inv, 1]}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def save_origins(per_split: Mapping[str, dict], cache_dir=None, extra_meta: Optional[dict] = None) -> dict:
    """Write ``<cache>/pion_s1_geometry_trainval.pt`` (+ ``.json``) from ``{"train": o, "val": o}`` with ``o`` the
    dict of ``origins_from_parquet`` in cache order. A split named "test" is refused."""
    if set(per_split) - {"train", "val"}:
        raise PermissionError("only the train and val splits may be written to the geometry file")
    d = Path(cache_dir or paths.CACHE_DIR)
    pt, js = d / f"{GEOMETRY_STEM}.pt", d / f"{GEOMETRY_STEM}.json"
    obj = {s: {"event_number": torch.from_numpy(np.asarray(o["event_number"], dtype=np.int64)),
               "origin": torch.from_numpy(np.asarray(o["origin"], dtype=np.float64)),
               "rows": torch.from_numpy(np.asarray(o["rows"], dtype=np.int64))} for s, o in per_split.items()}
    obj["definition"] = ("origin = (x, y, z) - R^T (x_local, y_local, z_local), R = common.rotation(theta, phi); "
                         "identical for every row of the event (no cuts)")
    obj["units"] = "mm"
    torch.save(obj, pt)
    meta = {"file": pt.name, "sha256": _sha256(pt), "bytes": pt.stat().st_size, "units": "mm",
            "definition": obj["definition"],
            "meaning": "absolute (global-frame) position of the shower-frame origin = energy-weighted centroid of the "
                       "lowest-z hit layer of the event over all rows, including deposits later than 100 ns",
            "convention": "global = origin + R^T . local; R = common.rotation(theta, phi) (rows = shower axes)",
            "order": "cache order of cache/pion_s1_trainval.pt (event_number stored per split)",
            "splits": {s: {"events": int(len(o["event_number"])), "rows_read": int(np.sum(o["rows"])),
                           "max_spread_mm": float(np.max(o["spread"])),
                           "origin_min_mm": [float(v) for v in np.min(o["origin"], 0)],
                           "origin_max_mm": [float(v) for v in np.max(o["origin"], 0)]} for s, o in per_split.items()},
            "contains_test_events": False,
            "dtypes": {"event_number": "int64", "origin": "float64 (n, 3)", "rows": "int64"}}
    meta.update(extra_meta or {})
    js.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load_origins(split: str, event_numbers=None, cache_dir=None, check_sha: bool = True) -> np.ndarray:
    """Origins (n, 3) float64 of ``split`` ("train" or "val") from the geometry file. With ``event_numbers`` (the
    cache extras of the same split) the stored event numbers must be identical, in order."""
    if split not in ("train", "val"):
        raise PermissionError("the geometry file holds train and val only")
    d = Path(cache_dir or paths.CACHE_DIR)
    pt, js = d / f"{GEOMETRY_STEM}.pt", d / f"{GEOMETRY_STEM}.json"
    if check_sha and js.exists():
        want = json.loads(js.read_text())["sha256"]
        if _sha256(pt) != want:
            raise RuntimeError(f"{pt.name}: SHA-256 differs from {js.name}")
    obj = torch.load(pt, map_location="cpu", weights_only=False)
    if event_numbers is not None and not np.array_equal(obj[split]["event_number"].numpy(), np.asarray(event_numbers)):
        raise ValueError(f"geometry file: event numbers of '{split}' differ from the cache")
    return obj[split]["origin"].numpy()


# ----------------------------------------------------------------------------------------------------------------------
# Instrumented volume (transverse outline; the depth is Z_FRONT ... Z_BACK for every column)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Volume:
    """Transverse outline of the instrumented volume.

    kind "stepped"  the envelope of the cells seen in training: per module row a = floor(|Y| / 101) the X interval
                    |X - 2.5| <= X_LAST[a] - 2.5 + 24.3, |Y| <= 2624.1; minus the beam hole (per-row edges).
         "octagon"  simpler alternative: |X - 2.5| <= min(w, c - s |Y|) with the straight line through the midpoints
                    of the steps, and a rectangular hole (mean edges). It cuts off the outer corner of every step.
         "box"      |X|, |Y| <= 2624.4 mm (Milestone 1 audit); wrong by the four corners and the hole (19.1 % of
                    its lattice positions have no cell).
    ``hole=False`` switches the hole off (diagnostic).
    """
    kind: str = "stepped"
    hole: bool = True

    def __post_init__(self):
        if self.kind not in ("stepped", "octagon", "box"):
            raise ValueError("kind must be 'stepped', 'octagon' or 'box'")

    @property
    def name(self) -> str:
        return self.kind + ("" if (self.hole or self.kind == "box") else "_nohole")

    def inside_xy(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Bool tensor: is the transverse position (X, Y) [mm, global] inside the outline (and outside the hole)?"""
        if self.kind == "box":
            return (X.abs() <= BOX_HALF) & (Y.abs() <= BOX_HALF)
        ay, dx = Y.abs(), (X - X_CENTRE).abs()
        hw = torch.as_tensor(X_LAST, dtype=X.dtype, device=X.device) - X_CENTRE + HALF_CELL
        if self.kind == "stepped":
            a = torch.div(ay, MODULE_PITCH_Y, rounding_mode="floor").long().clamp(0, len(X_LAST) - 1)
            inside = (dx <= hw[a]) & (ay <= Y_MAX)
            if self.hole:
                hl = torch.as_tensor(HOLE_LEFT, dtype=X.dtype, device=X.device) + HALF_CELL
                hr = torch.as_tensor(HOLE_RIGHT, dtype=X.dtype, device=X.device) - HALF_CELL
                ah = a.clamp(max=len(HOLE_LEFT) - 1)
                inside &= ~((ay < HOLE_Y) & (X > hl[ah]) & (X < hr[ah]))
            return inside
        w = float(hw[:11].mean())                                             # full half-width (rows 0..10)
        slope = float((hw[25] - hw[11]) / (14 * MODULE_PITCH_Y))              # -0.99945: one module per module row
        y11 = 11.5 * MODULE_PITCH_Y                                           # midpoint of the first narrower row
        inside = (dx <= torch.clamp(float(hw[11]) + slope * (ay - y11), max=w)) & (ay <= Y_MAX)
        if self.hole:
            hl = sum(HOLE_LEFT) / len(HOLE_LEFT) + HALF_CELL
            hr = sum(HOLE_RIGHT) / len(HOLE_RIGHT) - HALF_CELL
            inside &= ~((ay < HOLE_Y) & (X > hl) & (X < hr))
        return inside

    def inside_xy_np(self, X, Y) -> np.ndarray:
        return self.inside_xy(torch.as_tensor(np.asarray(X, dtype=np.float64)),
                              torch.as_tensor(np.asarray(Y, dtype=np.float64))).numpy()


def columns_inside(X, Y, ux, uy, half: float = HALF_CELL, max_step: float = 60.0) -> np.ndarray:
    """Independent point-in-volume test built directly on a list of cell columns (centres ``ux, uy`` [mm]), without
    the constants of this module: a point belongs to the cell row nearest in Y; it is inside if it lies within
    ``half`` of a column centre of that row, or between two neighbouring columns of the row that are less than
    ``max_step`` apart (module gaps are filled, the beam hole is not); in Y it must lie within ``half`` of the
    nearest row or between two rows. NumPy, float64."""
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    shape = X.shape
    X, Y = X.ravel(), Y.ravel()
    ys = np.unique(np.round(np.asarray(uy, dtype=np.float64), 3))
    row_of_col = np.searchsorted(ys, np.round(np.asarray(uy, dtype=np.float64), 3))
    j = np.clip(np.searchsorted(ys, Y), 1, len(ys) - 1)
    row = np.where(np.abs(Y - ys[j - 1]) <= np.abs(ys[j] - Y), j - 1, j)
    ok_y = (np.abs(Y - ys[row]) <= half) | ((Y >= ys[0]) & (Y <= ys[-1]))
    span = 1.0e5                                                               # > any |x| range: one key per (row, x)
    key_c = row_of_col * span + (np.asarray(ux, dtype=np.float64) + span / 2)
    o = np.argsort(key_c)
    key_c, row_c, x_c = key_c[o], row_of_col[o], np.asarray(ux, dtype=np.float64)[o]
    key_p = row * span + (X + span / 2)
    r = np.searchsorted(key_c, key_p)                                          # first column with key >= point
    n = len(key_c)
    ri, li = np.minimum(r, n - 1), np.maximum(r - 1, 0)
    has_r = (r < n) & (row_c[ri] == row)
    has_l = (r > 0) & (row_c[li] == row)
    dr = np.where(has_r, x_c[ri] - X, np.inf)
    dl = np.where(has_l, X - x_c[li], np.inf)
    ok_x = (dr <= half) | (dl <= half) | (has_r & has_l & (x_c[ri] - x_c[li] < max_step))
    return (ok_x & ok_y).reshape(shape)


# ----------------------------------------------------------------------------------------------------------------------
# Per-event acceptance
# ----------------------------------------------------------------------------------------------------------------------
def ring_nodes(nsub: int, nphi: int, device=None, dtype=torch.float32):
    """Quadrature nodes of the grid cells: z' (N_Z * nsub,), r (N_R * nsub,) - the sub-points of
    ``shapelab.slab_accept`` (midpoints in z', equal-area sub-rings in r) - and azimuths (nphi,), midpoint rule.
    ``device=None`` means the CPU (the grid tensors of ``common`` live on its own device)."""
    device = torch.device("cpu") if device is None else device
    ze = C.Z_EDGES.to(device=device, dtype=dtype)
    re = C.R_EDGES.to(device=device, dtype=dtype)
    k = (torch.arange(nsub, device=device, dtype=dtype) + 0.5) / nsub
    zs = (ze[:-1, None] + C.DZ * k).reshape(-1)
    rs = torch.sqrt(re[:-1, None] ** 2 + (re[1:, None] ** 2 - re[:-1, None] ** 2) * k).reshape(-1)
    al = (torch.arange(nphi, device=device, dtype=dtype) + 0.5) / nphi * (2 * math.pi)
    return zs, rs, al


def _check_variants(variants) -> dict:
    out = {}
    for name, faces in dict(variants).items():
        faces = tuple(faces)
        if not faces or set(faces) - set(FACES):
            raise ValueError(f"variant {name!r}: faces must be a non-empty subset of {FACES}")
        out[name] = faces
    return out


@torch.no_grad()
def acceptance(R: torch.Tensor, origin: torch.Tensor, volume: Volume = Volume(), variants: Mapping = VARIANTS, *,
               nsub: int = 2, nphi: int = 64, front_plane: str = "first_hit_layer", z_back: float = Z_BACK) -> dict:
    """Acceptance maps of a batch of events: ``{variant: (B, N_Z, N_R) float32}``, values in [0, 1].

    R        (B, 3, 3) rotation global -> shower (``Showers.ev["R"]``)
    origin   (B, 3) absolute origin of the shower frame [mm]
    variants name -> faces, a subset of ("front", "side", "back")
    front_plane  "first_hit_layer": Z >= origin_z - LAYER / 2 (default; equals the detector front face when the
                 first layer is hit); "detector": Z >= Z_FRONT.
    Memory: B * N_Z * N_R * nsub^2 * nphi values per intermediate tensor (4.2 MB per event at nsub 2, nphi 64).
    """
    variants = _check_variants(variants)
    if front_plane not in ("first_hit_layer", "detector"):
        raise ValueError("front_plane must be 'first_hit_layer' or 'detector'")
    dev, dt = R.device, torch.float32
    R = R.to(dt)
    origin = origin.to(device=dev)
    B = len(R)
    zs, rs, al = ring_nodes(nsub, nphi, dev, dt)
    rc, rsn = rs[:, None] * torch.cos(al), rs[:, None] * torch.sin(al)                  # (Rs, A)

    def glob(i):                                # global offset component i of (r cos a, r sin a, z'): sum_j R[j, i] x_j
        return (R[:, 0, i, None, None, None] * rc + R[:, 1, i, None, None, None] * rsn
                + R[:, 2, i, None, None, None] * zs[None, :, None, None])               # (B, Zs, Rs, A)

    need = set().union(*variants.values())
    masks = {}
    if need & {"front", "back"}:
        gz = glob(2)
        oz = origin[:, 2].to(dt)[:, None, None, None]
        if "front" in need:
            masks["front"] = gz >= (-LAYER / 2 if front_plane == "first_hit_layer" else (Z_FRONT - oz))
        if "back" in need:
            masks["back"] = gz <= (z_back - oz)
        del gz
    if "side" in need:
        masks["side"] = volume.inside_xy(origin[:, 0].to(dt)[:, None, None, None] + glob(0),
                                         origin[:, 1].to(dt)[:, None, None, None] + glob(1))
    out = {}
    for name, faces in variants.items():
        m = masks[faces[0]]
        for f in faces[1:]:
            m = m & masks[f]
        out[name] = m.to(dt).mean(3).reshape(B, C.N_Z, nsub, C.N_R, nsub).mean((2, 4))
    return out


@torch.no_grad()
def event_acceptance(R, origin, volume: Volume = Volume(), variants: Mapping = VARIANTS, *, nsub: int = 2,
                     nphi: int = 64, front_plane: str = "first_hit_layer", device=None, chunk: int = 32) -> dict:
    """``acceptance`` for many events in chunks; inputs and outputs live on the CPU, the work runs on ``device``
    (hold ``sfm.gpulock.gpu_lock`` when that is the GPU). Returns ``{variant: (N, N_Z, N_R) float32 CPU tensor}``."""
    R = torch.as_tensor(R)
    origin = torch.as_tensor(np.asarray(origin, dtype=np.float64)) if not torch.is_tensor(origin) else origin
    variants = _check_variants(variants)
    out = {k: [] for k in variants}
    for s in range(0, len(R), chunk):
        A = acceptance(R[s:s + chunk].to(device), origin[s:s + chunk].to(device), volume, variants, nsub=nsub,
                       nphi=nphi, front_plane=front_plane)
        for k in variants:
            out[k].append(A[k].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def mean_acceptance(R, origin, volume: Volume = Volume(), variants: Mapping = VARIANTS, *, nsub: int = 2,
                    nphi: int = 64, front_plane: str = "first_hit_layer", device=None, chunk: int = 32) -> dict:
    """Template Abar(z', r) = mean over the given events of A_e(z', r), per variant: ``{variant: (N_Z, N_R) float32}``
    on the CPU (accumulated in float64). Call it with TRAINING events; it is the ``template`` of the ratio form of
    ``apply_acceptance``. Pure geometry (theta, phi, origin of the events) - no hit and no fit enters."""
    R = torch.as_tensor(R)
    origin = torch.as_tensor(np.asarray(origin, dtype=np.float64)) if not torch.is_tensor(origin) else origin
    variants = _check_variants(variants)
    tot = {k: torch.zeros(C.N_Z, C.N_R, dtype=torch.float64) for k in variants}
    for s in range(0, len(R), chunk):
        A = acceptance(R[s:s + chunk].to(device), origin[s:s + chunk].to(device), volume, variants, nsub=nsub,
                       nphi=nphi, front_plane=front_plane)
        for k in variants:
            tot[k] += A[k].double().sum(0).cpu()
    return {k: (v / len(R)).float() for k, v in tot.items()}


def axis_exit(R, origin, volume: Volume = Volume(), z_max: float = float(C.Z_MAX), step: float = 2.0) -> np.ndarray:
    """Depth z' >= 0 [mm] at which the shower AXIS (r = 0) first leaves the transverse outline or enters the beam
    hole, scanned in steps of ``step`` up to ``z_max``; ``inf`` if it stays inside. An axis that starts outside
    (origin in the hole or beyond the outline) returns 0. NumPy / CPU; R (n, 3, 3), origin (n, 3)."""
    R = torch.as_tensor(np.asarray(R, dtype=np.float64))
    o = torch.as_tensor(np.asarray(origin, dtype=np.float64))
    z = torch.arange(0.0, z_max + step / 2, step, dtype=torch.float64)
    X = o[:, 0, None] + R[:, 2, 0, None] * z
    Y = o[:, 1, None] + R[:, 2, 1, None] * z
    out = ~volume.inside_xy(X, Y)
    first = torch.where(out.any(1), out.to(torch.int8).argmax(1), torch.full((len(o),), -1))
    res = np.where(first.numpy() >= 0, z.numpy()[first.clamp(min=0).numpy()], np.inf)
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Applying an acceptance to a map and scoring it on the fold points of common.test_nll
# ----------------------------------------------------------------------------------------------------------------------
def apply_acceptance(logP: torch.Tensor, A: torch.Tensor, floor: float = 1e-3,
                     template: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The map of an event after applying its acceptance, renormalised over the whole grid.

        naive form (template=None)   log P_A = log_softmax(log P + ln max(A, floor))
        ratio form                   log P_A = log_softmax(log P + ln max(A, floor) - ln max(template, floor))

    The naive form is right for a map of the shower itself (e.g. the core of the component formula). A map that was
    fitted to event-averaged data (a theta-blind network; a formula that carries the template ``abar``) already
    contains the mean acceptance of the training events; for it the event's acceptance has to replace that mean:
    ``template`` = mean over the training events of A_e for the same faces, shape (N_Z, N_R) or (B, N_Z, N_R).
    The template is pure geometry - no parameter is fitted. Where both A and the template are below ``floor`` the
    map is left as it is.

    ``floor`` > 0 keeps cells whose ring is estimated to be entirely outside from getting density zero (a fold
    point of a boundary hit can land there; the estimate of A has quadrature error of order 1 / (nsub^2 nphi)),
    and bounds the boost of the ratio form by 1 / floor."""
    if not floor > 0:
        raise ValueError("floor must be positive")
    if A.shape != logP.shape:
        raise ValueError(f"acceptance has shape {tuple(A.shape)}, log P {tuple(logP.shape)}")
    lp = logP + torch.log(A.to(logP.dtype).clamp_min(floor))
    if template is not None:
        if template.shape[-2:] != logP.shape[-2:]:
            raise ValueError(f"template has shape {tuple(template.shape)}, log P {tuple(logP.shape)}")
        lp = lp - torch.log(template.to(logP.dtype).clamp_min(floor))
    return F.log_softmax(lp.reshape(len(lp), -1), 1).reshape(lp.shape)


def score_variants(make_logPs: Callable, data, *, K: int = C.N_FOLD_EVAL, seed: int = 12345, batch: int = 128,
                   bulk=None, monitors: bool = True, chunk: Optional[int] = None) -> dict:
    """Per-event NLL of several per-event maps on the fold points of ``common.test_nll``.

    ``make_logPs(b, idx) -> {name: log P (B, N_Z, N_R)}`` is called once per batch; ``idx`` are the positions of the
    batch's events in ``data`` (storage order, batches of ``batch`` - the walk of ``Showers.batches``), so per-event
    quantities such as acceptance maps can be looked up. One generator (``seed``) and one ``torch.rand(H, K, 3)``
    per batch, drawn inside ``metric.event_nll_multi``: a map equal to a model's own output gets the bits of
    ``common.test_nll`` (the function runs inside ``metric.deterministic()`` and ``torch.inference_mode()``).

    Returns ``{name: {"nll", "hit", "log_mass"[, monitors]}}`` plus the key ``"_shared"``; all (N,) float64 arrays.
        nll        per-event NLL (report metric, or the bulk-restricted one with ``bulk=metric.Bulk(...)``)
        hit        sum over the event's hits of omega_i ln ghat_i
        log_mass   ln M_e (or ln M_A,e);   nll = weight_sum * log_mass - hit   (``metric.Parts``)
        monitors   (standard metric only) beyond_zb, last_ring, outside_bulk, total  - ``metric.mass_monitors``
        _shared    weight_sum, and for bulk retained, z_limit
    """
    n = len(data)
    dev = data.off.device
    res, shared = {}, {"weight_sum": []}
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        stack.enter_context(metric.deterministic())
        gen = torch.Generator(device=C.DEVICE).manual_seed(seed)
        for s in range(0, n, batch):
            idx = torch.arange(s, min(s + batch, n), device=dev)
            b = data.batch(idx)
            logPs = make_logPs(b, idx)
            names = list(logPs)
            out = metric.event_nll_multi([logPs[k] for k in names], b, data.box, K, gen, bulk=bulk, chunk=chunk,
                                         return_parts=True)
            for name, (nll, logP, parts) in zip(names, out):
                r = res.setdefault(name, {"nll": [], "hit": [], "log_mass": []})
                r["nll"].append(nll.double().cpu())
                r["hit"].append(torch.zeros(len(idx), device=dev, dtype=torch.float64)
                                .index_add_(0, b.ev, parts.hit.double()).cpu())
                r["log_mass"].append(parts.log_mass.double().cpu())
                if monitors and bulk is None:
                    mon = metric.mass_monitors(logP, b)
                    for k in ("beyond_zb", "last_ring", "outside_bulk", "total"):
                        r.setdefault(k, []).append(mon[k].cpu())
            p0 = out[0][2]
            shared["weight_sum"].append(p0.weight_sum.double().cpu())
            if p0.retained is not None:
                shared.setdefault("retained", []).append(p0.retained.double().cpu())
                shared.setdefault("z_limit", []).append(p0.z_limit.double().cpu())
    final = {name: {k: torch.cat(v).numpy() for k, v in r.items()} for name, r in res.items()}
    final["_shared"] = {k: torch.cat(v).numpy() for k, v in shared.items()}
    return final
