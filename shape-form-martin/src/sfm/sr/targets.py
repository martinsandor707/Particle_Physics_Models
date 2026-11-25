"""sfm.sr.targets - the regression targets of Milestone 3 (PLAN.md section 7), written as `.npz` files that the
search process reads (``sfm.sr.search.load_target``; that process never imports torch).

The constructions are NumPy functions of maps and counts, so they are tested on synthetic inputs. The two functions
that touch events (``estimate_response``, ``hit_loglik`` / ``fisher_scores``) import torch and ``common`` on demand and
take a ``common.Showers`` object: pass the TRAINING split (``sfm.data.load_sets(("train",))``). Nothing here opens the
test split, and nothing here computes a likelihood of a candidate equation.

Target (iii) - residual (``build_residual_target``)
    y_s(c') = ln (R_s rho_T,s)(c') - ln (R_s rho_B,s)(c')  for momentum slice s and centre cell c', both maps
    evaluated at the slice momentum p_s and folded with the train-estimated, energy-weighted cell-box response
        R_s[c', c] = sum_{i in s, cell(h_i) = c'} w_i (1/K) sum_k 1{cell(h_i + R_e u_ik) = c} / W_s(c'),
        W_s(c') = sum of w_i over the training hits of the slice centred in c'.
    Teacher: probability mean of the ensemble, ln rho_T = logsumexp_m ln rho_m - ln M.
    Abscissae: the centroid of the tilted response S_s[c', c] = R_s[c', c] rho_B(c) / (R_s rho_B)(c') - the exact
    first-order point at which a correction f enters ln R(rho_B e^f) - in the search variables zt, rh; x = x(p_s)
    is identical for all rows of a slice (the gauge loss finds the slices by runs of equal x).
    Masks: z' < 1150 mm, not the first z row, not the last ring, A > 0.05, at least ``min_hits`` hits in the cell,
    standard error of the teacher mean <= 0.05 nat (skipped and flagged for a single teacher), finite values.
    Slices: p_s >= 0.0895 GeV is ENFORCED (the benchmark maps are not teachers below the training minimum).
    Sampling: ``n_per_slice`` (~650) cells per slice, fixed-size systematic sampling with inclusion probability
    pi_c proportional to omega_c^gamma (omega_c = W_s(c') / sum W_s over the masked cells; gamma = 0.5 so that tail
    cells are represented; units with pi >= 1 are taken with certainty), Horvitz-Thompson weight omega_c / pi_c.
    The residual itself is not used in the design. Gauge: y has HT-weighted mean zero per slice, so the weighted L2
    of f = 0 equals the gauge loss of f = 0 (the null loss).
    Seed-null (control C3): y_null = kappa [ln R rho_A - ln R rho_B] for two halves of the ensemble on the same rows
    and weights, kappa = sqrt((1/M) / (1/a + 1/b)), so that its noise variance equals that of the full teacher.

Target (i) - trajectories (``build_trajectory_target``)
    X (G, 1) = x at G points, y (G, n_out) = the raw parameters of the Track-A finalist, weights (G, n_out) =
    n_g F_jj(x_g): momentum density (fraction of events at point g) times the diagonal Fisher information, so that
    sum_g w_gj Delta_j(x_g)^2 ~ 2 Delta NLL_j. The Fisher information is the EXPECTED HESSIAN of the energy-weighted
    pseudo-likelihood -sum_i w_i l_i (l_i = per-hit log-likelihood):
        F_jk(x) = E[ sum_i w_i d_j l_i d_k l_i | x ]                    (``fisher_from_hit_scores``)
    and NOT the covariance of the per-event scores E[(sum_i w_i d_j l_i)(sum_i w_i d_k l_i)], which is smaller by
    about sum_i w_i^2 (the effective number of hits) because the event NLL is not a log-likelihood of the event.
    ``guesses``: weighted-least-squares cubics in Horner form, ``list[list[str]]`` (one list per output).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from . import operators as O
from . import translate as T

__all__ = ["SCHEMA_VERSION", "event_slices", "ht_sample", "build_residual_target", "seed_null_kappa",
           "fisher_from_hit_scores", "score_covariance_from_hit_scores", "trajectory_weights",
           "build_trajectory_target", "cubic_guesses", "save_target", "load_target_arrays", "null_loss",
           "estimate_response", "hit_loglik", "fisher_scores", "ShiftedProvider", "RESIDUAL_MASK_DEFAULTS"]

SCHEMA_VERSION = "sfm.sr.target/1"
RESIDUAL_MASK_DEFAULTS = {"z_max_mm": 1150.0, "a_min": 0.05, "min_hits": 30, "max_se": 0.05}


# ----------------------------------------------------------------------------------------------------------------------
# Slices and sampling
# ----------------------------------------------------------------------------------------------------------------------
def event_slices(p, n_slices: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Slices at event quantiles of the momentum: (slice index per event, median momentum per slice)."""
    p = np.asarray(p, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_slices + 1))
    sid = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_slices - 1)
    med = np.array([np.median(p[sid == s]) for s in range(n_slices)])
    return sid, med


def inclusion_probabilities(size: np.ndarray, n: int) -> np.ndarray:
    """pi_c = min(1, n size_c / sum size) with the certainty units removed iteratively (sum pi = min(n, #units))."""
    size = np.asarray(size, dtype=np.float64)
    if np.any(size < 0) or not np.all(np.isfinite(size)):
        raise ValueError("sizes must be finite and non-negative")
    pos = size > 0
    pi = np.zeros_like(size)
    if pos.sum() <= n:
        pi[pos] = 1.0
        return pi
    certain = np.zeros(len(size), bool)
    while True:
        free = pos & ~certain
        m = n - int(certain.sum())
        pi_free = m * size[free] / size[free].sum()
        over = pi_free >= 1.0
        if not over.any():
            pi[free], pi[certain] = pi_free, 1.0
            return pi
        certain[np.flatnonzero(free)[over]] = True


def ht_sample(omega: np.ndarray, n: int, gamma: float = 0.5, rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-size systematic sample with pi proportional to omega^gamma. Returns (indices, pi of all units).
    The Horvitz-Thompson weight of a sampled unit is omega / pi: sum over the sample of (omega/pi) g is an unbiased
    estimate of sum over all units of omega g."""
    rng = np.random.default_rng(rng)
    omega = np.asarray(omega, dtype=np.float64)
    pi = inclusion_probabilities(np.where(omega > 0, omega ** gamma, 0.0), int(n))
    order = rng.permutation(len(omega))                    # random order, then systematic: fixed size, pi exact
    cum = np.cumsum(pi[order])
    total = int(round(cum[-1])) if len(cum) else 0
    if total:
        cum = cum * (total / cum[-1])                      # exact integer total: the sample size is fixed
    u = rng.uniform(0.0, 1.0)
    picks = np.searchsorted(cum, u + np.arange(total), side="right")
    picks = np.clip(picks, 0, len(order) - 1)
    return np.sort(order[np.unique(picks)]), pi


def seed_null_kappa(m: int, a: int, b: int) -> float:
    """kappa with Var[kappa (mean_A - mean_B)] = Var[mean of all M] for independent members: sqrt((1/M)/(1/a+1/b))."""
    return math.sqrt((1.0 / m) / (1.0 / a + 1.0 / b))


def _apply(R, v: np.ndarray) -> np.ndarray:
    return np.asarray(R @ v).reshape(-1) if R is not None else v


# ----------------------------------------------------------------------------------------------------------------------
# Target (iii)
# ----------------------------------------------------------------------------------------------------------------------
def build_residual_target(log_rho_teacher, log_rho_base, slice_p, energy_w, n_hits, z_mm, r_mm, *, response=None,
                          a_bar=None, n_per_slice: int = 650, gamma: float = 0.5,
                          z_max_mm: float = RESIDUAL_MASK_DEFAULTS["z_max_mm"],
                          a_min: float = RESIDUAL_MASK_DEFAULTS["a_min"],
                          min_hits: int = RESIDUAL_MASK_DEFAULTS["min_hits"],
                          max_se: float = RESIDUAL_MASK_DEFAULTS["max_se"], halves=None, seed: int = 0,
                          p_min: float = O.TRAIN_P_MIN, meta: Optional[Mapping] = None) -> Dict[str, np.ndarray]:
    """Build target (iii). See the module docstring.

    log_rho_teacher   (M, S, NZ, NR) member log-densities at the slice momenta, or (S, NZ, NR) for one teacher
    log_rho_base      (S, NZ, NR) log-density of the final baseline at the slice momenta
    slice_p           (S,) slice momenta in GeV (medians of the event-quantile bins), each >= p_min
    energy_w, n_hits  (S, NZ, NR) summed energy weight and number of TRAINING hits centred in each cell, per slice
    z_mm, r_mm        (NZ,), (NR,) cell centres
    response          None (no smoothing) or a sequence of S matrices (dense or scipy.sparse), R[c', c] row-stochastic
                      on the rows with hits, cells flattened as iz * NR + ir (``estimate_response``)
    a_bar             (NZ, NR) acceptance template, or None (mask skipped and recorded)
    halves            (index array A, index array B) of ensemble members for the seed-null target, or None
    Returns the arrays of the `.npz` schema (``save_target``).
    """
    lt = np.asarray(log_rho_teacher, dtype=np.float64)
    single = lt.ndim == 3
    if single:
        lt = lt[None]
    lb = np.asarray(log_rho_base, dtype=np.float64)
    M, S, NZ, NR = lt.shape
    slice_p = np.asarray(slice_p, dtype=np.float64)
    W = np.asarray(energy_w, dtype=np.float64)
    NH = np.asarray(n_hits)
    z_mm, r_mm = np.asarray(z_mm, dtype=np.float64), np.asarray(r_mm, dtype=np.float64)
    if lb.shape != (S, NZ, NR) or W.shape != (S, NZ, NR) or NH.shape != (S, NZ, NR) or slice_p.shape != (S,):
        raise ValueError("shapes: teacher (M, S, NZ, NR), base / energy_w / n_hits (S, NZ, NR), slice_p (S,)")
    if z_mm.shape != (NZ,) or r_mm.shape != (NR,):
        raise ValueError("z_mm (NZ,) and r_mm (NR,) are the cell centres")
    if np.any(slice_p < p_min * (1 - 1e-9)):
        raise ValueError(f"slice momenta below the training minimum {p_min:.4f} GeV: the benchmark maps are not "
                         f"used as teachers there (PLAN section 13); got {slice_p.min():.4f} GeV")
    if response is not None and len(response) != S:
        raise ValueError("response: one matrix per slice")
    if len(np.unique(slice_p)) != S:
        raise ValueError("slice momenta must be distinct (the gauge loss separates slices by the value of x)")
    rng = np.random.default_rng(seed)
    zz, rr = np.meshgrid(z_mm, r_mm, indexing="ij")
    zf, rf = zz.reshape(-1), rr.reshape(-1)
    geom = (zz < z_max_mm) & (np.arange(NZ)[:, None] > 0) & (np.arange(NR)[None, :] < NR - 1)
    if a_bar is not None:
        geom &= np.asarray(a_bar, dtype=np.float64) > a_min
    x_s = O.x_of_p(slice_p)
    order = np.argsort(x_s)
    out: Dict[str, list] = {k: [] for k in ("X", "y", "w", "sid", "iz", "ir", "pi", "omega", "se", "ynull")}
    counts = {"geometry": int(geom.sum()), "per_slice": []}
    kappa = None
    if halves is not None:
        ia, ib = np.asarray(halves[0]), np.asarray(halves[1])
        if len(np.intersect1d(ia, ib)) or len(ia) == 0 or len(ib) == 0:
            raise ValueError("halves must be two disjoint, non-empty sets of member indices")
        kappa = seed_null_kappa(M, len(ia), len(ib))
    for new_id, s in enumerate(order):
        R = response[s] if response is not None else None
        rho_b = np.exp(lb[s].reshape(-1))
        fb = _apply(R, rho_b)
        fm = np.stack([_apply(R, np.exp(lt[m, s].reshape(-1))) for m in range(M)])       # (M, C) folded members
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_fm = np.log(fm)
            ln_t = logsumexp(ln_fm, axis=0) - math.log(M)
            y = ln_t - np.log(fb)
            se = ln_fm.std(axis=0, ddof=1) / math.sqrt(M) if M > 1 else np.zeros_like(y)
            if R is not None:                               # centroid of the tilted response
                zc = _apply(R, rho_b * zf) / fb
                rc = _apply(R, rho_b * rf) / fb
            else:
                zc, rc = zf, rf
            yn = kappa * (np.log(fm[ia].mean(0)) - np.log(fm[ib].mean(0))) if halves is not None else None
        mask = geom.reshape(-1) & (NH[s].reshape(-1) >= min_hits) & (W[s].reshape(-1) > 0) & np.isfinite(y) & \
            np.isfinite(zc) & np.isfinite(rc)
        if M > 1:
            mask &= se <= max_se
        if yn is not None:
            mask &= np.isfinite(yn)
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            raise ValueError(f"slice {s} (p = {slice_p[s]:.3f} GeV): fewer than two cells survive the masks")
        omega = W[s].reshape(-1)[idx]
        omega = omega / omega.sum()
        pick, pi = ht_sample(omega, n_per_slice, gamma, rng)
        w = omega[pick] / pi[pick]
        yy = y[idx][pick]
        yy = yy - np.sum(w * yy) / np.sum(w)               # gauge: HT-weighted mean zero per slice
        cells = idx[pick]
        out["X"].append(np.column_stack([zc[cells] / O.ZETA_SCALE_MM, rc[cells] / O.RHO_SCALE_MM,
                                         np.full(len(cells), x_s[s])]))
        out["y"].append(yy)
        out["w"].append(w)
        out["sid"].append(np.full(len(cells), new_id))
        out["iz"].append(cells // NR)
        out["ir"].append(cells % NR)
        out["pi"].append(pi[pick])
        out["omega"].append(omega[pick])
        out["se"].append(se[cells])
        if yn is not None:
            v = yn[cells]
            out["ynull"].append(v - np.sum(w * v) / np.sum(w))
        counts["per_slice"].append({"p": float(slice_p[s]), "x": float(x_s[s]), "masked_cells": int(len(idx)),
                                    "sampled": int(len(cells)), "certain": int((pi[pick] >= 1.0).sum()),
                                    "ht_weight_sum": float(w.sum())})
    X = np.concatenate(out["X"])
    info = {"kind": "residual", "n_members": int(M), "single_teacher": bool(single or M == 1),
            "se_mask_applied": bool(M > 1), "acceptance_mask_applied": a_bar is not None,
            "response": "none" if response is None else "train-estimated cell-box response",
            "masks": {"z_max_mm": z_max_mm, "a_min": a_min, "min_hits": int(min_hits), "max_se": max_se,
                      "first_z_row": "excluded", "last_ring": "excluded"},
            "n_per_slice": int(n_per_slice), "gamma": float(gamma), "seed": int(seed), "p_min": float(p_min),
            "counts": counts, "seed_null_kappa": kappa, "scales_mm": [O.ZETA_SCALE_MM, O.RHO_SCALE_MM],
            "loss_note": "rows of a slice are contiguous and share one x: use operators.gauge_loss_julia(3) with "
                         "batching=False; weights are Horvitz-Thompson weights (sum ~ 1 per slice)"}
    info.update(dict(meta or {}))
    res = {"X": X, "y": np.concatenate(out["y"]), "weights": np.concatenate(out["w"]),
           "variable_names": np.array(O.VARIABLE_NAMES), "slice_id": np.concatenate(out["sid"]).astype(np.int64),
           "slice_p": slice_p[order], "slice_x": x_s[order], "cell_iz": np.concatenate(out["iz"]).astype(np.int64),
           "cell_ir": np.concatenate(out["ir"]).astype(np.int64), "pi": np.concatenate(out["pi"]),
           "omega": np.concatenate(out["omega"]), "seed_se": np.concatenate(out["se"]),
           "kind": np.array("residual"), "loss": np.array("gauge"), "schema_version": np.array(SCHEMA_VERSION),
           "meta_json": np.array(json.dumps(info))}
    if halves is not None:
        res["y_null"] = np.concatenate(out["ynull"])
    return res


def null_loss(target: Mapping[str, np.ndarray], key: str = "y") -> float:
    """Loss of f = 0: the scale against which every search loss is read (for the residual target it equals both
    the gauge loss and the weighted L2, because y is centred per slice)."""
    y, w = np.asarray(target[key]), np.asarray(target["weights"])
    if str(target["loss"]) == "gauge":
        return O.gauge_loss_numpy(np.zeros_like(y), y, w, np.asarray(target["X"])[:, 2])
    return float(np.sum(w * y * y, axis=0).sum() / np.sum(w, axis=0).sum()) if y.ndim == 1 else \
        [O.wl2_loss_numpy(np.zeros(len(y)), y[:, j], w[:, j]) for j in range(y.shape[1])]


# ----------------------------------------------------------------------------------------------------------------------
# Target (i)
# ----------------------------------------------------------------------------------------------------------------------
def fisher_from_hit_scores(dl, w, ev, n_events: int) -> np.ndarray:
    """Per-event diagonal of the expected Hessian of the energy-weighted pseudo-likelihood:
    F_e,j = sum_{i in e} w_i (d_j l_i)^2, for per-hit score derivatives ``dl`` (H, P), hit weights ``w`` (H,) and
    hit -> event index ``ev`` (H,). Its mean over the events at a momentum estimates F_jj(x)."""
    dl, w, ev = np.asarray(dl, dtype=np.float64), np.asarray(w, dtype=np.float64), np.asarray(ev)
    out = np.zeros((int(n_events), dl.shape[1]))
    np.add.at(out, ev, w[:, None] * dl * dl)
    return out


def score_covariance_from_hit_scores(dl, w, ev, n_events: int) -> np.ndarray:
    """For comparison only: the per-event squared score (sum_i w_i d_j l_i)^2. NOT the curvature of the loss."""
    dl, w, ev = np.asarray(dl, dtype=np.float64), np.asarray(w, dtype=np.float64), np.asarray(ev)
    s = np.zeros((int(n_events), dl.shape[1]))
    np.add.at(s, ev, w[:, None] * dl)
    return s * s


def trajectory_weights(x_events, fisher_events, n_points: int = 40) -> Dict[str, np.ndarray]:
    """Points at event quantiles of x and the weights n_g F_jj(x_g): events are cut into ``n_points`` quantile bins;
    x_g = bin median, n_g = fraction of events in the bin, F_jj(x_g) = bin mean of the per-event Fisher diagonal.
    sum_g w_gj Delta_j(x_g)^2 then approximates 2 Delta NLL_j (mean per event) for a perturbation Delta_j(x)."""
    x = np.asarray(x_events, dtype=np.float64)
    F = np.asarray(fisher_events, dtype=np.float64)
    edges = np.quantile(x, np.linspace(0, 1, n_points + 1))
    g = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_points - 1)
    xg = np.array([np.median(x[g == k]) for k in range(n_points)])
    ng = np.array([(g == k).mean() for k in range(n_points)])
    Fg = np.stack([F[g == k].mean(0) for k in range(n_points)])
    return {"x": xg, "n": ng, "fisher": Fg, "weights": ng[:, None] * Fg}


def cubic_guesses(x, y, weights, degree: int = 3, variable: str = "x") -> List[List[str]]:
    """Weighted-least-squares polynomials per output in Horner form: ``guesses`` for a multi-output PySR fit,
    ``list[list[str]]`` with one list per output."""
    x, y, w = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64), np.asarray(weights)
    out = []
    for j in range(y.shape[1]):
        sw = np.sqrt(np.maximum(w[:, j], 0.0))
        V = np.vander(x, degree + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(V * sw[:, None], y[:, j] * sw, rcond=None)
        out.append([T.horner_string(coef, variable)])
    return out


def build_trajectory_target(x_points, values, n_fraction, fisher_diag, names: Sequence[str], *,
                            x_range: Tuple[float, float] = O.TRAIN_X_RANGE, meta: Optional[Mapping] = None
                            ) -> Dict[str, np.ndarray]:
    """Target (i): X (G, 1), y (G, n_out), weights (G, n_out) = n_g F_jj(x_g) normalised to mean 1 per output (the
    factor back to nats is stored in ``nat_factor``: sum_g weights_gj Delta^2 * nat_factor_j ~ 2 Delta NLL_j),
    ``guesses`` as JSON (list of lists). Points outside the training range of x are refused."""
    x = np.asarray(x_points, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64)
    n = np.asarray(n_fraction, dtype=np.float64).reshape(-1)
    F = np.asarray(fisher_diag, dtype=np.float64)
    G = len(x)
    if y.shape != (G, len(names)) or F.shape != y.shape or n.shape != (G,):
        raise ValueError("shapes: x (G,), values and fisher_diag (G, n_out), n_fraction (G,), names (n_out,)")
    tol = 1e-3
    if x.min() < x_range[0] - tol or x.max() > x_range[1] + tol:
        raise ValueError(f"trajectory points must lie in the training range of x {x_range}; got "
                         f"[{x.min():.3f}, {x.max():.3f}]")
    if np.any(F < 0) or np.any(n < 0) or not (np.all(np.isfinite(F)) and np.all(np.isfinite(y))):
        raise ValueError("Fisher information and event fractions must be finite and non-negative")
    raw = n[:, None] * F
    mean = raw.mean(0)
    if np.any(mean <= 0):
        bad = [names[j] for j in np.flatnonzero(mean <= 0)]
        raise ValueError(f"outputs without Fisher information (the likelihood does not depend on them): {bad}")
    w = raw / mean
    guesses = cubic_guesses(x, y, w)
    info = {"kind": "trajectory", "names": list(names), "nat_factor": mean.tolist(),
            "weights": "event fraction x diagonal expected Hessian of the energy-weighted pseudo-likelihood, "
                       "normalised to mean 1 per output", "guesses": guesses}
    info.update(dict(meta or {}))
    order = np.argsort(x)
    return {"X": x[order, None], "y": y[order], "weights": w[order], "variable_names": np.array(["x"]),
            "output_names": np.array(list(names)), "nat_factor": mean, "event_fraction": n[order],
            "fisher_diag": F[order], "guesses_json": np.array(json.dumps(guesses)), "kind": np.array("trajectory"),
            "loss": np.array("wl2"), "schema_version": np.array(SCHEMA_VERSION), "meta_json": np.array(json.dumps(info))}


# ----------------------------------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------------------------------
def save_target(path, target: Mapping[str, np.ndarray]) -> Path:
    """Write the `.npz` (no pickles). ``sfm.sr.search.load_target`` reads X, y, weights, variable_names from it."""
    path = Path(path)
    for k in ("X", "y", "weights", "variable_names", "kind", "loss", "schema_version", "meta_json"):
        if k not in target:
            raise KeyError(f"target is missing `{k}`")
    X, y, w = (np.asarray(target[k]) for k in ("X", "y", "weights"))
    if w.shape != y.shape or len(X) != len(y) or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must have the shape of y and be finite and non-negative")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("X and y must be finite")
    if str(target["loss"]) == "gauge":
        key = X[:, 2]
        runs = 1 + int(np.sum(key[1:] != key[:-1]))
        if runs != len(np.unique(key)):
            raise ValueError("gauge loss: the rows of a slice must be contiguous (one run of equal x per slice)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in target.items()})
    return path


def load_target_arrays(path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------------------------------
# Event-level estimators (torch; training split only)
# ----------------------------------------------------------------------------------------------------------------------
def estimate_response(showers, slice_of_event, n_slices: int, K: int = 16, seed: int = 0, batch: int = 256) -> dict:
    """The energy-weighted cell-box response of each slice from hits of ``showers`` (pass the TRAINING split):
    {"R": [scipy.sparse.csr_matrix (C, C)] * S, "energy_w": (S, NZ, NR), "n_hits": (S, NZ, NR), "n_events": (S,)}.
    Fold points as in the metric (``common.fold``, K per hit, seeded generator on the data's device)."""
    import scipy.sparse as sp
    import torch

    import common as C

    dev = showers.xyz.device
    sid_ev = torch.as_tensor(np.asarray(slice_of_event), dtype=torch.long, device=dev)
    if len(sid_ev) != len(showers):
        raise ValueError("slice_of_event: one entry per event")
    ncell = C.N_Z * C.N_R
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    rows, cols, vals, sl = [], [], [], []
    W = torch.zeros(n_slices, ncell, dtype=torch.float64, device=dev)
    NH = torch.zeros(n_slices, ncell, dtype=torch.float64, device=dev)
    pos = 0
    with torch.inference_mode():
        for b in showers.batches(int(batch)):
            n_b = len(b.lnp)
            s_hit = sid_ev[pos:pos + n_b][b.ev]
            pos += n_b
            iz0, ir0 = C.locate(b.xyz[:, 2], torch.sqrt(b.xyz[:, 0] ** 2 + b.xyz[:, 1] ** 2))
            c0 = iz0 * C.N_R + ir0
            pts = C.fold(b, showers.box, int(K), gen)
            iz, ir = C.locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
            c1 = iz * C.N_R + ir
            w = b.w.double()
            W.index_put_((s_hit, c0), w, accumulate=True)
            NH.index_put_((s_hit, c0), torch.ones_like(w), accumulate=True)
            rows.append(c0[:, None].expand_as(c1).reshape(-1).cpu().numpy())
            cols.append(c1.reshape(-1).cpu().numpy())
            vals.append((w[:, None] / K).expand(-1, int(K)).reshape(-1).cpu().numpy())
            sl.append(s_hit[:, None].expand_as(c1).reshape(-1).cpu().numpy())
    rows, cols, vals, sl = (np.concatenate(a) for a in (rows, cols, vals, sl))
    Wn = W.cpu().numpy()
    R = []
    for s in range(n_slices):
        m = sl == s
        mat = sp.coo_matrix((vals[m], (rows[m], cols[m])), shape=(ncell, ncell)).tocsr()
        inv = np.where(Wn[s] > 0, 1.0 / np.where(Wn[s] > 0, Wn[s], 1.0), 0.0)
        R.append(sp.diags(inv) @ mat)
    return {"R": R, "energy_w": Wn.reshape(n_slices, C.N_Z, C.N_R),
            "n_hits": NH.cpu().numpy().reshape(n_slices, C.N_Z, C.N_R).astype(np.int64),
            "n_events": np.bincount(sid_ev.cpu().numpy(), minlength=n_slices)}


def hit_loglik(model, b, box, K: int, gen=None, pts=None):
    """Per-hit folded log-likelihood l_i = ln g_i - ln M_e (the summand of ``common.event_nll``) and the fold
    points used, so that several models can be evaluated on COMMON fold points (``pts=``)."""
    import torch

    import common as C

    logP = model(b.lnp)
    logrho = logP - C.LOG_VOL
    if pts is None:
        pts = C.fold(b, box, K, gen)
    iz, ir = C.locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(pts.shape[1])
    Pz = logP.exp().sum(2)
    cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
    t = ((b.zb - C.Z_MIN) / C.DZ).clamp(0, C.N_Z)
    k = t.floor().long().clamp(max=C.N_Z - 1)
    ar = torch.arange(len(Pz), device=Pz.device)
    mass = cdf[ar, k] + Pz[ar, k] * (t - k)
    return logg - torch.log(mass.clamp_min(1e-8))[b.ev], pts


def ShiftedProvider(provider, column: int, eps: float):
    """Wrap a parameter provider: raw_j -> raw_j + eps (one column), for central differences of the likelihood.
    An ``nn.Module`` with ``names`` (so it can stand in for a registered sub-module for one evaluation)."""
    import torch.nn as nn

    class _Shifted(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.column, self.eps = provider, int(column), float(eps)
            self.names = list(provider.names)

        def forward(self, lnp):
            raw = self.inner(lnp).clone()
            raw[:, self.column] = raw[:, self.column] + self.eps
            return raw

    return _Shifted()


def fisher_scores(model, showers, columns: Sequence[int], *, provider_attr: str = "provider", eps: float = 1e-3,
                  K: int = 16, seed: int = 0, batch: int = 128, max_events: Optional[int] = None) -> dict:
    """Per-event Fisher diagonal of the raw parameters ``columns`` of ``getattr(model, provider_attr)``:
    d_j l_i by central differences (step ``eps`` in the raw parameter) of the per-hit folded log-likelihood on
    common fold points; F_e,j = sum_i w_i (d_j l_i)^2. Also returns the squared per-event score for comparison.
    Pass the TRAINING split. Returns {"fisher": (E, P), "score_sq": (E, P), "x": (E,), "p": (E,)}."""
    import torch

    prov = getattr(model, provider_attr)
    dev = showers.xyz.device
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    was = model.training
    model.eval()
    fis, sco, ps = [], [], []
    done = 0
    try:
        with torch.inference_mode():
            for b in showers.batches(int(batch)):
                _, pts = hit_loglik(model, b, showers.box, int(K), gen)
                dl = []
                for j in columns:
                    vals = []
                    for sgn in (+1.0, -1.0):
                        setattr(model, provider_attr, ShiftedProvider(prov, j, sgn * eps))
                        try:
                            vals.append(hit_loglik(model, b, showers.box, int(K), pts=pts)[0].double())
                        finally:
                            setattr(model, provider_attr, prov)
                    dl.append((vals[0] - vals[1]) / (2.0 * eps))
                dl = torch.stack(dl, 1).cpu().numpy()
                w, ev = b.w.double().cpu().numpy(), b.ev.cpu().numpy()
                fis.append(fisher_from_hit_scores(dl, w, ev, len(b.lnp)))
                sco.append(score_covariance_from_hit_scores(dl, w, ev, len(b.lnp)))
                ps.append(b.p.cpu().numpy())
                done += len(b.lnp)
                if max_events is not None and done >= max_events:
                    break
    finally:
        setattr(model, provider_attr, prov)
        model.train(was)
    p = np.concatenate(ps)
    return {"fisher": np.concatenate(fis), "score_sq": np.concatenate(sco), "p": p, "x": O.x_of_p(p)}
