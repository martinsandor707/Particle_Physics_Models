"""sfm.metric - the per-event NLL of ``common.py``, generalised (fold scheme, number of points, bulk restriction,
exact decomposition, several models on the same points).

The metric of record stays ``common.event_nll`` / ``common.test_nll``. At default settings the functions here
execute the same tensor operations in the same order on the same random numbers, so they return the same
numbers; every headline value is still produced by ``common`` itself. Everything else in this module is a
*diagnostic or guard* and must never share a column or a delta with the report metric.

Definition (common.py:287-314)
    model(lnp) -> log P, shape (B, N_Z, N_R), a log-softmax over the ring cells of the (z', r) grid
    log rho = log P - LOG_VOL                                       density, piecewise constant on the grid
    x_ik    = c_i + R_e u_ik,  u_ik uniform in the detector cell     K points per hit, in the shower frame
    g_i     = (1/K) sum_k rho(x_ik)                                 look-up; ``locate`` clamps to the border cells
    M_e     = model mass with z' < zb_e                             flat cut, linear inside the z-bin of zb_e
    NLL_e   = - sum_i w_i [ln g_i - ln M_e]

Random numbers
    One call ``torch.rand(H, K, 3, device=..., generator=gen)`` per batch - the call of ``common.fold`` - is the
    only consumer of the generator, whatever ``fold``, ``chunk`` and ``bulk`` are. Consequences:
    * two models scored with separate ``common.test_nll`` calls see identical points (each call re-creates the
      generator with seed 12345 and walks the same batches), and ``test_nll_multi`` reproduces those calls while
      drawing the points once;
    * chunked evaluation works on slices of the tensor drawn up front, so it uses the same points as the
      unchunked one. Drawing chunk by chunk would not: ``rand(H, K/2, 3)`` twice differs from ``rand(H, K, 3)``
      on CPU and on CUDA (checked). The price is that the offsets themselves (H*K*12 bytes) are held in memory;
      only the 5-6 times larger intermediate tensors are chunked.

Bitwise equality with ``common`` - what holds and what does not
    Everything up to and including ``w_i * (ln g_i - ln M_e)`` is bit-identical to ``common`` on the same device.
    The last step, ``index_add_`` (sum over the hits of an event), is *not run-to-run deterministic on CUDA*:
    two consecutive ``common.test_nll`` calls with the same model differ by up to ~2e-5 per event (float32
    rounding of the accumulation order; the mean moves by ~1e-7). Inside ``with deterministic():`` (PyTorch's
    deterministic algorithms) the reduction is reproducible and this module equals ``common`` bit for bit; on CPU
    it always does. ``torch.inference_mode`` and ``torch.no_grad`` give identical bits (checked).

Jensen bias of the K-point average
    ghat_i is unbiased for the cell average g_i, but ln is concave:
        E[-ln ghat_K] = -ln g + Var_cell(rho) / (2 K g^2) + O(K^-2)
    so the NLL is biased upwards by ~c/K with a model-dependent c (about 0.16-0.24 nat for the benchmark
    models; sharper maps have larger c). ``k_scan`` / ``jensen_fit`` measure it; ``fold="stratified"`` reduces it.

Contents
    draw, strata_shape          fold points (i.i.d. as in common, or jittered strata)
    event_nll, event_nll_multi  per-event NLL of one batch (one / several models on the same points)
    Parts                       exact split of the NLL into per-hit terms and the normalisation term
    test_nll, test_nll_multi    fixed-seed evaluation of a data set, mirroring ``common.test_nll``
    Bulk                        bulk-restricted likelihood (guard G-bulk)
    mass_monitors               model mass beyond zb, in the last ring, in the border z-bins (guard G-mass)
    k_scan, jensen_fit, jensen_bias   NLL(K) and its K -> infinity extrapolation
    deterministic               context manager for a reproducible per-event reduction
"""
from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch

from . import paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402  (shape-form-martin/common.py, the metric of record)

__all__ = ["Bulk", "Parts", "KFit", "draw", "strata_shape", "event_nll", "event_nll_multi", "test_nll",
           "test_nll_multi", "mass_monitors", "k_scan", "jensen_fit", "jensen_bias", "deterministic"]

FOLDS = ("iid", "stratified")


# ----------------------------------------------------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def deterministic():
    """Reproducible per-event sums: switches PyTorch's deterministic algorithms on (warn-only for operations
    without a deterministic implementation) and restores the previous setting on exit. A session that already
    runs deterministically is left as it is.

    Needed on CUDA because ``index_add_``, the last operation of ``common.event_nll``, accumulates in an
    unspecified order there. Inside this context ``common.test_nll`` and ``sfm.metric.test_nll`` are each
    reproducible and equal to each other bit for bit; outside they agree to ~2e-5 per event, which is also how
    far two runs of ``common.test_nll`` are from one another. Costs about 5 % of the evaluation time.
    """
    if torch.are_deterministic_algorithms_enabled():                            # already on (strict or warn-only): keep
        yield
        return
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False, warn_only=prev_warn)


# ----------------------------------------------------------------------------------------------------------------------
# Fold points
# ----------------------------------------------------------------------------------------------------------------------
def strata_shape(K: int, box) -> tuple:
    """Strata (n, n, m) of the detector cell used by ``fold="stratified"``: n*n*m = K, the same number in the two
    transverse cell directions, chosen to make the longest stratum edge (in mm) as short as possible.

    For the 48.6 x 48.6 x 20.5 mm cell: K=4 -> (2,2,1), 8 -> (2,2,2), 16 -> (4,4,1), 32 -> (4,4,2),
    64 -> (4,4,4), 256 -> (8,8,4), 1024 -> (16,16,4). A K without a square factor gives (1, 1, K): still unbiased,
    stratified along the layer direction only.
    """
    K = int(K)
    if K < 1:
        raise ValueError("K must be a positive integer")
    bx, by, bz = (float(v) for v in box)
    best = None
    for n in range(1, math.isqrt(K) + 1):
        if K % (n * n):
            continue
        m = K // (n * n)
        edges = sorted((bx / n, by / n, bz / m), reverse=True)
        if best is None or edges < best[0]:
            best = (edges, (n, n, m))
    return best[1]


def draw(b, box, K: int, gen=None, fold: str = "iid") -> torch.Tensor:
    """Offsets u (H, K, 3) of the K fold points of every hit, in the cell frame [mm]. The only consumer of ``gen``:
    one ``torch.rand(H, K, 3)`` call for either scheme.

    "iid"         u = (rand - 1/2) * box. The first line of ``common.fold``, verbatim.
    "stratified"  jittered strata: the cell is cut into K = n*n*m congruent sub-boxes S_k (``strata_shape``) and
                  point k is uniform in S_k:  u_k = ((index_k + rand_k) / (n, n, m) - 1/2) * box.

    Stratified points - what is guaranteed
      * Unbiased: E[ghat] = (1/K) sum_k E[rho(u_k)] = (1/K) sum_k (K/V) int_{S_k} rho = (1/V) int_cell rho = g,
        because every stratum has volume V/K. (A single point is uniform only in its own stratum; it is the
        equally weighted average over the K equal-volume strata that integrates over the whole cell.)
      * Never worse than i.i.d.: Var(ghat_strat) = (1/K^2) sum_k Var_{S_k}(rho) = (1/K) [mean within-stratum
        variance] <= (1/K) Var_cell(rho) = Var(ghat_iid); the between-strata part of the variance is removed.
      * Faster rate for this metric: rho is piecewise constant on the (z', r) grid, so Var_{S_k}(rho) = 0 for
        every stratum that does not touch a grid-cell boundary. With strata refined in all three directions the
        number cut by the boundary surfaces grows like K^(2/3), hence Var(ghat_strat) = O(K^(-4/3)), not O(1/K).
      * Jensen bias: E[-ln ghat] + ln g = Var(ghat)/(2 g^2) + higher orders, so to leading order the bias shrinks
        by the same factor as the variance at fixed K, and like K^(-4/3) instead of 1/K asymptotically.
    A stratified NLL is therefore systematically *lower* than the i.i.d. one at equal K. It is a different
    estimator of the same K -> infinity quantity and is not the metric of record.
    Size of the effect at small K: on the synthetic map of the tests, whose structure sits at the grid scale
    (the least favourable case), variance and bias fall to 0.86 of the i.i.d. values at K = 16 (strata 4x4x1)
    and to 0.69 at K = 64 (4x4x4). The gain on the real models has to be measured (audit D6).
    """
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}, got {fold!r}")
    if fold == "iid":
        return (torch.rand(len(b.xyz), K, 3, device=b.xyz.device, generator=gen) - 0.5) * box
    n, _, m = strata_shape(K, box.tolist())
    dev = b.xyz.device
    grid = torch.meshgrid(torch.arange(n, device=dev), torch.arange(n, device=dev), torch.arange(m, device=dev),
                          indexing="ij")
    index = torch.stack(grid, -1).reshape(K, 3).to(torch.float32)                 # stratum of point k
    shape = torch.tensor([n, n, m], device=dev, dtype=torch.float32)
    u = torch.rand(len(b.xyz), K, 3, device=dev, generator=gen)
    return u.add_(index).div_(shape).sub_(0.5).mul_(box)                         # ((index + rand)/shape - 1/2) * box


def _locate(b, Rh, u):
    """Rotate offsets into the shower frame and find their grid cells: the expressions of ``common.fold`` and
    ``common.event_nll``. Returns (z', iz, ir)."""
    pts = b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", Rh, u)
    z = pts[..., 2]
    iz, ir = C.locate(z, torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    return z, iz, ir


# ----------------------------------------------------------------------------------------------------------------------
# Bulk-restricted likelihood
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Bulk:
    """Bulk-restricted likelihood (guard G-bulk; headline of the theta track).

    Region. For event e
        A_e = { Z_MIN <= z' < min(z_cut, zb_e) }  x  { rings 0 .. N_R-1-drop_rings },
    a union of (parts of) grid cells that does not depend on the model. It leaves out the deep tail and the back
    face, the last ring (which also receives every point clamped from r >= R_MAX), and points in front of the
    grid (clamped into the first z-bin by ``locate``). No clamped look-up remains inside A_e.

    Definition. The K fold points of a hit are a K-point quadrature of the metric's own detector model ("the
    deposit is uniform in the cell"). The bulk metric keeps the quadrature points that lie in A_e, on the data
    side and on the model side alike:
        phi_i  = (1/K) #{k : x_ik in A_e}                       retained fraction of hit i
        gA_i   = (1/K) sum_{k : x_ik in A_e} rho(x_ik)          estimates (1/V) int_{cell_i n A_e} rho
        W_e    = sum_i w_i phi_i                                retained energy fraction of the event
        omega_i = w_i phi_i / W_e                               (renormalise=True; else omega_i = w_i phi_i)
        M_A,e  = model mass in A_e                              exact for the piecewise-constant density (linear
                                                                 in the z-bin that contains the upper limit)
        NLL_e  = - sum_i omega_i [ln gA_i - ln M_A,e]
    A hit without a point in A_e has omega_i = 0 and drops out (its term is defined as 0, the limit of
    phi ln phi). An event with W_e = 0 has no bulk likelihood: its NLL is NaN (with renormalise=True) and must be
    masked explicitly by the caller; ``Parts.retained`` holds W_e.

    Why this choice
      * "Hit centre inside" assigns the whole energy of a straddling cell to one side: a zeroth-order mismatch
        with the model integral over cell n A_e, and an infinite NLL whenever all K points of a retained hit fall
        outside (probability ~2^-K at a face, (3/4)^K at an edge).
      * "All fold points inside" makes the selection of hits depend on K and on the seed, and leaves the model
        mass of the boundary layer in M_A without any hit to pay for it: the score then responds at first order
        to the density in that layer (a model is rewarded for emptying it).
      * The fractional rule is exact for every cell that lies entirely inside or outside A_e, and for a
        straddling cell it is wrong only to the extent that the true energy is not uniform across the cell.

    In which sense this is a proper truncated likelihood
      Let the detector cells c_i tile the instrumented volume and q_i = int_{c_i n A_e} rho / M_A,e. Then
      sum_i q_i = 1 (when A_e is instrumented), and NLL_e = - sum_i omega_i ln q_i + const is the cross-entropy
      between the event's energy distribution over the pieces c_i n A_e and the model truncated to A_e and
      renormalised there. Exact statements: (1) A_e is model-independent and the same for hits and normaliser;
      (2) M_A,e is the exact grid mass of A_e; (3) the value is invariant under any change of the model outside
      A_e (mass beyond z_cut or zb, in the dropped rings), up to float rounding - the back-face exploit of a
      theta-aware model has no handle; (4) for a model that vanishes outside A_e and hits whose cells lie inside
      A_e (with zb_e >= z_cut) it equals the standard NLL on the same points.
      Approximations: (a) energy of cells cut by the boundary of A_e is split by volume fraction; (b) phi_i and
      the cell integral come from the same K points: on top of the usual Jensen bias this adds a noise term and
      an O(1/K) offset on straddling hits (E[phi^ ln phi^] != phi ln phi) that is *the same for every model* and
      cancels in paired differences on common points, which is the only intended use; (c) the parts of A_e
      outside the instrumented volume (beyond the tilted faces) count in M_A,e, exactly as in the standard
      metric - acceptance is the model's job.

    Events whose back face lies inside the nominal region (zb_e < z_cut)
      The plan assumed min zb ~ 1221 mm > z_cut. The cache metadata shows zb down to ~10 mm in train and val
      (showers whose first hit layer is deep in the calorimeter). ``short="clip"`` (default) uses the upper limit
      min(z_cut, zb_e) for hits and normaliser alike, which keeps (1)-(3); ``short="raise"`` refuses such events.
      To restrict a guard to full-depth events, mask on ``Parts.z_limit == z_cut`` (same mask for both models).

    ``renormalise=False`` returns the bulk *part* W_e * NLL_e (omega_i = w_i phi_i), which is 0 instead of NaN for
    events without retained energy and weights events by their bulk energy fraction.
    """
    z_cut: float = 1150.0
    drop_rings: int = 1
    renormalise: bool = True
    short: str = "clip"

    def __post_init__(self):
        if not 0 <= int(self.drop_rings) < C.N_R:
            raise ValueError("drop_rings must be in [0, N_R)")
        if self.short not in ("clip", "raise"):
            raise ValueError("short must be 'clip' or 'raise'")
        if not C.Z_MIN < float(self.z_cut) <= C.Z_MAX:
            raise ValueError("z_cut must lie inside the grid")

    def z_limit(self, zb: torch.Tensor) -> torch.Tensor:
        """Per-event upper limit min(z_cut, zb_e) of the region (or an error with short='raise')."""
        if self.short == "raise" and bool((zb < self.z_cut).any()):
            raise ValueError(f"Bulk(short='raise'): {int((zb < self.z_cut).sum())} event(s) have zb < z_cut = "
                             f"{self.z_cut} mm (min zb = {float(zb.min()):.1f} mm)")
        return zb.clamp(max=float(self.z_cut))


def _as_bulk(bulk) -> Optional[Bulk]:
    if bulk is None or bulk is False:
        return None
    if bulk is True:
        return Bulk()
    if isinstance(bulk, Bulk):
        return bulk
    raise TypeError("bulk must be None, True or a Bulk instance")


def _mass_below(P_z: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Mass of a longitudinal profile P_z (B, N_Z) with z' < z (B,): ``common.event_nll`` lines 301-305."""
    cdf = torch.cat([P_z.new_zeros(len(P_z), 1), torch.cumsum(P_z, 1)], 1)
    t = ((z - C.Z_MIN) / C.DZ).clamp(0, C.N_Z)
    k = t.floor().long().clamp(max=C.N_Z - 1)
    ar = torch.arange(len(P_z), device=P_z.device)
    return cdf[ar, k] + P_z[ar, k] * (t - k)


# ----------------------------------------------------------------------------------------------------------------------
# Per-event NLL
# ----------------------------------------------------------------------------------------------------------------------
class Parts(NamedTuple):
    """Exact decomposition of the per-event NLL:  nll_e = weight_sum_e * log_mass_e - sum_{i in e} hit_i.

    hit         (H,)  omega_i * ln ghat_i   - histogram differences of this in (z', r) for the "where" question
    log_mass    (B,)  ln M_e (standard) or ln M_A,e (bulk), with the clamp of ``common`` (1e-8)
    weight      (H,)  omega_i: b.w (standard) or the bulk weights
    weight_sum  (B,)  sum of omega_i per event (1 up to float rounding, or W_e with Bulk(renormalise=False))
    retained    (B,)  bulk only: W_e, the event's energy fraction inside A_e; None otherwise
    z_limit     (B,)  bulk only: upper z' limit of A_e; None otherwise

    For two models scored on the same points, weight, weight_sum, retained and z_limit are identical, so
        nll^a_e - nll^b_e = sum_i [hit^b_i - hit^a_i] + weight_sum_e (log_mass^a_e - log_mass^b_e)
    holds term by term (to float rounding): a hit term and a normalisation term.
    """
    hit: torch.Tensor
    log_mass: torch.Tensor
    weight: torch.Tensor
    weight_sum: torch.Tensor
    retained: Optional[torch.Tensor]
    z_limit: Optional[torch.Tensor]


def _segment_sum(values: torch.Tensor, ev: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, device=values.device, dtype=values.dtype).index_add_(0, ev, values)


def _masked_logsumexp(v: torch.Tensor, mask: torch.Tensor):
    """logsumexp over dim 1 of the entries where ``mask`` is True. Returns (value, any): rows without a True entry
    get a finite dummy value and any=False (the caller gives them zero weight); no -inf or NaN is produced, also
    not in the gradient."""
    has = mask.any(1)
    v = torch.where(mask, v, torch.full_like(v, -math.inf))
    v = torch.where(has[:, None], v, torch.zeros_like(v))
    return torch.logsumexp(v, 1), has


def event_nll_multi(models_or_logPs: Sequence, b, box, K: int = C.N_FOLD, gen=None, *, fold: str = "iid",
                    chunk: Optional[int] = None, bulk=None, return_parts: bool = False) -> list:
    """``event_nll`` for several models on the *same* fold points (drawn once). Returns one result per model, each
    the tuple ``event_nll`` would return. With the default options every element is what a separate
    ``common.event_nll(model, b, box, K, gen')`` returns when gen' is in the state ``gen`` had on entry."""
    K = int(K)
    if K < 1:
        raise ValueError("K must be a positive integer")
    if chunk is not None and int(chunk) < 1:
        raise ValueError("chunk must be None or a positive integer")
    region = _as_bulk(bulk)
    logPs = [m(b.lnp) if callable(m) else m for m in models_or_logPs]
    if not logPs:
        return []
    for logP in logPs:
        if logP.shape != (len(b.lnp), C.N_Z, C.N_R):
            raise ValueError(f"log P has shape {tuple(logP.shape)}, expected {(len(b.lnp), C.N_Z, C.N_R)}")
    B, H = len(b.lnp), len(b.xyz)
    logrhos = [logP - C.LOG_VOL for logP in logPs]
    u = draw(b, box, K, gen, fold)                                         # the only use of the generator
    Rh = b.R[b.ev]
    step = K if chunk is None else min(int(chunk), K)

    if region is None:
        if step == K:                                                      # common.event_nll, line by line
            _, iz, ir = _locate(b, Rh, u)
            ev = b.ev[:, None].expand_as(iz)
            loggs = [torch.logsumexp(lr[ev, iz, ir], 1) - math.log(K) for lr in logrhos]
        else:                                                              # same points, K-chunks, exact up to rounding
            partial = [[] for _ in logrhos]
            for a in range(0, K, step):
                _, iz, ir = _locate(b, Rh, u[:, a:a + step])
                ev = b.ev[:, None].expand_as(iz)
                for j, lr in enumerate(logrhos):
                    partial[j].append(torch.logsumexp(lr[ev, iz, ir], 1))
            loggs = [torch.logsumexp(torch.stack(p, 1), 1) - math.log(K) for p in partial]
        out = []
        for logP, logg in zip(logPs, loggs):
            Pz = logP.exp().sum(2)                                         # longitudinal profile
            log_mass = torch.log(_mass_below(Pz, b.zb).clamp_min(1e-8))
            ll = logg - log_mass[b.ev]
            nll = -torch.zeros(B, device=ll.device).index_add_(0, b.ev, b.w * ll)
            if return_parts:
                out.append((nll, logP, Parts(b.w * logg, log_mass, b.w, _segment_sum(b.w, b.ev, B), None, None)))
            else:
                out.append((nll, logP))
        return out

    # ---- bulk-restricted -------------------------------------------------------------------------------------------
    n_keep = C.N_R - int(region.drop_rings)
    z_lim = region.z_limit(b.zb)                                           # (B,)
    z_hit = z_lim[b.ev][:, None]
    n_in = torch.zeros(H, device=u.device)
    partial, flags = [[] for _ in logrhos], []
    for a in range(0, K, step):
        z, iz, ir = _locate(b, Rh, u[:, a:a + step])
        inside = (z >= C.Z_MIN) & (z < z_hit) & (ir < n_keep)
        n_in = n_in + inside.sum(1)
        flags.append(inside.any(1))
        ev = b.ev[:, None].expand_as(iz)
        for j, lr in enumerate(logrhos):
            partial[j].append(_masked_logsumexp(lr[ev, iz, ir], inside)[0])
    flags = torch.stack(flags, 1)                                          # (H, chunks): chunk has a point inside
    phi = n_in / K
    w_phi = b.w * phi
    retained = _segment_sum(w_phi, b.ev, B)                                # W_e
    if region.renormalise:
        weight = w_phi / retained.clamp_min(torch.finfo(w_phi.dtype).tiny)[b.ev]
    else:
        weight = w_phi
    weight_sum = _segment_sum(weight, b.ev, B)
    out = []
    for logP, p in zip(logPs, partial):
        logg = _masked_logsumexp(torch.stack(p, 1), flags)[0] - math.log(K)   # finite dummy where phi = 0 (weight 0)
        PzA = logP[:, :, :n_keep].exp().sum(2)
        log_mass = torch.log(_mass_below(PzA, z_lim).clamp_min(1e-8))
        nll = -torch.zeros(B, device=logg.device).index_add_(0, b.ev, weight * (logg - log_mass[b.ev]))
        if region.renormalise:                                             # no retained energy: undefined, not 0
            nll = torch.where(retained > 0, nll, torch.full_like(nll, math.nan))
        if return_parts:
            out.append((nll, logP, Parts(weight * logg, log_mass, weight, weight_sum, retained, z_lim)))
        else:
            out.append((nll, logP))
    return out


def event_nll(model_or_logP, b, box, K: int = C.N_FOLD, gen=None, *, fold: str = "iid", chunk: Optional[int] = None,
              bulk=None, return_parts: bool = False):
    """Per-event NLL of one batch; ``common.event_nll`` at the defaults.

    model_or_logP  a model (callable ``lnp -> log P``) or a precomputed ``log P`` tensor (B, N_Z, N_R)
    b, box, K, gen as in ``common.event_nll`` (``b`` from ``Showers.batch``; ``gen=None`` = global generator)
    fold           "iid" (common) or "stratified" (jittered strata, see ``draw``)
    chunk          evaluate the K points in slices of this size to bound memory; same points, the partial sums
                   are combined with ``logsumexp``, so the result equals the unchunked one up to rounding.
                   ``None`` (or >= K) is the single-pass code path of ``common``
    bulk           ``None`` | ``True`` | ``Bulk(...)``: bulk-restricted likelihood (see ``Bulk``)
    return_parts   also return ``Parts`` (per-hit ``w_i ln g_i`` and per-event ``ln M_e``)

    Returns ``(nll, logP)`` like ``common.event_nll``, or ``(nll, logP, parts)``. Differentiable like the original.
    """
    return event_nll_multi([model_or_logP], b, box, K, gen, fold=fold, chunk=chunk, bulk=bulk,
                           return_parts=return_parts)[0]


def test_nll_multi(models: Sequence, data, K: int = C.N_FOLD_EVAL, seed: int = 12345, batch: int = 128, *,
                   return_parts: bool = False, deterministic_sum: bool = False, inference: bool = True, **kw):
    """Fixed-seed per-event NLL of several models on identical fold points.

    Generator ``torch.Generator(device=common.DEVICE).manual_seed(seed)`` and ``data.batches(batch)`` in storage
    order, as in ``common.test_nll``; the points of every batch are drawn once and used for all models. Row j of
    the result equals ``common.test_nll(models[j], data, K)`` (see the module docstring for "equals").

    Returns a float32 array (n_models, n_events), or with ``return_parts=True`` a dict of arrays:
        nll (M, N), hit (M, H_total), log_mass (M, N), weight (H_total,), weight_sum (N,),
        and for bulk evaluations retained (N,), z_limit (N,).
    Per-hit arrays are in the order of ``data.xyz`` / ``data.w``.

    deterministic_sum  run inside ``deterministic()`` (bit-reproducible per-event sums on CUDA)
    inference          ``torch.inference_mode()`` (default; same bits as ``no_grad``). Pass False for
                       ``torch.no_grad()`` if a model caches tensors during the forward pass that are later
                       needed by autograd.
    kw                 fold, chunk, bulk of ``event_nll``
    """
    models = list(models)
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.inference_mode() if inference else torch.no_grad())
        if deterministic_sum:
            stack.enter_context(deterministic())
        gen = torch.Generator(device=C.DEVICE).manual_seed(seed)
        nll = [[] for _ in models]
        parts = [[] for _ in models]
        for b in data.batches(batch):
            res = event_nll_multi(models, b, data.box, K, gen, return_parts=return_parts, **kw)
            for j, r in enumerate(res):
                nll[j].append(r[0])
                if return_parts:
                    parts[j].append(r[2])
        nll = torch.stack([torch.cat(x) for x in nll]).cpu().numpy()
        if not return_parts:
            return nll
        cat = lambda j, name: torch.cat([getattr(p, name) for p in parts[j]])        # noqa: E731
        out = {"nll": nll,
               "hit": torch.stack([cat(j, "hit") for j in range(len(models))]).cpu().numpy(),
               "log_mass": torch.stack([cat(j, "log_mass") for j in range(len(models))]).cpu().numpy(),
               "weight": cat(0, "weight").cpu().numpy(), "weight_sum": cat(0, "weight_sum").cpu().numpy()}
        if parts[0][0].retained is not None:
            out["retained"] = cat(0, "retained").cpu().numpy()
            out["z_limit"] = cat(0, "z_limit").cpu().numpy()
        return out


def test_nll(model, data, K: int = C.N_FOLD_EVAL, seed: int = 12345, batch: int = 128, **kw):
    """Per-event NLL with a fixed random seed; ``common.test_nll(model, data, K)`` at the defaults.
    Returns a float32 array (n_events,), or the dict of ``test_nll_multi`` (leading model axis kept) with
    ``return_parts=True``."""
    out = test_nll_multi([model], data, K, seed, batch, **kw)
    return out if isinstance(out, dict) else out[0]


test_nll.__test__ = False            # not a pytest test, whatever module imports it
test_nll_multi.__test__ = False


# ----------------------------------------------------------------------------------------------------------------------
# Mass monitors
# ----------------------------------------------------------------------------------------------------------------------
def mass_monitors(logP: torch.Tensor, b=None, *, zb: Optional[torch.Tensor] = None, bulk=None, masks=None) -> dict:
    """Per-event model masses that the NLL does not police (guard G-mass). float64 tensors of shape (B,).

    total         sum of exp(log P) (1 for a normalised map)
    beyond_zb     mass with z' >= zb_e: outside the normaliser M_e, yet looked up by hits behind the flat cut
    inside_zb     M_e itself (total - beyond_zb)
    last_ring     mass in ring N_R-1 (also the density of every point clamped from r >= R_MAX)
    first_z, last_z   mass in the first / last z-bin (density of points clamped from outside the grid)
    outside_bulk  mass outside the region A_e of ``bulk`` (default ``Bulk()``)
    mask:<name>   mass in the cells where ``masks[name]`` (bool, (N_Z, N_R)) is True, e.g. the cells with
                  acceptance Abar <= 0.01

    ``zb`` defaults to ``b.zb``; without either, the zb-dependent entries are omitted and the bulk region uses
    z_cut for every event. Small masses are summed directly (never as 1 - large), so values of 1e-12 are meaningful.
    """
    P = logP.detach().double().exp()
    Pz = P.sum(2)
    out = {"total": Pz.sum(1), "last_ring": P[:, :, -1].sum(1), "first_z": Pz[:, 0], "last_z": Pz[:, -1]}
    zb = b.zb if (zb is None and b is not None) else zb

    def split(prof, z):                                                    # (mass below z, mass at or above z)
        t = ((z.double() - C.Z_MIN) / C.DZ).clamp(0, C.N_Z)
        k = t.floor().long().clamp(max=C.N_Z - 1)
        ar = torch.arange(len(prof), device=prof.device)
        zero = prof.new_zeros(len(prof), 1)
        head = torch.cat([zero, torch.cumsum(prof, 1)], 1)
        tail = torch.cat([torch.flip(torch.cumsum(torch.flip(prof, [1]), 1), [1]), zero], 1)
        f = t - k
        return head[ar, k] + prof[ar, k] * f, tail[ar, k + 1] + prof[ar, k] * (1 - f)

    if zb is not None:
        out["inside_zb"], out["beyond_zb"] = split(Pz, zb)
    region = _as_bulk(True if bulk is None else bulk)
    n_keep = C.N_R - int(region.drop_rings)
    z_lim = region.z_limit(zb) if zb is not None else torch.full((len(P),), float(region.z_cut), device=P.device)
    out["outside_bulk"] = split(P[:, :, :n_keep].sum(2), z_lim)[1] + P[:, :, n_keep:].sum((1, 2))
    for name, mask in (masks or {}).items():
        mask = torch.as_tensor(mask, device=P.device)
        if mask.shape != (C.N_Z, C.N_R) or mask.dtype != torch.bool:
            raise ValueError(f"masks[{name!r}] must be a bool tensor of shape {(C.N_Z, C.N_R)}")
        out[f"mask:{name}"] = (P * mask).sum((1, 2))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# K dependence and the Jensen bias
# ----------------------------------------------------------------------------------------------------------------------
def k_scan(models: Sequence, data, Ks: Sequence[int] = (16, 64, 256, 1024), seeds: Sequence[int] = (12345, 12346, 12347),
           batch: int = 128, chunk: Optional[int] = 64, **kw) -> np.ndarray:
    """Per-event NLL at several K and several fold seeds: float64 array (n_models, len(Ks), len(seeds), n_events).

    All models share the points of each (K, seed), so differences between models are paired. With the default
    seed 12345, batch 128 and chunk >= 64 the K = 64 slice is the report metric, K = 16 the training-time metric.
    ``chunk`` bounds memory at large K (the offsets, H*K*12 bytes per batch, remain: lower ``batch`` if needed -
    that changes the points, not the estimator). ``kw``: fold, bulk, deterministic_sum, inference.
    """
    models = list(models)
    out = np.empty((len(models), len(Ks), len(seeds), len(data)), dtype=np.float64)
    for i, K in enumerate(Ks):
        for j, seed in enumerate(seeds):
            out[:, i, j] = test_nll_multi(models, data, int(K), int(seed), batch, chunk=chunk, **kw)
    return out


@dataclass(frozen=True)
class KFit:
    Ks: tuple
    powers: tuple
    mean: np.ndarray           # (len(Ks),) mean over events and seeds
    mean_se: np.ndarray        # (len(Ks),) fold-noise standard error of `mean` (not the event-sampling error)
    value_inf: float           # extrapolation K -> infinity
    value_inf_se: float
    coef: np.ndarray           # coefficients c_p of K^-p, in the order of `powers`
    coef_se: np.ndarray
    cov: np.ndarray            # covariance of (value_inf, coef...)
    chi2: float
    dof: int

    def bias(self, K: float) -> float:
        """Fitted bias at K: sum_p c_p K^-p (what the K-point value exceeds the K -> infinity value by)."""
        return float(sum(c * float(K) ** -p for c, p in zip(self.coef, self.powers)))

    def bias_se(self, K: float) -> float:
        x = np.array([float(K) ** -p for p in self.powers])
        return float(np.sqrt(x @ self.cov[1:, 1:] @ x))


def jensen_fit(values, Ks: Sequence[int], powers: Sequence[float] = (1.0,)) -> KFit:
    """Extrapolate a per-event quantity measured at several K to K -> infinity.

    ``values``: array (len(Ks), n_seeds, n_events) - one model's NLL from ``k_scan``, or the difference of two
    models' NLLs (then the result is the K dependence of their gap). At least two seeds.

    Model:      mean_K = v_inf + sum_p c_p K^-p,  default p = 1: the leading Jensen term
                c_1 = mean_e sum_i w_i Var_cell_i(rho) / (2 g_i^2). Use powers=(1, 2) with four or more K to
                absorb the next order (it matters at K = 16 for sharp maps), or (4/3,) for stratified points.
    Data:       mean_K = mean over events and seeds of values[K].
    Errors:     only the fold noise is random here - the events are fixed, and their sampling error is common to
                all K and cancels in the K dependence. Fold points of different events and seeds are
                independent, so with s2_e(K) = sample variance over seeds of values[K, :, e]
                    Var(mean_K) = sum_e s2_e(K) / (N^2 n_seeds).
    Fit:        weighted least squares with weights 1/Var(mean_K); the covariance is (X' W X)^-1 (errors
                propagated from the fold noise, not rescaled by chi2). ``chi2``/``dof`` tests the K^-p model.

    ``value_inf_se`` is therefore the uncertainty of the extrapolated value *for these events*; the usual
    event-sampling SE (``stats.paired``) is separate and much larger for a single model.
    """
    y = np.asarray(values, dtype=np.float64)
    Ks = tuple(int(k) for k in Ks)
    powers = tuple(float(p) for p in powers)
    if y.ndim != 3 or y.shape[0] != len(Ks):
        raise ValueError(f"values: expected shape (len(Ks)={len(Ks)}, n_seeds, n_events), got {y.shape}")
    if y.shape[1] < 2:
        raise ValueError("jensen_fit: at least two fold seeds are needed to estimate the fold noise")
    if not np.isfinite(y).all():
        raise ValueError("jensen_fit: non-finite values (mask events without a defined NLL first)")
    if len(Ks) < len(powers) + 1 or len(set(Ks)) != len(Ks) or any(p <= 0 for p in powers):
        raise ValueError("jensen_fit: need distinct K values, more of them than powers, and positive powers")
    n_seeds, n_ev = y.shape[1], y.shape[2]
    mean = y.mean((1, 2))
    var = y.var(axis=1, ddof=1).sum(1) / (n_ev ** 2 * n_seeds)
    if not (var > 0).all():
        raise ValueError("jensen_fit: zero fold noise at some K (identical values for all seeds)")
    X = np.stack([np.ones(len(Ks))] + [np.array(Ks, dtype=np.float64) ** -p for p in powers], 1)
    cov = np.linalg.inv(X.T @ (X / var[:, None]))
    beta = cov @ (X.T @ (mean / var))
    resid = mean - X @ beta
    se = np.sqrt(np.diag(cov))
    return KFit(Ks=Ks, powers=powers, mean=mean, mean_se=np.sqrt(var), value_inf=float(beta[0]),
                value_inf_se=float(se[0]), coef=beta[1:], coef_se=se[1:], cov=cov,
                chi2=float((resid ** 2 / var).sum()), dof=len(Ks) - X.shape[1])


def jensen_bias(model, data, Ks: Sequence[int] = (16, 64, 256, 1024), seeds: Sequence[int] = (12345, 12346, 12347),
                powers: Sequence[float] = (1.0,), batch: int = 128, chunk: Optional[int] = 64, **kw) -> KFit:
    """Empirical Jensen bias of one model on given events: ``k_scan`` followed by ``jensen_fit``.

    ``fit.value_inf`` +- ``fit.value_inf_se`` is the NLL the model would have with exact cell averages,
    ``fit.bias(64)`` what the report metric (K = 64, i.i.d.) adds to it, ``fit.coef[0]`` the constant c of c/K.
    """
    return jensen_fit(k_scan([model], data, Ks, seeds, batch, chunk, **kw)[0], Ks, powers)
