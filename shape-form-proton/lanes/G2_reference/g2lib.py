"""g2lib - helpers of lane G2_reference (no side effects at import).

setup_both(batch)      one common namespace C in which the likelihood can be switched between the report's standard one and the
                       3D slab one (use("std") / use("slab")); identical to shapelab.setup(slab=False/True) respectively.
acceptance(...)        the fixed, parameter-free acceptance maps used by the acceptance-aware models.
build(...)             model factory for the names used in this lane.
"""
import math
import os
import sys

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
LANE = "/home/user/tamas/shape-form-proton/lanes/G2_reference"
sys.path.insert(0, LANE)

import numpy as np
import torch

import shapelab

OUT = {"std": LANE + "/results_std", "slab": LANE + "/results_slab"}


def setup_both(batch=256, lik="std"):
    """(C, geo, data, use): C with both likelihoods; use(lik) switches event_nll / C['SLAB'] / C['OUT_DIR'].

    The slab event_nll is created exactly as shapelab.setup(slab=True) does (exec of SLAB_NLL_SRC in C); the standard one is the
    untouched 00_common function. test_nll / fit / fit_fast look `event_nll` up in C at call time, so switching is exact."""
    C, geo, data = shapelab.setup(batch=batch, out_dir=OUT["std"], slab=False)
    fn = {"std": C["event_nll"]}
    exec(shapelab.SLAB_NLL_SRC, C)
    fn["slab"] = C["event_nll"]

    def use(which):
        C["event_nll"] = fn[which]
        C["SLAB"] = which == "slab"
        C["OUT_DIR"] = OUT[which]
        return which

    use(lik)
    return C, geo, data, use


P_EDGES_LNP = [math.log(0.2)] + [math.log(float(k)) for k in range(1, 20)] + [math.log(20.5)]     # the 20 one-GeV momentum bins


@torch.no_grad()
def mean_acceptance_binned(C, sh, edges_lnp=P_EDGES_LNP, nsub=4, front=True, back=False):
    """Abar_k(z', r): event average of the azimuthal slab acceptance per momentum bin (training events) -> (n_bins, N_Z, N_R)."""
    dev = C["DEVICE"]
    edges = torch.tensor(edges_lnp, dtype=torch.float32, device=dev)
    nb = len(edges) - 1
    tot = torch.zeros(nb, C["N_Z"], C["N_R"], device=dev)
    cnt = torch.zeros(nb, device=dev)
    for b in sh.batches(256):
        A = C["event_slab"](b, sh.box, nsub, front, back)
        idx = torch.bucketize(b.lnp.contiguous(), edges[1:-1].contiguous())
        tot.index_add_(0, idx, A)
        cnt.index_add_(0, idx, torch.ones_like(b.lnp))
    return tot / cnt.clamp_min(1)[:, None, None], edges, cnt


@torch.no_grad()
def representative_acceptance(C, sh, K=64, nsub=4, front=True, back=False):
    """A_k(z', r) of K representative training events: the events at the (k + 1/2)/K quantiles of the incidence angle."""
    cos_t = sh.ev["R"][:, 2, 2]
    order = torch.argsort(cos_t, descending=True)                      # increasing theta
    pos = ((torch.arange(K, device=order.device) + 0.5) / K * len(order)).long().clamp(max=len(order) - 1)
    b = sh.batch(order[pos])
    return C["event_slab"](b, sh.box, nsub, front, back)


def acceptance(C, data, kind):
    """kind: none | front | both | frontP | bothP | margF | margB  ->  dict of constructor keyword arguments (fixed geometry)."""
    tr = data["train"]
    if kind == "none":
        return {}
    if kind in ("front", "both"):
        return {"A": shapelab.mean_acceptance(C, tr, front=True, back=(kind == "both"))}
    if kind in ("frontP", "bothP"):
        A, edges, _ = mean_acceptance_binned(C, tr, front=True, back=(kind == "bothP"))
        return {"A": A, "edges": edges}
    if kind in ("margF", "margB"):
        return {"A": representative_acceptance(C, tr, K=64, front=True, back=(kind == "margB"))}
    raise ValueError(kind)


def build(C, kind, degree, c, s, acc_kw=None):
    """Model of this lane for the acceptance kind (see `acceptance`)."""
    import fam
    kw = dict(degree=degree, lnp0=c, lnp_s=s, **(acc_kw or {}))
    cls = {"none": fam.GFlashRef, "front": fam.GFlashAcc, "both": fam.GFlashAcc, "frontP": fam.GFlashAccP, "bothP": fam.GFlashAccP,
           "margF": fam.GFlashSlabMarg, "margB": fam.GFlashSlabMarg}[kind]
    return cls(C, **kw).to(C["DEVICE"])


def ctor(kind, degree):
    """The exact constructor call that rebuilds the model before load_state_dict (acceptance maps are buffers)."""
    cls = {"none": "GFlashRef", "front": "GFlashAcc", "both": "GFlashAcc", "frontP": "GFlashAccP", "bothP": "GFlashAccP",
           "margF": "GFlashSlabMarg", "margB": "GFlashSlabMarg"}[kind]
    return f"fam.{cls}(C, degree={degree}, lnp0=c, lnp_s=s)"


def one_se(val):
    """val = {n: per-event validation NLL}. Returns (n_star, n_selected, {n: (delta, se)}) - the report's 1-SE rule."""
    n_star = min(val, key=lambda k: float(np.mean(val[k])))
    diff = {k: (shapelab.paired(val[k], val[n_star]) if k != n_star else (0.0, 0.0)) for k in val}
    ok = [k for k in val if k == n_star or diff[k][0] < diff[k][1]]
    return n_star, min(ok), diff
