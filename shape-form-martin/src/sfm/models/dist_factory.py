"""Track A (M2A) model factories: the ``component`` and ``mixture`` factories of AUDIT.md section 4.1, the two extra
acceptance templates ``back_corrected`` / ``full_box`` and the last-ring wrapper of rung A12.

Importing this module registers two factories in ``sfm.models.FACTORIES`` (thin wrappers, no new mathematics):

    component(variant, provider, shower_long, shower_long_kw, shower_lat, far, det_halo, template, start, ...)
        -> sfm.families.build_variant(variant, core, logA, ...) with core, logA = load_legacy_core()
    mixture(K, long, long_kw, lat, depth, order, fine, template, provider, target, ...)
        -> sfm.families.MixtureModel(...) with a PolyProvider (degree 3; 0 for the c{k}_lat_* shape rows)

Providers (AUDIT 4.1 "Providers")
    "poly"   PolyProvider: legacy degrees; new depth-profile rows degree 3, new radial-shape rows degree 0 (the rule of
             ``build_variant``), ``degrees`` overrides single rows (A6b: sh_rk degree 0). ``start`` = a finished run
             ("<stage>/<name>") whose provider coefficients are copied BY NAME into the new PolyProvider; rows that
             do not exist there keep the family defaults.
    "net"    ParamNet(names_net = momentum-dependent names minus radial-shape names, names_const = legacy-constant
             names + radial-shape names, width 16, depth 2, tanh, v0/vs of the polynomial provider), fit_to(the
             polynomial provider of ``net_from`` (a finished poly run; None = the model's own legacy provider),
             default_lnp(48), steps=2000, lbfgs_iter=500), then ``model.set_provider(net)``.
             With ``net_from`` the whole fitted poly model is rebuilt first (``load_run``), so trainable scalars that
             live outside the provider (the detector-halo L and R of A9) start at their fitted values as well.
             Literal-reading note: the audit lists the radial shape names under names_const while they also belong to
             ``model.momentum_dependent``; a name cannot be both, so they are removed from names_net.
             For the MixtureModel, which has no legacy split, names_net = rows of polynomial degree > 0 and
             names_const = the c{k}_lat_* rows (degree 0 in the poly version).

Initialisations (run only when the context does not ask to skip them, i.e. not when weights are loaded right after)
    init="shower_long"  ``init_shower_long(model, reference=load_run(init_ref))`` with its defaults (A4, A5, A4t, A5h,
                        A6*): KL matching of the new shower depth profile to the reference's depth marginal
    init="target"       mixture only: ``init_from_target(model, target=ref.core_map, names=model.names,
                        model_fn=model.core_map, steps=init_steps, n_search=init_search)`` with ref = load_run(init_ref)
    init="none"         nothing beyond the coefficient copy
    In every case ``model.init_info`` records ``kl_map`` = mean over ``default_lnp(48)`` of KL(P_ref || P_model), the
    reference being the model the new one was started from (the start / init / net_from run, or the legacy formula
    itself for runs that start from the legacy coefficients), plus the KL trail of the initialiser.

Acceptance templates (``acceptance_template``)
    "stored", "front", "front_back"  -> ``sfm.models.field.acceptance_template`` (existing files)
    "back_corrected" = mean_e A_front^back / mean_e 1[z' < zb_e]
    "full_box"       = mean_e A_front^back^side / mean_e 1[z' < zb_e]
    (AUDIT 4.1 "Template definitions"), event means over the TRAINING events. ``compute_templates`` evaluates them
    with ``acceptances`` / ``exit_depth``, ported from ``results/m1_audit/checks/c_side.py::acceptances`` and
    ``::exit_depth`` (M1-d reviewer code; geometry: box |X|, |Y| <= 54 cells = 2624.4 mm, front face half a layer in
    front of the first-layer centre, back face at z_back, straight line from the global origin; 2 x 2 sub-points per
    cell and 24 azimuth nodes). 1[z' < zb_e] is averaged over the same two z sub-points per cell. The results are
    stored once per machine (``scripts/run_m2a.py templates``) in ``template_dir()`` =
    ``OUT_RESULTS/machines/<machine id>/runs/m2a_dist/templates/`` (pristine-clone rule: under the output root, never
    in the clone) and read from there. ``set_template_dir(path)`` points the factory elsewhere (e.g. to a machine
    directory copied back to the laptop, to rebuild an A3c / A3x / A9 model there).

ParamNet warm start: ``make_paramnet`` calls ``ParamNet.fit_to`` directly. The replica of fit_to that worked around
the constant-column bug (a column constant in momentum, e.g. logit_dh of A9n, received a weight 1/(float rounding))
was REMOVED on 25 Nov 2026 when fit_to itself got the scale floor (``sfm.models.paramnet.ParamNet.fit_to``); the
fit report records the constant columns (``init_info["fit_to"]["constant_columns"]``).

Last-ring wrapper (rung A12): ``LastRing(base)`` adds one trainable scalar (start 0) to the log-probabilities of
ring 31 and renormalises: log P' = log_softmax(log P + d 1[ring = 31]).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths
from . import FACTORIES, BuildContext, load_run, register
from .field import C as _C
from .field import acceptance_template as _field_acceptance_template
from .paramnet import ParamNet

__all__ = ["template_dir", "theta_file", "set_template_dir", "NEW_TEMPLATES", "acceptance_template", "acceptances",
           "exit_depth", "compute_templates", "LastRing", "radial_shape_names", "net_name_split", "make_paramnet",
           "copy_coefficients_by_name", "kl_map", "legacy_core"]

STAGE = "m2a_dist"
NEW_TEMPLATES = ("back_corrected", "full_box")
_TEMPLATE_DIR_OVERRIDE: Optional[Path] = None


def set_template_dir(path) -> None:
    """Read (and write) the Track A templates in ``path`` instead of this machine's directory (None: reset)."""
    global _TEMPLATE_DIR_OVERRIDE
    _TEMPLATE_DIR_OVERRIDE = Path(path) if path is not None else None


def template_dir() -> Path:
    """OUT_RESULTS/machines/<live machine id>/runs/m2a_dist/templates (or the ``set_template_dir`` override)."""
    if _TEMPLATE_DIR_OVERRIDE is not None:
        return _TEMPLATE_DIR_OVERRIDE
    from ..machine import machine_dir_live              # the machine directory comes from the API, never built here
    return Path(machine_dir_live()) / "runs" / STAGE / "templates"


def theta_file() -> Path:
    """The training events' theta (for the detector-frame halo), next to the templates."""
    return template_dir() / "train_theta.npy"


def __getattr__(name):                     # PEP 562: the former module constants, now per machine
    if name == "TEMPLATE_DIR":
        return template_dir()
    if name == "THETA_FILE":
        return theta_file()
    raise AttributeError(name)

# geometry of results/m1_audit/checks/c_side.py
Z_BACK, X_E, LAYER = 4882.15, 54 * 48.6, 20.5
NSUB, NPHI = 2, 24


# ----------------------------------------------------------------------------------------------------------------------
# geometry (ported from results/m1_audit/checks/c_side.py)
# ----------------------------------------------------------------------------------------------------------------------
def exit_depth(theta, phi, z0):
    """Depth along the shower axis at which the axis crosses |X| = X_E or |Y| = X_E (inf if never).
    Port of ``results/m1_audit/checks/c_side.py::exit_depth`` (unchanged). Returns (z_exit, X0, Y0)."""
    theta, phi, z0 = (np.asarray(a, dtype=np.float64) for a in (theta, phi, z0))
    rho0 = z0 * np.tan(theta)
    X0, Y0 = rho0 * np.cos(phi), rho0 * np.sin(phi)
    cx, cy = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(np.abs(cx) > 1e-12, (X_E - np.sign(cx) * X0) / np.abs(cx), np.inf)
        ty = np.where(np.abs(cy) > 1e-12, (X_E - np.sign(cy) * Y0) / np.abs(cy), np.inf)
    return np.minimum(tx, ty), X0, Y0


@torch.no_grad()
def acceptances(b, X0, Y0, zbcos, nsub: int = NSUB, nphi: int = NPHI) -> dict:
    """dict of A (B, N_Z, N_R): 'front', 'fb', 'side', 'all', 'front_side', and 'below_zb' (the fraction of the
    cell's z sub-points with z' < zb_e; no azimuth dependence).

    Port of ``results/m1_audit/checks/c_side.py::acceptances`` with the device taken from the batch instead of a
    module constant, ``nsub`` / ``nphi`` as arguments and the extra 'below_zb' entry. ``b`` is a
    ``Showers.batch`` (needs R, lnp, zb), X0, Y0 the transverse position of the shower origin in the global frame,
    zbcos = zb cos(theta) the depth of the back face below the shower origin (all tensors (B,))."""
    dev = b.R.device
    ze, re, dz = _C.Z_EDGES.to(dev), _C.R_EDGES.to(dev), float(_C.DZ)
    k = (torch.arange(nsub, device=dev, dtype=torch.float32) + 0.5) / nsub
    zs = (ze[:-1, None] + dz * k).reshape(-1)                                                   # (N_Z * nsub,)
    rs = torch.sqrt(re[:-1, None] ** 2 + (re[1:, None] ** 2 - re[:-1, None] ** 2) * k).reshape(-1)
    al = (torch.arange(nphi, device=dev, dtype=torch.float32) + 0.5) / nphi * 2 * math.pi
    R = b.R                                                                                     # global -> shower
    ca, sa = torch.cos(al), torch.sin(al)
    B = len(b.lnp)
    gx = (R[:, 0, 0, None, None, None] * rs[None, None, :, None] * ca + R[:, 1, 0, None, None, None] * rs[None, None, :, None] * sa
          + R[:, 2, 0, None, None, None] * zs[None, :, None, None])                             # (B, Z, Rr, A)
    gy = (R[:, 0, 1, None, None, None] * rs[None, None, :, None] * ca + R[:, 1, 1, None, None, None] * rs[None, None, :, None] * sa
          + R[:, 2, 1, None, None, None] * zs[None, :, None, None])
    gz = (R[:, 0, 2, None, None, None] * rs[None, None, :, None] * ca + R[:, 1, 2, None, None, None] * rs[None, None, :, None] * sa
          + R[:, 2, 2, None, None, None] * zs[None, :, None, None])
    front = gz >= -LAYER / 2
    back = gz <= zbcos[:, None, None, None]
    side = ((X0[:, None, None, None] + gx).abs() <= X_E) & ((Y0[:, None, None, None] + gy).abs() <= X_E)

    def red(m):
        return m.float().mean(3).reshape(B, _C.N_Z, nsub, _C.N_R, nsub).mean((2, 4))
    out = {"front": red(front), "fb": red(front & back), "side": red(side), "all": red(front & back & side),
           "front_side": red(front & side)}
    below = (zs[None, :] < b.zb[:, None]).float().reshape(B, _C.N_Z, nsub).mean(2)            # (B, N_Z)
    out["below_zb"] = below[:, :, None].expand(B, _C.N_Z, _C.N_R)
    return out


@torch.no_grad()
def compute_templates(train, theta, phi, device=None, batch: int = 32, log=print) -> dict:
    """Event means over ``train`` (a ``common.Showers`` of the TRAINING split; theta, phi its per-event angles) of the
    acceptances above, and the two new templates. Returns numpy float64 arrays (N_Z, N_R) keyed by
    'front', 'fb', 'side', 'all', 'front_side', 'below_zb', 'back_corrected', 'full_box'."""
    device = device or _C.DEVICE
    zb = train.ev["zb"].detach().cpu().numpy().astype(np.float64)
    th, ph = np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    z0 = Z_BACK - zb * np.cos(th)
    _, X0, Y0 = exit_depth(th, ph, z0)
    sh = train.to(device)
    X0t = torch.tensor(X0, dtype=torch.float32, device=device)
    Y0t = torch.tensor(Y0, dtype=torch.float32, device=device)
    zbc = torch.tensor(zb * np.cos(th), dtype=torch.float32, device=device)
    names = ("front", "fb", "side", "all", "front_side", "below_zb")
    acc = {n: torch.zeros(_C.N_Z, _C.N_R, dtype=torch.float64, device=device) for n in names}
    N = len(sh)
    idx_all = torch.arange(N, device=device)
    for s in range(0, N, batch):
        idx = idx_all[s:s + batch]
        A = acceptances(sh.batch(idx), X0t[idx], Y0t[idx], zbc[idx])
        for n in names:
            acc[n] += A[n].double().sum(0)
        if log and (s // batch) % 500 == 0:
            log(f"  templates: {s}/{N} events")
    out = {n: (acc[n] / N).cpu().numpy() for n in names}
    den = out["below_zb"]
    if not (den > 0).all():
        raise RuntimeError("mean_e 1[z' < zb_e] is zero in some cells: the template ratio is undefined there")
    out["back_corrected"] = out["fb"] / den
    out["full_box"] = out["all"] / den
    return out


def acceptance_template(kind: str = "stored", device=None) -> torch.Tensor:
    """Acceptance template (N_Z, N_R): the existing kinds of ``sfm.models.field.acceptance_template`` and the two new
    ones (``NEW_TEMPLATES``), read from ``TEMPLATE_DIR`` (written by ``scripts/run_m2a.py templates``)."""
    device = device or _C.DEVICE
    if kind not in NEW_TEMPLATES:
        return _field_acceptance_template(kind, device)
    f = template_dir() / f"{kind}.pt"
    if not f.exists():
        raise FileNotFoundError(f"{paths.portable(f)} missing: run `uv run python -B shape-form-martin/scripts/run_m2a.py "
                                "templates`")
    A = torch.load(f, map_location=device, weights_only=False)
    if A.shape != (_C.N_Z, _C.N_R) or not torch.isfinite(A).all():
        raise ValueError(f"template {kind}: bad shape {tuple(A.shape)} or non-finite values")
    return A.float()


# ----------------------------------------------------------------------------------------------------------------------
# last-ring wrapper (A12)
# ----------------------------------------------------------------------------------------------------------------------
class LastRing(nn.Module):
    """log P' = log_softmax(log P_base + d 1[ring = N_R - 1]); d is one trainable scalar starting at ``init`` (0)."""

    def __init__(self, base: nn.Module, init: float = 0.0):
        super().__init__()
        self.base = base
        self.log_f_last = nn.Parameter(torch.tensor(float(init)))

    def forward(self, lnp):
        out = self.base(lnp)
        add = torch.zeros(out.shape[-1], device=out.device, dtype=out.dtype)
        add = add + F.one_hot(torch.tensor(out.shape[-1] - 1, device=out.device), out.shape[-1]).to(out.dtype) * self.log_f_last
        B = out.shape[0]
        return F.log_softmax((out + add).reshape(B, -1), 1).reshape(out.shape)

    def n_params(self) -> int:
        return int(self.base.n_params()) + 1

    @property
    def provider(self):
        return self.base.provider


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
_CORE = {}


def legacy_core():
    """(core, logA) of the final pion formula on the CPU (``sfm.families.load_legacy_core``), cached."""
    if "core" not in _CORE:
        from ..families import load_legacy_core
        _CORE["core"] = load_legacy_core("cpu")
    return _CORE["core"]


def radial_shape_names(model) -> list:
    """Raw names of the shape parameters of the radial families (shower cross-section of a ComponentModel; the
    c{k}_lat_* rows of a MixtureModel)."""
    from ..families.components import ComponentModel, MixtureModel
    base = model.base if isinstance(model, LastRing) else model
    if isinstance(base, ComponentModel):
        return [t.shape_prefix + n for t in base.sh_lat.terms for n, _ in t.family.SHAPES]
    if isinstance(base, MixtureModel):
        return [n for n in base.names if "_lat_" in n]
    raise TypeError(f"radial_shape_names: unsupported model {type(base).__name__}")


def net_name_split(model) -> tuple:
    """(names_net, names_const) of the ParamNet provider (module docstring)."""
    from ..families.components import ComponentModel, MixtureModel
    base = model.base if isinstance(model, LastRing) else model
    shapes = radial_shape_names(base)
    if isinstance(base, ComponentModel):
        net = [n for n in base.momentum_dependent if n not in shapes]
        const = list(base.legacy_constant) + [n for n in shapes if n not in base.legacy_constant]
    elif isinstance(base, MixtureModel):
        deg = base.provider.deg
        net = [n for n in base.names if deg[n] > 0 and n not in shapes]
        const = [n for n in base.names if n not in net]
    else:
        raise TypeError(f"net_name_split: unsupported model {type(base).__name__}")
    if sorted(net + const) != sorted(base.names):
        raise RuntimeError("net_name_split: the split does not cover the model's names exactly")
    return net, const


def make_paramnet(model, poly, lnp_grid=None, steps: int = 2000, lbfgs_iter: int = 500, fit: bool = True,
                  log=None, scale_floor="default") -> tuple:
    """ParamNet for ``model`` (names from ``net_name_split``), v0/vs/mass of ``poly``; fitted to ``poly`` unless
    ``fit`` is False. Returns (net, fit report or None). Seed the global RNG before calling."""
    from ..families.components import default_lnp
    names_net, names_const = net_name_split(model)
    dev = poly.coef.device
    net = ParamNet(names_net, names_const, width=16, depth=2, activation="tanh", v0=float(poly.v0),
                   vs=float(poly.vs), mass=float(poly.mass)).to(dev)
    with torch.no_grad():                                        # the exact float32 buffers of the provider
        net.v0.copy_(poly.v0)
        net.vs.copy_(poly.vs)
        net.mass.copy_(poly.mass)
    rep = None
    if fit:
        grid = default_lnp(48, device=dev) if lnp_grid is None else lnp_grid
        # fit_to itself handles columns that are constant in momentum (scale floor, sfm.models.paramnet); the replica
        # that worked around the earlier bug is gone (module docstring)
        sf = ParamNet.SCALE_FLOOR_ABS if scale_floor == "default" else scale_floor
        rep = net.fit_to(poly, grid, steps=steps, lbfgs_iter=lbfgs_iter, log=log, scale_floor=sf)
    return net, rep


@torch.no_grad()
def copy_coefficients_by_name(dst, src) -> list:
    """Copy the polynomial coefficients of every name present in both PolyProviders (up to the destination row's
    degree; higher source coefficients are dropped, missing ones are zero). Returns the names copied."""
    done = []
    for i, n in enumerate(dst.names):
        if n not in src.names:
            continue
        j = src.names.index(n)
        d = dst.deg[n]
        row = (src.coef[j] * src.mask[j]).to(dst.coef.device, dst.coef.dtype)
        dst.coef[i].zero_()
        m = min(d + 1, row.numel())
        dst.coef[i, :m] = row[:m]
        done.append(n)
    return done


@torch.no_grad()
def kl_map(reference, model, lnp=None) -> float:
    """Mean over ``default_lnp(48)`` of KL(P_reference || P_model) (full maps, acceptance included)."""
    from ..families.components import default_lnp, kl_div
    dev = next(iter(model.buffers())).device
    lnp = default_lnp(48, device=dev) if lnp is None else lnp
    was_r, was_m = reference.training, model.training
    reference.eval()
    model.eval()
    out = float(kl_div(reference(lnp), model(lnp)).mean())
    reference.train(was_r)
    model.train(was_m)
    return out


def _legacy_reference(device):
    """The legacy formula's map as a model (legacy_centre with the legacy coefficients and the stored template)."""
    from ..families import build_variant
    core, logA = legacy_core()
    return build_variant("legacy_centre", core, logA).to(device).eval()


def _poly_for_component(model, core, degrees: Optional[dict]):
    """PolyProvider copied from the legacy fit with the degree rule of ``build_variant`` plus overrides."""
    from ..families.params import PolyProvider
    deg = {n: (0 if "_lat" in n or n.startswith("lat") or n in ("logit_dh",) else int(core.degree))
           for n in model.new_names}
    for n, d in (degrees or {}).items():
        if n not in model.names:
            raise KeyError(f"degrees: '{n}' is not a raw parameter of this model")
        if n not in model.new_names:
            raise KeyError(f"degrees: '{n}' is a legacy row; its degree is fixed by the legacy fit")
        deg[n] = int(d)
    return PolyProvider.from_legacy(core, names=model.names, defaults=model.defaults, degrees=deg)


def _theta_train():
    f = theta_file()
    if not f.exists():
        raise FileNotFoundError(f"{paths.portable(f)} missing: run `uv run python -B shape-form-martin/scripts/run_m2a.py "
                                "templates` (stores the training events' theta for the detector-frame halo)")
    return np.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# factories
# ----------------------------------------------------------------------------------------------------------------------
@register("component")
def component(ctx: BuildContext, variant: str = "survival_exact", provider: str = "poly", shower_long="plateau",
              shower_long_kw: Optional[dict] = None, shower_lat=None, lat_family_kw: Optional[list] = None,
              far: bool = True, det_halo: bool = False, template: str = "stored", start: Optional[str] = None,
              degrees: Optional[dict] = None, coef_shift: Optional[dict] = None, logit_dh_from_far: bool = False,
              init: str = "none", init_ref: Optional[str] = None, net_from: Optional[str] = None,
              last_ring: bool = False, fit_to_steps: int = 2000, fit_to_lbfgs: int = 500,
              fit_to_scale_floor="default"):
    """ComponentModel of AUDIT 4.1 (module docstring). Keyword arguments beyond the audit's list:
    lat_family_kw      ``family_kw`` of ``legacy_shower_lateral`` (A7l), used with a tuple ``shower_lat``
    degrees            polynomial degrees of new rows (A6b: {"sh_rk": 0})
    coef_shift         {name: added to the constant coefficient after the start copy} (A7l: log_lam += ln 20)
    logit_dh_from_far  A9r: logit_dh = logit(mean over the training events of sigmoid(logit_far(p))) of ``start``
    init, init_ref     "none" | "shower_long" (reference run ``init_ref``, default ``start``)
    net_from           provider="net": the finished poly run to fit the ParamNet to (None = legacy provider)
    last_ring          wrap in ``LastRing`` (A12)
    fit_to_steps, fit_to_lbfgs   Adam steps / L-BFGS iterations of the ParamNet warm start (AUDIT 4.1: 2000 / 500;
                       other values only in the smoke mode of scripts/run_m2a.py, registered under another stage)
    fit_to_scale_floor "default" = ParamNet.SCALE_FLOOR_ABS (1e-2, fit_to version 3, user decision of 25 Nov 2026);
                       None = fit_to version 2 (used only to rebuild a warm start made before version 3)
    """
    from ..families import build_variant
    from ..families.components import init_shower_long
    from ..families.lateral import legacy_shower_lateral
    from ..families.halo_det import DetHalo
    dev = ctx.device
    info = {"provider": provider, "start": start, "init": init}

    if provider == "net" and net_from is not None:
        # the fitted poly model (all its trainable scalars), then the provider replaced by a ParamNet fitted to it
        model = load_run(net_from, BuildContext(device=dev, train=ctx.train, seed=ctx.seed, log=ctx.log,
                                                skip_init=True, results_dir=ctx.results_dir)).train()
        base = model.base if isinstance(model, LastRing) else model
        poly = base.provider
        net, rep = make_paramnet(base, poly, fit=not ctx.skip_init, log=None, steps=fit_to_steps,
                                 lbfgs_iter=fit_to_lbfgs, scale_floor=fit_to_scale_floor)
        ref = None
        if not ctx.skip_init:
            import copy
            ref = copy.deepcopy(model).eval()
        base.set_provider(net)
        info.update({"net_from": net_from, "fit_to": _fit_report(rep)})
        if ref is not None:
            info["kl_ref"] = net_from
            info["kl_map"] = kl_map(ref, model)
        model.init_info = info
        return model.to(dev)

    core, logA0 = legacy_core()
    A = acceptance_template(template, "cpu")
    logA = torch.log(A.clamp_min(1e-6)) if template != "stored" else logA0
    info["template"] = template
    if isinstance(shower_lat, (list, tuple)) and lat_family_kw is not None:
        shower_lat = legacy_shower_lateral(tuple(shower_lat), family_kw=[dict(k) for k in lat_family_kw])
    elif isinstance(shower_lat, list):
        shower_lat = tuple(shower_lat)
    halo = DetHalo.from_samples(_theta_train()) if det_halo else None
    kw = dict(shower_long=shower_long, shower_long_kw=shower_long_kw, shower_lat=shower_lat, far=far, det_halo=halo)
    # build the structure once without a provider to learn its names, then the provider with the degree rule
    from ..families.components import ComponentModel, VARIANTS
    from ..families.cellint import Grid
    from .. import legacy as _legacy
    cfg = dict(VARIANTS[variant])
    cfg.update(kw)
    grid = Grid.from_common(_legacy.C)
    shell = ComponentModel(None, logA, grid=grid, **cfg)
    poly = _poly_for_component(shell, core, degrees)
    model = build_variant(variant, core, logA, grid=grid, provider=poly, **kw)
    model = model.to(dev)
    poly = model.provider
    if provider not in ("poly", "net"):
        raise ValueError("provider must be 'poly' or 'net'")

    ref = None
    if not ctx.skip_init:
        if start is not None:
            src = load_run(start, BuildContext(device=dev, train=ctx.train, seed=ctx.seed, log=ctx.log,
                                               skip_init=True, results_dir=ctx.results_dir)).eval()
            src_base = src.base if isinstance(src, LastRing) else src
            info["copied"] = len(copy_coefficients_by_name(poly, src_base.provider))
            info["not_copied"] = [n for n in poly.names if n not in src_base.provider.names]
            ref = src
            if logit_dh_from_far:
                if ctx.train is None:
                    raise ValueError("logit_dh_from_far needs the training set in the build context")
                with torch.no_grad():
                    lnp = ctx.train.ev["lnp"].to(dev)
                    raw = src_base.provider(lnp)[:, src_base.provider.names.index("logit_far")]
                    wbar = float(torch.sigmoid(raw.double()).mean())
                    poly.coef[poly.index("logit_dh")].zero_()
                    poly.coef[poly.index("logit_dh"), 0] = math.log(wbar / (1 - wbar))
                info["logit_dh_init"] = {"mean_far_weight": wbar, "logit": math.log(wbar / (1 - wbar))}
        else:
            ref = _legacy_reference(dev)
        for n, v in (coef_shift or {}).items():
            with torch.no_grad():
                poly.coef[poly.index(n), 0] += float(v)
        info["coef_shift"] = coef_shift
        if init == "shower_long":
            rname = init_ref or start
            rmod = ref if rname == start else load_run(rname, BuildContext(device=dev, train=ctx.train, seed=ctx.seed,
                                                                             log=ctx.log, skip_init=True,
                                                                             results_dir=ctx.results_dir)).eval()
            res = init_shower_long(model, rmod)
            info["init_shower_long"] = res
        elif init != "none":
            raise ValueError("component: init must be 'none' or 'shower_long'")

    if last_ring:
        model = LastRing(model).to(dev)
    if provider == "net":                         # net_from is None: fit to the model's own (legacy) provider
        base = model.base if isinstance(model, LastRing) else model
        net, rep = make_paramnet(base, base.provider, fit=not ctx.skip_init, steps=fit_to_steps,
                                 lbfgs_iter=fit_to_lbfgs, scale_floor=fit_to_scale_floor)
        base.set_provider(net)
        info.update({"net_from": None, "fit_to": _fit_report(rep)})
    if ref is not None:
        info["kl_ref"] = start or "legacy formula (legacy_centre, legacy coefficients, stored template)"
        info["kl_map"] = kl_map(ref, model)
    model.init_info = info
    return model.to(dev)


@register("mixture")
def mixture(ctx: BuildContext, K: int = 3, long="emg", long_kw: Optional[dict] = None, lat="burr",
            lat_kw: Optional[dict] = None, depth: str = "linear", order: bool = True, fine: bool = True,
            template: str = "stored", provider: str = "poly", init: str = "target", init_ref: Optional[str] = None,
            init_steps: Sequence[int] = (1500, 1500), init_search: int = 400, net_from: Optional[str] = None,
            fit_to_steps: int = 2000, fit_to_lbfgs: int = 500, fit_to_scale_floor="default"):
    """MixtureModel of AUDIT 4.1 (A10 and its exploratory variants). The PolyProvider has degree 3 for every row and
    degree 0 for the c{k}_lat_* shape rows, starts at the family defaults (v0, vs, mass of the legacy fit), and is
    moved onto the map of ``init_ref`` by ``init_from_target`` on the core maps (before the acceptance)."""
    from ..families.components import MixtureModel, init_from_target
    from ..families.cellint import Grid
    from ..families.params import PolyProvider
    from .. import legacy as _legacy
    dev = ctx.device
    info = {"provider": provider, "init": init, "init_ref": init_ref, "template": template}
    if provider == "net":
        if net_from is None:
            raise ValueError("mixture with provider='net' needs net_from (the fitted poly run)")
        model = load_run(net_from, BuildContext(device=dev, train=ctx.train, seed=ctx.seed, log=ctx.log,
                                                skip_init=True, results_dir=ctx.results_dir)).train()
        poly = model.provider
        net, rep = make_paramnet(model, poly, fit=not ctx.skip_init, steps=fit_to_steps, lbfgs_iter=fit_to_lbfgs,
                                 scale_floor=fit_to_scale_floor)
        ref = None
        if not ctx.skip_init:
            import copy
            ref = copy.deepcopy(model).eval()
        model.set_provider(net)
        info.update({"net_from": net_from, "fit_to": _fit_report(rep)})
        if ref is not None:
            info["kl_ref"] = net_from
            info["kl_map"] = kl_map(ref, model)
        model.init_info = info
        return model.to(dev)

    core, logA0 = legacy_core()
    A = acceptance_template(template, "cpu")
    logA = torch.log(A.clamp_min(1e-6)) if template != "stored" else logA0
    mm = MixtureModel(None, K=int(K), long=long, lat=lat, logA=logA, order=order, depth=depth, long_kw=long_kw,
                      lat_kw=lat_kw, fine=fine, grid=Grid.from_common(_legacy.C))
    deg = {n: (0 if "_lat_" in n else 3) for n in mm.names}
    poly = PolyProvider(mm.names, deg, mm.defaults, v0=float(core.v0), vs=float(core.vs), mass=float(core.mass))
    with torch.no_grad():
        poly.v0.copy_(core.v0.detach().cpu())
        poly.vs.copy_(core.vs.detach().cpu())
        poly.mass.copy_(core.mass.detach().cpu())
    mm.set_provider(poly)
    mm = mm.to(dev)
    if not ctx.skip_init and init == "target":
        if init_ref is None:
            raise ValueError("mixture: init='target' needs init_ref (the finished run whose map is the target)")
        ref = load_run(init_ref, BuildContext(device=dev, train=ctx.train, seed=ctx.seed, log=ctx.log, skip_init=True,
                                              results_dir=ctx.results_dir)).eval()
        res = init_from_target(mm, target=ref.core_map, names=mm.names, model_fn=mm.core_map,
                               steps=tuple(int(s) for s in init_steps), n_search=int(init_search), seed=0)
        info["init_from_target"] = res
        info["kl_ref"] = init_ref
        info["kl_map"] = kl_map(ref, mm)
    elif init not in ("target", "none"):
        raise ValueError("mixture: init must be 'target' or 'none'")
    mm.init_info = info
    return mm


def _fit_report(rep):
    if rep is None:
        return None
    rms = rep["rms"]
    lb = rep.get("lbfgs") or {}
    return {"loss": rep["loss"], "rms_max": max(rms.values()) if rms else None,
            "rms_max_name": max(rms, key=rms.get) if rms else None,
            "constant_columns": rep.get("constant_columns"), "near_floor_columns": rep.get("near_floor_columns"),
            "scale_floor_rel": rep.get("scale_floor_rel"), "lbfgs_stopped": lb.get("stopped"),
            "fit_to": "sfm.models.paramnet.ParamNet.fit_to (scale floor; no replica)",
            "fit_to_version": rep.get("fit_to_version"), "scale_floor_abs": rep.get("scale_floor_abs"),
            "column_std": rep.get("column_std"), "below_scale_floor": rep.get("below_scale_floor"),
            "column_weight": rep.get("column_weight"), "rms": rms}
