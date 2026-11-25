"""sfm.models - model wrappers for the training harness, and the small factory registry used by ``scripts/fit.py``.

Every model maps ``lnp (B,) -> log P (B, N_Z, N_R)`` (a normalised map on the grid of ``common.py``), has
``n_params()`` and works in ``eval()`` under ``torch.inference_mode()``.

    paramnet   ParamProvider (contract), ParamNet, centring, kinetic
    field      FieldNet, field_model, mlp_std_start, FieldAcc, kl_warm_start, acceptance_template
    ensemble   Ensemble, count_params
    table      ResidualTable

Factory registry. A config names a factory and its keyword arguments::

    [model]
    factory = "field_acc"
    template = "front"
    warm_steps = 500

``build(name, ctx, **kwargs)`` calls the registered function with a ``BuildContext`` (device, training set, seed,
log, and ``skip_init`` - set when the weights are loaded from a checkpoint right after building, so expensive
initialisations such as warm starts can be skipped). New factories register with ``@register("name")``.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .. import paths
from .ensemble import Ensemble, count_params
from .field import C as _C
from .field import (TRAIN_P_MAX, TRAIN_P_MIN, FieldAcc, FieldNet, acceptance_template, default_lnp_grid, field_model,
                    kl_warm_start, mlp_std_start)
from .paramnet import ParamNet, ParamProvider, centring, is_param_provider, kinetic
from .table import ResidualTable

__all__ = ["BuildContext", "FACTORIES", "register", "build", "available", "load_run", "Ensemble", "count_params",
           "FieldAcc", "FieldNet", "acceptance_template", "default_lnp_grid", "field_model", "kl_warm_start",
           "mlp_std_start", "ParamNet", "ParamProvider", "centring", "is_param_provider", "kinetic", "ResidualTable"]


@dataclass
class BuildContext:
    device: Any = None
    train: Any = None                    # common.Showers of the training split (some factories need its ln p range)
    seed: int = 0
    log: Callable = print
    skip_init: bool = False
    results_dir: Path = dataclasses.field(default_factory=lambda: paths.RESULTS_DIR)

    def __post_init__(self):
        if self.device is None:
            self.device = _C.DEVICE


FACTORIES: dict = {}


def register(name: str):
    def deco(fn):
        if name in FACTORIES:
            raise ValueError(f"factory {name!r} registered twice")
        FACTORIES[name] = fn
        return fn
    return deco


def available() -> list:
    return sorted(FACTORIES)


def build(name: str, ctx: Optional[BuildContext] = None, **kwargs) -> nn.Module:
    """Build a model with the registered factory ``name``."""
    if name not in FACTORIES:
        raise KeyError(f"unknown model factory {name!r}; available: {available()}")
    ctx = ctx or BuildContext()
    model = FACTORIES[name](ctx, **kwargs)
    return model.to(ctx.device)


def _run_dir(run: str, ctx: BuildContext) -> Path:
    """Directory of a finished run. "repo:<stage>/<name>" = a record in the clone (``paths.REPO_RESULTS``, e.g. a
    laptop-trained network); "out:machines/<id>/runs/<stage>/<name>" = an output of a machine under the output root
    (``paths.OUT_RESULTS``; on the laptop this is the repository, where returned machine directories are placed);
    otherwise relative to ``ctx.results_dir`` (or absolute)."""
    if run.startswith("repo:"):
        return paths.REPO_RESULTS / run[len("repo:"):]
    if run.startswith("out:"):
        return paths.OUT_RESULTS / run[len("out:"):]
    p = Path(run)
    return p if p.is_absolute() else Path(ctx.results_dir) / p


def load_run(run: str, ctx: Optional[BuildContext] = None) -> nn.Module:
    """Rebuild a finished fit from ``<results>/<run>/model.pt`` (its stored factory spec and selected weights)."""
    ctx = ctx or BuildContext()
    path = _run_dir(run, ctx) / "model.pt"
    obj = torch.load(path, map_location=ctx.device, weights_only=False)
    spec = (obj.get("meta") or {}).get("model")
    if not spec or "factory" not in spec:
        raise ValueError(f"{path}: no model spec stored (meta.model.factory)")
    sub = BuildContext(device=ctx.device, train=ctx.train, seed=ctx.seed, log=ctx.log, skip_init=True,
                       results_dir=ctx.results_dir)
    model = build(spec["factory"], sub, **dict(spec.get("kwargs") or {}))
    model.load_state_dict(obj["state"])
    return model.eval()


# ----------------------------------------------------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------------------------------------------------
@register("field")
def _field(ctx, hidden: int = 64, n_layers: int = 3, n_freq: int = 4):
    """Freshly initialised ``common.FieldModel`` architecture (seeded by the caller)."""
    return field_model(hidden, n_layers, n_freq, device=ctx.device)


@register("field_mlp_std")
def _field_mlp_std(ctx):
    """FieldNet(32, 4) starting from the weights of ``mlp_std.pt`` (plan B0, B2, B6 starting point)."""
    return mlp_std_start(ctx.device)


@register("field_acc")
def _field_acc(ctx, template: str = "stored", base: str = "mlp_std", hidden: int = 32, n_layers: int = 4,
               n_freq: int = 4, warm_steps: int = 500, warm_lr: float = 1e-3, grid_points: int = 64):
    """FieldAcc(base, A): logits + ln A (plan B4). base="mlp_std": start from mlp_std's weights and warm-start the
    wrapper by KL to mlp_std's own map (so training starts from the same function); base="fresh": random init,
    no warm start."""
    A = acceptance_template(template, ctx.device)
    if base == "mlp_std":
        net = mlp_std_start(ctx.device)
    elif base == "fresh":
        net = field_model(hidden, n_layers, n_freq, device=ctx.device)
    else:
        raise ValueError("field_acc: base must be 'mlp_std' or 'fresh'")
    model = FieldAcc(net, A=A)
    if base == "mlp_std" and warm_steps > 0 and not ctx.skip_init:
        target = mlp_std_start(ctx.device).eval()
        res = model.warm_start_from(target, default_lnp_grid(grid_points, ctx.device), warm_steps, warm_lr,
                                    log=ctx.log)
        ctx.log(f"field_acc warm start ({template}): KL {res['kl_initial']:.4e} -> {res['kl_final']:.4e}")
    return model


def _knots(ctx, n_knots: int, knots: str):
    if knots == "uniform":
        return np.linspace(math.log(TRAIN_P_MIN), math.log(TRAIN_P_MAX), int(n_knots))
    if knots == "quantile":
        if ctx.train is None:
            raise ValueError("residual_table: knots='quantile' needs the training set in the build context")
        lnp = ctx.train.ev["lnp"].detach().cpu().numpy().astype(np.float64)
        return np.quantile(lnp, np.linspace(0, 1, int(n_knots)))
    raise ValueError("knots must be 'uniform' or 'quantile'")


@register("residual_table")
def _residual_table(ctx, base: str = "mlp_std", n_knots: int = 12, knots: str = "quantile", freeze_base: bool = True):
    """ResidualTable(base, knots): base = "mlp_std" or a finished run "<stage>/<name>"."""
    b = mlp_std_start(ctx.device) if base == "mlp_std" else load_run(base, ctx)
    return ResidualTable(b, _knots(ctx, n_knots, knots), freeze_base=freeze_base)


@register("ensemble")
def _ensemble(ctx, members: list = (), log_weights: Optional[list] = None, trainable_weights: bool = False,
              include_mlp_std: bool = False):
    """Ensemble of finished runs ("<stage>/<name>" each), optionally with mlp_std as a member."""
    ms = [load_run(r, ctx) for r in members]
    if include_mlp_std:
        ms.insert(0, mlp_std_start(ctx.device).eval())
    return Ensemble(ms, log_weights, trainable_weights)


@register("from_run")
def _from_run(ctx, run: str):
    """The selected weights of a finished run, e.g. to continue training from them."""
    return load_run(run, ctx).train()
