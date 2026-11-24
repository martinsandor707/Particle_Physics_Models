"""Read-only adapter to the follow-up study in ``shape-form-proton/``.

Loads the two pion benchmarks and the acceptance template from
``shape-form-proton/results/particles/pion/`` without writing anything there:
  * ``sys.dont_write_bytecode = True`` is set before ``shapelab`` / ``particle_shape`` are imported
    (no .pyc files appear in shape-form-proton/);
  * only the geometry helpers ``SLAB_GEOM_SRC`` (slab_accept, event_slab) are exec'd into the namespace
    ``C = dict(vars(common))``; ``SLAB_NLL_SRC`` is never used, so ``C["event_nll"]`` stays common.event_nll;
  * ``shapelab.setup`` / ``setup_particle`` / ``protonlab.load_*`` are never called (they write a cache there).

Benchmarks:
  load_mlp_std()                  fine-tuned 4x32 MLP (``mlp_std.pt``, key "state")
  load_component_formula(train)   final 101-parameter component formula (``new3_std_n3.pt``) wrapped with ``abar.pt``
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import torch  # noqa: E402

from . import paths  # noqa: E402

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
if str(paths.PROTON_DIR) not in sys.path:
    sys.path.append(str(paths.PROTON_DIR))       # appended: never shadows anything else on sys.path

import common  # noqa: E402
import particle_shape  # noqa: E402,F401  (shape-form-proton/particle_shape.py)
import shapelab  # noqa: E402  (shape-form-proton/shapelab.py)

PION_DIR = paths.PION_RESULTS_DIR

C = dict(vars(common))
exec(shapelab.SLAB_GEOM_SRC, C)                  # geometry helpers only (slab_accept, event_slab)
assert C["event_nll"] is common.event_nll and C["test_nll"] is common.test_nll


def load_mlp_std(device=None):
    """``common.FieldModel(hidden=32, n_layers=4)`` with the weights of ``mlp_std.pt`` (eval mode)."""
    device = device or common.DEVICE
    net = common.FieldModel(hidden=32, n_layers=4).to(device)
    ck = torch.load(PION_DIR / "mlp_std.pt", map_location=device, weights_only=False)
    net.load_state_dict(ck["state"])
    return net.eval()


def load_component_formula(train_set):
    """The final pion component formula under the report metric: ``shapelab.build_final(C, "pion", ..., metric="std")``.
    ``c0, s0`` come from the training set's ln p range (``shapelab.lnp_center_scale``), as in the original pipeline;
    the checkpoint's buffers (lnp0, lnp_s, v0, vs) are restored by ``load_state_dict``. Returns the wrapped model
    (``WithAcceptance(core, abar)``) in eval mode."""
    c0, s0 = shapelab.lnp_center_scale({"train": train_set})
    model, core, t, k = shapelab.build_final(C, "pion", str(PION_DIR), c0, s0, metric="std")
    return model.eval()


def n_params(model) -> int:
    return int(model.n_params()) if hasattr(model, "n_params") else sum(p.numel() for p in model.parameters())


def load_abar(device=None):
    """The stored acceptance template Ā(z', r) (128 x 32), built from training events with front and back faces."""
    return torch.load(PION_DIR / "abar.pt", map_location=device or common.DEVICE, weights_only=False)


def mean_acceptance(train_set, front=True, back=False, nsub=4):
    """Ā recomputed with ``shapelab.mean_acceptance`` (event mean of the azimuthal slab acceptance)."""
    return shapelab.mean_acceptance(C, train_set, nsub=nsub, front=front, back=back)
