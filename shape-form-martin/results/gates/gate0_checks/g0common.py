"""Gate 0 helper: shared setup for the check scripts (paths, validation/training data, the benchmark models).
Never loads the test cache."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MARTIN = HERE.parents[2]
REPO = MARTIN.parent
sys.path.insert(0, str(MARTIN / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data, legacy, paths  # noqa: E402

common = legacy.common
shapelab = legacy.shapelab
PION = paths.PION_RESULTS_DIR
M0 = paths.M0_DIR
DEV = common.DEVICE


def load_trainval(device=DEV):
    return data.load_sets(("train", "val"), device=device)        # allow_test stays False


def jload(p):
    return json.loads(Path(p).read_text())


def new3_std(n, train_set):
    """Rung n of the final formula ladder (new3_std_n{n}.pt x abar.pt), as particle_pipeline.py builds it."""
    c0, s0 = shapelab.lnp_center_scale({"train": train_set})
    m, core, t, k = shapelab.build_final(legacy.C, "pion", str(PION), c0, s0, metric="std", n=n)
    return m.eval()


class BaseWrap(torch.nn.Module):
    """particle_pipeline.BaseWrap: the report's Formula (gamma x core_tail) of degree n."""

    def __init__(self, long, n, c0, s0):
        super().__init__()
        self.f = common.Formula(long, "core_tail", degree=n, lnp0=c0, lnp_s=s0)

    def forward(self, lnp):
        return self.f(lnp)


def base_std(n, train_set):
    """Rung n of the baseline ladder (base_std_n{n}.pt): WithAcceptance(BaseWrap('gamma', n), abar)."""
    c0, s0 = shapelab.lnp_center_scale({"train": train_set})
    core = BaseWrap("gamma", n, c0, s0).to(DEV)
    m = shapelab.WithAcceptance(core, torch.load(PION / "abar.pt", map_location=DEV)).to(DEV)
    m.load_state_dict(torch.load(PION / f"base_std_n{n}.pt", map_location=DEV, weights_only=False)["state"])
    return m.eval()


def f32mean(a):
    return float(np.asarray(a, dtype=np.float32).mean())


def f64mean(a):
    return float(np.asarray(a, dtype=np.float64).mean())
