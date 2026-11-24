"""Model loading for the R1 audit (read-only use of the study's checkpoints)."""
import os
import sys

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import torch

import shapelab
from proton_shape import ProtonShape

FINAL = "/home/user/tamas/shape-form-proton/results/final/proton"
DEG = {1: dict(degree=1, degrees={"logit_tr": 2}), **{n: dict(degree=n) for n in range(2, 7)}}


def _strip(sd, prefix="base."):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_all(C, data, which=None):
    """dict name -> model (eval mode, on the device)."""
    dev = C["DEVICE"]
    c, s = shapelab.lnp_center_scale(data)
    out = {}
    rep = shapelab.baseline_models(C, data)
    for k in ("gflash_n1", "gflash_n6", "mlp", "cnn"):
        out["report_" + k] = rep[k]
    ld = lambda f: torch.load(os.path.join(FINAL, f), map_location=dev, weights_only=False)
    for n in (1, 2, 6):
        m = ProtonShape(C, lnp0=c, lnp_s=s, lateral="gce", **DEG[n]).to(dev)
        m.load_state_dict(ld(f"slab_gce_n{n}.pt")["state"])
        out[f"new_slab_gce_n{n}"] = m.eval()
        core = ProtonShape(C, lnp0=c, lnp_s=s, lateral="gce", **DEG[n]).to(dev)
        w = shapelab.WithAcceptance(core, torch.ones(C["N_Z"], C["N_R"], device=dev)).to(dev)
        w.load_state_dict(ld(f"stdacc_gce_n{n}.pt")["state"])          # logA buffer comes from the checkpoint
        out[f"new_stdacc_gce_n{n}"] = w.eval()
    f = C["Formula"]("gamma", "core_tail", degree=6, lnp0=c, lnp_s=s).to(dev)
    f.load_state_dict(_strip(ld("ref_slab_gflash_n6.pt")["state"]))
    out["ref_slab_gflash_n6"] = f.eval()
    m = C["FieldModel"](16, 3).to(dev)
    m.load_state_dict(_strip(ld("ref_slab_mlp_L3xH16.pt")["state"]))
    out["ref_slab_mlp"] = m.eval()
    f = C["Formula"]("gamma", "core_tail", degree=6, lnp0=c, lnp_s=s).to(dev)
    w = shapelab.WithAcceptance(f, torch.ones(C["N_Z"], C["N_R"], device=dev)).to(dev)
    w.load_state_dict(ld("ref_std_gflash_acc_n6.pt")["state"])
    out["ref_std_gflash_acc_n6"] = w.eval()
    if which is not None:
        out = {k: v for k, v in out.items() if k in which}
    return out
