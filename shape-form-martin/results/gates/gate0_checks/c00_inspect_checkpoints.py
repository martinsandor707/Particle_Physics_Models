"""Gate 0 helper: what is stored in the reference checkpoints (read-only; no test data, no model evaluation)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch

PION = Path(__file__).resolve().parents[4] / "shape-form-proton" / "results" / "particles" / "pion"


def describe(v, ind=0):
    pad = " " * ind
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, dict):
                print(f"{pad}{k}: dict({len(x)})")
                if ind < 4:
                    describe(x, ind + 2)
            else:
                describe_leaf(k, x, pad)
    else:
        describe_leaf("", v, pad)


def describe_leaf(k, x, pad):
    if isinstance(x, torch.Tensor):
        print(f"{pad}{k}: tensor {tuple(x.shape)} {x.dtype} {x.device}" + (f" value={x.item()!r}" if x.numel() == 1 else ""))
    elif isinstance(x, np.ndarray):
        print(f"{pad}{k}: ndarray {x.shape} {x.dtype} mean={x.astype(np.float64).mean():.9f}")
    else:
        s = repr(x)
        print(f"{pad}{k}: {type(x).__name__} {s[:160]}")


for f in ("mlp_std.pt", "new3_std_n3.pt", "new3_std_n2.pt", "new3_std_n1.pt", "base_std_n5.pt", "base_std_n3.pt", "abar.pt"):
    ck = torch.load(PION / f, map_location="cpu", weights_only=False)
    print("=" * 20, f)
    describe(ck)
