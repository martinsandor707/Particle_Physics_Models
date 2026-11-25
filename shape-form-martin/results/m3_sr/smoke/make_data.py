"""Synthetic targets for the M3 runner smoke tests (no real data is read; nothing here is a campaign target).

    uv run python -B shape-form-martin/results/m3_sr/smoke/make_data.py

Writes data/*.npz next to this file (small, regenerated deterministically):
  toy2d.npz        a, b ~ U(-2, 2), 400 rows; y = tanh(1.3a - 0.5b) exp(-0.3 b^2) + 0.15 a b + N(0, 0.01)
                   (smoke a: kill / resume / fallback; PySR's default L2)
  residual3d.npz   12 slices x 650 rows = 7800, inputs (zt, rh, x) on the data box, rows slice-contiguous, x constant
                   per slice; y = delta(zt, rh, x) + c(x) + noise, c(x) a gauge term the gauge loss must ignore;
                   positive Horvitz-Thompson-like weights; written with targets.save_target (smoke b, c, d, f)
  traj3.npz        40 points x in [-1, 1], three outputs, weights (40, 3), cubic guesses list[list[str]] from
                   targets.cubic_guesses (smoke e: multi-output)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from sfm.sr import operators as O  # noqa: E402
from sfm.sr import targets as TG  # noqa: E402

DATA = HERE / "data"


def toy2d(rng):
    X = rng.uniform(-2, 2, size=(400, 2))
    a, b = X[:, 0], X[:, 1]
    y = np.tanh(1.3 * a - 0.5 * b) * np.exp(-0.3 * b ** 2) + 0.15 * a * b + rng.normal(0, 0.01, len(a))
    np.savez(DATA / "toy2d.npz", X=X, y=y, variable_names=np.array(["a", "b"]))


def delta_true(zt, rh, x):
    """The synthetic 'correction' (Stage-1 representable): bounded, depends on all three inputs."""
    return 0.5 * np.tanh(1.2 * zt - 1.0 + 0.6 * x) - 0.4 / (1.0 + (0.15 * rh) ** 2) * (1.0 + 0.3 * x)


def residual3d(rng):
    S, n = 12, 650
    xs = np.linspace(-0.95, 0.95, S)
    zt = rng.uniform(-200.0, 1450.0, size=(S, n)) / O.ZETA_SCALE_MM
    r = np.minimum(np.abs(rng.normal(0.0, 150.0, size=(S, n))), 700.0)
    rh = r / O.RHO_SCALE_MM
    x = np.repeat(xs[:, None], n, axis=1)
    w = rng.gamma(2.0, 0.5, size=(S, n))
    gauge = 0.7 * x + 0.2                                     # c(x): invisible to the softmax and to the loss
    y = delta_true(zt, rh, x) + gauge + rng.normal(0.0, 0.02, size=(S, n))
    # HT-weighted mean zero per slice, as targets.build_residual_target stores it
    y = y - (np.sum(w * y, axis=1, keepdims=True) / np.sum(w, axis=1, keepdims=True))
    X = np.stack([zt.ravel(), rh.ravel(), x.ravel()], axis=1)
    target = {"X": X, "y": y.ravel(), "weights": w.ravel(), "variable_names": np.array(list(O.VARIABLE_NAMES)),
              "kind": np.array("residual"), "loss": np.array("gauge"),
              "schema_version": np.array(TG.SCHEMA_VERSION), "slice_id": np.repeat(np.arange(S), n),
              "slice_x": xs,
              "meta_json": np.array(json.dumps({"synthetic": True, "delta": "0.5 tanh(1.2 zt - 1 + 0.6 x) - 0.4 "
                                                "linv(0.15 rh) (1 + 0.3 x)", "gauge_term": "0.7 x + 0.2",
                                                "noise_sd": 0.02}))}
    TG.save_target(DATA / "residual3d.npz", target)


def traj3(rng):
    G = 40
    x = np.linspace(-1.0, 1.0, G)
    y = np.stack([0.3 + 0.5 * x - 0.2 * x ** 2 + 0.1 * x ** 3, 0.8 * np.tanh(1.5 * x), np.exp(0.5 * x) - 1.0], axis=1)
    y = y + rng.normal(0.0, 0.002, size=y.shape)
    w = rng.uniform(0.5, 1.5, size=(G, 3))
    w = w / w.mean(axis=0, keepdims=True)
    guesses = TG.cubic_guesses(x, y, w)
    np.savez(DATA / "traj3.npz", X=x[:, None], y=y, weights=w, variable_names=np.array(["x"]),
             guesses_json=np.array(json.dumps(guesses)), kind=np.array("trajectory"), loss=np.array("wl2"))
    (DATA / "traj3_guesses.json").write_text(json.dumps(guesses, indent=1) + "\n")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261125)
    toy2d(rng)
    residual3d(rng)
    traj3(rng)
    for p in sorted(DATA.iterdir()):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
