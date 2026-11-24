"""M1-d Part C helper for D9 - is the oscillating part of ln P_MLP - ln P_formula along z' a property of the data
(shared by independently trained networks) or of the individual network (its sin/cos depth features)?

    uv run python -B shape-form-martin/results/m1_audit/checks/c_ringing.py

Maps at the 12 event-quantile momenta of the audit; window 20 < z' < 1150 mm (88 bins, 1134 mm); rings 0, 2, 4, 8, 12,
16, 24. For each pair of maps (a, b):  D(z') = ln P_a - ln P_b  on one ring; a cubic polynomial in z' is removed by
least squares, the residual is multiplied by a Hann window and projected on exp(-2 pi i z'/T) for
T = 550, 412.5 (features k = 3, 4 of every FieldModel), 330, 275, 235.7, 206.25 mm (k = 5..8: only the
nonlinearity can produce them) and 165 mm. Amplitude = 2 |sum w D e^{-i..}| / sum w  [nat] (that of a pure sinusoid).
Phase coherence between the four networks (each minus the formula):
    coherence_T = |sum_n c_n| / sum_n |c_n|     (1 = the same oscillation in every network; ~0.44 expected for
                                                 4 independent random phases of equal amplitude)
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

common, legacy = chk.common, chk.legacy
RINGS = (0, 2, 4, 8, 12, 16, 24)
NH = 8


def local_mlp(H):
    ck = torch.load(chk.paths.ORIGINAL_DIR / "results" / "pion" / f"mlp_L3xH{H}.pt", map_location=chk.DEV, weights_only=False)
    m = common.FieldModel(hidden=H, n_layers=3).to(chk.DEV)
    m.load_state_dict(ck["model"])
    return m.eval()


def main():
    t0 = time.time()
    sets, _ = chk.load_cpu(("train",))
    p12 = np.quantile(sets["train"].ev["p"].numpy().astype(np.float64), (np.arange(12) + 0.5) / 12)
    with chk.gpu_lock("m1d D9 ringing"):
        nets = {"mlp_std": legacy.load_mlp_std(), "L3xH16": local_mlp(16), "L3xH32": local_mlp(32), "L3xH64": local_mlp(64)}
        formula = legacy.load_component_formula(sets["train"])
        lnp = torch.log(torch.tensor(p12, dtype=torch.float32, device=chk.DEV))
        with torch.no_grad():
            maps = {k: m(lnp).double().cpu().numpy() for k, m in nets.items()}
            maps["formula"] = formula(lnp).double().cpu().numpy()
    zc = common.ZC.cpu().numpy().astype(np.float64)
    win = (zc > 20) & (zc < 1150)
    z = zc[win]
    zn = (z - z.mean()) / (z.max() - z.min())
    X = np.stack([np.ones_like(z), zn, zn ** 2, zn ** 3], 1)
    proj = X @ np.linalg.pinv(X)
    hann = np.hanning(len(z) + 2)[1:-1]
    periods = [550.0, 412.5, 330.0, 275.0, 1650.0 / 7, 206.25, 165.0]
    E = np.stack([np.exp(-2j * math.pi * z / T) for T in periods], 1) * hann[:, None]
    cond = float(np.linalg.cond(X))

    def harmonics(D):
        """D (n_p, n_z_window) -> complex amplitudes (n_p, n_periods), rms of the cubic-detrended residual (n_p,)."""
        res = D - D @ proj.T
        return 2 * (res @ E) / hann.sum(), np.sqrt((res ** 2).mean(1)), np.sqrt((res ** 2).mean(1))

    out = {"p_GeV": p12, "rings": RINGS, "design_condition_number": cond, "window_mm": [float(z.min()), float(z.max())],
           "harmonic_periods_mm": periods, "pairs": {}, "coherence": {}}
    names = list(nets)
    coef = {}
    for a in names:
        for b in ["formula"] + names:
            if a == b or (b != "formula" and names.index(b) <= names.index(a)):
                continue
            amp, rr, rq = [], [], []
            for ring in RINGS:
                D = maps[a][:, win, ring] - maps[b][:, win, ring]
                c, r_res, r_quad = harmonics(D)
                coef[(a, b, ring)] = c
                amp.append(np.abs(c))
                rr.append(r_res)
                rq.append(r_quad)
            amp = np.array(amp)                                           # (ring, p, k)
            out["pairs"][f"{a} - {b}"] = {"amplitude_mean_over_p_and_rings": amp.mean((0, 1)),
                                         "amplitude_by_ring_mean_over_p": amp.mean(1),
                                         "amplitude_p_ge_5GeV": amp[:, 3:].mean((0, 1)),
                                         "rms_after_cubic_detrend": float(np.mean(rq))}
    for ring in RINGS:
        C = np.stack([coef[(a, "formula", ring)] for a in names])          # (net, p, k)
        coh = np.abs(C.sum(0)) / np.abs(C).sum(0)
        out["coherence"][str(ring)] = {"mean_over_p": coh.mean(0), "p_ge_5GeV": coh[3:].mean(0)}
    chk.write("c_ringing.json", out, t0)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print("periods", np.round(out["harmonic_periods_mm"], 1), "cond", cond)
    for k, v in out["pairs"].items():
        print(f"{k:22s} amp [nat]:", v["amplitude_mean_over_p_and_rings"], "p>=5:", v["amplitude_p_ge_5GeV"], " rms(detrended)", round(v["rms_after_cubic_detrend"], 4))
    for ring, v in out["coherence"].items():
        print("coherence ring", ring, v["mean_over_p"], "p>=5:", v["p_ge_5GeV"])
    # amplitude of the part common to the four networks vs their spread, per period (mean over p >= 5 GeV)
    for ring in RINGS:
        C = np.stack([coef[(a, "formula", ring)] for a in names])[:, 3:]
        common_amp = np.abs(C.mean(0)).mean(0)
        spread = np.sqrt((np.abs(C - C.mean(0)) ** 2).mean(0)).mean(0)
        out.setdefault("common_vs_spread", {})[str(ring)] = {"common_amplitude": common_amp, "network_spread_rms": spread}
        print("ring", ring, "common", common_amp, "spread", spread)
    chk.write("c_ringing.json", out, t0)


if __name__ == "__main__":
    main()
