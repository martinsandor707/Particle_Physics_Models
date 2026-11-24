"""A detektor anyagából várt értékek az ePIC LFHCal szimulált rétegszerkezetére (eic/epic compact/hcal/lfhcal.xml, ePIC 25.12–26.08):
60 réteg × [15,2 mm Steel235 + 4,0 mm szcintillátor (C19H21, 1,032 g/cm³) + 0,8 mm Kapton + 0,5 mm levegő], előtte 40,4 mm Steel235.

Számol: a réteg-keverék X0, λ_I, λ_π, R_M, Ec értékét (PDG 2025 anyagállandók, vastagság-súlyozással), a 40,4 mm-es előlap
megfelelőjét, a müon Highland-szélességét (vetített σ_y, β = 1, rögzített p, az előlappal és nélküle; hatványillesztés c·(z'/1 m)^k
több mélységtartományban), az elektron zápormaximumát (PDG: t_max = ln(E/Ec) − 0,5, az előlap után), és a megállási küszöböket.
A CSDA-hatótávokat (proton, müon) nem számolja: azok a NIST PSTAR / PDG-táblákból homogenizálva készültek (független ellenőrzés,
2026-11-17), és itt dokumentált bemenetként szerepelnek.
    python lfhcal_expectations.py          kimenet: results/particles/lfhcal_expectations.json"""
import json
import math
import os

import numpy as np

OUT = "/home/user/tamas/shape-form-proton/results/particles/lfhcal_expectations.json"
# PDG 2025: X0, lambda_I, lambda_pi, R_M [g/cm2], Ec (e-) [MeV], minimum dE/dx [MeV cm2/g]
PDG = {"Fe": (13.84, 132.1, 160.7, 13.53, 21.68, 1.451), "C": (42.70, 85.8, 117.8, 11.08, 81.74, 1.742),
       "PVT": (43.90, 81.3, 113.6, 9.89, 94.11, 1.956), "Kapton": (40.58, 85.5, 117.5, 10.62, 81.01, 1.820),
       "Air": (36.62, 90.1, 122.0, 8.83, 87.92, 1.815)}
ES = 21.2052                                                      # MeV (Rossi Ec definition)


def mix_mass(parts):
    """Mass-fraction mixture: 1/X = sum w/X_i for X0, lambda, R_M; dE/dx additive."""
    q = [1 / sum(w / PDG[m][i] for m, w in parts) for i in range(4)]
    dedx = sum(w * PDG[m][5] for m, w in parts)
    return q, dedx


steel, steel_dedx = mix_mass([("Fe", 0.998), ("C", 0.002)])
MATS = {"steel": (steel, steel_dedx, 7.85), "scint": ([PDG["PVT"][i] for i in range(4)], PDG["PVT"][5], 1.032),
        "kapton": ([PDG["Kapton"][i] for i in range(4)], PDG["Kapton"][5], 1.43), "air": ([PDG["Air"][i] for i in range(4)], PDG["Air"][5], 0.0012)}
THICK = {"steel": 15.2, "scint": 4.0, "kapton": 0.8, "air": 0.5}
PITCH, FRONT = 20.5, 40.4
L = {m: [10 * v / rho for v in q] for m, (q, d, rho) in MATS.items()}       # lengths in mm: X0, lambda_I, lambda_pi, R_M
f = {m: t / PITCH for m, t in THICK.items()}
eff = [1 / sum(f[m] / L[m][i] for m in f) for i in range(4)]
X0, LI, LPI, RM = eff
EC = ES * X0 / RM
dedx_mm = sum(f[m] * MATS[m][1] * MATS[m][2] / 10 for m in f)                # MeV/mm (minimum ionising)
steel_dedx_mm = steel_dedx * 7.85 / 10
front_X0 = FRONT / L["steel"][0]
front_em_mm, front_dedx_mm = front_X0 * X0, FRONT * steel_dedx_mm / dedx_mm


def highland_sigma(z_mm, p=1.0, with_front=True):
    """Projected rms displacement sigma_y [mm] at depth z' for a beta = 1 muon of momentum p [GeV]; F applied once (total X/X0)."""
    s_lo = -FRONT if with_front else 0.0
    out = []
    for z in np.atleast_1d(z_mm):
        s = np.linspace(s_lo, z, 4001)
        inv_x0 = np.where(s < 0, 1 / L["steel"][0], 1 / X0)
        integral = np.trapezoid((z - s) ** 2 * inv_x0, s)                   # mm^2 (per unit radiation length)
        xtot = (front_X0 if with_front else 0.0) + z / X0
        F = 1 + 0.038 * math.log(max(xtot, 1e-6))
        out.append(0.0136 / p * F * math.sqrt(integral))
    return np.array(out)


def powerfit(lo, hi, with_front=True):
    z = np.linspace(lo, hi, 40)
    sg = highland_sigma(z, with_front=with_front)
    k, lnc = np.polyfit(np.log(z / 1000), np.log(sg), 1)
    return {"c_mm": float(math.exp(lnc)), "k": float(k)}


res = {"layer": {"pitch_mm": PITCH, "thickness_mm": THICK, "front_steel_mm": FRONT},
       "stack": {"X0_mm": X0, "lambda_I_mm": LI, "lambda_pi_mm": LPI, "R_M_mm": RM, "Ec_MeV": EC, "mip_dEdx_MeV_per_mm": dedx_mm,
                 "density_g_cm3": sum(f[m] * MATS[m][2] for m in f)},
       "front": {"X0": front_X0, "lambda_I": FRONT / L["steel"][1], "lambda_pi": FRONT / L["steel"][2],
                 "stack_equiv_em_mm": front_em_mm, "stack_equiv_dEdx_mm": front_dedx_mm},
       "highland_muon": {"with_front": {f"{lo}-{hi} mm": powerfit(lo, hi) for lo, hi in ((300, 1200), (100, 1200), (200, 1300), (300, 1000))},
                         "without_front": {f"{lo}-{hi} mm": powerfit(lo, hi, False) for lo, hi in ((300, 1200),)},
                         "sigma_at_1m_p1GeV_mm": {"with_front": float(highland_sigma(1000.0)[0]), "without_front": float(highland_sigma(1000.0, with_front=False)[0])},
                         "note": "beta = 1, fixed p (no energy loss), normal incidence; energy loss along the track raises c and k"},
       "electron_tmax_behind_first_tile_mm": {str(E): (math.log(E * 1000 / EC) - 0.5) * X0 - front_em_mm for E in (1, 2, 5, 10, 20)},
       "csda_external": {
           "source": "NIST PSTAR / PDG muon tables, homogenised over the layer mix, after the 40.4 mm front steel at normal incidence "
                     "(independent check, 2026-11-17); Bethe estimate agrees within ~1 %",
           "proton_mm_after_front": {"T=0.433 GeV (p=1 GeV/c)": 165, "T=0.5 GeV": 219, "T=1.0 GeV": 694},
           "proton_local_exponent": {"0.05-0.1 GeV": 1.75, "0.1-0.2": 1.73, "0.2-0.5": 1.63, "0.5-1": 1.46, "1-2": 1.31},
           "muon_linear_fit": "R ≈ 1.06–1.07 m/GeV · T − 75…80 mm for T = 0.15–1.25 GeV",
           "stopping_in_front_steel_GeV_c": {"proton": 0.61, "muon": 0.153, "note": "normal incidence; 0.64 / 0.165 at theta = 0.57 rad"}}}
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ("stack", "front", "highland_muon", "electron_tmax_behind_first_tile_mm")}, indent=1, ensure_ascii=False))
