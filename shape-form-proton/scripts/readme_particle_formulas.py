"""A README „Részecskénkénti képletek” szakaszát állítja elő a végleges (3D-mérce) illesztésből.

Forrás: results/particles/<p>/formula_values.json és formula_coefficients_slab.csv (a scripts/formula_card.py írja).
Kimenet: a README két jelölő (<!-- KEPLETEK:START/END -->) közötti része.
    python readme_particle_formulas.py"""
import csv
import json
import math
import os
import re

ROOT = "/home/user/tamas/shape-form-proton"
HU = {"electron": "Elektron", "muon": "Müon", "proton": "Proton", "pion": "Pion"}
NPAR = {"electron": "n = 3, 64", "muon": "n = 6, 129", "proton": "n = 6, 143", "pion": "n = 3, 101"}
MASS = {"electron": "0,511 MeV", "muon": "105,7 MeV", "proton": "938,3 MeV", "pion": "139,6 MeV"}
P_COLS = [1.0, 5.0, 20.0]


def num(x, unit=""):
    if isinstance(x, str):
        return x.replace(".", ",")
    a = abs(x)
    s = f"{x:.3g}" if a < 1000 else f"{x:.0f}"
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}·10^{int(e)}"
    return s.replace(".", ",").replace("-", "−") + unit


def pct(x):
    return "< 0,01 %" if x < 1e-4 else num(100 * x) + " %"


def coef(D):
    out = {}
    for r in csv.DictReader(open(os.path.join(D, "formula_coefficients_slab.csv"))):
        out[r["parameter"]] = float(r["c0"])
    return out


def block(particle):
    D = os.path.join(ROOT, "results", "particles", particle)
    v = json.load(open(os.path.join(D, "formula_values.json")))["slab"]
    c = coef(D)
    idx = [v["p [GeV]"].index(p) for p in P_COLS]
    col = lambda k: [v[k][i] for i in idx]
    track = "w nyom" in v
    gamma = "t_max [mm]" in v
    end = "w folt" in v
    terms = ["w_zápor·S"] + (["w_nyom·U"] if track else []) + ["w_csúcs·D"] + (["w_folt·E"] if end else []) + ["w_távoli·H"]
    L = [f"**{HU[particle]}** (m = {MASS[particle]}; polinomfok és paraméterszám: {NPAR[particle]})", "",
         f"`ρ(z', r | p) = {' + '.join(terms)}`", ""]
    if gamma:
        L.append("- Zápor mélységprofilja gamma: `L(z') ∝ s^(α−1)·e^(−(α−1)s/t_max)`, s = z' − z0 (értékek lent).")
    else:
        L.append("- Zápor mélységprofilja plató: `L(z') ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ·z'/1 m)` (értékek lent).")
    L.append("- Zápor keresztmetszete: `T(r│z') = f_mag·Gauss(r; σ) + f_test·Cauchy(r; R) + f_udvar·e^(−r/λ)/(2πλ(r + 1 mm))`; "
             "σ, R, λ és az arányok a mélységgel változnak (lent z' = 100 mm-en).")
    if track:
        R1, kR, R0 = math.exp(c["log_R1"]), c["k_R"], c["R0"]
        lat = col("λ_att [mm]")[0]
        L.append(f"- Nyom: hatótáv `R(T) = {num(R1)} mm · (T/GeV)^{num(kR)} − {num(R0)} mm`; csillapítás `e^(−z'/{num(lat)} mm)`; "
                 f"Bragg-felfutás `(1 − z'/R + 0,03)^(−{num(col('q Bragg')[0])})`; szélesség "
                 f"`σ(z')² = ({num(col('σ0 nyom [mm]')[0])} mm)² + ({num(col('c_ms [mm]')[0])} mm · (z'/1 m)^{num(col('k_ms')[0])} / βp)²`; "
                 f"delta-elektron-udvar `e^(−r/{num(col('l delta-udvar [mm]')[0])} mm)` (aránya lent).")
    if end:
        L.append(f"- Folt a hatótáv végén: Gauss, σ_z = {num(col('folt σ_z [mm]')[0])} mm, σ_r = {num(col('folt σ_r [mm]')[0])} mm.")
    L.append(f"- Csúcs a belépésnél: Gauss (hely és szélességek lent). Távoli udvar: `H ∝ σ(z'/10 mm)·e^(−z'/{num(col('távoli udvar L [mm]')[0])} mm)/(r + 1 cm)`.")
    L += ["", "| | " + " | ".join(f"p = {p:g} GeV/c" for p in P_COLS) + " |", "|---|" + "---:|" * len(P_COLS)]
    rows = [("T (mozgási energia)", "T [GeV]", lambda x: num(x, " GeV"))]
    for k, lab in (("zápor", "zápor"), ("nyom", "nyom"), ("csúcs", "csúcs"), ("folt", "folt"), ("távoli udvar", "távoli udvar")):
        if f"a detektorban: {k}" in v:
            rows.append((f"energiarész: {lab}", f"a detektorban: {k}", pct))
    if gamma:
        rows += [("zápor kezdete z0", "z0 [mm]", lambda x: num(x, " mm")), ("zápor maximuma t_max", "t_max [mm]", lambda x: num(x, " mm")),
                 ("alak α − 1", "α−1", num)]
    else:
        rows += [("zápor kezdete z_s / felfutás w1", ("z_s [mm]", "w1 [mm]"), None),
                 ("zápor vége ℓ / lecsengés w2", ("ℓ [mm]", "w2 [mm]"), None),
                 ("lassú meredekség κ", "κ [1/m]", lambda x: num(x, " /m"))]
    rows += [("mag σ / Cauchy R / udvar λ (z' = 100 mm)", ("σ mag, z'=100 [mm]", "R Cauchy, z'=100 [mm]", "λ udvar, z'=100 [mm]"), None),
             ("arány mag / test / udvar (z' = 100 mm)", "f mag / test / udvar, z'=100", num)]
    if track:
        rows += [("nyom hatótáva R(T)", "R(T) hatótáv [mm]", lambda x: num(x, " mm")),
                 ("delta-udvar aránya", "f delta-udvar", pct)]
    rows += [("csúcs helye / σ_z / σ_r", ("csúcs z [mm]", "csúcs σ_z [mm]", "csúcs σ_r [mm]"), None)]
    for lab, key, f in rows:
        if isinstance(key, tuple):
            cells = [" / ".join(num(v[k][i]) for k in key) + " mm" for i in idx]
        else:
            cells = [f(x) for x in col(key)]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


out = ["<!-- KEPLETEK:START (scripts/readme_particle_formulas.py generálja) -->", "",
       "### Részecskénkénti képletek (3D-mérce, a beírt számok az illesztett értékek)", "",
       "A fenti általános képlet részecskénként: melyik tag van benne, az állandó paraméterek behelyettesítve, a p-függő "
       "paraméterek 1, 5 és 20 GeV/c-nél. Az „energiarész” a detektorba (a ferde belépési sík mögé) eső energia megoszlása. "
       "A polinom-együtthatók: `results/particles/<részecske>/formula_coefficients_slab.csv`; minden impulzus és a "
       "report-mérce: `formula.md`. Ha a hatótáv nagyobb a detektornál (~1,3 m), a részecske átmegy rajta.", ""]
for p in ("electron", "muon", "proton", "pion"):
    out += block(p)
out.append("<!-- KEPLETEK:END -->")
txt = "\n".join(out)
f = os.path.join(ROOT, "README.md")
s = open(f).read()
if "<!-- KEPLETEK:START" in s:
    s = re.sub(r"<!-- KEPLETEK:START.*?<!-- KEPLETEK:END -->", lambda m: txt, s, flags=re.S)
else:
    anchor = "Hadronoknál a nyom szélesedése, delta-udvara és Bragg-kitevője effektív"
    i = s.index(anchor)
    j = s.index("\n\n", i) + 2
    s = s[:j] + txt + "\n\n" + s[j:]
open(f, "w").write(s)
print(txt)
