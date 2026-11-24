"""A végleges formula „kártyája” részecskénként: a képlet, a bekapcsolt tagok, az illesztett paraméterek néhány impulzusnál
és a teljes együttható-táblázat (a polinomok az x = (ln(T/GeV) − v0)/vs változóban, T = √(p² + m²) − m).
Adat nem kell hozzá: a checkpointokból és a rács-állandókból dolgozik.
    python formula_card.py            kimenet: results/particles/<p>/formula.md, formula_coefficients_<mérce>.csv, formula_values.json"""
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn.functional as F

import protonlab
import shapelab
from particle_shape import MASS

P_SHOW = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
HU = {"electron": "elektron", "muon": "müon", "proton": "proton", "pion": "pion"}
TERMS = [("far", "távoli udvar"), ("endblob", "folt a hatótáv végén"), ("ms", "többszörös szórás (a nyom szélesedése)"),
         ("track_halo", "delta-elektron-udvar a nyom körül"), ("bragg", "Bragg-felfutás"), ("curv", "négyzetes mélységtagok a keresztmetszetben")]


def fmt(v):
    a = abs(v)
    return f"{v:.3g}" if (a >= 1e4 or (0 < a < 1e-2)) else (f"{v:.0f}" if a >= 100 else f"{v:.3g}")


def evaluate(core, p, abar=None):
    """Fizikai paraméterek a p [GeV] impulzusoknál (lista). abar: a szögekre átlagolt detektor-elfogadás (Ā) – ezzel súlyozva a
    komponensek részaránya a detektorba (a belépési sík mögé) eső energiában."""
    lnp = torch.log(torch.tensor(p, device=core.coef.device))
    with torch.no_grad():
        lw, comp, th = core.components(lnp)
    g = {k: v.flatten().cpu().numpy() for k, v in th.items()}
    ex = lambda k: np.exp(g[k])
    names = ["zápor", "nyom", "csúcs"] + (["folt"] if core.endblob else []) + (["távoli udvar"] if core.far else [])
    out = {"p [GeV]": list(p), "T [GeV]": list(np.exp(g["lnT"]))}
    for nm, w in zip(names, lw):
        if w is not None:
            out[f"w {nm}"] = list(torch.exp(w).flatten().cpu().numpy())
    if abar is not None:
        with torch.no_grad():
            e = {nm: (torch.exp(w).flatten() * (torch.exp(c) * abar).sum((1, 2))) for nm, w, c in zip(names, lw, comp) if w is not None}
            tot = sum(e.values())
        for nm, v in e.items():
            out[f"a detektorban: {nm}"] = list((v / tot).cpu().numpy())
    if core.long == "plateau":
        out.update({"z_s [mm]": g["zs"], "w1 [mm]": ex("log_w1"), "ℓ [mm]": ex("log_ell"), "w2 [mm]": ex("log_w2"), "κ [1/m]": g["kappa"]})
    elif core.long == "gamma":
        out.update({"z0 [mm]": g["z0"], "t_max [mm]": ex("log_tmax"), "α−1": ex("log_am1")})
    for zz in (100.0, 600.0):
        zk = zz / 1000
        q = lambda k: g[k] * zk * zk if k in g else 0.0
        l1 = g["logit_1"] + g["s_1"] * zk + q("q_1")
        l2 = g["logit_2"] + g["s_2"] * zk + q("q_2")
        s1, s2 = 1 / (1 + np.exp(-l1)), 1 / (1 + np.exp(-l2))
        out.update({f"σ mag, z'={zz:.0f} [mm]": np.exp(g["log_sig"] + g["s_sig"] * zk + q("q_sig")),
                    f"R Cauchy, z'={zz:.0f} [mm]": np.exp(g["log_R"] + g["s_R"] * zk),
                    f"λ udvar, z'={zz:.0f} [mm]": np.exp(g["log_lam"] + g["s_lam"] * zk),
                    f"f mag / test / udvar, z'={zz:.0f}": [f"{a:.2f} / {b:.2f} / {c:.2f}" for a, b, c in zip(s1, (1 - s1) * s2, (1 - s1) * (1 - s2))]})
    if core.use_track:
        out["R(T) hatótáv [mm]"] = core.range_mm(th).flatten().cpu().numpy()
        out["σ0 nyom [mm]"] = ex("log_st")
        if core.att:
            out["λ_att [mm]"] = ex("log_latt")
        if core.ms:
            out["c_ms [mm]"], out["k_ms"] = ex("log_cms"), g["ms_pow"]
        if core.track_halo:
            out["f delta-udvar"], out["l delta-udvar [mm]"] = 1 / (1 + np.exp(-g["logit_th"])), ex("log_lh")
        if core.bragg:
            out["q Bragg"] = g["q_bk"]
    if core.use_spike:
        out.update({"csúcs z [mm]": g["sp_z"], "csúcs σ_z [mm]": ex("log_sp_sz"), "csúcs σ_r [mm]": ex("log_sp_sr")})
    if core.endblob:
        out.update({"folt σ_z [mm]": ex("log_sez"), "folt σ_r [mm]": ex("log_ser")})
    if core.far:
        out.update({"távoli udvar L [mm]": ex("log_Lf"), "távoli udvar R [mm]": ex("log_Rf")})
    return {k: [x if isinstance(x, str) else float(x) for x in (v if hasattr(v, "__len__") else [v] * len(p))] for k, v in out.items()}


def coef_table(core):
    rows = []
    cm = (core.coef * core.mask).detach().cpu().numpy()
    for i, n in enumerate(core.names):
        d = int(core.mask[i].sum().item()) - 1
        rows.append([n, d] + [float(v) for v in cm[i, :d + 1]])
    return rows


NOTE = {
    "electron": "Elektromágneses zápor, nyom nincs. Kis impulzuson a belépés körüli csúcs jelentős, nagyobb impulzuson a zápor dominál. "
                "A ~1 %-os távoli udvar hozta a legnagyobb javulást a kiegészítő tagok közül; a szimulációs igazság szerint ez azonnali "
                "elektromágneses (e±/γ) energia, nem neutron, és a modell 5–20 GeV-en ~30 %-kal alulbecsli.",
    "muon": "Minimálisan ionizáló nyom: az energia nagy része a nyomban van (a primer saját ionizációja és a közeli delta-elektronok). "
            "A nyom a mélységgel a többszörös szórás szerint szélesedik; a Highland-képlet az LFHCal rétegeire a 40 mm-es acél előlappal "
            "≈ 62–63 mm-t és 1,35–1,41-es kitevőt ad. A végén Bragg-felfutás (Bragg–Kleeman: q ≈ 0,43); a hatótáv közel lineáris T-ben.",
    "proton": "Kis impulzuson megálló proton (nyom + csúcs + folt), nagyobb impulzuson hadronzápor. A hatótáv T = 0,5–1 GeV-en 4–16 %-on "
              "belül egyezik a CSDA-hatótávval (az előlap után); a k_R kitevő globális, a CSDA helyi kitevője itt 1,3–1,6. A csillapítási "
              "hossz ≈ 1,26 λ_I. A nyom szélesedése, delta-udvara és Bragg-kitevője itt effektív: a hadronmag egy részét is ezek írják le.",
    "pion": "Mint a proton: nyom az első magkölcsönhatásig, utána hadronzápor. A csillapítási hossz ≈ 1,31 λ_π, a protonénál hosszabb, "
            "ahogy a pion kölcsönhatási hossza is. A nyom kiegészítőinek paraméterei effektívek.",
}
COMP = ("zápor", "nyom", "csúcs", "folt", "távoli udvar")
Z100 = {k: f"{k}, z'=100 [mm]" for k in ("σ mag", "R Cauchy", "λ udvar")}


def pct(v):
    if v < 1e-4:
        return "< 0.01 %"
    return f"{100 * v:.2g} %" if v < 0.01 else f"{100 * v:.0f} %"


def vol(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def summary(particle, D, models, vals):
    """Rövid, a számokból generált összefoglaló a táblázatok elé."""
    t = json.load(open(os.path.join(D, "test_v3.json")))
    s2 = json.load(open(os.path.join(D, "s2_v3.json")))
    n, g = t["nll"], t["gap_closed"]
    ev = vals["slab"]
    i1, i20 = P_SHOW.index(1.0), P_SHOW.index(20.0)
    shares = "; ".join(f"{c} {pct(ev['a detektorban: ' + c][i1])} → {pct(ev['a detektorban: ' + c][i20])}"
                       for c in COMP if ("a detektorban: " + c) in ev)
    out = ["## Röviden", "",
           f"- **Mi ez:** {NOTE[particle]}",
           f"- **Pontosság (teszt-NLL, report-mérce, kisebb = jobb):** új formula {n['std:new']:.4f}, az ugyanígy illesztett baseline × Ā "
           f"{n['std:base_acc']:.4f}, a finomhangolt háló {n['std:mlp_ft']:.4f}. A baseline és a háló közti rés {100 * g['std:new']:.0f} %-át "
           f"zárja be (3D-mérce: {100 * g['slab:new']:.0f} %; független s2 minta: {100 * s2['gap_closed']['std:new']:.0f} % / "
           f"{100 * s2['gap_closed']['slab:new']:.0f} %).",
           f"- **A detektorba eső energia megoszlása, 1 GeV → 20 GeV** (3D-mérce): {shares}."]
    core = models["slab"][1]
    if core.use_track:
        with torch.no_grad():
            kR = float(core.theta(torch.log(torch.tensor([1.0], device=core.coef.device)))["k_R"].flatten()[0])
        out.append(f"- **Nyom:** hatótáv R ≈ {ev['R(T) hatótáv [mm]'][i1]:.0f} mm 1 GeV-en (kitevő k_R ≈ {kR:.2f}); csillapítási hossz "
                   f"λ_att ≈ {ev['λ_att [mm]'][0]:.0f} mm; szélesedés c ≈ {ev['c_ms [mm]'][0]:.0f} mm, k ≈ {ev['k_ms'][0]:.2f}; "
                   f"Bragg-kitevő q ≈ {ev['q Bragg'][0]:.2f}.")
    f100 = ev["f mag / test / udvar, z'=100"]
    out.append(f"- **Keresztmetszet z' = 100 mm-en, 1 → 20 GeV:** Gauss-mag σ ≈ {ev[Z100['σ mag']][i1]:.0f} → {ev[Z100['σ mag']][i20]:.0f} mm, "
               f"Cauchy-test R ≈ {ev[Z100['R Cauchy']][i1]:.0f} → {ev[Z100['R Cauchy']][i20]:.0f} mm, udvar λ ≈ {ev[Z100['λ udvar']][i1]:.0f} mm; "
               f"arányok (mag / test / udvar) {f100[i1]} → {f100[i20]}.")
    sh = {r["shape"]: r for r in json.load(open(os.path.join(D, "shapes.json")))["shapes"]}
    parts = []
    for lab, k in (("szuper-tojás", "98% superegg (refit)"), ("csepp", "98% drop (refit)"), ("kapszula", "98% capsule (refit)"),
                   ("kalibrált izoszint", "98% iso-density (new formula)")):
        if k in sh:
            r = sh[k]
            parts.append(f"{lab} {vol(r['V_in'])} L ({vol(r['V_slab'])} L a detektorban, lefedés {r['cov']:.2f})")
    out.append("- **98 %-os burkok** (a legkisebb test, amelyben a záporok 98 %-ában az energia ≥ 95 %-a bent van; 20 bin átlaga): "
               + "; ".join(parts) + ".")
    out += measurement_bullets(D)
    import re
    return [re.sub(r"(\d)\.(\d)", r"\1,\2", x) for x in out] + [""]


def rng(vals, scale=100, fmt_="{:.0f}"):
    lo, hi = min(vals) * scale, max(vals) * scale
    a, b = fmt_.format(lo), fmt_.format(hi)
    return a if a == b else f"{a}–{b}"


def measurement_bullets(D):
    """Az új mérések (független s2 minta) részecskénkénti eredményei, ha megvannak."""
    out = []
    f = os.path.join(D, "profile_metrics.json")
    if os.path.exists(f):
        s = json.load(open(f))["summary"]["all"]
        g = lambda m, q: f"{1e3 * s[f'{m}:{q}']:.2f}"
        out.append(f"- **Átlagos profilok** (független s2 minta, szeparációs erő × 10⁻³, mélység / sugár, kisebb = jobb): új formula "
                   f"{g('new formula', 'S_z')} / {g('new formula', 'S_r')}, háló {g('MLP', 'S_z')} / {g('MLP', 'S_r')}, régi képlet "
                   f"{g('baseline', 'S_z')} / {g('baseline', 'S_r')}; statisztikai padló {1e3 * s['floor_S_z']:.2f} / {1e3 * s['floor_S_r']:.2f}."
                   + (" A pion 3D-hálója degenerált (tömege gyakorlatilag a detektor-szeleten kívül van), az értékei nem megbízhatók." if "pion" in D else ""))
    f = os.path.join(D, "front_region.json")
    if os.path.exists(f):
        b = json.load(open(f))["bins"]
        out.append(f"- **A belépés előtti tartomány (z' < 0) energiája a beesési szöggel** (θ ≈ {b[0]['theta_mean']:.2f} → "
                   f"{b[-1]['theta_mean']:.2f} rad, p ≥ 2 GeV): adat {100 * b[0]['data']:.1f} → {100 * b[-1]['data']:.1f} %, új formula "
                   f"{100 * b[0]['new formula']:.1f} → {100 * b[-1]['new formula']:.1f} %, háló {100 * b[0]['MLP']:.1f} → {100 * b[-1]['MLP']:.1f} %.")
    f = os.path.join(D, "truth_decomposition.json")
    if os.path.exists(f):
        G = json.load(open(f))["groups"]
        far = "távoli (r≥300)"
        fr = [g["region"][far] for g in G.values()]
        neut = [g["region_cat"][far]["neutronból"] / max(g["region"][far], 1e-12) for g in G.values()]
        em = [g["region_cat"][far]["EM"] / max(g["region"][far], 1e-12) for g in G.values()]
        prompt = [g["region_time"][far]["<5 ns"] / max(g["region"][far], 1e-12) for g in G.values()]
        mfar = [g["model"]["region"][far] for g in G.values()]
        mhalo = [g["model"]["region_comp"][far].get("távoli udvar", 0.0) / max(g["model"]["region"][far], 1e-12) for g in G.values()]
        prim = [g["cat"]["primer"] for g in G.values()]
        late = [g["late_frac_of_all"] for g in G.values()]
        out.append(f"- **Szimulációs igazság** (s2, 0–20 GeV négy impulzus-tartományában): a beeső részecske saját lerakódása "
                   f"{rng(prim)} %, a neutronok leányaié {rng([g['cat']['neutronból'] for g in G.values()])} %; a 100 ns után érkező (a fitből "
                   f"kihagyott) energia {rng(late, fmt_='{:.1f}')} %. Az r ≥ 300 mm-es tartomány az energia {rng(fr, fmt_='{:.2f}')} %-a "
                   f"(a modellben {rng(mfar, fmt_='{:.2f}')} %), ennek {rng(neut)} %-a neutronból, {rng(em)} %-a elektromágneses és "
                   f"{rng(prompt)} %-a azonnali (< 5 ns); a modellben ennek {rng(mhalo)} %-át viszi a távoli udvar tag.")
    f = os.path.join(D, "theta_entrance_two.json")
    if os.path.exists(f):
        t = json.load(open(f))
        d = t["s2"]["theta_minus_control"]
        b = json.load(open(os.path.join(D, "theta_entrance_beta.json"))) if os.path.exists(os.path.join(D, "theta_entrance_beta.json")) else None
        fr = t.get("front") or []
        fr_txt = ""
        if fr:
            a = lambda k: f"{100 * fr[0][k]:.1f} → {100 * fr[-1][k]:.1f} %"
            fr_txt = f" A belépés előtti energia (θ ≈ 0,25 → 0,57 rad): adat {a('data')}, végleges formula {a('final')}, szögfüggő {a('theta')}."
        out.append(f"- **Szögfüggő kiegészítés – nem modellpredikció.** A beesési szöget eseményenként bemenetként megadjuk, és "
                   f"{len(t['beta'])} belépési paraméter (csúcs, zápor- és nyomkezdet) a szöggel lineárisan változhat: P + β·(1/cosθ − 1,1). "
                   f"A β-kat az adatra illesztettük (empirikus korrekció, a formula fizikája nem jósolja); a többi szám ezen a kártyán szög "
                   f"nélküli. 3D teszt-NLL: {t['test']['control']:.4f} → {t['test']['theta']:.4f} (s2: {d[0]:+.4f} ± {d[1]:.4f} nat/esemény)."
                   + fr_txt + (" Müonnál az illesztés átrendezi a tagokat, ott a tagok fizikailag nem értelmezhetők." if "muon" in D else ""))
    f = os.path.join(D, "time_split.json")
    if os.path.exists(f):
        t = json.load(open(f))
        fd = [v["mean"] for v in t["delayed_fraction_test"].values()]
        pr, de = t["prompt"]["describe"], t["delayed"]["describe"]
        out.append(f"- **Időben szétválasztva** (cellánként: gyors az a cella, amelyet az első találat után < 5 ns-on belül, késleltetett, amelyet "
                   f"később talált el először valami; a késleltetett hányad így alsó becslés): a késleltetett cellák energiája eseményenként "
                   f"átlagosan {rng(fd, fmt_='{:.1f}')} %. Az 5–20 GeV-es átlagos sugár / r90 (modell): gyors {rng(pr['mean_r_mm'][2:], scale=1)} / "
                   f"{rng(pr['r90_mm'][2:], scale=1)} mm, késleltetett {rng(de['mean_r_mm'][2:], scale=1)} / {rng(de['r90_mm'][2:], scale=1)} mm; "
                   f"az udvar hossza z' = 100 mm-en: gyors {rng(t['prompt']['halo_lambda_z100_mm'], scale=1)} mm, késleltetett "
                   f"{rng(t['delayed']['halo_lambda_z100_mm'], scale=1)} mm.")
    f = os.path.join(D, "fluctuations_pca.json")
    if os.path.exists(f):
        t = json.load(open(f))["groups"]
        ev3 = [sum(g["explained_variance"][:3]) for g in t.values()]
        mean_S = lambda key: [1e3 * np.nanmean(list(g["S"][key].values())) for g in t.values()]
        fl = [1e3 * np.nanmean(list(g["floor"].values())) for g in t.values()]
        out.append(f"- **Záporonkénti ingadozás** (PCA az eseményenkénti profilokon): az első 3 főkomponens a variancia {rng(ev3)} %-a. Az eseményenkénti "
                   f"mennyiségek szeparációs ereje (× 10⁻³, a két félminta padlója {rng(fl, scale=1, fmt_='{:.0f}')}): bármely átlag-modell "
                   f"{rng(mean_S('gauss, k=0'), scale=1)} (binelési állandó), "
                   f"3 főkomponens Gauss-együtthatókkal {rng(mean_S('gauss, k=3'), scale=1)}, 10 főkomponens a tanító együtthatóiból "
                   f"{rng(mean_S('empirical, k=10'), scale=1)}.")
    import re
    return [re.sub(r"(\d)\.(\d)", r"\1,\2", x) for x in out]


GLOSSARY = ["## Hogyan olvasd a táblázatokat", "",
            "- Oszlopok: az impulzus p [GeV]; minden sor egy illesztett mennyiség értéke ennél az impulzusnál.",
            "- **w …**: a komponens súlya a vágatlan forgástestben. **a detektorban: …**: a komponens aránya a detektorba (a ferde "
            "belépési sík mögé) eső energiában – ez a fizikailag értelmezhető energiaarány.",
            "- **z_s, w1, ℓ, w2, κ** (plató) ill. **z0, t_max, α−1** (gamma): a zápor mélységprofilja. **σ mag, R Cauchy, λ udvar** és "
            "**f mag / test / udvar**: a keresztmetszet három tagjának szélessége és aránya az adott mélységben.",
            "- **R(T)**: a nyom hossza (hatótáv); **σ0**: a nyom szélessége a belépésnél; **c_ms, k_ms**: a többszörös szórás miatti "
            "szélesedés; **λ_att**: csillapítási hossz; **f, l delta-udvar**: a nyom körüli udvar aránya és hossza; **q Bragg**: a "
            "Bragg-felfutás kitevője; **csúcs …, folt …**: a két Gauss-folt helye és szélessége; **távoli udvar L, R**: mélységi és "
            "sugárirányú hossz (R → ∞: sugárirányban nincs levágás).",
            "- Két illesztés van: a **3D-mérce** (eseményenként a detektor-szeletre normálva; ez írja le a forgástestet) és a "
            "**report-mérce** (× Ā; ebből számolódik az r95- és az izoszint-alakot). A kettő értékei kissé eltérnek.", ""]


for particle in ("electron", "muon", "proton", "pion"):
    D = os.path.join(ROOT, "results", "particles", particle)
    C = protonlab.load_common_for(particle, "s1", OUT_DIR=D)
    models = {m: shapelab.build_final(C, particle, D, 0.0, 1.0, metric=m) for m in ("slab", "std")}   # v0, vs: from the checkpoint
    t = models["slab"][2]
    lines = [f"# A végleges formula – {HU[particle]}", "",
             f"Tömeg m = {MASS[particle]} GeV; változat `{t['variant']}`; a paraméterek az x = (ln(T/GeV) − v0)/vs változó polinomjai, "
             f"T = √(p² + m²) − m. Illesztés: a tanítóhalmaz találataira, eseményenkénti likelihooddal (`shapelab.fit_fast`), a fokszám "
             f"és a tagok a validáción választva.", ""]
    core = models["slab"][1]
    lines += ["## A képlet", "",
              "ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H — minden tag külön a (z', r) rácsra normálva "
              "(σ(·) = logisztikus függvény; ami ennél a részecskénél nincs bekapcsolva, az kimarad):", "",
              "- **Zápor** S = L(z')·T(r | z'). Plató-mélységprofil: L ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ z'/1 m); "
              "gamma: L ∝ s^(α−1)·e^(−(α−1) s/t_max), s ≈ z' − z0. Keresztmetszet: "
              "T = f1·e^(−r²/2σ²)/(2πσ²) + f2·R²/(π(r² + R²)²) + f3·e^(−r/λ)/(2πλ(r + 1 mm)); ln σ, ln R, ln λ és a keverési logitok "
              "lineárisak a mélységben (σ-nál és a logitoknál négyzetes taggal is, ha „görbület” be van kapcsolva); minden mélységben "
              "az r < 700 mm korongra normálva.",
              "- **Nyom** U ∝ σ((z' − z_t0)/4 mm)·σ((R(T) − z')/w_R)·e^(−κ_t z'/R)·e^(−z'/λ_att)·(1 − z'/R + 0,03)^(−q) "
              "× [(1 − f)·Gauss(r; σ(z')) + f·e^(−r/l)/(2πl(r + 1 mm))], ahol R(T) = R1·T^(k_R) − R0 (hatótáv), w_R = w·R + 3 mm, "
              "σ(z')² = σ0² + (c·(z'/1 m)^k / βp)² (többszörös szórás), βp = (T² + 2Tm)/(T + m).",
              "- **Csúcs** a belépésnél: D ∝ e^(−(z' − z_sp)²/2σ_z²)·e^(−r²/2σ_r²); **folt** a hatótáv végén: "
              "E ∝ e^(−(z' − R(T))²/2σ_ez²)·e^(−r²/2σ_er²); **távoli udvar**: H ∝ σ(z'/10 mm)·e^(−z'/L)·e^(−r/R)/(r + 10 mm).",
              "- **Súlyok**: w_távoli = σ(a_far), w_folt = σ(a_end)·(1 − w_távoli); a zápor, a nyom és a csúcs a maradékon osztozik "
              "(a csúcs a nyomhoz kötve: f1 = d1/(d1 + L_t), L_t a nyom hossza).",
              "- Minden nyers paraméter P(x) = Σ c_k x^k, x = (ln(T/GeV) − v0)/vs, T = √(p² + m²) − m; „log_” = a mennyiség "
              "logaritmusa, „logit_” = arány logitja.", ""]
    on = [hu for k, hu in TERMS if getattr(core, k, False)]
    lines += [f"Tagok: zápor ({'plató' if core.long == 'plateau' else 'gamma'}-mélységprofil × [Gauss-mag + Cauchy-test + e^(−r/λ)/r udvar])"
              + (" + nyom (hatótáv, csillapítás)" if core.use_track else "") + " + csúcs a belépésnél"
              + "".join(f" + {x}" for x in on) + ".", ""]
    vals, blocks = {}, []
    for metric, label in (("slab", "3D-mérce (eseményenként a detektor-szeletre normálva) – ez a forgástest"),
                          ("std", "report-mérce (× Ā; ebből számolódik az r95- és az izoszint-alakot)")):
        m, core, t, K = models[metric]
        v0, vs = float(core.v0), float(core.vs)
        n = t["degrees"]["new_slab" if metric == "slab" else "new_std"]
        ev = evaluate(core, P_SHOW, torch.load(os.path.join(D, "abar.pt"), map_location=core.coef.device))
        vals[metric] = ev
        blocks += [f"## {label}", "", f"Fokszám n = {n}, {core.n_params()} paraméter; v0 = {v0:.4f}, vs = {vs:.4f}. A „w” sorok a "
                  "vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé "
                  "(a szögekre átlagolt elfogadással, Ā-val súlyozva).", "",
                  "| paraméter | " + " | ".join(f"p = {p:g}" for p in P_SHOW) + " |", "|---|" + "---:|" * len(P_SHOW)]
        for k, row in ev.items():
            if k == "p [GeV]":
                continue
            blocks.append(f"| {k} | " + " | ".join(x if isinstance(x, str) else fmt(x) for x in row) + " |")
        blocks.append("")
        rows = coef_table(core)
        path = os.path.join(D, f"formula_coefficients_{metric}.csv")
        with open(path, "w") as fh:
            fh.write("parameter,degree," + ",".join(f"c{k}" for k in range(core.degree + 1)) + "\n")
            for r in rows:
                fh.write(",".join([r[0], str(r[1])] + [f"{v:.8g}" for v in r[2:]]) + "\n")
        blocks += [f"Együtthatók: `{os.path.basename(path)}` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).", ""]
    lines = lines[:4] + summary(particle, D, models, vals) + lines[4:] + GLOSSARY + blocks
    json.dump(vals, open(os.path.join(D, "formula_values.json"), "w"), indent=1)
    open(os.path.join(D, "formula.md"), "w").write("\n".join(lines))
    print(particle, "ok", flush=True)
