"""A README táblázatai közvetlenül az eredményfájlokból (results/particles/*). Kimenet: markdown a stdout-ra."""
import json
import os

import numpy as np

ROOT = "/home/user/tamas/shape-form-proton/results/particles"
HU = {"electron": "elektron", "pion": "pion", "proton": "proton", "muon": "müon"}
BASE = {"electron": "GFlash", "pion": "GFlash", "proton": "GFlash", "muon": "henger"}
VAR = {"full": "zápor + nyom + csúcs", "full_att": "zápor + nyom (csillapítással) + csúcs", "notrack": "zápor + csúcs",
       "plateau": "plató-zápor + csúcs", "plateau_track": "plató-zápor + nyom + csúcs"}
f4 = lambda x: f"{x:.4f}".replace(".", ",")
d4 = lambda d: f"{d[0]:+.4f} ± {d[1]:.4f}".replace(".", ",")
pc = lambda x: f"{100 * x:.0f} %"


def J(p, n):
    for k in (3, 2):                                        # the final round (physics extensions) when available
        if n in ("test", "s2") and os.path.exists(os.path.join(ROOT, p, f"{n}_v{k}.json")):
            n = f"{n}_v{k}"
            break
    f = os.path.join(ROOT, p, f"{n}.json")
    return json.load(open(f)) if os.path.exists(f) else None


rows_std, rows_3d, rows_s2, rows_var, rows_round = [], [], [], [], []
for p in ("electron", "pion", "proton", "muon"):
    t, s2, rep, var = J(p, "test"), J(p, "s2"), J(p, "repro"), J(p, "variants")
    if t is None:
        continue
    n, P = t["nll"], t["params"]
    rows_std.append(f"| {HU[p]} | {f4(n['std:report_baseline'])} ({P['report_baseline']}) | {f4(n['std:base_acc'])} | "
                    f"{f4(n['std:new_n2'])} ({P['new_n2']}) | **{f4(n['std:new'])}** ({P['new']}) | {d4(t['delta_vs_base']['std:new'])} | "
                    f"{f4(n['std:report_mlp'])} / {f4(n['std:mlp_ft'])} ({P['mlp_ft']}) | {pc(t['gap_closed']['std:new'])} |")
    rows_3d.append(f"| {HU[p]} | {f4(n['slab:base'])} | {f4(n['slab:new_n2'])} | **{f4(n['slab:new'])}** ({P['new_slab']}) | "
                   f"{d4(t['delta_vs_base']['slab:new'])} | {f4(n['slab:mlp_ft'])} | {pc(t['gap_closed']['slab:new'])} |")
    if s2 is not None:
        rows_s2.append(f"| {HU[p]} | {s2['n_events']} | {d4(s2['delta_vs_base']['std:new'])} | {d4(s2['delta_vs_base']['slab:new'])} | "
                       f"{pc(s2['gap_closed']['std:new'])} / {pc(s2['gap_closed']['slab:new'])} |")
    parts = t['variant'].split('+')
    rows_var.append(f"| {HU[p]} | {VAR.get(parts[0], parts[0])}{''.join(' + ' + q for q in parts[1:])} | "
                    + " | ".join(f4(var['val'][k]) for k in VAR) + f" | {t['degrees']['new_std']} / {t['degrees']['new_slab']} |")
    k_ = 3 if os.path.exists(os.path.join(ROOT, p, "test_v3_nll.npz")) else 2
    f1, f2 = os.path.join(ROOT, p, "test_nll.npz"), os.path.join(ROOT, p, f"test_v{k_}_nll.npz")
    if os.path.exists(f1) and os.path.exists(f2):
        a, b = dict(np.load(f1)), dict(np.load(f2))
        dd = {m: b[f"{m}:new"] - a[f"{m}:new"] for m in ("std", "slab")}
        rows_round.append(f"| {HU[p]} | {t['variant']} | " + " | ".join(f"{dd[m].mean():+.4f} ± {dd[m].std(ddof=1) / np.sqrt(len(dd[m])):.4f}".replace('.', ',') for m in ("std", "slab")) + " |")

print("Report-mérce (teszt-NLL, kisebb = jobb; zárójelben a paraméterszám; Δ = páros különbség a × Ā-val újraillesztett baseline-hoz):\n")
print("| részecske | baseline (report) | baseline × Ā | új, n = 2 | új (1-SE) | Δ új − baseline × Ā | MLP report / finomhangolt | rés bezárva |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
print("\n".join(rows_std))
print("\n3D-mérce (a detektor-szeletre normálva; minden modell így tanítva):\n")
print("| részecske | baseline | új, n = 2 | új (1-SE) | Δ új − baseline | MLP | rés bezárva |")
print("|---|---:|---:|---:|---:|---:|---:|")
print("\n".join(rows_3d))
print("\nFüggetlen második minta (s2, más véletlenmag; Δ = új − baseline, report-mérce / 3D):\n")
print("| részecske | események | Δ report-mérce | Δ 3D | rés bezárva |")
print("|---|---:|---:|---:|---:|")
print("\n".join(rows_s2))
if rows_round:
    print("\nA kiegészítő tagok hozzájárulása (végleges formula − alapváltozat, teszt, páros):\n")
    print("| részecske | változat | report-mérce | 3D |")
    print("|---|---|---:|---:|")
    print("\n".join(rows_round))
print("\nVáltozatválasztás (validációs NLL, report-mérce × Ā, n = 2; a választott a legkevesebb paraméteres a legjobbtól 1 SE-n belül):\n")
print("| részecske | választott | " + " | ".join(VAR[k] for k in VAR) + " | fokszám (report / 3D) |")
print("|---|---|" + "---:|" * len(VAR) + "---:|")
print("\n".join(rows_var))

# envelope bodies
print("\nBurok-testek (95 % energia a záporok 98 %-ában; átlag a 20 binre, liter; zárójelben a teszt-lefedés):\n")
print("| részecske | csepp (report 03, újraillesztve): tartomány / szelet | szuper-tojás: tartomány / szelet | különbség a szeletben |")
print("|---|---:|---:|---:|")
for p in ("electron", "pion", "proton"):
    D = os.path.join(ROOT, p, "bodies_E95_EV98")
    if not os.path.isdir(D):
        continue
    v = {}
    for fam in ("drop", "superegg"):
        rs = [json.load(open(os.path.join(D, f"{fam}_bin{b:02d}.json"))) for b in range(20) if os.path.exists(os.path.join(D, f"{fam}_bin{b:02d}.json"))]
        if len(rs) == 20:
            v[fam] = (np.mean([r["V_in_L"] for r in rs]), np.mean([r["V_slab_L"] for r in rs]), np.mean([r["cov_test"] for r in rs]))
    if len(v) == 2:
        dd, se = v["drop"], v["superegg"]
        g = lambda x: f"{x:.1f}".replace(".", ",") if x < 100 else f"{x:.0f}"
        print(f"| {HU[p]} | {g(dd[0])} / {g(dd[1])} ({dd[2]:.3f}) | {g(se[0])} / {g(se[1])} ({se[2]:.3f}) | {100 * (se[1] / dd[1] - 1):+.1f} % |".replace("0.", "0,"))
v = os.path.join(ROOT, "verify.json")
if os.path.exists(v):
    print("\nEllenőrzés (verify.json):")
    for p, r in json.load(open(v)).items():
        if isinstance(r, dict):
            print(f"- {p}: report-mérce újraszámolva max |eltérés| {r['std_new_max_abs_diff_vs_pipeline']:.1e}; 3D nsub 8 vs 2: "
                  f"{r['slab_new_mean_diff_nsub8_minus_nsub2']:+.1e}; konzisztens: {r['consistent']}; normáltság {r['max_abs_sum_P_minus_1']:.1e}; "
                  f"választás tesztet nem használ: {not r['selection_files_mention_test']}; teszt és s2 előjele egyezik: {r['same_sign_test_s2']}")
