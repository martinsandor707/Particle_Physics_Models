"""protocol – a report fokszám-választási protokollja (01_gflash: `select_degree`, 1-SE szabály) tetszőleges formulacsaládra.

A `00_common.select_degree` csak a `Formula(long, lat)` osztályt ismeri; ez a változat bármely `make(n)` gyárral működik,
de ugyanazt csinálja: minden n fokszámra tanít (az n-edik modell az (n-1)-edik megoldásából indul), a validációs NLL
alapján n* a legjobb, és a választott n a LEGKISEBB fokszám, amelynek páros validációs különbsége n*-hoz képest
egy standard hibán belül van. A tesztet csak a kiválasztott modellre nézzük.
"""
import os

import numpy as np
import pandas as pd
import torch

import shapelab


def one_se_choice(val):
    """val = {kulcs: eseményenkénti validációs NLL}. Vissza: (legjobb kulcs, {kulcs: (Δ, SE)})."""
    best = min(val, key=lambda k: float(np.mean(val[k])))
    diff = {k: shapelab.paired(val[k], val[best]) if k != best else (0.0, 0.0) for k in val}
    return best, diff


def select_degree_generic(C, data, make, name, degrees=range(1, 7), protocol="report", epochs=150, lr=0.03,
                          fast_steps=3000, particle="proton", log=print):
    """`make(n)` -> n-edfokú modell a DEVICE-on (kezdőértékekkel); `model.warm_start(prev)` az előző fokszámból indít.

    protocol="report": a közös `fit` (Adam, állandó lr, korai leállás patience=15, C["BATCH"] méretű kötegek) – a report tanítása;
    protocol="fast":   shapelab.fit_fast (Adam + koszinuszos lr) – ugyanaz a veszteség, gyorsabb és pontosabb optimum.
    Vissza: (választott modell, táblázat, {n: modell}, {n: val NLL tömb})."""
    models, val, rows, prev = {}, {}, [], None
    for n in degrees:
        m = make(n)
        if prev is not None:
            m.warm_start(prev)
        if log:
            log(f"--- degree {n} ---")
        if protocol == "report":
            hist = C["fit"](m, data, os.path.join(C["OUT_DIR"], particle, f"{name}_n{n}.pt"), epochs=epochs, lr=lr)
            n_ep = len(hist["val"])
        else:
            hist = shapelab.fit_fast(C, data, m, steps=fast_steps, lr=lr, log=None)
            n_ep = fast_steps
        m.eval()
        val[n] = C["test_nll"](m, data["val"])
        rows.append({"n": n, "params": m.n_params(), "train NLL": float(C["test_nll"](m, data["train"]).mean()),
                     "val NLL": float(val[n].mean()), "epochs/steps": n_ep})
        models[n] = prev = m
        if log:
            log(f"    params {rows[-1]['params']}  val {rows[-1]['val NLL']:.4f}")
    tab = pd.DataFrame(rows).set_index("n")
    n_star, diff = one_se_choice(val)
    tab["Δval vs n*"] = [diff[n][0] for n in tab.index]
    tab["SE"] = [diff[n][1] for n in tab.index]
    ok = [n for n in tab.index if n == n_star or tab.loc[n, "Δval vs n*"] < tab.loc[n, "SE"]]
    n_sel = int(min(ok))
    tab["chosen"] = ["◀" if n == n_sel else "" for n in tab.index]
    if log:
        log(f"lowest validation NLL: n* = {n_star};  chosen by the 1-SE rule: n = {n_sel}")
    return models[n_sel], tab, models, val
