"""protonlab – a proton alak-kísérletek közös belépési pontja.

Sofiya `00_common.ipynb` notebookjának kódcelláit VÁLTOZATLANUL futtatja le egy névtérben
(ugyanaz a rács, detektor-konvolúció, likelihood, tanítás, export), és a proton-adatot
gyorsítótárból tölti (az első hívás a 16,7 GB-os CSV-ből építi fel, pontosan `load_data` szerint).

Használat:
    import protonlab
    C = protonlab.load_common()            # dict: a 00_common összes definíciója
    geo, data = protonlab.load_proton(C)   # data = {"train","val","test"} a DEVICE-on
"""
import json
import os

import sys

import matplotlib

if "ipykernel" not in sys.modules:          # scripts: no display; notebooks keep their inline backend
    matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON_NB = os.path.join(HERE, "00_common.ipynb")
BASELINE_DIR = "/home/user/sofiya/shape-form/results/proton"      # a report mentett proton-eredményei (csak olvasás)
DATA_DIR = "/home/user/data/simulations/regression-proton/"
CACHE = os.path.join(HERE, "cache", "proton_sets.pt")


def load_common(ns=None, **overrides):
    """A 00_common.ipynb kódcelláinak futtatása; a beállítások (OUT_DIR, BATCH, ...) felülírhatók."""
    ns = ns if ns is not None else {}
    ns.setdefault("PARTICLES", ["proton"])                           # így az áttekintő ábra cellája nem fut le
    ns.setdefault("DATA_DIR", DATA_DIR)
    ns.setdefault("OUT_DIR", os.path.join(HERE, "results"))
    ns.update(overrides)
    with open(COMMON_NB) as fh:
        nb = json.load(fh)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            exec(compile("".join(c["source"]), COMMON_NB, "exec"), ns)
    return ns


def load_proton(C):
    """(geo, {'train','val','test'}) – ugyanaz a felosztás, mint a notebookokban (load_data, SEED=1)."""
    import torch
    Showers, DEVICE = C["Showers"], C["DEVICE"]
    if os.path.exists(CACHE):
        ck = torch.load(CACHE, weights_only=False)
        sets = {k: Showers(v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(DEVICE) for k, v in ck["sets"].items()}
        return ck["geo"], sets
    df, geo, sets = C["load_data"]("proton")
    del df
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save({"geo": geo, "sets": {k: {"ev": {a: b.cpu() for a, b in s.ev.items()}, "off": s.off.cpu(),
                                         "xyz": s.xyz.cpu(), "w": s.w.cpu(), "box": s.box.cpu()}
                                     for k, s in sets.items()}}, CACHE)
    return geo, sets


# ----------------------------------------------------------------------------------------------------------------------
# Általánosítás a többi részecskére (electron, pion, muon, proton): ugyanaz a load_data, részecskénként külön gyorsítótár.
# ----------------------------------------------------------------------------------------------------------------------
DATA_ROOT = "/home/user/data/simulations"
REPORT_RESULTS = "/home/user/sofiya/shape-form/results"


def data_dir(particle):
    return os.path.join(DATA_ROOT, f"regression-{particle}") + "/"


def cache_path(particle, sample="s1"):
    if particle == "proton":
        return CACHE if sample == "s1" else os.path.join(HERE, "cache", "proton_s2_sets.pt")
    return os.path.join(HERE, "cache", f"{particle}_{sample}_sets.pt")


def load_common_for(particle, sample="s1", **overrides):
    """00_common névtér a részecske adatkönyvtárával és fájlmintájával."""
    return load_common(DATA_DIR=data_dir(particle), PARTICLES=[particle],
                       FILE_PATTERN="{particle}_standalone_N100000_M0_20GeV_reg_balanced_" + sample + "_v3.csv", **overrides)


def load_particle(C, particle, sample="s1"):
    """(geo, {'train','val','test'}) a részecske adott mintájára; az első hívás a CSV-ből építi (load_data, SEED=1)."""
    import torch
    Showers, DEVICE = C["Showers"], C["DEVICE"]
    path = cache_path(particle, sample)
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        return ck["geo"], {k: Showers(v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(DEVICE) for k, v in ck["sets"].items()}
    C2 = load_common_for(particle, sample, OUT_DIR=C["OUT_DIR"])
    df, geo, sets = C2["load_data"](particle)
    del df
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"geo": geo, "sets": {k: {"ev": {a: b.cpu() for a, b in s.ev.items()}, "off": s.off.cpu(), "xyz": s.xyz.cpu(),
                                         "w": s.w.cpu(), "box": s.box.cpu()} for k, s in sets.items()}}, path)
    return geo, {k: Showers(v.ev, v.off, v.xyz, v.w, v.box).to(DEVICE) for k, v in sets.items()}
