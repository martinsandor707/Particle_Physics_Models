"""Geometry audit, step 5c - controls and the information ladder on VALIDATION (no training, no registry row).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g5c_controls.py

(1) Negative controls: the ratio-form map with the acceptance of ANOTHER validation event (random permutation; and a
    permutation inside theta quintiles). If the gain came from a generic reshaping of the map and not from the
    event's own geometry, the controls would gain too.
(2) Information ladder, all in the ratio form  log P + ln A - ln Abar  with front + side faces:
        inputs (p, theta)                      A = mean of A_e over the TRAINING events of the same theta bin
                                               (64 equal-count bins; geometry only, nothing fitted)
        inputs (p, theta, phi, z_b)            A_e from the straight-line origin (no entry position)
        inputs (p, theta, phi, entry position) A_e from the true origin  [the geometry arm]
    The first two rows are DIAGNOSTICS for the design of the theta track; they are not part of the geometry arm.
(3) Share of the validation energy (K = 64 fold points) in cells with small acceptance.
(4) Bootstrap confidence intervals of the main deltas (event bootstrap, stratified by momentum bin).
AUDIT numbers on the laptop's fold points.
"""
import json
import time

import glib as L
from glib import G, common, legacy, metric, np, torch
import torch.nn.functional as F

t0 = time.time()
sets, extras, origin = L.load_cpu(("train", "val"))
tr, val = sets["train"], sets["val"]
N = len(val)
th_tr, th_va = extras["train"]["theta"], extras["val"]["theta"]
NB = 64
edges = np.quantile(th_tr, np.linspace(0, 1, NB + 1))
bin_tr = np.clip(np.searchsorted(edges[1:-1], th_tr, side="right"), 0, NB - 1)
bin_va = np.clip(np.searchsorted(edges[1:-1], th_va, side="right"), 0, NB - 1)
abar = {k: torch.from_numpy(v) for k, v in np.load(L.ARR / "abar_train.npz").items()}
V = {"F": ("front",), "FS": ("front", "side"), "FSB": ("front", "side", "back")}
rng = np.random.default_rng(20261126)
perm = rng.permutation(N)
qe = np.quantile(th_va, np.linspace(0, 1, 6))
qc = np.clip(np.searchsorted(qe[1:-1], th_va, side="right"), 0, 4)
perm_q = np.arange(N)
for q in range(5):
    i = np.flatnonzero(qc == q)
    perm_q[i] = rng.permutation(i)
out = {"split": "val", "n": N, "K": L.K, "seed": L.SEED, "floor": L.FLOOR, "theta_bins": NB,
       "control_fixed_points": {"random": int((perm == np.arange(N)).sum()), "within_quintile": int((perm_q == np.arange(N)).sum())}}

with L.gpulock.gpu_lock("geometry audit g5c (controls, theta-only template)"):
    out["machine"] = L.machine_tag()
    DEV = L.DEV
    # theta-binned templates from ALL training events (geometry only)
    tsum = {k: torch.zeros(NB, common.N_Z, common.N_R, dtype=torch.float64) for k in V}
    cnt = torch.zeros(NB, dtype=torch.float64)
    Otr = origin["train"]
    for s in range(0, len(tr), 4096):
        idx = np.arange(s, min(s + 4096, len(tr)))
        A = G.event_acceptance(tr.ev["R"][idx], torch.as_tensor(Otr[idx]), G.Volume(), V, nsub=L.NSUB, nphi=L.NPHI, device=DEV)
        b = torch.as_tensor(bin_tr[idx])
        for k in V:
            tsum[k].index_add_(0, b, A[k].double())
        cnt.index_add_(0, b, torch.ones(len(idx), dtype=torch.float64))
        print(f"theta templates {s + len(idx)}/{len(tr)}  {time.time() - t0:.0f} s", flush=True)
    ttheta = {k: (tsum[k] / cnt[:, None, None]).float() for k in V}
    for k in V:                                   # consistency: the bin-weighted mean of the theta templates is Abar
        back = (tsum[k].sum(0) / cnt.sum()).float()
        out.setdefault("theta_templates_recombine_to_abar_max_abs", {})[k] = float((back - abar[k]).abs().max())
    np.savez_compressed(L.ARR / "abar_train_by_theta.npz", edges=edges, **{k: v.numpy() for k, v in ttheta.items()})
    L.sidecar(L.ARR / "abar_train_by_theta.npz", split="train", n_events=len(tr), theta_bins=NB,
              note="mean of A_e over the training events of each theta bin (equal-count bins); geometry only; DIAGNOSTIC for the theta track")
    A = G.event_acceptance(val.ev["R"], torch.as_tensor(origin["val"]), G.Volume(), V, nsub=L.NSUB, nphi=L.NPHI, device=DEV)
    ol = torch.as_tensor(G.straight_line_origin(th_va, extras["val"]["phi"], val.ev["zb"].numpy()))
    A_line = G.event_acceptance(val.ev["R"], ol, G.Volume(), V, nsub=L.NSUB, nphi=L.NPHI, device=DEV)
    mlp = legacy.load_mlp_std()
    formula = legacy.load_component_formula(tr)
    ab = {k: abar[k].to(DEV) for k in V}
    tt = {k: ttheta[k].to(DEV) for k in V}
    bin_t, perm_t, permq_t = torch.as_tensor(bin_va), torch.as_tensor(perm), torch.as_tensor(perm_q)

    def make(b, idx):
        i = idx.cpu()
        d = {}
        for name, lp in (("mlp", mlp(b.lnp)), ("formula", formula(b.lnp))):
            d[name] = lp
            for k in V:
                d[f"{name}*{k}/abar"] = G.apply_acceptance(lp, A[k][i].to(DEV), L.FLOOR, template=ab[k])
                d[f"{name}*{k}_theta/abar"] = G.apply_acceptance(lp, tt[k][bin_t[i].to(DEV)], L.FLOOR, template=ab[k])
                d[f"{name}*{k}_line/abar"] = G.apply_acceptance(lp, A_line[k][i].to(DEV), L.FLOOR, template=ab[k])
            for k in ("FS", "FSB"):
                d[f"{name}*{k}_perm/abar"] = G.apply_acceptance(lp, A[k][perm_t[i]].to(DEV), L.FLOOR, template=ab[k])
                d[f"{name}*{k}_permq/abar"] = G.apply_acceptance(lp, A[k][permq_t[i]].to(DEV), L.FLOOR, template=ab[k])
        return d

    data = val.to(DEV)
    res = G.score_variants(make, data, K=L.K, seed=L.SEED, batch=L.BATCH, monitors=False)
    st = np.flatnonzero(val.ev["zb"].numpy() >= 1221.0)
    sub = data.subset(torch.as_tensor(st, device=DEV))
    st_t = torch.as_tensor(st)

    def make_b(b, idx):
        i = st_t[idx.cpu()]
        d = {}
        for name, lp in (("mlp", mlp(b.lnp)), ("formula", formula(b.lnp))):
            d[name] = lp
            for k in ("F", "FS"):
                d[f"{name}*{k}_theta/abar"] = G.apply_acceptance(lp, tt[k][bin_t[i].to(DEV)], L.FLOOR, template=ab[k])
                d[f"{name}*{k}_line/abar"] = G.apply_acceptance(lp, A_line[k][i].to(DEV), L.FLOOR, template=ab[k])
        return d

    blk = G.score_variants(make_b, sub, K=L.K, seed=L.SEED, batch=L.BATCH, bulk=metric.Bulk(short="raise"))

    # (3) validation energy in low-acceptance cells (fold points of the report metric)
    thr = (0.0, 1e-3, 1e-2, 0.1, 0.5)
    share = {k: np.zeros(len(thr)) for k in ("FS", "FSB")}
    with torch.inference_mode():
        gen = torch.Generator(device=DEV).manual_seed(L.SEED)
        for s in range(0, N, L.BATCH):
            idx = torch.arange(s, min(s + L.BATCH, N), device=DEV)
            b = data.batch(idx)
            u = metric.draw(b, data.box, L.K, gen)
            pts = b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)
            iz, ir = common.locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
            for k in share:
                af = A[k][idx.cpu()].to(DEV)[b.ev[:, None].expand_as(iz), iz, ir]
                for j, t in enumerate(thr):
                    share[k][j] += float((b.w[:, None] * (af <= t)).sum() / L.K)
    out["val_energy_share_of_fold_points_in_cells_with_A_le"] = {k: dict(zip(map(str, thr), (v / N).tolist())) for k, v in share.items()}
    del data, sub
    torch.cuda.empty_cache()

rep = np.load(L.ARR / "val_report.npz")
for name in ("mlp", "formula"):
    assert np.array_equal(res[name]["nll"], rep[f"{name}|nll"])                       # same fold points as g5
    assert np.array_equal(res[f"{name}*FS/abar"]["nll"], rep[f"{name}*FS/abar|nll"])
names = [k for k in res if not k.startswith("_")]
np.savez_compressed(L.ARR / "val_controls.npz", event_number=extras["val"]["event_number"], perm=perm, perm_within_quintile=perm_q,
                    theta_bin=bin_va, **{f"{k}|nll": res[k]["nll"] for k in names},
                    **{f"bulk|{k}|nll": blk[k]["nll"] for k in blk if not k.startswith("_")}, bulk_stratum_index=st)
L.sidecar(L.ARR / "val_controls.npz", split="val", n_events=N, K=L.K, seed=L.SEED, batch=L.BATCH, machine=out["machine"], floor=L.FLOOR,
          metric="report metric (keys '<map>|nll') and G-bulk on the stratum zb >= 1221 mm (keys 'bulk|<map>|nll')",
          label="AUDIT numbers; controls and diagnostics; '*_theta' and '*_line' maps are NOT the geometry arm")
tab = {}
for k in names:
    if k in ("mlp", "formula"):
        continue
    b = k.split("*")[0]
    tab[k] = L.paired0(res[k]["nll"] - res[b]["nll"])
out["report_metric_delta"] = tab
lad = {}
for name in ("mlp", "formula"):
    for k in ("F", "FS", "FSB"):
        a_th, a_li, a_tr = (res[f"{name}*{k}{s}/abar"]["nll"] for s in ("_theta", "_line", ""))
        lad[f"{name} {k}"] = {"theta_only": L.paired0(a_th - res[name]["nll"]), "theta_phi_zb (straight line)": L.paired0(a_li - res[name]["nll"]),
                              "theta_phi_entry (true origin)": L.paired0(a_tr - res[name]["nll"]),
                              "line_minus_theta": L.paired0(a_li - a_th), "true_minus_line": L.paired0(a_tr - a_li),
                              "theta_only_share_of_full": float((a_th - res[name]["nll"]).mean() / (a_tr - res[name]["nll"]).mean())}
out["information_ladder_report_metric"] = lad
okb = np.isfinite(blk["mlp"]["nll"]) & np.isfinite(blk["formula"]["nll"])
out["bulk_stratum_delta"] = {k: L.paired0((blk[k]["nll"] - blk[k.split('*')[0]]["nll"])[okb]) for k in blk if "*" in k}
out["bulk_persistence_theta_only"] = {}
for name in ("mlp", "formula"):
    main = (res[f"{name}*FS_theta/abar"]["nll"] - res[name]["nll"])[st][okb]
    gb = (blk[f"{name}*FS_theta/abar"]["nll"] - blk[name]["nll"])[okb]
    dec = L.stats.guard_persistence(main, gb)
    out["bulk_persistence_theta_only"][name] = {"report_on_stratum": L.paired0(main), "bulk": L.paired0(gb), "margin": dec.numbers["margin"],
                                                "se_margin": dec.numbers["se_margin"], "z_margin": dec.numbers["z_margin"], "passed": bool(dec.passed)}
# (4) bootstrap of the main deltas, stratified by momentum bin
pbin = val.ev["bin"].numpy()
out["bootstrap_95"] = {}
for k in ("mlp*FS/abar", "mlp*FSB/abar", "formula*FS/abar", "formula*FSB/abar", "mlp*F/abar", "mlp*FS_theta/abar"):
    b = k.split("*")[0]
    bs = L.stats.bootstrap_paired(res[k]["nll"], res[b]["nll"], n_boot=2000, seed=0, strata=pbin)
    out["bootstrap_95"][k] = {"mean": bs.mean, "se_boot": bs.se, "lo": bs.lo, "hi": bs.hi, "se_plain": tab[k]["se"]}
bs = L.stats.bootstrap_paired(res["mlp*FSB/abar"]["nll"], res["mlp*FS/abar"]["nll"], n_boot=2000, seed=0, strata=pbin)
out["bootstrap_95"]["back-face share mlp (FSB - FS, ratio)"] = {"mean": bs.mean, "se_boot": bs.se, "lo": bs.lo, "hi": bs.hi}
for k, v in tab.items():
    print(f"{k:28s} {L.fmt(v)}", flush=True)
print(json.dumps(lad, indent=1, default=str)[:100])
for k, v in lad.items():
    print(k, {a: (L.fmt(b) if isinstance(b, dict) else round(b, 3)) for a, b in v.items()}, flush=True)
print("bulk", {k: L.fmt(v) for k, v in out["bulk_stratum_delta"].items()})
print(json.dumps(out["bulk_persistence_theta_only"]))
print(json.dumps(out["val_energy_share_of_fold_points_in_cells_with_A_le"]))
print(json.dumps(out["bootstrap_95"], indent=1))
out["seconds"] = time.time() - t0
(L.HERE / "g5c_controls.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
