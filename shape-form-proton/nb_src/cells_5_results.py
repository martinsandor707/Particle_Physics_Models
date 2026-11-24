# %% [markdown]
# ## 6. Results
#
# ### 6.1 Accuracy, like with like
#
# Every model is compared with models **trained under the same likelihood**. In the report's metric a pure $(z', r \mid p)$ model may
# contain the fixed geometric template $\bar A$ ("× Ā", 4096 numbers computed from the training events' angles and depths, not fitted).
# Reference refits use exactly the same optimiser and protocol as the new formula (`fit_fast`, 4000 steps per rung, degree ladder
# 1–6, warm start, 1-SE rule; `scripts/particle_pipeline.py`); the network reference is the report's MLP fine-tuned with the same
# optimiser. The report's own models appear only in the report's metric. "NEW, compact" is the degree-2 rung.

# %%
def strip(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


NB = PIPE_TEST["degrees"]                                        # chosen degrees of the reference ladders (1-SE on validation)
ck = torch.load(f"{FINAL}/base_std_n{NB['base_std']}.pt", map_location=DEVICE, weights_only=False)["state"]
gacc = shapelab.WithAcceptance(Formula("gamma", "core_tail", degree=NB["base_std"], lnp0=LNP0, lnp_s=LNP_S), A_BAR_DEV).to(DEVICE)
gacc.load_state_dict({**{"base." + k: v for k, v in strip(ck, "base.f.").items()}, "logA": ck["logA"]})
g6 = Formula("gamma", "core_tail", degree=NB["base_slab"], lnp0=LNP0, lnp_s=LNP_S).to(DEVICE)
g6.load_state_dict(strip(torch.load(f"{FINAL}/base_slab_n{NB['base_slab']}.pt", map_location=DEVICE, weights_only=False)["state"], "f."))
REF = {"GFlash × Ā (refit)": gacc.eval(), "GFlash (3D refit)": g6.eval()}
for tag, key in (("mlp_std", "MLP fine-tuned (report metric)"), ("mlp_slab", "MLP fine-tuned (3D)")):
    m_ = FieldModel(16, 3).to(DEVICE)
    m_.load_state_dict(torch.load(f"{FINAL}/{tag}.pt", map_location=DEVICE, weights_only=False)["state"])
    REF[key] = m_.eval()
NEW = {"std": LADDER[("std", CHOSEN["std"])], "slab": LADDER[("slab", CHOSEN["slab"])]}

TABLE = [  # (name, params, model trained in the report's metric, model trained in the 3D metric)
    ("GFlash n=6 (report)", 45, REPORT["gflash_n6"], None),
    (f"GFlash (× Ā / 3D refit, n = {NB['base_std']} / {NB['base_slab']})", REF["GFlash × Ā (refit)"].n_params(), REF["GFlash × Ā (refit)"], REF["GFlash (3D refit)"]),
    ("NEW, compact (n = 2)", LADDER[("std", 2)].n_params(), LADDER[("std", 2)], LADDER[("slab", 2)]),
    ("NEW (1-SE)", NEW["std"].n_params(), NEW["std"], NEW["slab"]),
    ("MLP L3xH16 (report)", 881, REPORT["mlp"], None),
    ("MLP L3xH16 (fine-tuned)", 881, REF["MLP fine-tuned (report metric)"], REF["MLP fine-tuned (3D)"]),
    ("CNN (report)", 121939, REPORT["cnn"], None),
]
BASE = {"std": TABLE[1][0], "3D": TABLE[1][0]}
rows, PER = [], {}
for name, npar, ms, m3d in TABLE:
    for metric, m, fn in (("std", ms, test_nll), ("3D", m3d, test_nll_slab)):
        if m is not None:
            for split in ("val", "test"):
                PER[(name, metric, split)] = fn(m, data[split])
for name, npar, ms, m3d in TABLE:
    r = {"model": name, "params": npar}
    for metric in ("std", "3D"):
        if (name, metric, "test") not in PER:
            r.update({f"{metric}: val": np.nan, f"{metric}: test": np.nan, f"{metric}: Δtest vs GFlash refit": ""})
            continue
        r[f"{metric}: val"] = PER[(name, metric, "val")].mean()
        r[f"{metric}: test"] = PER[(name, metric, "test")].mean()
        d, se = paired(PER[(name, metric, "test")], PER[(BASE[metric], metric, "test")])
        r[f"{metric}: Δtest vs GFlash refit"] = f"{d:+.4f} ± {se:.4f}"
    rows.append(r)
ACC = pd.DataFrame(rows).set_index("model")
display(ACC.style.format({c: "{:.4f}" for c in ACC.columns if c.endswith("val") or c.endswith("test")}, na_rep="–"))
ACC.to_csv(f"{FINAL}/notebook_accuracy_table.csv")

T = lambda name, metric: PER[(name, metric, "test")].mean()
MLPN = "MLP L3xH16 (fine-tuned)"
for metric, label in (("std", "report metric"), ("3D", "3D metric")):
    g, n2, n6, mlp = T(BASE[metric], metric), T("NEW, compact (n = 2)", metric), T("NEW (1-SE)", metric), T(MLPN, metric)
    print(f"{label}: share of the (GFlash refit → fine-tuned MLP) gap closed by the new formula: 1-SE {(g - n6) / (g - mlp):.0%}, n = 2 {(g - n2) / (g - mlp):.0%}"
          f"  | remaining gap to the MLP: {paired(PER[('NEW (1-SE)', metric, 'test')], PER[(MLPN, metric, 'test')])[0]:+.4f}")
r0 = T("GFlash n=6 (report)", "std")
print(f"report metric, decomposition of the report's GFlash → report MLP gap ({r0 - T('MLP L3xH16 (report)', 'std'):.4f} nat): geometric template Ā "
      f"{r0 - T(BASE['std'], 'std'):.4f}, new formula {T(BASE['std'], 'std') - T('NEW (1-SE)', 'std'):.4f}, left {T('NEW (1-SE)', 'std') - T('MLP L3xH16 (report)', 'std'):.4f}")

# %% [markdown]
# ### 6.2 Where the gain comes from: per momentum bin (3D metric, test set)
#
# Paired difference to the GFlash formula refitted with the same 3D likelihood, in every 1 GeV bin. Negative = better.

# %%
bins_te = data["test"].ev["bin"].cpu().numpy()
SER = {"new formula (3D)": ("NEW (1-SE)", "#008300", "o"), "MLP (fine-tuned, 3D)": (MLPN, "#ff7f0e", "^")}
base = PER[(BASE["3D"], "3D", "test")]
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 2.2]})
for lab, (key, col, mk) in SER.items():
    d = PER[(key, "3D", "test")] - base
    mean = np.array([d[bins_te == b].mean() for b in range(20)])
    se = np.array([d[bins_te == b].std(ddof=1) / np.sqrt((bins_te == b).sum()) for b in range(20)])
    for a_, sl in ((ax[0], slice(0, 3)), (ax[1], slice(2, 20))):
        a_.errorbar(np.arange(20)[sl] + 0.5, mean[sl], se[sl], fmt=mk + "-", color=col, ms=4, lw=1.5, capsize=2, label=lab)
for a_ in ax:
    a_.axhline(0, color="0.3", lw=0.8); a_.grid(alpha=0.25); a_.set_xlabel("p [GeV] (bin centre)")
ax[0].set(title="0–3 GeV (stopping protons)", ylabel="Δ NLL vs GFlash (3D refit) [nat]")
ax[1].set(title="2–20 GeV (hadronic showers)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
dn = PER[("NEW (1-SE)", "3D", "test")] - PER[(MLPN, "3D", "test")]
hi = bins_te >= 2
print(f"new formula − fine-tuned MLP (3D): bins 2–19 {dn[hi].mean():+.4f} ± {dn[hi].std(ddof=1) / np.sqrt(hi.sum()):.4f} per event "
      f"(positive in {sum(dn[bins_te == b].mean() > 0 for b in range(2, 20))} of 18 bins); "
      f"bins 0–1 hold {dn[~hi].sum() / dn.sum():.0%} of the remaining gap")

# %% [markdown]
# **What to look for:** the new formula is better than the refitted GFlash in every bin: by ≈ 0.3 nat below 1 GeV, 0.06 nat at
# 1–2 GeV and 0.01–0.025 nat above 2 GeV. The network trained with the same likelihood is still better: by a small but systematic
# amount in almost every bin above 2 GeV (printed above), and more below 2 GeV, where about 60 % of the remaining gap sits.
