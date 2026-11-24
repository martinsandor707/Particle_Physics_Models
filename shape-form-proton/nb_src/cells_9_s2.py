# %% [markdown]
# ## 8. Independent check on a second simulation sample
#
# The components, the lateral form and the degree were chosen on the validation set. Test values were logged during the exploration
# as well, but an independent review (`review/R2_numbers`) found that every choice agrees with the validation ranking. As a fully
# independent check the models are evaluated on the **second proton sample** of the same simulation (`..._balanced_s2_v3.csv`,
# another random seed: 99 948 events that none of the models has seen). Loading it from the CSV takes ≈ 7 minutes; the compact
# tensors are cached in `cache/proton_s2_sets.pt`.

# %%
S2 = "cache/proton_s2_sets.pt"
if os.path.exists(S2):
    ck2 = torch.load(S2, weights_only=False)
    s2 = {k: Showers(v["ev"], v["off"], v["xyz"], v["w"], v["box"]) for k, v in ck2["sets"].items()}

    def nll_s2(model, fn):
        return np.concatenate([fn(model, sh.to(DEVICE)) for sh in s2.values()])

    for name, _, ms, m3d in TABLE[:6]:
        for metric, m, fn in (("std", ms, test_nll), ("3D", m3d, test_nll_slab)):
            if m is not None:
                PER[(name, metric, "s2")] = nll_s2(m, fn)
    rows = []
    for name, _, ms, m3d in TABLE[:6]:
        r = {"model": name}
        for metric in ("std", "3D"):
            if (name, metric, "s2") not in PER:
                r.update({f"{metric}: NLL (s2)": np.nan, f"{metric}: Δ vs GFlash refit (s2)": ""}); continue
            r[f"{metric}: NLL (s2)"] = PER[(name, metric, "s2")].mean()
            d, se = paired(PER[(name, metric, "s2")], PER[(BASE[metric], metric, "s2")])
            r[f"{metric}: Δ vs GFlash refit (s2)"] = f"{d:+.4f} ± {se:.4f}"
        rows.append(r)
    S2TAB = pd.DataFrame(rows).set_index("model")
    print(f"s2 sample: {sum(len(v) for v in s2.values())} events")
    display(S2TAB.style.format({c: "{:.4f}" for c in S2TAB.columns if "NLL" in c}, na_rep="–"))
    S2TAB.to_csv(f"{FINAL}/notebook_accuracy_s2.csv")
else:
    print("second sample not cached – run scripts/build_cache_s2.py once (≈ 7 min)")
