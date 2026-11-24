import sys, torch, numpy as np
F = "/home/user/tamas/shape-form-proton/results/final/proton"
S = "/home/user/tamas/shape-form-proton/review/R2_numbers/snapshot"
def ld(p): return torch.load(p, map_location="cpu", weights_only=False)
for tag in ("slab", "stdacc"):
    ch = ld(f"{F}/{tag}_gce_chosen.pt")
    print(tag, "chosen keys:", list(ch.keys()), "class", ch.get("class"), "n_params", ch.get("n_params"))
    print("  state keys:", list(ch["state"].keys()))
    for n in range(1, 7):
        r = ld(f"{F}/{tag}_gce_n{n}.pt")
        same = all(torch.equal(r["state"][k], ch["state"][k]) for k in ch["state"]) if set(r["state"]) == set(ch["state"]) else "keys differ"
        print(f"  n{n}: keys {list(r.keys())} deg {r.get('deg')} n_params {r.get('n_params')} val {float(np.mean(r['val_nll'])):.6f} same_as_chosen {same}")
    # snapshot (first run, NMAX=3)
    for n in range(1, 4):
        try:
            r0 = ld(f"{S}/{tag}_gce_n{n}.pt"); r = ld(f"{F}/{tag}_gce_n{n}.pt")
            same = all(torch.equal(r["state"][k], r0["state"][k]) for k in r0["state"])
            print(f"  snapshot n{n} identical state to current: {same}")
        except Exception as e:
            print("  snapshot", n, e)
    c0 = ld(f"{S}/{tag}_gce_chosen.pt")
    print("  snapshot chosen n_params", c0.get("n_params"), "test", float(np.mean(c0["test_nll"])), "val", float(np.mean(c0["val_nll"])))
    print("  current chosen n_params", ch.get("n_params"), "test", float(np.mean(ch["test_nll"])), "val", float(np.mean(ch["val_nll"])))
for f in ("ref_std_gflash_acc_n6", "ref_slab_gflash_n6", "ref_slab_mlp_L3xH16"):
    r = ld(f"{F}/{f}.pt")
    print(f, list(r.keys()), r.get("class"), r.get("n_params"), "val", float(np.mean(r["val_nll"])), "test", float(np.mean(r["test_nll"])))
    print("   state keys:", list(r["state"].keys())[:12])
