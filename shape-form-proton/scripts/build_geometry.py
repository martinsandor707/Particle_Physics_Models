"""Eseményenkénti origó (x0, y0, z0) és a detektor cellarácsa a CSV-ből, a gyorsítótár eseménysorrendjéhez igazítva.

A gyorsítótár (proton_sets.pt) nem tárolja a globális koordinátákat; itt a `load_data` felosztását pontosan megismételjük
(ugyanazok a vágások, SEED = 1), és ellenőrizzük, hogy az impulzusok egyeznek a gyorsítótáréval."""
import os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, pandas as pd, torch
import protonlab

t0 = time.time()
TIME_CUT_NS, Z_MIN, Z_MAX, R_MAX, SEED = 100.0, -200.0, 1450.0, 700.0, 1
cols = ["event_number", "x", "y", "z", "x_trans", "y_trans", "z_trans", "x_local", "y_local", "z_local",
        "earliest_contribution_time", "incoming_momentum", "incoming_momentum_bin", "incoming_theta", "incoming_phi"]
df = pd.read_csv(os.path.join(protonlab.DATA_DIR, "proton_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.csv"), usecols=cols)
print(f"read {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
# a detektor cellarácsa: minden valaha megszólalt cella (vágások előtt)
xy = np.unique(np.stack([np.round(df.x.values, 1), np.round(df.y.values, 1)], 1), axis=0)
zs = np.unique(np.round(df.z.values, 2))
xyz_cnt = len(np.unique(np.stack([np.round(df.x.values, 1), np.round(df.y.values, 1), np.round(df.z.values, 1)], 1), axis=0))
print("unique (x,y) cell centres:", len(xy), "| layers:", len(zs), "| unique (x,y,z):", xyz_cnt, flush=True)
# ugyanazok a vágások és sorrend, mint Showers.from_dataframe
d = df[df.earliest_contribution_time < TIME_CUT_NS].sort_values("event_number", kind="stable")
r = np.hypot(d.x_local, d.y_local)
d = d[(d.z_local >= Z_MIN) & (d.z_local <= Z_MAX) & (r < R_MAX)]
first = d.groupby("event_number").first()
first = first.loc[np.sort(first.index.values)]
x0, y0, z0 = (first.x - first.x_trans).values, (first.y - first.y_trans).values, (first.z - first.z_trans).values
p, bins = first.incoming_momentum.values, first.incoming_momentum_bin.values.astype(np.int64)
theta, phi, evn = first.incoming_theta.values, first.incoming_phi.values, first.index.values
rng = np.random.default_rng(SEED)
parts = {"train": [], "val": [], "test": []}
for b in np.unique(bins):
    idx = rng.permutation(np.where(bins == b)[0])
    n1, n2 = round(0.70 * len(idx)), round(0.85 * len(idx))
    parts["train"] += idx[:n1].tolist(); parts["val"] += idx[n1:n2].tolist(); parts["test"] += idx[n2:].tolist()
ck = torch.load(protonlab.CACHE, weights_only=False)
out = {"cells_xy": torch.tensor(xy, dtype=torch.float32), "layers_z": torch.tensor(zs, dtype=torch.float32), "sets": {}}
for k, v in parts.items():
    i = np.sort(np.array(v))
    ok = np.allclose(p[i], ck["sets"][k]["ev"]["p"].numpy(), rtol=0, atol=1e-5)
    print(k, len(i), "aligned with cache:", ok, flush=True)
    assert ok
    out["sets"][k] = {"origin": torch.tensor(np.stack([x0[i], y0[i], z0[i]], 1), dtype=torch.float32),
                      "theta": torch.tensor(theta[i], dtype=torch.float32), "phi": torch.tensor(phi[i], dtype=torch.float32),
                      "event_number": torch.tensor(evn[i])}
torch.save(out, os.path.join(protonlab.HERE, "cache", "proton_geometry.pt"))
print(f"done in {time.time() - t0:.0f}s")
