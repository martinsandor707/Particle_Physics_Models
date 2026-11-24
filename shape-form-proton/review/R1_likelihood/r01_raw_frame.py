"""R1 audit, step 1: the shower frame on the RAW columns (all hits of the 14 992 test events, before any cut).

Checks (independent of the cached tensors and of shapelab):
  a) local = R(theta, phi) . trans  with the rows of 00_common.rotation (e_theta, e_phi, e_r);
  b) z_trans = z' cos(theta) - x' sin(theta)  (inverse rotation, third row of R^T);
  c) trans = global - origin with ONE origin per event;
  d) z0 = min z over ALL hits of the event (pre-cut), i.e. the origin layer is the first hit layer;
  e) the origin (x0, y0) is the energy-weighted centroid of the first-layer hits - with or without the time cut;
  f) events whose origin layer loses all its hits through the report's cuts: which cut removes them."""
import json
import os

import numpy as np
import pandas as pd

OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
df = pd.read_parquet(os.path.join(OUT, "raw_test_rows.parquet"))
print("rows", len(df), "events", df.event_number.nunique(), flush=True)
res = {"rows (all hits, pre-cut)": int(len(df)), "events": int(df.event_number.nunique())}
th, ph = df.incoming_theta.values, df.incoming_phi.values
ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
R = np.stack([np.stack([ct * cp, ct * sp, -st], -1), np.stack([-sp, cp, 0 * sp], -1), np.stack([st * cp, st * sp, ct], -1)], 1)
trans = df[["x_trans", "y_trans", "z_trans"]].values
loc = df[["x_local", "y_local", "z_local"]].values
pred = np.einsum("nij,nj->ni", R, trans)
res["a) max |R.trans - local| [mm]"] = float(np.abs(pred - loc).max())
zt_formula = loc[:, 2] * ct - loc[:, 0] * st
res["b) max |z' cos t - x' sin t - z_trans| [mm]"] = float(np.abs(zt_formula - trans[:, 2]).max())
res["b) min z_trans over all raw hits [mm]"] = float(trans[:, 2].min())
o = df[["x", "y", "z"]].values - trans
df["ox"], df["oy"], df["oz"] = o[:, 0], o[:, 1], o[:, 2]
g = df.groupby("event_number")
spread = (g[["ox", "oy", "oz"]].max() - g[["ox", "oy", "oz"]].min()).max()
res["c) max spread of (global - trans) within an event [mm] (x, y, z)"] = [float(v) for v in spread.values]
zmin = g.z.min()
oz = g.oz.first()
res["d) max |z0 - min z (all raw hits)| [mm]"] = float((oz - zmin).abs().max())
# e) energy-weighted centroid of the first layer
first = df[np.isclose(df.z.values, df.oz.values, atol=0.05)]
def centroid(d):
    w = d.energy.values
    return pd.Series({"cx": (w * d.x.values).sum() / w.sum(), "cy": (w * d.y.values).sum() / w.sum(), "ox": d.ox.iloc[0], "oy": d.oy.iloc[0]})
cen_all = first.groupby("event_number").apply(centroid)
d_all = np.hypot(cen_all.cx - cen_all.ox, cen_all.cy - cen_all.oy)
res["e) origin vs energy-weighted centroid of the first-layer hits (all times): median / 99% / max distance [mm]"] = [float(np.median(d_all)), float(np.quantile(d_all, 0.99)), float(d_all.max())]
prompt = first[first.earliest_contribution_time < 100.0]
cen_p = prompt.groupby("event_number").apply(centroid)
d_p = np.hypot(cen_p.cx - cen_p.ox, cen_p.cy - cen_p.oy)
res["e) same with only the prompt (t < 100 ns) first-layer hits: median / 99% / max [mm]"] = [float(np.median(d_p)), float(np.quantile(d_p, 0.99)), float(d_p.max())]
res["e) events whose first (origin) layer has no prompt hit at all"] = int(len(cen_all) - len(cen_p))
# f) what removes the origin-layer hits?  (cuts of Showers.from_dataframe)
r = np.hypot(df.x_local.values, df.y_local.values)
keep_t = df.earliest_contribution_time.values < 100.0
keep_s = (df.z_local.values >= -200.0) & (df.z_local.values <= 1450.0) & (r < 700.0)
df["keep"] = keep_t & keep_s
df["is_first"] = np.isclose(df.z.values, df.oz.values, atol=0.05)
ev_has_first_kept = df[df.is_first & df.keep].event_number.unique()
lost = np.setdiff1d(df.event_number.unique(), ev_has_first_kept)
res["f) events whose origin layer has no hit after the cuts"] = int(len(lost))
sub = df[df.event_number.isin(lost) & df.is_first]
res["f) of their origin-layer hits: removed by the time cut only / by the spatial cut only / by both"] = [
    int(((~keep_t[sub.index]) & keep_s[sub.index]).sum()) if False else int(((sub.earliest_contribution_time >= 100) & ((sub.z_local >= -200) & (sub.z_local <= 1450) & (np.hypot(sub.x_local, sub.y_local) < 700))).sum()),
    int(((sub.earliest_contribution_time < 100) & ~((sub.z_local >= -200) & (sub.z_local <= 1450) & (np.hypot(sub.x_local, sub.y_local) < 700))).sum()),
    int(((sub.earliest_contribution_time >= 100) & ~((sub.z_local >= -200) & (sub.z_local <= 1450) & (np.hypot(sub.x_local, sub.y_local) < 700))).sum())]
res["f) particle types of those origin-layer hits"] = sub.particle_type.value_counts().head(8).to_dict()
res["f) their origin-layer hit times [ns]: median, min"] = [float(sub.earliest_contribution_time.median()), float(sub.earliest_contribution_time.min())]
# distance (along the axis) between the origin layer and the first KEPT layer for those events
kept = df[df.keep]
zk = kept.groupby("event_number").z_trans.min()
gap = zk.loc[np.intersect1d(lost, zk.index)]
res["f) for those events: first kept layer behind the origin [mm]: median / 90% / max"] = [float(gap.median()), float(gap.quantile(0.9)), float(gap.max())]
# energy fraction removed by the time cut, by the spatial cut (per event, averaged)
e_tot = g.energy.sum()
e_t = df[~keep_t].groupby("event_number").energy.sum().reindex(e_tot.index, fill_value=0) / e_tot
e_s = df[keep_t & ~keep_s].groupby("event_number").energy.sum().reindex(e_tot.index, fill_value=0) / e_tot
res["energy fraction removed per event: time cut (mean, max)"] = [float(e_t.mean()), float(e_t.max())]
res["energy fraction removed per event: spatial cuts after the time cut (mean, max)"] = [float(e_s.mean()), float(e_s.max())]
# spatial cut by side: z' < -200, z' > 1450, r >= 700 (prompt hits)
pr = df[keep_t]
rr = np.hypot(pr.x_local, pr.y_local)
for lab, m in (("z'<-200", pr.z_local < -200), ("z'>1450", pr.z_local > 1450), ("r>=700", (rr >= 700) & (pr.z_local >= -200) & (pr.z_local <= 1450))):
    ef = pr[m].groupby("event_number").energy.sum().reindex(e_tot.index, fill_value=0) / e_tot
    res[f"prompt energy fraction cut by {lab} (mean per event)"] = float(ef.mean())
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "r01_raw_frame.json"), "w"), indent=1)
