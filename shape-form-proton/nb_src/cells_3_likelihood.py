# %% [markdown]
# ## 4. Two metrics: the report's likelihood and the 3D (slab) likelihood
#
# **The report's metric** (00, `event_nll`): the folded model density at each hit, normalised by the mass of the model in front of
# the back face, $M_e = \int_{z' < z_b} \rho$. A body of revolution has mass in front of the tilted front face as well; this mass
# is counted in $M_e$ although no hit can ever be recorded there.
#
# **The 3D likelihood:** the same folded density at each hit, but normalised over the region where the event's hits can be,
# the slab:
#
# $$\mathrm{NLL}_e = -\sum_i w_i\left[\log g(c_i) - \log M_e^{slab}\right],\qquad
# M_e^{slab} = \sum_{(z', r)} P(z', r \mid p)\, A_{\theta_e}(z', r).$$
#
# The two versions differ only in the region over which the model is normalised. Neither is exact: hits can only be recorded in
# instrumented cells, and both regions also contain the space outside the calorimeter's transverse outline (outer edge, beam hole)
# and the module gaps. An independent check (`review/R1_likelihood`) found that about 0.22–0.24 nat of every model's slab mass, and
# ≈ 0.27 nat of its mass in the report's normalisation, lies where no hit can be recorded. The part that remains is model-dependent,
# so **models are compared only with models trained under the same likelihood**: among slab-trained models the leftover bias of a
# paired difference is ≤ 0.001 nat. Re-normalising the report's models over the slab (below) is not an improvement of those models;
# it only measures how much mass each of them places in front of the entrance face.
#
# Every number below uses the same split, the same per-event evaluation (`test_nll`: $K = 64$ cell points, fixed seed) and paired
# standard errors.

# %%
def event_nll_slab(model, b, box, K=N_FOLD, gen=None):
    """Per-event NLL, detector-folded, normalised over the detector slab (tilted front and back faces)."""
    logP = model(b.lnp)
    logrho = logP - LOG_VOL
    pts = fold(b, box, K, gen)
    iz, ir = locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(K)
    mass = (logP.exp() * event_slab(b, box)).sum((1, 2))
    ll = logg - torch.log(mass.clamp_min(1e-8))[b.ev]
    return -torch.zeros(len(b.lnp), device=ll.device).index_add_(0, b.ev, b.w * ll), logP


@torch.no_grad()
def test_nll_slab(model, data_, K=N_FOLD_EVAL):
    gen = torch.Generator(device=DEVICE).manual_seed(12345)
    return torch.cat([event_nll_slab(model, b, data_.box, K, gen)[0] for b in data_.batches(128)]).cpu().numpy()


C_STD = globals()                                              # the report's namespace (00_common)
C_SLAB = {**globals(), "event_nll": event_nll_slab, "test_nll": test_nll_slab}


def paired(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


# %% [markdown]
# ### The report's saved proton models, evaluated in both metrics (no refit)
#
# The models are loaded from the report's results folder (`/home/user/sofiya/shape-form/results/proton`, read only). The first
# column reproduces the report's table exactly.

# %%
import sys
sys.path.insert(0, ".")
import shapelab                                                 # helper module of this folder (model loading, fast fit, wrappers)

REPORT = shapelab.baseline_models(C_STD, data)                  # gflash_n1..n6, mlp (L3xH16), cnn, mlp_formula, cnn_formula
NLL = {}                                                        # (model, metric, split) -> per-event NLL
for k in ("gflash_n1", "gflash_n2", "gflash_n6", "mlp_formula", "mlp", "cnn"):
    for metric, fn in (("std", test_nll), ("slab", test_nll_slab)):
        for split in ("val", "test"):
            NLL[(k, metric, split)] = fn(REPORT[k], data[split])
rows = []
for k in ("gflash_n1", "gflash_n2", "gflash_n6", "mlp_formula", "mlp", "cnn"):
    rows.append({"model (report)": k, "params": sum(p.numel() for p in REPORT[k].parameters()),
                 "std: test NLL": NLL[(k, "std", "test")].mean(), "3D: test NLL": NLL[(k, "slab", "test")].mean(),
                 "3D − std": paired(NLL[(k, "slab", "test")], NLL[(k, "std", "test")])[0]})
display(pd.DataFrame(rows).set_index("model (report)").round(4))

# %% [markdown]
# **What to look for:** the first column is the report's table (GFlash 18.2536, MLP 18.1922, CNN 18.1908). The third column is the
# mass each model places in front of the entrance face, where no hit can be recorded: ≈ 0.031 nat for GFlash, ≈ 0.020 for the networks,
# which learned the cut as a cone. The product formula $L(z')\,T(r)$ cannot represent the cut; about 18 % of the GFlash–MLP gap of the
# report (0.0614 → 0.0502 nat) is this geometric effect.
