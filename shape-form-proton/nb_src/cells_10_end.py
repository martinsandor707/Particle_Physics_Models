# %% [markdown]
# ## 9. Summary
#
# The numbers below are printed from this run (test set, like with like).

# %%
def tt(name, metric):
    return PER[(name, metric, "test")].mean()


lines = [
    f"* Report metric: GFlash (report) {tt('GFlash n=6 (report)', 'std'):.4f} (45 par.) → GFlash × Ā refit {tt(BASE['std'], 'std'):.4f} (45, fixed geometric template)"
    f" → new formula × Ā {tt('NEW (1-SE)', 'std'):.4f} ({NEW['std'].n_params()} par.; compact {tt('NEW, compact (n = 2)', 'std'):.4f} with"
    f" {LADDER[('std', 2)].n_params()}) → MLP {tt(MLPN, 'std'):.4f} (881).",
    f"* 3D metric: GFlash refit {tt(BASE['3D'], '3D'):.4f} → new formula {tt('NEW (1-SE)', '3D'):.4f} (compact {tt('NEW, compact (n = 2)', '3D'):.4f})"
    f" → MLP trained in 3D {tt(MLPN, '3D'):.4f}.",
]
for metric, label in (("std", "report metric"), ("3D", "3D metric")):
    g, n6, mlp = tt(BASE[metric], metric), tt("NEW (1-SE)", metric), tt(MLPN, metric)
    lines.append(f"* {label}: the new formula closes {(g - n6) / (g - mlp):.0%} of the gap between the GFlash formula and the MLP trained the same way.")
display(Markdown("\n".join(lines)))

# %% [markdown]
# 1. **Geometry.** In this detector the proton shower is a body of revolution **cut by the tilted entrance face** (and the back face):
#    the origin of the shower frame is the first hit layer, so every hit satisfies $z'\cos\theta - x'\sin\theta \ge 0$. The networks of
#    05/06 learned the cut as a cone; a product $L(z')\,T(r)$ cannot. In the report's metric the fixed template $\bar A$ (the cut averaged
#    over the incidence angles) improves the GFlash formula by ≈ 0.020 nat without new parameters; about half of that comes from the
#    report metric counting mass where no hit can be recorded (with a normalisation over the instrumented cells the gain is ≈ 0.010,
#    `review/R1_likelihood`).
# 2. **A better closed form for the proton**: shower (plateau depth profile × [Gauss core + Cauchy body + exponential-$dE/dr$ halo])
#    + primary track (attenuated until the first nuclear interaction, ending at the range $R(T)$, widening with depth by multiple
#    scattering) + entry spike + end-of-range blob + diffuse far halo. Against GFlash refitted the same way it is better at every
#    momentum, most below 2 GeV (stopping protons); overall by ≈ 0.03–0.04 nat, already with the degree-2 version. It closes about
#    90 % of the gap to the network trained the same way (printed above), with a sixth of its parameters; the network is still slightly
#    better above 2 GeV. The literature's Bock two-gamma profile is not better. The ranking and the sizes replicate on a second,
#    independent simulation sample (≈ 100 000 events).
# 3. **Envelope bodies.** For the "smallest body holding 95 % of the energy in 98 % of the showers" the report's drop is within ≈ 4 % of
#    the best possible body of revolution; the best closed form (super-egg) gains ≈ 3.5 %. The volume is set by the criterion and the
#    data domain, not by the shape family; the volume the body really occupies in the detector (inside the slab) is 8–11 % smaller.
#    The calibrated iso-density body recommended by the report is not smaller than the drop.
#
# **Caveats.**
# (1) Neither likelihood is normalised exactly over the instrumented cells: ≈ 0.22–0.27 nat of every model's mass lies where no hit can
# be recorded (outer edge, beam hole, module gaps). Comparisons are therefore made only between models trained under the same
# likelihood; the independent review found the remaining bias of such paired differences ≤ 0.001 nat in the 3D metric.
# (2) The network reference is the report's MLP fine-tuned with the same optimiser as the formulas (it improves the report's MLP by
# ≈ 0.002 nat); a larger network could still be somewhat better.
# (3) The 1-SE rule picks the top rung of nested, warm-started ladders (as for GFlash in 01); the degree-2 version (87 parameters) is
# within ≈ 0.0015 nat of it and is the better choice when interpretability matters.
# (4) The cross-section is normalised over the grid's disc at every depth. Normalised over the infinite plane, the fit is 0.001–0.002
# nat better, but a very wide halo then acts as a depth-dependent sink of energy with scales of tens of metres; we prefer the physical
# version.
# (5) The track parameters describe the mean energy profile effectively (the track end is shorter than the CSDA range, because protons
# that interact before stopping are averaged in); below ≈ 0.6 GeV the component split is effective as well, and above 2 GeV the
# track's halo also takes part of the hadronic core (for the proton the multiple-scattering term is effective, unlike for the muon).
# (6) The origin of the shower frame is defined from all hits, including the late (> 100 ns) ones that the likelihood drops; in 1.2 % of
# the events the first recorded layer lies behind the origin layer. This is inherited from the report and affects all models alike.
