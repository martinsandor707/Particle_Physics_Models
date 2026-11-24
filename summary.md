# Repository summary: shower-shape models for a calorimeter

*State of the repository on 24 November 2026. All scores are test-set values taken from `report_en.pdf`, the saved notebook
outputs and the result files in `shape-form-proton/results/`. Scores are negative log-likelihoods (NLL): lower is better.*

## In brief

- **Goal.** Describe where a simulated particle (electron, pion, proton or muon, 0.1–20 GeV) deposits its energy in a
  sampling hadron calorimeter, as a function of the particle's momentum. Two products are wanted: the average
  energy-density shape of a shower, and a small 3D body that contains almost all of a shower's energy.
- **Approach.** Closed-form formulas and small neural networks are fitted to the same simulated hits with the same
  per-event likelihood, so their scores can be compared directly. The networks serve as the accuracy reference; the
  formulas are the interpretable product.
- **Best networks.** One small MLP per particle (881–3,841 weights). A CNN with 40–380 times more weights is not better.
- **Best closed-form formulas.** The physically motivated *component formula* of the follow-up study (53–143
  parameters). It closes 70–90 % of the distance between the GFlash-type baseline and the networks.
- **Best containment body.** The *super-egg* hull, 3.5–3.8 % smaller than the first stage's drop hull for hadrons.
- **Main open problem.** Every model here describes the *average* shower. Shower-to-shower fluctuations are not modelled.

## 1. What the project is trying to achieve

The data are standalone DD4hep simulations of the ePIC LFHCal (60 layers of 4 mm scintillator + 15.2 mm steel + 1.3 mm
Kapton/air, behind a 40.4 mm steel front plate). For each of four particle types there are about 100,000 events with
momentum spread over 0.1–20 GeV and incidence angles of roughly 0.06–0.63 rad.

The work answers three questions:

1. **How well can a compact formula describe the average shower shape?** The target is a normalised energy density
   `ρ(z', r | p)`: depth `z'` along the particle's direction, distance `r` from its axis, momentum `p`.
2. **How much better can a free-form model do?** Neural networks with no prescribed shape give an estimate of the best
   achievable score, and so measure what the formulas miss.
3. **What is the smallest body that holds a shower?** The criterion is: at least 95 % of the energy lies inside the body
   for 98 % of showers, with minimal volume.

The work came in two stages. The first is a team report (`report_en.pdf`, notebooks in `shape-form-original/`). The second
is a follow-up study (`shape-form-proton/`) that set out to find a better proton shape and was then extended to all four
particles.

## 2. What is in the repository

| Path | Contents |
|---|---|
| [report_en.pdf](report_en.pdf) | The 3-page team report (12 November 2026): methods, results and conclusions of the first stage. |
| [shape-form-original/](shape-form-original/) | The first-stage notebooks, with saved outputs. `00_common` holds the shared code; `01` GFlash formula; `02` envelope hull; `03` drop hull; `04` cylinder formula (muon); `05` MLP + distillation; `06` CNN + distillation; `07` summary. Most exist once per particle. |
| [shape-form-original/common.py](shape-form-original/common.py) | Importable version of `00_common.ipynb` (added locally, see section 7). |
| [shape-form-proton/](shape-form-proton/) | The follow-up study: Hungarian `README.md` (also as PDF), `LAB_GUIDE.md`, the formula code (`particle_shape.py`, `proton_shape.py`), the experiment harness (`shapelab.py`, `families.py`, `envlab.py`), `scripts/`, per-topic experiment folders in `lanes/`, independent checks in `review/`, and all results and checkpoints in `results/`. |
| `shape-form-codes.rar` | An archive of the first-stage notebooks. |
| `pyproject.toml`, `uv.lock` | Python environment (managed with `uv`): PyTorch, pandas, pyarrow, PySR, matplotlib, plotly, scipy. |

## 3. The common framework

Both stages use the same data handling, grid and score. This is what makes all numbers in this document comparable.

- **Input data.** One table row is one detector-cell hit (cell size 48.6 × 48.6 × 20.5 mm). Hits later than 100 ns are dropped.
- **Shower coordinates.** Every hit is expressed in the shower's own frame. The origin is the energy-weighted centre of the
  first hit layer; `z'` runs along the true direction of incidence; `r` is the distance from that axis. The shape is
  assumed to be symmetric around the axis.
- **Grid.** `z'` from −200 to 1450 mm in 128 bins and `r` from 0 to 700 mm in 32 rings (finer near the axis). Every model,
  formula or network, has the same interface: given `ln p` it returns the share of the shower's energy in each of the
  128 × 32 ring-shaped cells.
- **Score: per-event NLL.** For one event, `NLL = −Σ_i w_i [ln g_i − ln M]`.
  - `w_i` is hit *i*'s share of the event's energy.
  - `g_i` is the model density averaged over that hit's detector cell (random points inside the cell, rotated into the
    shower frame). This is how detector resolution enters.
  - `M` is the part of the model that lies in front of the detector's back face for that event.
  - Only differences between models matter. They are quoted as *paired* differences on the same test events with their
    standard error.
- **Protocol.** 70/15/15 % train/validation/test split, stratified per 1 GeV momentum bin, fixed seed. Parameters are
  fitted on train; early stopping and every model choice use validation; test is only used for the final numbers.
- **Selection rule (1-SE rule).** Among candidates, take the simplest one whose validation NLL is within one standard
  error of the best.

## 4. Methods

### 4.1 Baseline formulas (first stage)

The density is a product of a depth profile and a lateral profile, `ρ ∝ L(z') · T(r)`.

| Part | Name | Form |
|---|---|---|
| Depth | gamma (GFlash) | `L ∝ s^(α−1) · e^(−(α−1)s/t_max)`, `s = z' − z0` |
| Depth | plateau ("cylinder") | `L ∝ σ((z'−z_s)/w1) · σ((ℓ−z')/w2) · e^(κz')`, σ = logistic function |
| Lateral | core + tail (GFlash) | `T = p_c/π · R_C²/(r²+R_C²)² + (1−p_c)/π · R_T²/(r²+R_T²)²` |
| Lateral | Gauss + exponential | `T = p_c · e^(−r²/2σ²)/(2πσ²) + (1−p_c) · e^(−r/λ)/(2πλ²)` |

Every parameter is a polynomial of degree *n* in `ln p`; lateral parameters also change linearly with depth. The degree
(1–6) is chosen with the 1-SE rule. The baseline is gamma × core+tail for electron, pion and proton, and plateau ×
core+tail for the muon. This is a GFlash-*like* formula refitted to this simulation, not the Geant4 GFlash library.

### 4.2 Neural density fields: MLP and CNN (first stage)

- **MLP.** One small network is evaluated separately for each of the 4,096 grid cells. Its 19 inputs are the cell's
  rescaled depth and radius, four sine/cosine frequencies of each, and `ln p / 3`. It has `N` hidden layers of `H`
  neurons with SiLU activation and one output; a softmax over all cells turns the outputs into energy shares.
  - *Training:* Adam, learning rate 3e-3, 32 events per batch, at most 60 epochs, early stopping, plus a small
    smoothness penalty along depth.
  - *Size search:* width `H` ∈ {16, 32, 64, 128, 256} at 3 layers, then depth `N` ∈ {1…5} at the chosen width, both with
    the 1-SE rule.
- **CNN.** An architecture derived from the task rather than searched: an 8 × 2 "sketch" upsampled four times to the grid,
  with the momentum fed in through per-channel scaling (FiLM). It only has to match the MLP's validation score.

### 4.3 Distillation (first stage)

The trained network is "translated" into a formula. Eight candidates (the four families of 4.1 at degree 1 and 2) are
fitted to the network's output maps at 30 momenta by minimising the KL divergence. The simplest candidate within 0.005 nat
of the best is kept. No data and no detector smearing are used in this step. The report calls this restricted symbolic
regression: it selects from a fixed library instead of searching over formula structures.

### 4.4 Detector geometry: acceptance factor and 3D likelihood (follow-up)

The follow-up study found that the detector's front face is the largest single effect missing from the first stage.

- **The effect.** The particle enters at an angle, so in the shower's frame the front face is a tilted plane through the
  origin. No energy can be recorded in front of it. A product `L(z') · T(r)` cannot represent this cone-shaped front.
- **Acceptance factor `Ā(z', r)`.** For every ring, the fraction that lies inside the detector, averaged over the events'
  angles. It has a closed form and no free parameters. Multiplying a model by it improves every formula under the
  original score ("report metric").
- **3D ("slab") likelihood.** A second score that normalises each event over its actual detector slab (tilted front and
  back faces). Scores under the two likelihoods are not comparable with each other. A formula trained with the 3D
  likelihood must not be used as a stand-alone `(z', r)` model.
- **`fit_fast`.** A cleaner training routine (Adam with decaying learning rate) that reproduces to about 1e-4 nat. The
  first stage's routine is noisier for formulas: their validation NLL jumps by up to ±0.03 nat between epochs.

### 4.5 Component formula (follow-up)

Instead of one product, the density is a weighted sum of separately normalised, physically motivated components:

`ρ(z', r | p) = w_shower·S + w_track·U + w_peak·D + w_blob·E + w_far·H`

| Component | Form | Meaning |
|---|---|---|
| Shower `S` | `L(z') · T(r│z')` | the hadronic or electromagnetic shower |
| – depth | plateau (4.1); gamma for the muon | start, rise, end, decay, slow slope |
| – lateral | `f1·Gauss(r;σ) + f2·R²/(π(r²+R²)²) + f3·e^(−r/λ)/(2πλ(r+1 mm))` | narrow core, heavy-tailed body, exponential halo; all vary with depth |
| Track `U` | `σ((z'−z_t0)/4 mm) · σ((R(T)−z')/w_R) · e^(−z'/λ_att) · (1−z'/R+0.03)^(−q)` × lateral | the primary particle's own track: range `R(T) = R1·T^k − R0`, attenuation length, Bragg rise |
| – track width | `σ(z')² = σ0² + (c·(z'/1 m)^k / βp)²`, plus an exponential halo | multiple scattering; delta electrons |
| Entry peak `D` | Gaussian at the entry point | energy in the first layer |
| End blob `E` | Gaussian at `z' = R(T)` | a particle that stops |
| Far halo `H` | `σ(z'/10 mm) · e^(−z'/L)/(r+1 cm)` | diffuse energy far from the axis |

- Every parameter is a polynomial in `x = (ln T − v0)/vs`, where `T` is the kinetic energy.
- The formula is fitted directly to the training hits by maximum likelihood, not to a network's output.
- Which components are used, and the degree, were chosen on validation.

### 4.6 Containment bodies ("hulls")

All are bodies of revolution `r(z')`, fitted per 1 GeV momentum bin to the 95 %-energy / 98 %-of-showers criterion.

| Body | Form | Parameters per bin |
|---|---|---|
| Envelope (first stage) | `r = A·u^a·e^(−bu)`, `u = z' − z0` | 4 |
| Drop (first stage) | `r = A·g((z'−z0)/L)`, `g ∝ t^(1/2)(1−t)^(3/2)` | 3 |
| Super-egg (follow-up) | `r = A·(1 − |τ|^n)^(1/m)`, with separate front and back half-lengths | 6 |
| r95 | the smallest region of a model's *average* map holding 95 % of the energy | none (derived) |

The r95 bodies describe the average shower, so they cover far fewer individual showers and are not comparable with the hulls.

### 4.7 Further analyses in the follow-up

- **Symbolic regression with PySR** on the remaining difference between network and formula
  (`scripts/sr_residual.py`, with a control run).
- **Literature variants:** a two-gamma depth profile and GFlash's electromagnetic lateral profile.
- **Angle-dependent extension:** the incidence angle as an extra input for a few entry parameters.
- **Time split:** prompt versus delayed energy.
- **Fluctuations:** principal-component analysis of per-event profiles and a simple generator built on it
  (`scripts/fluctuations_pca.py`).
- **Simulation truth:** which particles deposit the energy (`scripts/truth_decomposition.py`).
- **Physics cross-checks** of fitted parameters against values expected from the detector material
  (`scripts/lfhcal_expectations.py`).

## 5. Best results to date

### 5.1 Overview (original score, test NLL; parameter count in brackets)

| Particle | Baseline formula | Baseline × `Ā` | Component formula | MLP (report) | MLP (fine-tuned) |
|---|---|---|---|---|---|
| Electron | 14.8479 (33) | 14.8406 (27) | **14.8167** (75) | 14.8068 (2,785) | **14.8067** |
| Pion | 18.1563 (27) | 18.1420 (39) | **18.1235** (101) | 18.1221 (3,841) | **18.1208** |
| Proton | 18.2536 (45) | 18.2331 (45) | **18.1940** (143) | 18.1922 (881) | **18.1898** |
| Muon | 16.0899 (59) | 16.0846 (59) | **16.0120** (129) | 16.0046 (2,785) | **16.0034** |

Each mean has a standard error of about 0.007–0.015 nat, but paired differences are known to 0.0003–0.002 nat.

### 5.2 Best MLPs

| Particle | Hidden layers × neurons | Weights | Report MLP | Gain over baseline (paired) | Fine-tuned | CNN (weights) |
|---|---|---|---|---|---|---|
| Electron | 3 × 32 | 2,785 | 14.8068 | −0.0411 ± 0.0010 | 14.8067 | 14.8076 (336,355) |
| Pion | 4 × 32 | 3,841 | 18.1221 | −0.0342 ± 0.0011 | 18.1208 | 18.1218 (121,939) |
| Proton | 3 × 16 | 881 | 18.1922 | −0.0614 ± 0.0020 | 18.1898 | 18.1908 (121,939) |
| Muon | 3 × 32 | 2,785 | 16.0046 | −0.0854 ± 0.0012 | 16.0034 | 16.0041 (121,939) |

- **The recipe that works** is the MLP of section 4.2 with the architecture above.
- **The best stored weights are the fine-tuned ones.** They start from the report's weights and continue with `fit_fast`
  (6,000 steps, learning rate 1e-3 falling to 1e-4). The gain is small (0.0001–0.0024 nat).
- **Network size barely matters.** Candidates' validation scores differ by at most 0.0025 nat (0.012 for the proton).
- **The CNN adds nothing.** It agrees with the MLP to within 0.0015 nat.
- **MLP ≈ CNN suggests a ceiling.** Two unrelated architectures reach the same score, so this is probably close to the
  best any average-shape model in `(z', r)` can do.
- **Under the 3D likelihood** the fine-tuned MLPs score 14.7976 (electron), 18.1614 (proton) and 16.0000 (muon). The pion
  network trained this way is degenerate and its score (18.0920) is not reliable.

Where the weights are:

- Fine-tuned: `shape-form-proton/results/particles/<particle>/mlp_std.pt` (original score) and `mlp_slab.pt` (3D
  likelihood). The weights sit under the key `"state"`.
- The report's own checkpoints are not in this repository.

Loading the pion network with `common.py`, run from `shape-form-original/` (checked on 24 November 2026: it reproduces
the test NLL of 18.1208):

```python
import torch, common
common.configure(data_dir="data/", out_dir="results")
net = common.FieldModel(hidden=32, n_layers=4).to(common.DEVICE)
net.load_state_dict(torch.load("../shape-form-proton/results/particles/pion/mlp_std.pt",
                               map_location=common.DEVICE, weights_only=False)["state"])
log_P = net.eval()(torch.log(torch.tensor([5.0], device=common.DEVICE)))   # (1, 128, 32): log energy shares at 5 GeV
```

### 5.3 Best closed-form formulas

**The component formula (section 4.5) is the best closed form to date.**

| Particle | Components | Degree | Parameters | Test NLL | Gain over baseline × `Ā` | Gap closed | Still behind report MLP |
|---|---|---|---|---|---|---|---|
| Electron | shower (plateau) + entry peak + far halo | 4 | 75 | 14.8167 | −0.0239 ± 0.0009 | 70 % | +0.0099 ± 0.0008 |
| Pion | shower (plateau) + track + entry peak + end blob + far halo | 3 | 101 | 18.1235 | −0.0186 ± 0.0007 | 88 % | +0.0014 ± 0.0004 |
| Proton | same as pion | 6 | 143 | 18.1940 | −0.0391 ± 0.0016 | 90 % | +0.0018 ± 0.0005 |
| Muon | shower (gamma) + track + entry peak + end blob + far halo | 6 | 129 | 16.0120 | −0.0726 ± 0.0011 | 89 % | +0.0074 ± 0.0004 |

"Gap closed" is the share of the distance between the baseline × `Ā` and the fine-tuned MLP that the formula recovers.

- **A smaller variant is almost as good.** At degree 2 the scores are 14.8173 (53 parameters), 18.1239 (87), 18.1952 (87)
  and 16.0132 (81): at most 0.0012 nat worse.
- **An independent second sample confirms the gains** (about 100,000 events per particle, never used for any choice):
  −0.0236 ± 0.0003, −0.0212 ± 0.0004, −0.0409 ± 0.0007 and −0.0739 ± 0.0007.
- **Under the 3D likelihood** the formula scores 14.8072 (64 parameters), 18.1014, 18.1660 and 16.0144.
- **The formula has saturated.** Further terms and the PySR-found corrections add at most 0.0025 nat. The literature
  variants were worse.
- **Fitted parameters are physically sensible.** Examples: the muon's range of about 1.06 m per GeV of kinetic energy and
  its multiple-scattering exponent match expectations; attenuation lengths are about 1.3 nuclear interaction lengths
  (proton 266 mm, pion 339 mm); the electron's core width is 16–19 mm, about 0.8 Molière radii.
- **Caveats.** For hadrons the track's extra parameters are effective rather than literal. Test-set diagnostics played a
  part in choosing components, which is why the second sample is the clean check.

Where the formulas are:

- Code: `shape-form-proton/particle_shape.py`; loading: `shapelab.build_final`.
- Fitted values and all coefficients per particle: `shape-form-proton/results/particles/<particle>/formula.md`,
  `formula_coefficients_std.csv` (original score) and `formula_coefficients_slab.csv` (3D likelihood).
- The acceptance factor needed with the original score: `abar.pt` in the same folder.

**First-stage formulas**, for reference:

| Particle | Baseline (family, degree, parameters) | Test NLL | Distilled from the MLP (family, degree, parameters) | Test NLL |
|---|---|---|---|---|
| Electron | gamma × core+tail, 4, 33 | 14.8479 | plateau × Gauss+exp, 1, 19 | 14.8516 |
| Pion | gamma × core+tail, 3, 27 | 18.1563 | gamma × core+tail, 2, 21 | 18.1634 |
| Proton | gamma × core+tail, 6, 45 | 18.2536 | gamma × core+tail, 2, 21 | 18.2677 |
| Muon | plateau × core+tail, 6, 59 | 16.0899 | plateau × Gauss+exp, 2, 27 | 16.0712 |

The distilled formulas match the baseline with about half the parameters but do not recover the networks' extra
accuracy. They are slightly worse than the baseline for hadrons and better only for the muon.

### 5.4 Best containment bodies

Volume in litres inside the data range, with the part inside the detector in brackets. All reach the 0.98 coverage target
on test showers. Means over the 20 momentum bins.

| Body | Electron | Pion | Proton | Muon |
|---|---|---|---|---|
| Drop (report) | 26.1 (24.6) | 1486 (1362) | 1616 (1469) | – |
| Drop, refitted | 25.9 (24.4) | 1478 (1335) | 1600 (1428) | 46.5 (43.5) |
| **Super-egg** | 25.9 (24.2) | **1423** (1302) | **1544** (1395) | **37.7** (35.8) |
| Iso-density surface of the component formula, calibrated to 98 % | 31.7 (30.7) | 1491 (1368) | 1596 (1449) | 48.9 (44.9) |

- The super-egg is the smallest body for hadrons and the muon. For the electron it equals the drop.
- The report's drop was already close to optimal. The calibrated iso-density surface is not smaller, because the 98 %
  criterion is decided by extreme showers, not by the average shape.
- The first stage's envelope is larger: 26.5, 1755 and 1965 L for electron, pion and proton.

## 6. Limitations and open problems

- **Only the average is modelled.** Three principal components explain about three quarters of the per-shower variance
  for electrons and muons, but only 40–50 % for hadrons above 2 GeV. A 10-component generator still misses by several
  times the statistical floor. An event generator needs a separate, non-Gaussian fluctuation model.
- **Angle dependence at the entry is too weak.** The optional angle-dependent extension gains 0.002–0.009 nat (0.045 for
  the muon) but is an empirical correction and not part of the final formula.
- **Late energy is excluded.** Hits after 100 ns are cut; for hadrons that is at least 3–5 % of the energy.
- **The scores are not exact.** Neither likelihood normalises exactly to the instrumented cells (about 0.2 nat), so only
  models trained the same way should be compared.
- **The grid is finite.** It ends at `r` = 700 mm and `z'` = 1450 mm; hadronic hulls reach that edge, so their volumes
  are underestimated.
- **Detector cells are treated as uniform boxes.** At module boundaries the real segmentation is coarser.
- **Distillation is limited.** It follows the network's map rather than the data, and only tries eight candidates of
  degree 2 or less.
- **The MLP size search is greedy** (width first, then depth), and on flat validation curves the 1-SE rule is close to
  simply taking the best.

## 7. State of this checkout

- **Version control.** Only `.gitignore` and `README.md` are committed. Everything else is untracked.
- **First-stage notebooks.** The saved outputs come from the original run on another machine. That run's checkpoints are
  not here.
- **Local data.** Only the pion sample is present: `shape-form-original/data/` holds the CSV (15 GB, 49.1 million hits)
  and a Parquet copy with just the 14 columns the code uses (2.6 GB).
- **Local results.** `shape-form-original/results/pion/` holds a partial rerun of the MLP width search: three finished
  candidates and one interrupted.
- **Local additions.**
  - `common.py`: the shared notebook as an importable module. Settings go through `common.configure(...)`; it can also
    read Parquet. On synthetic data it gives bit-identical results to the notebook.
  - `05_mlp_pion_copy.ipynb`: a copy of the pion MLP notebook that imports `common.py`.
- **Follow-up study.** Its results and checkpoints are included (145 MB), but its scripts contain absolute paths of the
  machine they were written on and its data cache is absent. It will not run here without adapting those paths and
  rebuilding the cache from the CSV files of all four particles.
- **Project metadata.** `pyproject.toml` describes the project as voxel-wise segmentation of overlapping calorimeter
  showers. No such code is in the repository yet.

## 8. Terms used

| Term | Meaning |
|---|---|
| `z'`, `r` | Depth along the particle's direction from the shower start, and distance from the shower axis, in mm. |
| NLL (nat) | Per-event average negative log-likelihood of the deposits under the model. Only differences are meaningful. |
| Paired difference | Difference of two models' NLL on the same events, ± its standard error. Negative means the first is better. |
| Baseline | The first stage's GFlash-type formula (cylinder for the muon). |
| Original score / report metric | The first stage's likelihood; with `× Ā` when the acceptance factor is applied. |
| 3D likelihood / slab metric | The follow-up's likelihood, normalised to each event's actual detector slab. |
| Second sample (s2) | An independent simulation sample of about 100,000 events per particle, used only for checking. |
| Hull | A body of revolution that holds at least 95 % of the energy for 98 % of showers. |
