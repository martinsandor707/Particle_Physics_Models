# Proton shape lab – shared brief for experiment agents

Goal of the project: find a **better 3D shape for the proton shower** than the ones in the team report
(`/home/user/tamas/models/lib/report_hu.pdf`, built by the notebooks in `/home/user/sofiya/shape-form/`).
Everything here must stay directly comparable with the report, so all experiments use the *unchanged* common code
of `00_common.ipynb` (grid, detector folding, likelihood, split).

## Hard rules

- **Read-only:** `/home/user/sofiya/**`, `/home/user/data/**`, `/home/user/robi/**`, `/home/user/tamas/models/**`.
  Never write, move or delete anything there.
- **Write only inside your own lane directory** `/home/user/tamas/shape-form-proton/lanes/<lane>/` (scripts, logs, results, figures).
  Do not edit `protonlab.py`, `shapelab.py`, `envlab.py`, `families.py`, `00_common.ipynb` (shared by all agents) – if you need a
  change, copy the function into your lane and say so in your report.
- Python: **`/home/user/.venv/bin/python`** (torch 2.10 + CUDA). Set `MPLBACKEND=Agg`. Figures: PNG only.
- One shared GPU (H200). Keep each process modest (< 12 GB GPU memory), at most **2 concurrent python processes per agent**.
  A single foreground command may run at most ~9 minutes: use short runs, or `run_in_background` + a log file.
- **No test-set leakage:** fit on `data["train"]`, choose everything (family, degree, hyper-parameters, early stopping) on
  `data["val"]`. The test set is evaluated only for candidates you finally report. Never tune on test.
- Report numbers exactly as measured; if something fails or is inconclusive, say so. No rounding in your favour.

## The data and the report's pipeline (proton)

- 99 925 proton events, p = 0.24–20 GeV, split 69 946 / 14 987 / 14 992 (train/val/test), stratified per 1-GeV bin, `SEED=1`.
- Hits are in the shower frame: origin = energy-weighted centroid of the hits in the event's first hit layer
  (`z0 = min z`), z' along the true direction of incidence, `r = hypot(x', y')`. Local = R·trans with rows of R =
  (e_theta, e_phi, e_r); hence the detector-axis depth of a point is `z_trans = z' cos(theta) - x' sin(theta) >= 0`.
- Cuts: time < 100 ns, −200 ≤ z' ≤ 1450 mm, r < 700 mm. Hit weight = energy / event energy (each event sums to 1).
- Grid: 128 × 32 cells in (z', r); r edges = 700·(i/32)^1.7. Every density model is `model(lnp) -> log P (B, 128, 32)`
  (cell probabilities, sums to 1).
- Metric: per-event NLL (nat/deposit) = −Σ_i w_i [log g(c_i) − log M_e]; g = model density averaged over the (rotated)
  48.6 × 48.6 × 20.5 mm cell by K Monte-Carlo points; M_e = model mass in front of the back face (z' < z_b).
  `test_nll(model, data[split])` uses K = 64 and a fixed seed, so per-event values are comparable (paired) between models.
- Report numbers for the proton (test NLL, parameters): GFlash baseline = gamma × core_tail, degree 6: **18.2536 (45)**;
  degree 1: 18.2869 (15); MLP L3xH16: **18.1922 (881)**; CNN: 18.1908 (121 939); distilled formulas 18.2677 / 18.2662 (21).
  Validation NLL: GFlash n=1…6: 18.2889, 18.2646, 18.2627, 18.2620, 18.2595, 18.2587; MLP 18.1958.
- Envelope bodies (criterion: ≥ 95 % of the energy inside for 98 % of the showers, per 1-GeV bin, minimal volume):
  envelope `r = A u^a e^{-bu}`: mean V = 1965 L, test coverage 0.976; drop `r = A g(t)`, `g ∝ t^{1/2}(1-t)^{3/2}`: 1707 L, 0.977.
  r95 bodies of the mean maps: GFlash 470 L (cov 0.73), MLP 466 L (0.75).

## Harness (import from `/home/user/tamas/shape-form-proton`)

```python
import sys; sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import shapelab, families, envlab, protonlab

C, geo, data = shapelab.setup(batch=256, out_dir="<your lane>/results", slab=False)   # C = namespace of 00_common
base = shapelab.baseline_nll(C, data)          # per-event val/test NLL of the report's models (cached): gflash_n1..6, mlp, cnn, ...
c, s = shapelab.lnp_center_scale(data)         # L = (ln p − c)/s, same centring as the report's select_degree

class MyFamily(shapelab.PolyFormula):          # every raw parameter = polynomial in L; per-parameter degree via `degrees={...}`
    SPEC = [("z0", -20.0), ("log_tmax", 4.6), ...]
    def log_density(self, th, Z, R):           # th[name]: (B,1,1); Z, R: (1,128,32) cell centres [mm]; return log dE/dV, un-normalised
        ...
m = MyFamily(C, degree=2, lnp0=c, lnp_s=s).to(C["DEVICE"])
shapelab.fit_fast(C, data, m, steps=3000)      # Adam + cosine lr, keeps the best validation state (uses C["event_nll"])
res, val, test = shapelab.evaluate(C, data, m, "my_name", base)   # JSON + checkpoint in <out_dir>/proton/, paired Δ vs gflash_n6 and mlp
```

- `C["fit"]`, `C["select_degree"]`, `C["test_nll"]`, `C["export"]`, `C["Formula"]`, `C["FieldModel"]`, `C["distill"]`,
  `C["iso_radius"]` … are the report's own functions. `families.py` has building blocks and `GammaCoreTail` (the baseline rewritten
  on `PolyFormula`; use it as a template and as a sanity check).
- `shapelab.setup(..., slab=True)` switches to the **3D (slab) likelihood**: identical folded per-hit density, but the model is
  normalised over the detector slab of the event (`−layer/2 ≤ z' cosθ − x' sinθ ≤ D`, tilted front and back faces) instead of
  `z' < z_b`. `C["event_slab"](b, box)` gives the azimuthal acceptance A (B,128,32) of every (z', r) ring; closed form with arccos.
  `shapelab.mean_acceptance(C, data["train"])` is its event average Ā(z', r); `shapelab.WithAcceptance(base, Ā)` multiplies a model by it
  (a pure (z', r | p) model usable with the standard likelihood).
- A model is *not* allowed to see anything but `lnp` (plus fixed, parameter-free geometry). Parameter count = number of free
  coefficients (`model.n_params()`).
- `envlab`: `deposits`, `shower_quantile`, `threshold`, `coverage`, `Body` (+ `Envelope`, `BetaDrop`), `bin_sets`, `grid_volume_L`
  for the envelope criterion (exactly the functions of notebooks 02/03).

## What to return

Your final message is consumed by a program: fill the requested structured fields precisely (numbers as measured, absolute
paths of scripts/checkpoints/figures you created). Put longer tables in files inside your lane and reference them.

## Established facts (measured by the lead on 2026-11-16; you can rely on them)

1. **Reproduction.** With this harness the report's proton numbers come back exactly (per-event differences ~3e-5):
   GFlash n=6 test 18.2536, MLP 18.1922, MLP-formula 18.2677.
2. **Speed / noise.** Harness start ≈ 2 s (cached data). `fit_fast(steps=3000)` at batch 256 ≈ 15–25 s for a formula with
   ≤ 60 parameters. Repeating a fit with another seed changes the validation NLL by ~1e-4; 9000 steps gain ~3e-4.
   So validation differences above ~0.001 nat are real; always quote the paired SE (`shapelab.paired`).
3. **The front face is tilted (3D slab geometry) – the largest single effect.** Because `z0 = min z` of the event, every hit
   satisfies `z_trans = z' cosθ − x' sinθ ≥ 0` (cell boxes reach −10.25 mm). In (z', r) this cuts the body of revolution by an
   oblique plane through the origin (θ = 0.055–0.628 rad, density rising with θ, independent of p). The closed-form acceptance
   (`C["slab_accept"]`) agrees with brute-force Monte Carlo to 4e-4 nat in ln M; all fold points are inside the slab.
   - Report models re-normalised over the slab, *no refit* (val / test): GFlash n=1 18.2537 / 18.2518, n=2 18.2318 / 18.2273,
     n=6 18.2272 / 18.2222, MLP 18.1755 / 18.1720, CNN 18.1748 / 18.1707.
   - `families.GammaCoreTail` degree 2 (21 params) refitted with `fit_fast`: standard likelihood val 18.2637 / test 18.2593
     (the report's own n=2 fit: 18.2646 / 18.2600); **slab likelihood val 18.2115 / test 18.2057**.
   - At p ≥ 5 GeV nearly all of KL(MLP ‖ GFlash) (0.04–0.055 nat) comes from z' < 30 mm: the cone-shaped front of the data,
     which a product L(z')·T(r) cannot represent, plus the entry spike.
4. **Momentum dependence of the baseline's deficit** (test, MLP − GFlash n=6, standard likelihood): bin 0 (p < 1 GeV) −0.49,
   bin 1 −0.11, bins 2–4 −0.05…−0.07, bins 5–19 −0.02…−0.04 nat. Low-momentum protons mostly *range out* (kinetic energy
   T = sqrt(p² + 0.938²) − 0.938 GeV: 30 MeV at p = 0.24, 0.43 GeV at p = 1, 1.27 GeV at p = 2): a short track ending in a
   stopping blob, not a hadronic shower. KL(MLP ‖ GFlash) is 1.7 nat at p = 0.5 GeV and 0.15 at 1.5 GeV.
5. **Entry spike.** The origin is the energy-weighted centroid of the first hit layer, so there is always energy at
   (z' ≈ 0, r ≈ 0): the first hit layer holds 43 % of the visible energy in bin 0, 12 % in bin 1, 7 % in bin 2 and 2.5–5 % above
   5 GeV; 2.6 % of all energy sits exactly at the origin (r < 1 mm). 97.4 % of the events start in the first physical layer.
6. **Lateral profile.** dE/dr of the data is an almost perfect exponential between r ≈ 50 and 350 mm (slope ≈ 95–100 mm at
   5–20 GeV), i.e. T(r) ∝ e^{−r/λ}/r rather than the r^{-4} tails of the two Cauchy terms.
7. **Residual 3D asymmetry (detector edges).** Energy-weighted ⟨x'⟩ (x' = along e_theta, +x' points radially outward and upstream):
   θ ∈ [0.5, 0.7]: −31 mm at z' = 150–400, −47 at 400–800, −77 at 800–1450 (outer edge of the calorimeter, cell centres reach
   |x|, |y| ≈ 2600 mm while the axis is at ρ = z·tanθ up to ≈ 2700 mm); θ < 0.2: +40 mm at 400–800 and +94 at 800–1450 (inner
   beam hole). ⟨y'⟩ ≈ 0 everywhere. `cache/proton_geometry.pt` (being built; may appear a few minutes after you start) holds the
   per-event origin (x0, y0, z0), θ, φ aligned with the cached train/val/test events, plus all cell centres (`cells_xy`) and
   layer positions (`layers_z`) of the detector.
8. **Where the remaining gap is (slab metric, validation).** `GammaCoreTail` degree 2 refitted under the slab likelihood versus the
   report's MLP re-normalised over the slab: +0.036 nat overall, of which bins 0–1 (p < 2 GeV) contribute 0.026 (+0.45 nat per
   event in bin 0, +0.08 in bin 1), bins 2–4 0.002 and bins 5–19 0.008 (≈ +0.011 per event). So above 2 GeV the plain formula is
   already within ≈ 0.01 nat of the network once the geometry is right; the big prize is the stopping-proton regime.
9. **Network upper bound under the slab likelihood.** The report's MLP (L3xH16, report protocol, batch 32, 60 epochs) trained
   with the slab likelihood: test 18.1624 (val ≈ 18.1675), i.e. 0.0096 better than the standard-trained MLP re-normalised (18.1720).
10. **Report's own (standard) metric with the parameter-free acceptance factor.** `WithAcceptance(GammaCoreTail, Ā)`, `fit_fast`:
    front only: n=1 val 18.2667 / test 18.2645, n=2 18.2452 / 18.2406, n=3 18.2434 / 18.2365; front+back: n=1 18.2648 / 18.2624,
    n=2 18.2437 / 18.2389, n=3 18.2420 / 18.2349 (report baseline n=6, 45 parameters: 18.2587 / 18.2536).
11. A formula trained with the slab likelihood is **not** a valid stand-alone (z', r) model: it puts ≈ 15 % of its mass outside the
    slab and scores 18.37 under the standard likelihood. For the standard metric always refit (plain or with `WithAcceptance`).
12. The report's training protocol (`C["fit"]`: batch 32, constant lr 0.03) is noisy: the validation NLL of a formula jumps by up to
    ±0.03 nat between epochs and the kept "best epoch" is partly luck. `fit_fast` (cosine decay) is cleaner and reproducible to 1e-4.
