"""particle_shape – a végleges formula mind a négy részecskére (a ProtonShape általánosítása, a kiegészítő tagokkal).

    rho(z', r | p) = w_sh S + w_tr U + w_sp D + w_end E + w_far H        (mindegyik komponens külön a rácsra normálva)

  S  zápor:        mélységprofil (gamma; plató; vagy a Bock-féle kétgammás) × keresztmetszet (Gauss-mag + Cauchy-test +
                   exp(-r/lam)/(r + 1 mm) udvar), a keresztmetszet mélységenként a rács korongjára normálva (lat_norm="grid").
  U  nyom:         cső a tengelyen, az R(T) = R1 T^k - R0 hatótávig (T a mozgási energia); att: exp(-z'/lam_att) csillapítás;
                   bragg: (1 - z'/R + 0,03)^(-q) felfutás; ms: a szélesség a mélységgel nő (Highland, 1/(beta p));
                   track_halo: delta-elektron-udvar exp(-r/l)/(r + 1 mm).
  D  csúcs:        Gauss a belépésnél (az első találati réteg az origó).
  E  folt:         Gauss a hatótáv végén (endblob).          H  távoli udvar: exp(-z'/L) exp(-r/R)/(r + 10 mm) (far).

Minden nyers paraméter az x = (ln T - v0)/vs polinomja (T = sqrt(p^2 + m^2) - m a részecske tömegével); egyetlen bemenet: ln p.
Állandók (0. fok): CONST, log_latt, BOCK_CONST és EXTRA_CONST elemei – köztük a zápor udvarának hossza (log_lam) is.
"""
import math

import torch
import torch.nn.functional as F

import families
from proton_shape import ProtonShape
from shapelab import PolyFormula

MASS = {"electron": 0.000511, "muon": 0.105658, "pion": 0.139570, "proton": 0.938272}     # GeV


def kinetic(lnp, m):
    p = torch.exp(lnp)
    return p * p / (torch.sqrt(p * p + m * m) + m)


class ParticleShape(ProtonShape):
    PLATEAU = [("zs", -10.0), ("log_w1", math.log(15)), ("log_ell", math.log(3000)), ("log_w2", math.log(30)), ("kappa", 0.0)]
    ATT = [("log_latt", math.log(250.0))]
    # Bock et al. (NIM 186 (1981) 533): from the shower origin  w (x/X0)^(a-1) e^(-b x/X0) + (1-w) (x/lam)^(a-1) e^(-d x/lam);
    # from the calorimeter face: convolved with the exponential start distribution e^(-s/lam) (Kulchitsky & Vinogradov,
    # hep-ex/9903019); plus a small backward (backsplash) tail in front of the start, level f_b, length l_b.
    BOCK = [("logit_w", 1.0), ("log_a", math.log(1.2)), ("log_b", math.log(0.3)), ("log_d", math.log(1.0)),
            ("log_X0", math.log(20.0)), ("log_lamI", math.log(200.0)), ("z_st", 0.0), ("log_fb", -3.0), ("log_lb", math.log(50.0))]
    BOCK_CONST = ("log_X0", "log_lamI", "z_st", "log_fb", "log_lb")      # log_lamI: Bock's interaction length (own name, see below)

    # far: diffuse far halo (neutrons / photons travelling far): exp(-z'/L_f) exp(-r/R_f) / (r + 10 mm), from the entry on
    FAR = [("logit_far", -4.0), ("log_Lf", math.log(600.0)), ("log_Rf", math.log(300.0))]
    # endblob: deposit at the end of the range of a stopping particle (Bragg peak, capture products): Gaussian at z' = R(T)
    END = [("logit_end", -3.0), ("log_sez", math.log(40.0)), ("log_ser", math.log(40.0))]
    # track_halo: knock-on electrons / soft radiation around the primary track: (1-f) Gauss(st) + f e^(-r/lh) / (2 pi lh r)
    THALO = [("logit_th", -2.0), ("log_lh", math.log(30.0))]
    # bragg: Bragg-Kleeman rise of dE/dx towards the end of the range, (R - z')^(-q)  (q = 1 - 1/1.75 ~ 0.43)
    BRAGG = [("q_bk", 0.43)]
    # curv: quadratic depth terms of the lateral core width and of the two mixture logits
    CURV = [("q_sig", 0.0), ("q_1", 0.0), ("q_2", 0.0)]
    # ms: multiple-scattering broadening of the track (Highland): sigma(z')^2 = st^2 + (c (z'/1 m)^k / (beta p))^2, k ~ 1.5
    MS = [("log_cms", math.log(50.0)), ("ms_pow", 1.5)]
    # log_lam (the shower halo length) is constant in p: until 2026-11-17 this came from a name clash with Bock's interaction length;
    # all final fits were made this way, so it is kept explicitly (its depth slope s_lam can still depend on p, see 'slopes').
    EXTRA_CONST = ("log_Lf", "log_Rf", "log_sez", "log_ser", "log_lh", "q_bk", "q_sig", "q_1", "q_2", "log_cms", "ms_pow", "log_lam")

    def __init__(self, C, mass, degree=1, degrees=None, lnp0=0.0, lnp_s=1.0, spike=True, track=True, tie=True,
                 lateral="gce", lat_norm="grid", long="gamma", att=False, far=False, endblob=False, track_halo=False, bragg=False,
                 curv=False, ms=False):
        shower = list(self.SHOWER[lateral])
        if long == "plateau":
            shower = list(self.PLATEAU) + shower[3:]                     # the three gamma parameters -> five plateau parameters
        elif long == "bock":
            shower = list(self.BOCK) + shower[3:]
        spec = shower + (list(self.SPIKE) if spike else []) + (list(self.TRACK) if track else [])
        self.use_spike, self.use_track, self.tie = bool(spike), bool(track), bool(tie and spike and track)
        self.att = bool(att and track)
        if self.tie:
            spec += list(self.TIE)
        if self.att:
            spec += list(self.ATT)
        self.far, self.endblob = bool(far), bool(endblob and track)
        if self.far:
            spec += list(self.FAR)
        if self.endblob:
            spec += list(self.END)
        self.track_halo, self.bragg, self.curv = bool(track_halo and track), bool(bragg and track), bool(curv and lateral in ("gce", "cce"))
        if self.track_halo:
            spec += list(self.THALO)
        if self.bragg:
            spec += list(self.BRAGG)
        if self.curv:
            spec += list(self.CURV)
        self.ms = bool(ms and track)
        if self.ms:
            spec += list(self.MS)
        object.__setattr__(self, "SPEC", spec)
        d = {n: 0 for n, _ in spec if n in self.CONST or n == "log_latt" or n in self.BOCK_CONST or n in self.EXTRA_CONST}
        d.update(degrees or {})
        PolyFormula.__init__(self, C, degree, degrees=d, lnp0=lnp0, lnp_s=lnp_s)
        self.lateral, self.var, self.lat_norm, self.long = lateral, "lnT", lat_norm, long
        self.register_buffer("mass", torch.tensor(float(mass)))
        self.register_buffer("LOG_AREA", (C["LOG_VOL"] - math.log(C["DZ"])).clone(), persistent=False)
        if long == "bock":                                               # source points of the start convolution (2 per z' cell)
            ze = C["Z_EDGES"]
            src = torch.cat([ze[:-1] + 0.25 * C["DZ"], ze[:-1] + 0.75 * C["DZ"]]).sort().values
            self.register_buffer("SRC", src.clone(), persistent=False)
            self.register_buffer("DIFF", (C["ZC"][:, None] - src[None, :]).clone(), persistent=False)   # z'_i - s_j
            self.DS = float(C["DZ"]) / 2
        lo, hi = [float(torch.log(kinetic(torch.tensor(float(lnp0) + sg * float(lnp_s)), float(mass)))) for sg in (-1.0, 1.0)]
        self.register_buffer("v0", torch.tensor((hi + lo) / 2))
        self.register_buffer("vs", torch.tensor((hi - lo) / 2))

    def lnT(self, lnp):
        return torch.log(kinetic(lnp, self.mass))

    def params(self, lnp):
        x = (self.lnT(lnp) - self.v0) / self.vs
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ (self.coef * self.mask).T

    def theta(self, lnp):
        th = PolyFormula.theta(self, lnp)
        th["lnT"] = self.lnT(lnp)[:, None, None]
        return th

    def bock_face(self, th):
        """Bock profile from the calorimeter face on the z' grid: (B, N_Z), unnormalised."""
        g = lambda k: th[k][:, :, 0]                                         # (B, 1)
        w, a, b, d = torch.sigmoid(g("logit_w")), torch.exp(g("log_a")), torch.exp(g("log_b")), torch.exp(g("log_d"))
        X0, lam, zst = torch.exp(g("log_X0")), torch.exp(g("log_lamI")), g("z_st")
        x = self.DIFF[None]                                                  # (1, N_Z, Ns)
        pos = x > 0
        xc = x.clamp_min(1e-3)
        a3, b3, d3, w3, X03, lam3 = (v[:, :, None] for v in (a, b, d, w, X0, lam))
        lo = (w3 * torch.exp((a3 - 1) * torch.log(xc / X03) - b3 * xc / X03)
              + (1 - w3) * torch.exp((a3 - 1) * torch.log(xc / lam3) - d3 * xc / lam3)) * pos
        q = torch.sigmoid((self.SRC[None] - zst) / 3.0) * torch.exp(-(self.SRC[None] - zst).clamp_min(0) / lam) * self.DS / lam   # (B, Ns)
        face = (lo * q[:, None, :]).sum(-1)                                  # (B, N_Z)
        zc = self.ZZ[:, 0][None]                                             # (1, N_Z)
        back = torch.exp(g("log_fb")) * face.mean(1, keepdim=True) * torch.exp(-(zst - zc).clamp_min(0) / torch.exp(g("log_lb")))
        return face + back

    def log_shower(self, th, Z, R):
        if self.long == "gamma":
            return super().log_shower(th, Z, R)
        if self.long == "bock":
            th2 = dict(th, z0=torch.zeros_like(th["z_st"]), log_tmax=torch.zeros_like(th["z_st"]), log_am1=torch.zeros_like(th["z_st"]))
            lat = super().log_shower(th2, Z, R) - families.long_gamma(Z, th2["z0"], torch.exp(th2["log_tmax"]), torch.exp(th2["log_am1"]))
            return torch.log(self.bock_face(th) + 1e-30)[:, :, None] + lat
        # plateau depth profile (the report's cylinder, 04): fast rise at zs, slow change, cut-off at ell
        L = (F.logsigmoid((Z - th["zs"]) / torch.exp(th["log_w1"])) + F.logsigmoid((torch.exp(th["log_ell"]) - Z) / torch.exp(th["log_w2"]))
             + th["kappa"] * Z / 1000)
        th2 = dict(th, z0=torch.zeros_like(th["zs"]), log_tmax=torch.zeros_like(th["zs"]), log_am1=torch.zeros_like(th["zs"]))
        lat = super().log_shower(th2, Z, R) - families.long_gamma(Z, th2["z0"], torch.exp(th2["log_tmax"]), torch.exp(th2["log_am1"]))
        return L + lat

    def log_track(self, th, Z, R):
        lt = super().log_track(th, Z, R)                               # lon(z') - r^2 / (2 st^2)
        if self.att:
            lt = lt - Z.clamp_min(0) / torch.exp(th["log_latt"])
        if self.bragg:
            u = (Z / self.range_mm(th)).clamp(0.0, 1.0)
            lt = lt - th["q_bk"] * torch.log(1.0 - u + 0.03)
        if self.track_halo or self.ms:
            st = torch.exp(th["log_st"])
            lon = lt + R * R / (2 * st * st)
            sig = st
            if self.ms:                                                # depth-dependent width, normalised at every depth
                T = torch.exp(th["lnT"])
                bp = (T * T + 2 * T * self.mass) / (T + self.mass)     # beta * p [GeV]
                sig = torch.sqrt(st * st + (torch.exp(th["log_cms"]) * (Z.clamp_min(1.0) / 1000) ** th["ms_pow"] / bp) ** 2)
            lat = families.lat_gauss(R, sig)
            if self.track_halo:
                lat = torch.logaddexp(F.logsigmoid(-th["logit_th"]) + lat,
                                      F.logsigmoid(th["logit_th"]) + families.lat_exp_over_r(R, torch.exp(th["log_lh"])))
            lt = lon + lat
        return lt

    def components(self, lnp):
        lw, comp, th = super().components(lnp)
        if not (self.far or self.endblob):
            return lw, comp, th
        Z, R, B = self.ZZ[None], self.RR[None], len(lnp)
        norm = lambda ld: F.log_softmax((ld + self.LOG_VOL).reshape(B, -1), 1).reshape(B, self.N_Z, self.N_R)
        lw, comp = list(lw), list(comp)
        if self.endblob:                                               # w_end = sigma(logit_end) of what is left
            le, lne = F.logsigmoid(th["logit_end"]), F.logsigmoid(-th["logit_end"])
            Rr = self.range_mm(th)
            ce = norm(-(Z - Rr) ** 2 / (2 * torch.exp(th["log_sez"]) ** 2) - R * R / (2 * torch.exp(th["log_ser"]) ** 2))
            lw = [w + lne if w is not None else None for w in lw] + [le]
            comp = comp + [ce]
        if self.far:
            lf, lnf = F.logsigmoid(th["logit_far"]), F.logsigmoid(-th["logit_far"])
            cf = norm(F.logsigmoid(Z / 10.0) - Z.clamp_min(0) / torch.exp(th["log_Lf"]) - R / torch.exp(th["log_Rf"]) - torch.log(R + 10.0))
            lw = [w + lnf if w is not None else None for w in lw] + [lf]
            comp = comp + [cf]
        return tuple(lw), tuple(comp), th

    def init_from_data(self, C, data):
        h = data["train"].hits()
        long = "plateau" if self.long == "plateau" else "gamma"
        f0 = C["Formula"](long, "core_tail", degree=1).init_from(h.z.values, h.r.values, h.w.values)
        with torch.no_grad():
            for i, n in enumerate(f0.names):
                if n in self.names:
                    self.coef[self.names.index(n), 0] = f0.coef[i, 0]
            if self.lateral in ("gce", "cce"):
                rc0 = float(torch.exp(f0.coef[f0.names.index("log_rc"), 0]))
                self.coef[self.names.index("log_sig"), 0] = math.log(max(0.8 * rc0, 3.0))
                self.coef[self.names.index("log_R"), 0] = math.log(max(3.0 * rc0, 20.0))
        return self
