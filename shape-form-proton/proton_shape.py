"""proton_shape – az első protonformula (zápor + megálló nyom + belépési csúcs); a végleges formula a particle_shape.ParticleShape.

    rho(z', r | p) = w_sh(p) S(z', r | p) + w_tr(p) U(z', r | p) + w_sp(p) D(z', r),

mindhárom komponens a rácson 1-re normált, a súlyok energiahányadok.

  S  zápor:          gamma-profil mélységben × (mag + udvar) keresztirányban (a GFlash-forma);
                     lateral="cc": két Cauchy-tag (a report formulája), lateral="ce": Cauchy-mag + exp(-r/lam)/r udvar.
  U  megálló nyom:   keskeny cső a tengely mentén, amely az R(T) = R1 T^k − R0 hatótávnál ér véget
                     (T = sqrt(p² + m_p²) − m_p a mozgási energia: Bragg–Kleeman-típusú hatótáv–energia törvény).
  D  belépési csúcs: az első találati réteg (ennek súlypontja az origó), keskeny folt az origóban.

Minden nyers paraméter az x = (ln T − v0)/vs változó polinomja (a fokszám paraméterenként adható meg);
a modell egyetlen bemenete ln p.  A megálló-nyom és a csúcs ötlete és első megvalósítása az F4 kísérleti szálból származik
(lanes/F4_lowp/fam.py: StopMix); itt egy modulba rendezve, a laterális udvar választható alakjával.
"""
import math

import torch
import torch.nn.functional as F

import families
from shapelab import PolyFormula

M_P = 0.938272          # proton tömege [GeV]


def kinetic(lnp):
    """A proton mozgási energiája T [GeV], p = exp(lnp) [GeV] (kioltás nélküli alak)."""
    p = torch.exp(lnp)
    return p * p / (torch.sqrt(p * p + M_P ** 2) + M_P)


VARS = {"lnp": lambda lnp: lnp, "lnT": lambda lnp: torch.log(kinetic(lnp))}


class ProtonShape(PolyFormula):
    SHOWER = {"cc": [("z0", -20.0), ("log_tmax", math.log(100)), ("log_am1", math.log(3)),
                     ("log_rc", math.log(10)), ("log_ratio", math.log(3)), ("logit_pc", 0.85),
                     ("s_rc", 0.0), ("s_ratio", 0.0), ("s_pc", 0.0)],
              "ce": [("z0", -20.0), ("log_tmax", math.log(100)), ("log_am1", math.log(3)),
                     ("log_rc", math.log(10)), ("log_lam", math.log(90)), ("logit_pc", 0.85),
                     ("s_rc", 0.0), ("s_lam", 0.0), ("s_pc", 0.0)],
              # három tag: Gauss-mag + Cauchy-test + exp(-r/lam)/r udvar (az F1 szál legjobb laterális alakja)
              "gce": [("z0", -20.0), ("log_tmax", math.log(100)), ("log_am1", math.log(3)),
                      ("log_sig", math.log(15)), ("log_R", math.log(60)), ("log_lam", math.log(110)), ("logit_1", 0.0), ("logit_2", 0.0),
                      ("s_sig", 0.0), ("s_R", 0.0), ("s_lam", 0.0), ("s_1", 0.0), ("s_2", 0.0)],
              "cce": [("z0", -20.0), ("log_tmax", math.log(100)), ("log_am1", math.log(3)),
                      ("log_sig", math.log(12)), ("log_R", math.log(60)), ("log_lam", math.log(110)), ("logit_1", 0.0), ("logit_2", 0.0),
                      ("s_sig", 0.0), ("s_R", 0.0), ("s_lam", 0.0), ("s_1", 0.0), ("s_2", 0.0)]}
    SPIKE = [("logit_sp", -3.0), ("sp_z", 0.0), ("log_sp_sz", math.log(6.0)), ("log_sp_sr", math.log(10.0))]
    TRACK = [("logit_tr", -1.5), ("log_R1", math.log(700.0)), ("k_R", 1.77), ("R0", 15.0), ("log_wR", math.log(0.12)),
             ("kap", 0.3), ("zt0", 10.0), ("log_st", math.log(20.0))]
    TIE = [("log_d1", math.log(22.0))]
    CONST = ("s_rc", "s_ratio", "s_lam", "s_pc", "s_sig", "s_R", "s_1", "s_2", "sp_z", "log_sp_sz", "log_sp_sr", "log_R1", "k_R", "R0", "log_wR", "kap", "zt0",
             "log_st", "log_d1")

    def __init__(self, C, degree=1, degrees=None, lnp0=0.0, lnp_s=1.0, spike=True, track=True, tie=True, lateral="cc", var="lnT",
                 lat_norm="plane"):
        spec = list(self.SHOWER[lateral]) + (list(self.SPIKE) if spike else []) + (list(self.TRACK) if track else [])
        self.use_spike, self.use_track, self.tie = bool(spike), bool(track), bool(tie and spike and track)
        if self.tie:
            spec += list(self.TIE)
        object.__setattr__(self, "SPEC", spec)
        d = {n: 0 for n, _ in spec if n in self.CONST}
        d.update(degrees or {})
        super().__init__(C, degree, degrees=d, lnp0=lnp0, lnp_s=lnp_s)
        self.lateral, self.var, self.lat_norm = lateral, var, lat_norm
        # lat_norm="grid": the cross-section is normalised on the grid's disc (r < R_MAX) at every depth, so the depth profile of the
        # shower is set by L(z') alone (a lateral term cannot move energy out of the grid and act as a depth-dependent sink).
        # lat_norm="plane": every lateral term is normalised over the infinite plane (as the report's Formula).
        self.register_buffer("LOG_AREA", (C["LOG_VOL"] - math.log(C["DZ"])).clone(), persistent=False)
        lo, hi = [float(VARS[var](torch.tensor(float(lnp0) + sg * float(lnp_s)))) for sg in (-1.0, 1.0)]
        self.register_buffer("v0", torch.tensor((hi + lo) / 2))
        self.register_buffer("vs", torch.tensor((hi - lo) / 2))

    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        sd.pop(prefix + "LOG_AREA", None)            # derived from the grid; older checkpoints may or may not contain it
        super()._load_from_state_dict(sd, prefix, *args, **kw)

    # ---- paraméterek: polinom az x = (v(p) − v0)/vs változóban ----
    def params(self, lnp):
        x = (VARS[self.var](lnp) - self.v0) / self.vs
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ (self.coef * self.mask).T

    def theta(self, lnp):
        th = super().theta(lnp)
        th["lnT"] = VARS["lnT"](lnp)[:, None, None]
        return th

    # ---- komponensek (nem normált log-sűrűségek) ----
    def log_shower(self, th, Z, R):
        zk = Z / 1000
        L = families.long_gamma(Z, th["z0"], torch.exp(th["log_tmax"]), torch.exp(th["log_am1"]))
        if self.lateral in ("gce", "cce"):
            q = lambda k: th[k] * zk * zk if k in th else 0.0              # optional quadratic depth terms (curv)
            sig = torch.exp(th["log_sig"] + th["s_sig"] * zk + q("q_sig"))
            core = families.lat_gauss(R, sig) if self.lateral == "gce" else families.lat_cauchy(R, sig)
            mid = families.lat_cauchy(R, torch.exp(th["log_R"] + th["s_R"] * zk))
            halo = families.lat_exp_over_r(R, torch.exp(th["log_lam"] + th["s_lam"] * zk))
            l1, l2 = th["logit_1"] + th["s_1"] * zk + q("q_1"), th["logit_2"] + th["s_2"] * zk + q("q_2")
            lat = torch.logsumexp(torch.stack([F.logsigmoid(l1) + core, F.logsigmoid(-l1) + F.logsigmoid(l2) + mid,
                                               F.logsigmoid(-l1) + F.logsigmoid(-l2) + halo], 0), 0)
            if self.lat_norm == "grid":
                lat = lat - torch.logsumexp(lat + self.LOG_AREA[None], -1, keepdim=True)
            return L + lat
        rc = torch.exp(th["log_rc"] + th["s_rc"] * zk)
        lp = th["logit_pc"] + th["s_pc"] * zk
        if self.lateral == "cc":
            halo = families.lat_cauchy(R, rc * (1 + torch.exp(th["log_ratio"] + th["s_ratio"] * zk)))
        else:
            halo = families.lat_exp_over_r(R, torch.exp(th["log_lam"] + th["s_lam"] * zk))
        lat = torch.logaddexp(F.logsigmoid(lp) + families.lat_cauchy(R, rc), F.logsigmoid(-lp) + halo)
        if self.lat_norm == "grid":
            lat = lat - torch.logsumexp(lat + self.LOG_AREA[None], -1, keepdim=True)
        return L + lat

    def range_mm(self, th):
        """R(T) = R1 T^k − R0 [mm] (T GeV-ben), sima alsó korláttal."""
        raw = torch.exp(th["log_R1"] + th["k_R"] * th["lnT"]) - th["R0"]
        return F.softplus(raw / 5.0) * 5.0 + 1.0

    def log_track(self, th, Z, R):
        Rr = self.range_mm(th)
        wR = torch.exp(th["log_wR"]) * Rr + 3.0
        lon = F.logsigmoid((Z - th["zt0"]) / 4.0) + F.logsigmoid((Rr - Z) / wR) - th["kap"] * (Z / Rr).clamp(-1.0, 1.5)
        st = torch.exp(th["log_st"])
        return lon - R * R / (2 * st * st)

    def log_spike(self, th, Z, R):
        sz, sr = torch.exp(th["log_sp_sz"]), torch.exp(th["log_sp_sr"])
        return -(Z - th["sp_z"]) ** 2 / (2 * sz * sz) - R * R / (2 * sr * sr)

    def log_weights(self, th):
        """(ln w_zápor, ln w_nyom, ln w_csúcs), mind (B,1,1) vagy None."""
        zero = torch.zeros_like(th["lnT"])
        if self.use_track:
            l_tr, l_ntr = F.logsigmoid(th["logit_tr"]), F.logsigmoid(-th["logit_tr"])
        else:
            l_tr, l_ntr = None, zero
        if not self.use_spike:
            return l_ntr, l_tr, None
        l_sp, l_nsp = F.logsigmoid(th["logit_sp"]), F.logsigmoid(-th["logit_sp"])
        if not self.use_track:
            return l_nsp, None, l_sp
        if not self.tie:
            return l_nsp + l_ntr, l_nsp + l_tr, l_sp
        # kötött súlyok: a megálló proton csúcsa a nyom első rétege, f1 = d1 / (d1 + Lt)
        d1 = torch.exp(th["log_d1"])
        Lt = F.softplus((self.range_mm(th) - th["zt0"]) / 5.0) * 5.0
        l_f1, l_nf1 = torch.log(d1) - torch.log(d1 + Lt), torch.log(Lt.clamp_min(1e-12)) - torch.log(d1 + Lt)
        return l_ntr + l_nsp, l_tr + l_nf1, torch.logaddexp(l_tr + l_f1, l_ntr + l_sp)

    def components(self, lnp):
        """(log-súlyok, a komponensek normált log P térképei, paraméterek) – ábrákhoz és diagnosztikához."""
        th = self.theta(lnp)
        Z, R, B = self.ZZ[None], self.RR[None], len(lnp)
        norm = lambda ld: F.log_softmax((ld + self.LOG_VOL).reshape(B, -1), 1).reshape(B, self.N_Z, self.N_R)
        comp = [norm(self.log_shower(th, Z, R)),
                norm(self.log_track(th, Z, R)) if self.use_track else None,
                norm(self.log_spike(th, Z, R)) if self.use_spike else None]
        return self.log_weights(th), comp, th

    def forward(self, lnp):
        lw, comp, _ = self.components(lnp)
        return torch.logsumexp(torch.stack([w + c for w, c in zip(lw, comp) if c is not None], 0), 0)

    def describe(self, title=""):
        """Parameter table. The polynomial variable is x = (v(p) - v0)/vs with v = ln(T/GeV) (T: kinetic energy), not ln p."""
        rows = []
        for i, n in enumerate(self.names):
            terms = " ".join(f"{float(v):+.4g}" + ("" if k == 0 else r"\,x" if k == 1 else rf"\,x^{k}")
                             for k, v in enumerate((self.coef * self.mask)[i].detach()) if k <= self.deg[n])
            rows.append(f"| `{n}` | ${terms}$ |")
        v = "\\ln(p/\\mathrm{GeV})" if getattr(self, "var", "lnT") == "lnp" else "\\ln(T/\\mathrm{GeV})"
        var = f"$x=({v} {-float(self.v0):+.4g})/{float(self.vs):.4g}$"
        return "\n".join([f"### {title}", "", f"| parameter | value, {var} |", "|---|---|"] + rows
                         + ["", f"Number of parameters: **{self.n_params()}**"])

    def log_density(self, th, Z, R):          # a PolyFormula interfész kedvéért (a forward felül van írva)
        return self.log_shower(th, Z, R)

    def init_from_data(self, C, data):
        """Ugyanaz a kezdőérték-becslés, mint a 00_common Formula.init_from (átlagos mélység, szórás, medián sugár)."""
        h = data["train"].hits()
        f0 = C["Formula"]("gamma", "core_tail", degree=1).init_from(h.z.values, h.r.values, h.w.values)
        with torch.no_grad():
            for i, n in enumerate(f0.names):
                if n in self.names:
                    self.coef[self.names.index(n), 0] = f0.coef[i, 0]
            if self.lateral in ("gce", "cce"):                          # widths from the median radius of the training energy
                rc0 = float(torch.exp(f0.coef[f0.names.index("log_rc"), 0]))
                self.coef[self.names.index("log_sig"), 0] = math.log(max(0.8 * rc0, 3.0))
                self.coef[self.names.index("log_R"), 0] = math.log(max(3.0 * rc0, 20.0))
        return self
