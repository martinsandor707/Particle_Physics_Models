"""fam.py - lane F2 (longitudinal structure of the proton shower).

LongMix: a sum of separable components, each normalised on the (z', r) grid and mixed with energy fractions:

    P(z', r | p) = f_sh(p) * Shower(z', r | p) + f_sp(p) * Spike(z', r) + f_tr(p) * Track(z', r | p)

  Shower  = L(z'|p) * T(r | z', p);  T = two Cauchy terms with a depth slope (exactly the baseline's lateral part);
            L = "gamma"  : the baseline gamma (start z0, softplus-smoothed)
                "gamma2" : mixture of two gammas with a common start z0
                "conv"   : gamma(a, b) convolved with an exponential distribution of the shower start (attenuation length lam,
                           starting at zs) - numerical convolution on the 128-cell grid
                "conv2"  : (short gamma + long gamma) convolved with the same exponential start distribution
  Spike   = entry spike at the origin: cell-integrated logistic bump in z' (centre sp_mu, width sp_w) times exp(-(r/R_s)^k)
  Track   = primary track on the axis: exp(-kappa (z' - z_on)) for z' > z_on (cell-integrated turn-on), optional range cut-off
            sigmoid((R(p) - z)/w), lateral exp(-(r/R_t)^k)

Every raw parameter is a polynomial in L = (ln p - lnp0)/lnp_s (shapelab.PolyFormula); the component fractions are the softmax of
(0, lw_spike(L), lw_track(L)).  No side effects at import.
"""
import math

import torch
import torch.nn.functional as F

from shapelab import PolyFormula

LOG_PI = math.log(math.pi)


def lat_cauchy(r, R):
    return 2 * torch.log(R) - LOG_PI - 2 * torch.log(r * r + R * R)


def long_gamma(z, z0, tmax, am1):
    s = F.softplus((z - z0) / 2) * 2 + 1e-3
    return am1 * torch.log(s) - am1 / tmax * s


class LongMix(PolyFormula):
    """Shower (+ entry spike) (+ primary track), each normalised on the grid, mixed with p-dependent energy fractions."""

    def __init__(self, C, degree=2, long="gamma", spike=None, track=None, track_range=False, wdeg=2, lam_deg=1, kap_deg=1,
                 lat2=False, degrees=None, lnp0=0.0, lnp_s=1.0):
        self.long, self.spike, self.track, self.track_range, self.lat2 = long, spike, track, bool(track_range), bool(lat2)
        spec, deg = [], {}

        def add(name, init, d=None):
            spec.append((name, float(init)))
            if d is not None:
                deg[name] = d

        # shower: lateral part (baseline)
        add("log_rc", math.log(10)); add("log_ratio", math.log(3)); add("logit_pc", 0.85)
        add("s_rc", 0.0, 0); add("s_ratio", 0.0, 0); add("s_pc", 0.0, 0)
        # shower: longitudinal part
        if long in ("gamma", "gamma2"):
            add("z0", -20.0); add("log_tmax", math.log(100)); add("log_am1", math.log(3))
            if long == "gamma2":
                add("log_tmax2", math.log(350)); add("log_am1b", math.log(1.0)); add("logit_f2", -1.0)
        elif long in ("conv", "conv2"):
            add("zs", -15.0, 0); add("log_lam", math.log(200), lam_deg)
            add("log_a", math.log(1.5)); add("log_b", math.log(120))
            if long == "conv2":
                add("log_a2", math.log(1.5)); add("log_b2", math.log(30)); add("logit_f2", -1.0)
        else:
            raise ValueError(long)
        if lat2:                                         # the second gamma gets its own (p-independent-slope) lateral part
            add("b_log_rc", math.log(10)); add("b_log_ratio", math.log(3)); add("b_logit_pc", 0.85)
        if spike:
            add("sp_mu", 0.0, 0); add("sp_logw", math.log(3.0), 0); add("sp_logR", math.log(10.0), 0)
            if spike == "gg":
                add("sp_logk", math.log(2.0), 0)
            add("lw_spike", -3.0, wdeg)
        if track:
            add("tr_zon", -10.0, 0); add("tr_kap", 0.4, kap_deg); add("tr_logRt", math.log(20.0), 0)
            if track == "gg":
                add("tr_logk", math.log(2.0), 0)
            if track_range:
                add("tr_logR", 7.1, 2); add("tr_logwr", math.log(0.15), 0)
            add("lw_track", -2.5, wdeg)
        self.SPEC = spec
        deg.update(degrees or {})
        super().__init__(C, degree, degrees=deg, lnp0=lnp0, lnp_s=lnp_s)
        if track and track_range and self.deg["tr_logR"] >= 1:           # proton range rises steeply with p: start from a physical slope
            with torch.no_grad():
                self.coef[self.names.index("tr_logR"), 1] = 5.0
        ze = C["Z_EDGES"].detach().clone().float()
        self.register_buffer("z_lo", ze[:-1].clone())
        self.register_buffer("z_hi", ze[1:].clone())
        dz = float(C["DZ"])
        k = torch.arange(self.N_Z)
        u = k.float() * dz
        u[0] = dz / 3                                                    # same-cell term: mean forward distance, half weight
        logw = torch.zeros(self.N_Z)
        logw[0] = math.log(0.5)
        idx = k[:, None] - k[None, :]
        self.register_buffer("u_k", u.to(ze.device))
        self.register_buffer("logw_k", logw.to(ze.device))
        self.register_buffer("kidx", idx.clamp_min(0).to(ze.device))
        self.register_buffer("kmask", (idx < 0).to(ze.device))

    # ---- building blocks -------------------------------------------------------------------------------------------
    def _lat(self, th, Z, R, pre=""):
        zk = Z / 1000
        rc = torch.exp(th[pre + "log_rc"] + th["s_rc"] * zk)
        rt = rc * (1 + torch.exp(th[pre + "log_ratio"] + th["s_ratio"] * zk))
        lp = th[pre + "logit_pc"] + th["s_pc"] * zk
        return torch.logaddexp(F.logsigmoid(lp) + lat_cauchy(R, rc), F.logsigmoid(-lp) + lat_cauchy(R, rt))

    def _start(self, th):
        """log of the cell-integrated exponential start distribution, (B, N_Z)."""
        zs, lam = th["zs"][:, :, 0], torch.exp(th["log_lam"][:, :, 0])
        S = torch.exp(-F.relu(self.z_lo[None] - zs) / lam) - torch.exp(-F.relu(self.z_hi[None] - zs) / lam)
        return torch.log(S.clamp_min(1e-30))

    def _kernel(self, th, a_name, b_name):
        a, b = torch.exp(th[a_name][:, :, 0]), torch.exp(th[b_name][:, :, 0])
        return (a - 1) * torch.log(self.u_k[None]) - self.u_k[None] / b + self.logw_k[None]          # (B, N_Z) over offsets

    def _convolve(self, logS, logG):
        K = logG[:, self.kidx].masked_fill(self.kmask[None], -float("inf"))                          # (B, i, j): G(z_i - z_j)
        return torch.logsumexp(K + logS[:, None, :], 2)

    def _long_parts(self, th, Z):
        """List of (log L(z') (B, N_Z, 1) normalised over z, log weight (B,1,1) or None)."""
        Zc = Z[:, :, :1]
        if self.long == "gamma":
            return [(long_gamma(Zc, th["z0"], torch.exp(th["log_tmax"]), torch.exp(th["log_am1"])), None)]
        if self.long == "gamma2":
            l1 = long_gamma(Zc, th["z0"], torch.exp(th["log_tmax"]), torch.exp(th["log_am1"]))
            l2 = long_gamma(Zc, th["z0"], torch.exp(th["log_tmax2"]), torch.exp(th["log_am1b"]))
            f = th["logit_f2"]
            return [(F.log_softmax(l1, 1), F.logsigmoid(-f)), (F.log_softmax(l2, 1), F.logsigmoid(f))]
        logS = self._start(th)
        if self.long == "conv":
            return [(self._convolve(logS, self._kernel(th, "log_a", "log_b"))[:, :, None], None)]
        g1 = F.log_softmax(self._kernel(th, "log_a", "log_b"), 1)
        g2 = F.log_softmax(self._kernel(th, "log_a2", "log_b2"), 1)
        f = th["logit_f2"]
        if self.lat2:
            return [(F.log_softmax(self._convolve(logS, g1), 1)[:, :, None], F.logsigmoid(-f)),
                    (F.log_softmax(self._convolve(logS, g2), 1)[:, :, None], F.logsigmoid(f))]
        f = f[:, :, 0]
        return [(self._convolve(logS, torch.logaddexp(F.logsigmoid(-f) + g1, F.logsigmoid(f) + g2))[:, :, None], None)]

    def _shower(self, th, Z, R):
        parts = self._long_parts(th, Z)
        if len(parts) == 1:
            return parts[0][0] + self._lat(th, Z, R)
        if self.lat2:
            return torch.logaddexp(parts[0][1] + parts[0][0] + self._lat(th, Z, R),
                                   parts[1][1] + parts[1][0] + self._lat(th, Z, R, "b_"))
        return torch.logaddexp(parts[0][1] + parts[0][0], parts[1][1] + parts[1][0]) + self._lat(th, Z, R)

    def _spike(self, th, Z, R):
        mu, w = th["sp_mu"], torch.exp(th["sp_logw"])
        lo, hi = self.z_lo[None, :, None], self.z_hi[None, :, None]
        pz = torch.sigmoid((hi - mu) / w) - torch.sigmoid((lo - mu) / w)
        k = torch.exp(th["sp_logk"]) if self.spike == "gg" else (1.0 if self.spike == "exp" else 2.0)
        return torch.log(pz.clamp_min(1e-30)) - (R / torch.exp(th["sp_logR"])) ** k

    def _track(self, th, Z, R):
        zon = th["tr_zon"]
        lo, hi = self.z_lo[None, :, None], self.z_hi[None, :, None]
        a = torch.maximum(lo, zon)
        ov = (hi - a).clamp(min=0.0)                                        # part of the cell behind the start of the track
        zm = 0.5 * (hi + a) - zon
        lz = torch.log(ov.clamp_min(1e-12)) - th["tr_kap"] * zm / 100.0
        if self.track_range:
            Rg = torch.exp(th["tr_logR"])
            lz = lz + F.logsigmoid((Rg - zm) / (Rg * torch.exp(th["tr_logwr"]) + 1.0))
        k = torch.exp(th["tr_logk"]) if self.track == "gg" else (1.0 if self.track == "exp" else 2.0)
        return lz - (R / torch.exp(th["tr_logRt"])) ** k

    # ---- model -----------------------------------------------------------------------------------------------------
    def components(self, lnp):
        """(names, log component maps (B, n, N_Z*N_R) each normalised, log fractions (B, n))."""
        th, Z, R, B = self.theta(lnp), self.ZZ[None], self.RR[None], len(lnp)
        names = ["shower"]
        comps = [F.log_softmax((self._shower(th, Z, R) + self.LOG_VOL).reshape(B, -1), 1)]
        logits = [torch.zeros(B, device=lnp.device)]
        if self.spike:
            names.append("spike")
            comps.append(F.log_softmax((self._spike(th, Z, R) + self.LOG_VOL).reshape(B, -1), 1))
            logits.append(th["lw_spike"].reshape(B))
        if self.track:
            names.append("track")
            comps.append(F.log_softmax((self._track(th, Z, R) + self.LOG_VOL).reshape(B, -1), 1))
            logits.append(th["lw_track"].reshape(B))
        return names, torch.stack(comps, 1), F.log_softmax(torch.stack(logits, 1), 1)

    def forward(self, lnp):
        _, comps, lw = self.components(lnp)
        return torch.logsumexp(comps + lw[:, :, None], 1).reshape(len(lnp), self.N_Z, self.N_R)

    def log_density(self, th, Z, R):                                         # not used (forward is overridden)
        raise NotImplementedError

    def init_from(self, names, coef):
        """Copy the coefficients of equally named parameters from another PolyFormula (names, coef*mask)."""
        with torch.no_grad():
            for i, n in enumerate(self.names):
                if n in names:
                    src = coef[names.index(n)]
                    d = min(self.deg[n], len(src) - 1)
                    self.coef[i].zero_()
                    self.coef[i, :d + 1] = src[:d + 1].to(self.coef.device)
        return self

    @torch.no_grad()
    def physical(self, p_values=(0.5, 0.8, 1.5, 2.5, 5.0, 10.0, 20.0)):
        """Physical values of the parameters at a few momenta (for the notes)."""
        dev = self.coef.device
        lnp = torch.log(torch.tensor(p_values, dtype=torch.float32, device=dev))
        th = {n: v.reshape(-1).cpu().numpy() for n, v in self.theta(lnp).items()}
        names, _, lw = self.components(lnp)
        out = {"p": list(p_values)}
        for j, n in enumerate(names):
            out["f_" + n] = lw[:, j].exp().cpu().numpy().round(4).tolist()
        for n, v in th.items():
            if n.startswith("log_") or n.startswith("sp_log") or n.startswith("tr_log") or n.startswith("b_log_"):
                out[n.replace("log", "", 1).replace("__", "_").strip("_")] = [float(f"{x:.4g}") for x in __import__("numpy").exp(v)]
            elif n.startswith("logit_") or n.startswith("b_logit_"):
                out[n.replace("logit_", "")] = [float(f"{1 / (1 + math.exp(-x)):.4g}") for x in v]
            else:
                out[n] = [float(f"{x:.4g}") for x in v]
        if "tr_kap" in th:
            out["tr_lambda_mm"] = [float(f"{100.0 / x:.4g}") if abs(x) > 1e-6 else float("inf") for x in th["tr_kap"]]
        return out
