"""Depth profiles L(z'): log-densities with constrained parametrisations and stable evaluation.

Every family is a small stateless object

    fam = make_long("emg", prefix="sh_")
    fam.names                      raw parameter names (prefix + local name), fixed order
    fam.defaults                   {name: default raw value}
    nat = fam.natural(raw)         raw = {local name: tensor}; constraint links live here
    fam.log_pdf(z, nat)            ln L(z') (broadcast of z against the natural parameters)
    fam.log_pdf_s(s, nat, normalised=True)    positive-support families: the named density in s, for tests

The cell masses are normalised on the truncated grid afterwards, so ``log_pdf`` may drop additive constants (it does:
the normalising constants of some families are differences of large numbers and would only add float32 noise).

Units. Raw location and scale parameters of the new families are in units of U = 100 mm (location = U raw,
scale = floor + U softplus(raw)), raw shape parameters are O(1); an optimiser step is then of comparable relative
size for all of them. ``softplus + floor`` never overflows for any finite raw value (polynomials and networks
extrapolate to x = -4.1 at p = 0.0005 GeV). The two legacy families keep the legacy links (exp of a log parameter,
the exponent clamped to [-12, 35], which is inactive for the fitted coefficients).

Support and the start of the profile. The grid begins at z' = -200 mm, 200 mm in front of the first hit layer.
  * ``plateau`` and ``emg`` live on the whole line.
  * The positive-support families (gamma, gengamma, invgauss, lognormal, experlang, burr) are evaluated at the warped
    depth  s = delta softplus((z' - z0)/delta) + 1e-3 mm  (the legacy ``long_gamma`` soft start, delta = 2 mm):
    s = z' - z0 for z' - z0 >> delta, and s -> 1e-3 + delta exp((z' - z0)/delta) in front of z0. Consequences:
      - behind the start the density is the named one, literally; within ~3 delta of z0 it is not (L is f(s(z')),
        not the push-forward of f, and it is renormalised on the grid);
      - in front of z0 the profile is f evaluated near s = 0, i.e. ~ s^(d-1) -> exp((d-1)(z'-z0)/delta) for a
        power-law start s^(d-1). With delta = 2 mm that is a cut-off, not a model of the backward energy: only
        ``plateau`` (logistic rise of width w1 = 60-140 mm in the legacy fit) and ``emg`` (Gaussian edge) or a free
        delta (``soft="free"``) can put shower energy in front of the start;
      - a start exponent below 1 (f -> infinity at s -> 0) would turn the region in front of z0 into a flat shelf at
        the level f(1e-3 mm); all links below therefore keep the density finite at s -> 0 (mode > 0).

Families (density in s, natural parameters, limits; identifiability notes in each class)
    plateau      sigma((z-zs)/w1) sigma((ell-z)/w2) exp(kappa z / 1 m)                     legacy
    gamma        s^(alpha-1) exp(-(alpha-1) s / t_max)                                    legacy ``long_gamma``
    emg          Exp(rate 1/tau) (*) N(mu, sigma^2)
    gengamma     s^(d-1) exp(-(s/a)^p)   as (mode m, log-width w, p):  ln f = [q - expm1(q)] / (w p)^2, q = p ln(s/m)
    invgauss     s^(-3/2) exp(-(s-mu)^2 / (2 cv^2 mu s))
    lognormal    s^(-1) exp(-ln^2(s/med) / (2 sig^2))                                      = gengamma at p -> 0
    experlang    Exp(1/l1) (*) Erlang(n, 1/l2), n = 1, 2, 3                               Gamma(n+1) at l1 = l2
    burr         (s/lam)^(c-1) (1 + (s/lam)^c)^(-k-1)                                      Weibull at k -> inf
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cellint import log_erfcx, log_ndtr

__all__ = ["U_MM", "S_MIN", "softplus_floor", "inv_softplus", "soft_start", "LongFamily", "Plateau", "GammaLegacy",
           "EMG", "GenGamma", "InvGauss", "LogNormal", "ExpErlang", "BurrXII", "LONG_FAMILIES", "make_long"]

U_MM = 100.0                 # unit of raw location / scale parameters [mm]
S_MIN = 1e-3                 # floor of the warped depth s [mm] (legacy long_gamma)
_SQRT2 = math.sqrt(2.0)
_LOG_2PI = math.log(2.0 * math.pi)


def softplus_floor(raw, floor: float = 0.0, unit: float = 1.0):
    """floor + unit * softplus(raw): positive, never overflows, ~ floor + unit * raw for raw >> 1."""
    return floor + unit * F.softplus(raw)


def inv_softplus(v: float) -> float:
    """raw with softplus(raw) = v (v > 0)."""
    return float(v + math.log(-math.expm1(-v))) if v < 30 else float(v)


def raw_of(value: float, floor: float = 0.0, unit: float = 1.0) -> float:
    """raw with floor + unit * softplus(raw) = value."""
    return inv_softplus((value - floor) / unit)


def _clamp_exp(log_v):
    """exp with the exponent clamped to [-12, 35] (legacy log-parameters extrapolated far outside the fit range)."""
    return torch.exp(log_v.clamp(-12.0, 35.0))


def soft_start(z, z0, delta, s_min: float = S_MIN):
    """Warped depth s = delta softplus((z - z0)/delta) + s_min > 0."""
    return delta * F.softplus((z - z0) / delta) + s_min


# ----------------------------------------------------------------------------------------------------------------------
class LongFamily:
    """Base class. Subclasses set ``name``, ``PARAMS`` = ((local raw name, default raw), ...), ``ORDER_KEY`` and
    implement ``natural`` and ``log_pdf`` (or ``log_pdf_s`` for positive-support families)."""

    name = ""
    PARAMS: tuple = ()
    positive_support = False
    ORDER_KEY = None             # (natural name, local raw name, "real" | "positive"): the cumulative-ordering key
    KEY_STEP_RAW = 0.55          # default raw increment of the key for components k > 0 (softplus(0.55) = 1 -> +100 mm)

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    @property
    def local_names(self):
        return [n for n, _ in self.params()]

    @property
    def names(self):
        return [self.prefix + n for n, _ in self.params()]

    @property
    def defaults(self):
        return {self.prefix + n: float(v) for n, v in self.params()}

    def params(self):
        return self.PARAMS

    def pick(self, th: dict, shape=(-1, 1)):
        """Own raw parameters from a dict keyed by prefixed names, reshaped (default (B, 1))."""
        return {n: th[self.prefix + n].reshape(shape) for n in self.local_names}

    def natural(self, raw: dict, key_prev=None) -> dict:                      # pragma: no cover - interface
        raise NotImplementedError

    def log_pdf(self, z, nat: dict):                                          # pragma: no cover - interface
        raise NotImplementedError

    def _key(self, own, key_prev, raw_key, unit=U_MM):
        """Ordering key: its own link, or ``key_prev + unit softplus(raw)`` when a previous key is given."""
        return own if key_prev is None else key_prev + unit * F.softplus(raw_key)


class _SoftStart(LongFamily):
    """Positive-support family on the warped depth s(z') (see the module docstring).

    soft   softness delta in mm (default 2.0, the legacy value), or "free": delta = 0.5 mm + 10 mm softplus(rdelta)
           becomes a raw parameter (a backward tail of length delta / (start exponent))."""

    positive_support = True
    DELTA_UNIT = 10.0

    def __init__(self, prefix: str = "", soft=2.0, z0_default: float = -0.2):
        super().__init__(prefix)
        self.soft = soft
        self.z0_default = float(z0_default)

    def params(self):
        p = (("z0", self.z0_default),) + tuple(self.PARAMS)
        if self.soft == "free":
            p = p + (("rdelta", raw_of(2.0, 0.5, self.DELTA_UNIT)),)
        return p

    def _start(self, raw):
        z0 = U_MM * raw["z0"]
        delta = softplus_floor(raw["rdelta"], 0.5, self.DELTA_UNIT) if self.soft == "free" else float(self.soft)
        return z0, delta

    def log_pdf(self, z, nat):
        return self.log_pdf_s(soft_start(z, nat["z0"], nat["delta"]), nat, normalised=False)

    def log_pdf_s(self, s, nat, normalised=True):                             # pragma: no cover - interface
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
class Plateau(LongFamily):
    """Legacy plateau: ln L = ln sigma((z - zs)/w1) + ln sigma((ell - z)/w2) + kappa z / 1000.

    Raw (legacy names and links): zs [mm], log_w1, log_ell, log_w2 (exp), kappa [1/m].
    Support: the whole line. Tails: backward exp(+z (1/w1 + kappa/1 m)), forward exp(-z (1/w2 - kappa/1 m)); in
    between exp(kappa z / 1 m).
    Not identifiable / weakly identified
      * ell - zs << w1, w2: the two logistics overlap, the product is a sech^2-like bump whose position and width
        are (zs + ell)/2 and w; the separation ell - zs only rescales it (the pion fit below ~1 GeV: ell = 26-36 mm,
        w1 = 117-140 mm, w2 = 103-175 mm);
      * ell beyond the grid (ell = 1529 mm at 20 GeV with w2 = 1257 mm): only the combination entering the slope
        d ln L / dz ~ kappa/1 m - sigma((z - ell)/w2)/w2 inside the grid is constrained, a ridge in (ell, w2, kappa);
      * kappa and 1/w2 are exchangeable for z >> ell.
    """

    name = "plateau"
    PARAMS = (("zs", -10.0), ("log_w1", math.log(15.0)), ("log_ell", math.log(3000.0)), ("log_w2", math.log(30.0)),
              ("kappa", 0.0))
    ORDER_KEY = ("ell", "log_ell", "positive")
    KEY_STEP_RAW = math.log(100.0)
    RAW_SCALE = {"zs": 30.0}     # raw units for random search (zs is in mm, the others are logs or 1/m)

    def natural(self, raw, key_prev=None):
        ell = _clamp_exp(raw["log_ell"])
        return {"zs": raw["zs"], "w1": _clamp_exp(raw["log_w1"]), "ell": ell if key_prev is None else key_prev + ell,
                "w2": _clamp_exp(raw["log_w2"]), "kappa": raw["kappa"]}

    def log_pdf(self, z, nat):
        return (F.logsigmoid((z - nat["zs"]) / nat["w1"]) + F.logsigmoid((nat["ell"] - z) / nat["w2"])
                + nat["kappa"] * z / 1000)


class GammaLegacy(LongFamily):
    """Legacy ``families.long_gamma``: ln L = am1 ln s - (am1 / tmax) s,  s = 2 softplus((z - z0)/2) + 1e-3.

    Raw (legacy names and links): z0 [mm], log_tmax, log_am1 (exp). alpha - 1 = am1 > 0, mode of s = tmax, rate
    beta = am1 / tmax. Density in s: Gamma(alpha, beta), normaliser beta^alpha / Gamma(alpha) (not included).
    Tails: s^(alpha-1) at the start, exp(-beta s) behind.
    Not identifiable: am1 -> infinity (Gaussian limit N(z0 + tmax, tmax^2/am1): only z0 + tmax and tmax/sqrt(am1)
    remain, three parameters for two); am1 -> 0 at fixed am1/tmax (pure exponential, tmax and am1 only through their
    ratio).
    """

    name = "gamma"
    PARAMS = (("z0", -20.0), ("log_tmax", math.log(100.0)), ("log_am1", math.log(3.0)))
    positive_support = True
    ORDER_KEY = ("tmax", "log_tmax", "positive")
    KEY_STEP_RAW = math.log(100.0)
    RAW_SCALE = {"z0": 30.0}

    def natural(self, raw, key_prev=None):
        tmax = _clamp_exp(raw["log_tmax"])
        return {"z0": raw["z0"], "tmax": tmax if key_prev is None else key_prev + tmax,
                "am1": _clamp_exp(raw["log_am1"]), "delta": 2.0}

    def log_pdf(self, z, nat):
        return self.log_pdf_s(soft_start(z, nat["z0"], 2.0), nat, normalised=False)

    def log_pdf_s(self, s, nat, normalised=True):
        am1, beta = nat["am1"], nat["am1"] / nat["tmax"]
        out = am1 * torch.log(s) - beta * s
        if normalised:
            out = out + (am1 + 1) * torch.log(beta) - torch.lgamma(am1 + 1)
        return out


class EMG(LongFamily):
    """Exponentially modified Gaussian: Z = N(mu, sigma^2) + Exp(mean tau), i.e. an exponential attenuation with
    length tau that starts at mu, the start smeared by sigma.

        f(z) = (1/tau) exp(-K t + K^2/2) Phi(t - K),      t = (z - mu)/sigma,  K = sigma/tau (= lambda sigma)

    Evaluation, with u = t - K (no cancellation and no overflow for any K, t):
        u >= 0:  ln f = -ln tau - K u - K^2/2 + ln Phi(u)                    (every term <= 0)
        u <  0:  ln f = -ln(2 tau) - t^2/2 + ln erfcx(-u / sqrt 2)           (erfcx in (0, 1])
    The single ``log_ndtr`` form ln(1/tau) + (mu - z)/tau + K^2/2 + ln Phi(u) is the same function but adds
    K^2/2 to a number near -K^2/2 for u << 0: absolute float32 error ~6e-8 K^2 (3e-5 at K = 30, 0.03 at K = 1000);
    a pure erfcx form overflows for u > 13.3 (float32) / 37.7 (float64). Hence the two branches; no bound on K is
    needed numerically. ``k_max`` (default None) optionally caps K smoothly for identifiability only.

    Raw: mu (location, U), rsig (sigma = sig_floor + U softplus), rtau (tau = 1 mm + U softplus). With
    ``tau="external"`` the family has no rtau and takes ``tau`` from the caller (tied to the track attenuation).
    Support: the whole line. Tails: Gaussian exp(-(z - mu)^2 / 2 sigma^2) in front, exponential exp(-z/tau) behind.
    Mean mu + tau, variance sigma^2 + tau^2, skewness 2 (tau/sigma)^3 / (1 + tau^2/sigma^2)^(3/2).
    Not identifiable
      * K = sigma/tau >> 1: a Gaussian N(mu + tau, sigma^2); tau enters only through the mean (skewness 2/K^3,
        7e-5 at K = 30). Two effective parameters;
      * sigma << DZ (12.9 mm): the edge is narrower than a cell; cell masses depend on sigma only at second order
        (sigma is weakly identified below ~3 mm, and with ``integration="centre"`` not at all);
      * tau >> grid length: flat behind the edge, tau and the weight of the component trade off.
    Nested in / overlapping with the repo's forms: the plateau with ell -> infinity is a logistic edge times
    exp(kappa z): the same shape with a logistic instead of a Gaussian edge. The EMG has no second (cut-off) edge.
    """

    name = "emg"
    ORDER_KEY = ("mu", "mu", "real")

    def __init__(self, prefix: str = "", tau="free", sig_floor: float = 0.5, k_max=None,
                 init=(30.0, 60.0, 300.0)):
        super().__init__(prefix)
        self.tau, self.sig_floor, self.k_max, self.init = tau, float(sig_floor), k_max, init

    def params(self):
        mu, sig, tau = self.init
        p = (("mu", mu / U_MM), ("rsig", raw_of(sig, self.sig_floor, U_MM)))
        return p + ((("rtau", raw_of(tau, 1.0, U_MM)),) if self.tau == "free" else ())

    def natural(self, raw, key_prev=None, tau=None):
        mu = self._key(U_MM * raw["mu"], key_prev, raw["mu"])
        if self.tau == "free":
            tau = softplus_floor(raw["rtau"], 1.0, U_MM)
        elif tau is None:
            raise ValueError("EMG(tau='external') needs tau")
        return {"mu": mu, "sigma": softplus_floor(raw["rsig"], self.sig_floor, U_MM), "tau": tau}

    def log_pdf(self, z, nat):
        sigma, tau = nat["sigma"], nat["tau"]
        t = (z - nat["mu"]) / sigma
        K = sigma / tau
        if self.k_max is not None:
            K = self.k_max * torch.tanh(K / self.k_max)
        u = t - K
        up, un = u.clamp_min(0.0), u.clamp_max(0.0)
        a = -torch.log(tau) - K * up - 0.5 * K * K + log_ndtr(up)
        b = -torch.log(2.0 * tau) - 0.5 * t * t + log_erfcx(-un / _SQRT2)
        return torch.where(u >= 0, a, b)

    @staticmethod
    def log_pdf_naive(z, nat):
        """The single-branch ``log_ndtr`` form (PLAN.md section 5, A4), for comparison in tests."""
        sigma, tau = nat["sigma"], nat["tau"]
        K = sigma / tau
        return -torch.log(tau) + (nat["mu"] - z) / tau + 0.5 * K * K + torch.special.log_ndtr((z - nat["mu"]) / sigma - K)


class GenGamma(_SoftStart):
    """Generalised gamma (Stacy): f(s) = p / (a^d Gamma(d/p)) s^(d-1) exp(-(s/a)^p), s > 0.

    Parametrised by the mode m, the logarithmic width w at the mode and the stretch exponent p:

        ln f = [q - expm1(q)] / (w p)^2 + const,   q = p ln(s/m)                (-q^2/2 - q^3/6 - ... for |q| < 0.5)
        d = 1 + 1/(w^2 p),   a = m (w p)^(2/p),   d^2 ln f / d(ln s)^2 at the mode = -1/w^2

    so that (m, w) fix position and width and p only bends the two sides: p = 1 is the gamma density (am1 = 1/w^2,
    tmax = m), p -> 0 the log-normal with mode m and sigma = w (exactly: the limit of the bracket is -q^2/2), p = 2
    a Gaussian-type fall-off, large p a hard cut at a. The (a, d, p) form is badly conditioned (a ~ m (wp)^(2/p)
    varies over tens of decades at fixed shape); (m, w, p) is not. ``torch.special.gammainc`` is never used: the
    normaliser is not needed (cells are normalised on the grid) and has no shape gradient in torch.

    Raw: z0 (U), rm (m = 1 mm + U softplus), rw (w = 0.02 + softplus), rp (p = p_lo + (p_hi - p_lo) sigmoid;
    default range (0.1, 6), so (s/m)^p <= 1651^6 stays far below the float32 limit).
    Tails: s^(d-1) at the start (d > 1 always: the mode exists), exp(-(s/a)^p) behind (stretched for p < 1).
    Not identifiable
      * p for small p: at p -> 0 the density depends on p only through the relative correction p ln(s/m)/3; the
        direction rp -> -infinity is flat (log-normal limit);
      * w -> 0: a Gaussian at z0 + m of width m w; z0 and m then enter only through z0 + m (as for every
        shifted positive family);
      * the upper side p is seen only where s > m: if the mode lies beyond the grid, p is free.
    """

    name = "gengamma"
    ORDER_KEY = ("m", "rm", "positive")

    def __init__(self, prefix: str = "", soft=2.0, p_range=(0.1, 6.0), init=(150.0, 0.8, 1.0), z0_default=-0.2):
        super().__init__(prefix, soft, z0_default)
        self.p_lo, self.p_hi = float(p_range[0]), float(p_range[1])
        m, w, p = init
        f = (p - self.p_lo) / (self.p_hi - self.p_lo)
        self.PARAMS = (("rm", raw_of(m, 1.0, U_MM)), ("rw", raw_of(w, 0.02)), ("rp", math.log(f / (1 - f))))

    def natural(self, raw, key_prev=None):
        z0, delta = self._start(raw)
        m = self._key(softplus_floor(raw["rm"], 1.0, U_MM), key_prev, raw["rm"])
        w = softplus_floor(raw["rw"], 0.02)
        p = self.p_lo + (self.p_hi - self.p_lo) * torch.sigmoid(raw["rp"])
        return {"z0": z0, "delta": delta, "m": m, "w": w, "p": p}

    @staticmethod
    def stacy(nat):
        """(a, d, p) of the textbook form s^(d-1) exp(-(s/a)^p)."""
        m, w, p = nat["m"], nat["w"], nat["p"]
        return m * (w * p) ** (2.0 / p), 1.0 + 1.0 / (w * w * p), p

    def log_pdf_s(self, s, nat, normalised=True):
        m, w, p = nat["m"], nat["w"], nat["p"]
        q = p * torch.log(s / m)
        qs = q.clamp(-0.5, 0.5)
        ser = torch.zeros_like(qs)
        for k in range(12, 1, -1):                                     # -sum_{k=2}^{12} q^k / k!  (Horner)
            ser = (ser + 1.0 / math.factorial(k)) * qs
        ser = -ser * qs
        g = torch.where(q.abs() < 0.5, ser, q - torch.expm1(q))
        out = g / (w * p) ** 2
        if normalised:
            a, d, _ = self.stacy(nat)
            c = d - 1.0
            out = out + c * torch.log(m) - c / p + torch.log(p) - d * torch.log(a) - torch.lgamma(d / p)
        return out


class InvGauss(_SoftStart):
    """Inverse Gaussian (Wald; first-passage depth of a drifting diffusion):

        f(s) = sqrt(lam / (2 pi s^3)) exp(-lam (s - mu)^2 / (2 mu^2 s)),   lam = mu / cv^2

    Raw: z0 (U), rmu (mean mu = 1 mm + U softplus), rcv (coefficient of variation cv = 0.02 + softplus).
    Mean mu, variance mu^2 cv^2, mode mu [sqrt(1 + 9 cv^4/4) - 3 cv^2/2].
    Tails: exp(-lam / 2s) at the start (faster than any power: the profile is empty in front of z0 for every
    delta), s^(-3/2) exp(-s / (2 mu cv^2)) behind.
    Not identifiable: cv -> 0 (Gaussian at z0 + mu, width mu cv; z0 and mu only through their sum);
    cv >> 1 (Levy limit s^(-3/2) exp(-lam/2s): only lam = mu/cv^2 remains); mu >> grid (the exponential cut-off is
    outside the grid: again only lam).
    """

    name = "invgauss"
    ORDER_KEY = ("mu", "rmu", "positive")

    def __init__(self, prefix: str = "", soft=2.0, init=(250.0, 0.9), z0_default=-0.2):
        super().__init__(prefix, soft, z0_default)
        self.PARAMS = (("rmu", raw_of(init[0], 1.0, U_MM)), ("rcv", raw_of(init[1], 0.02)))

    def natural(self, raw, key_prev=None):
        z0, delta = self._start(raw)
        mu = self._key(softplus_floor(raw["rmu"], 1.0, U_MM), key_prev, raw["rmu"])
        return {"z0": z0, "delta": delta, "mu": mu, "cv": softplus_floor(raw["rcv"], 0.02)}

    def log_pdf_s(self, s, nat, normalised=True):
        mu, cv = nat["mu"], nat["cv"]
        out = -1.5 * torch.log(s) - (s - mu) ** 2 / (2.0 * cv * cv * mu * s)
        if normalised:
            out = out + 0.5 * (torch.log(mu) - 2.0 * torch.log(cv) - _LOG_2PI)
        return out


class LogNormal(_SoftStart):
    """Log-normal: f(s) = exp(-ln^2(s/med) / (2 sig^2)) / (s sig sqrt(2 pi)).

    Raw: z0 (U), rmed (median = 1 mm + U softplus), rsig (sig = 0.02 + softplus). Mode med exp(-sig^2).
    Tails: faster than any power at the start, slower than any exponential behind (log-quadratic).
    Nested: exactly the p -> 0 boundary of ``GenGamma`` (mode m = med exp(-sig^2), w = sig).
    Not identifiable: sig -> 0 (Gaussian at z0 + med, width med sig); the three-parameter (shifted) log-normal is the
    textbook case of a likelihood ridge in (z0, med, sig) when only the bulk is seen.
    """

    name = "lognormal"
    ORDER_KEY = ("med", "rmed", "positive")

    def __init__(self, prefix: str = "", soft=2.0, init=(200.0, 0.8), z0_default=-0.2):
        super().__init__(prefix, soft, z0_default)
        self.PARAMS = (("rmed", raw_of(init[0], 1.0, U_MM)), ("rsig", raw_of(init[1], 0.02)))

    def natural(self, raw, key_prev=None):
        z0, delta = self._start(raw)
        med = self._key(softplus_floor(raw["rmed"], 1.0, U_MM), key_prev, raw["rmed"])
        return {"z0": z0, "delta": delta, "med": med, "sig": softplus_floor(raw["rsig"], 0.02)}

    def log_pdf_s(self, s, nat, normalised=True):
        sig = nat["sig"]
        out = -torch.log(s) - torch.log(s / nat["med"]) ** 2 / (2.0 * sig * sig)
        if normalised:
            out = out - torch.log(sig) - 0.5 * _LOG_2PI
        return out


def log_hn(x, n: int):
    """ln h_n(x),  h_n(x) = P(n, x) / x^n = [1 - exp(-x) e_{n-1}(x)] / x^n,  e_m = truncated exponential series.

    h_n is entire and positive (h_n(0) = 1/n!); x = (1/l2 - 1/l1) s may have either sign. Three regions:
        |x| <= 2:  series (1/(n-1)!) sum_k (-x)^k / (k! (n + k))                 (the removable singularity)
        x  >  2:   ln(1 - exp(-x) e_{n-1}(x)) - n ln x
        x  < -2:   -x + ln([exp(x) - e_{n-1}(x)] / x^n)                          (no exp(|x|) overflow)
    """
    xs, xp, xn = x.clamp(-2.0, 2.0), x.clamp_min(2.0), x.clamp_max(-2.0)
    ser = torch.zeros_like(xs)
    for k in range(26, -1, -1):                                        # Horner in (-x): sum (-x)^k / (k! (n+k))
        ser = ser * (-xs) + 1.0 / (math.factorial(k) * (n + k))
    ser = torch.log(ser / math.factorial(n - 1))

    def e(v):                                                          # e_{n-1}(v) = sum_{k<n} v^k / k!
        out = torch.ones_like(v)
        for k in range(n - 1, 0, -1):
            out = 1.0 + out * v / k
        return out

    pos = torch.log1p(-torch.exp(-xp) * e(xp)) - n * torch.log(xp)
    neg = -xn + torch.log((torch.exp(xn) - e(xn)) / xn ** n)
    return torch.where(x > 2.0, pos, torch.where(x < -2.0, neg, ser))


class ExpErlang(_SoftStart):
    """Exponential start (*) Erlang cascade: s = s0 + g, s0 ~ Exp(mean l1), g ~ Gamma(n, scale l2), n in {1, 2, 3}.

        f(s) = (1/l1) (s/l2)^n exp(-s/l1) h_n((1/l2 - 1/l1) s),      h_n(x) = P(n, x) / x^n

    with P the regularised lower incomplete gamma function for integer n (elementary). At l1 = l2 the two rates
    coincide (x = 0, h_n = 1/n!) and f is Gamma(n + 1, l1); ``log_hn`` handles the removable singularity by a
    series, so nothing special happens there.

    Raw: z0 (U), rl1 (l1 = 1 mm + U softplus), rl2 (l2 = 0.5 mm + U softplus). ``ordered=True`` (default for n = 1)
    uses l1 = l2 + U softplus(rl1) instead, i.e. l1 > l2.
    Tails: s^n at the start; exp(-s / max(l1, l2)) behind.
    Not identifiable: for n = 1 the density is symmetric under l1 <-> l2 (a two-fold label symmetry; ``ordered``
    removes it); near l1 = l2 the two lengths enter only through l1 + l2 at first order (the density is
    Gamma(n + 1)); l1 >> grid: a flat shelf, l1 and the component weight trade off.
    Nested: Gamma with integer shape n + 1 at l1 = l2; the legacy ``bock`` variant is Exp (*) two gammas with real
    shapes, by numerical convolution.
    """

    name = "experlang"
    ORDER_KEY = ("l2", "rl2", "positive")

    def __init__(self, prefix: str = "", n: int = 2, soft=2.0, ordered=None, init=(250.0, 60.0), z0_default=-0.2):
        super().__init__(prefix, soft, z0_default)
        if n not in (1, 2, 3):
            raise ValueError("ExpErlang: n must be 1, 2 or 3")
        self.n = int(n)
        self.ordered = (n == 1) if ordered is None else bool(ordered)
        l1, l2 = init
        r1 = raw_of(l1 - l2, 0.0, U_MM) if self.ordered else raw_of(l1, 1.0, U_MM)
        self.PARAMS = (("rl1", r1), ("rl2", raw_of(l2, 0.5, U_MM)))

    def natural(self, raw, key_prev=None):
        z0, delta = self._start(raw)
        l2 = self._key(softplus_floor(raw["rl2"], 0.5, U_MM), key_prev, raw["rl2"])
        l1 = l2 + U_MM * F.softplus(raw["rl1"]) if self.ordered else softplus_floor(raw["rl1"], 1.0, U_MM)
        return {"z0": z0, "delta": delta, "l1": l1, "l2": l2}

    def log_pdf_s(self, s, nat, normalised=True):
        l1, l2, n = nat["l1"], nat["l2"], self.n
        return -torch.log(l1) + n * torch.log(s / l2) - s / l1 + log_hn((1.0 / l2 - 1.0 / l1) * s, n)


class BurrXII(_SoftStart):
    """Burr XII (Singh-Maddala): f(s) = (c k / lam) (s/lam)^(c-1) (1 + (s/lam)^c)^(-k-1), CDF 1 - (1 + (s/lam)^c)^(-k).

        ln f = ln(c k / lam) + (c - 1) y - (k + 1) softplus(c y),   y = ln(s/lam)            (no power is formed)

    Raw: z0 (U), rscale (lam = 1 mm + U softplus), rc (c = 1 + softplus: mode > 0), rk (k = 0.05 + softplus).
    Tails: s^(c-1) at the start; a power law s^(-ck-1) behind (heavier than any exponential: the mean exists only
    for ck > 1).
    Not identifiable: k -> infinity at fixed lam k^(-1/c) (Weibull limit exp(-k (s/lam)^c): a ridge in (lam, k));
    the tail index ck only from s >> lam, i.e. not at all if lam is comparable to the grid length.
    Nested: log-logistic at k = 1, Lomax-type at c = 1 (excluded by the link), Weibull as a limit.
    """

    name = "burr"
    ORDER_KEY = ("lam", "rscale", "positive")

    def __init__(self, prefix: str = "", soft=2.0, init=(300.0, 1.8, 3.0), z0_default=-0.2):
        super().__init__(prefix, soft, z0_default)
        lam, c, k = init
        self.PARAMS = (("rscale", raw_of(lam, 1.0, U_MM)), ("rc", raw_of(c, 1.0)), ("rk", raw_of(k, 0.05)))

    def natural(self, raw, key_prev=None):
        z0, delta = self._start(raw)
        lam = self._key(softplus_floor(raw["rscale"], 1.0, U_MM), key_prev, raw["rscale"])
        return {"z0": z0, "delta": delta, "lam": lam, "c": softplus_floor(raw["rc"], 1.0),
                "k": softplus_floor(raw["rk"], 0.05)}

    def log_pdf_s(self, s, nat, normalised=True):
        lam, c, k = nat["lam"], nat["c"], nat["k"]
        y = torch.log(s / lam)
        out = (c - 1.0) * y - (k + 1.0) * F.softplus(c * y)
        if normalised:
            out = out + torch.log(c * k / lam)
        return out


LONG_FAMILIES = {
    "plateau": Plateau, "gamma": GammaLegacy, "emg": EMG, "gengamma": GenGamma, "invgauss": InvGauss,
    "lognormal": LogNormal, "burr": BurrXII,
    "experlang1": lambda prefix="", **kw: ExpErlang(prefix, n=1, **kw),
    "experlang2": lambda prefix="", **kw: ExpErlang(prefix, n=2, **kw),
    "experlang3": lambda prefix="", **kw: ExpErlang(prefix, n=3, **kw),
}


def make_long(name, prefix: str = "", **kw) -> LongFamily:
    """Depth-profile family by name (see ``LONG_FAMILIES``) or pass an instance through."""
    if isinstance(name, LongFamily):
        return name
    if name not in LONG_FAMILIES:
        raise KeyError(f"unknown longitudinal family '{name}'; available: {sorted(LONG_FAMILIES)}")
    return LONG_FAMILIES[name](prefix, **kw)
