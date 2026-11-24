"""fam - bodies of revolution r(z') = A g(z'; shape) for the report's envelope criterion (lane E1_bodies).

All classes derive from envlab.Body (score s = ln r - ln g(z'), level c = ln A, `log_i2` = ln of the analytic
integral of g^2 over the whole body, so ln V[mm^3] = ln(pi) + 2 ln A + log_i2).  No side effects at import.

Families (parameters per momentum bin = len(names) + 1, the +1 is the size A):
  Envelope()                       report 02: r = A u^a exp(-b u)                                   (envlab)
  GenDrop(fixed=(0.5, 1.5))        report 03: the drop                                              3
  GenDrop()                        generalised drop, free alpha, beta                               5
  GenDrop(fixed=(0.5, 0.5))        ellipsoid; (0, 0) cylinder; (0, 1) cone; (0, 0.5) paraboloid     3
  GenDrop(fixed=(None, 0))         power horn A t^alpha with a flat back; (0, None): flat front, power tip   4
  GenDrop(..., trunc=True)         the same, cut by a plane at z' = z_cut (finite, no thin tip)     +1
  TruncEnvelope()                  the report's gamma envelope cut at z' = z_cut                    5
  SuperEllipsoid(...)              r = A (1 - |tau|^n)^(1/m); two_sided: different half lengths (egg)   3..8
  Capsule(front, back)             cylinder with rounded / conical / flat ends                      3..5
  Frustum()                        truncated cone (r linear between two end discs)                  4
  TwoPower()                       A (t/tm)^p rising, A ((1-t)/(1-tm))^q falling                    6
  Knots(zk)                        non-parametric: ln r piecewise linear between fixed knots        len(zk)
  BodyOfLnP                        a family whose shape parameters and ln A are polynomials in ln p (closed form in p)
"""
import math

import numpy as np
import torch
from scipy import special

try:
    import envlab
except ImportError:                                    # the harness directory is not on the path yet
    import sys
    sys.path.insert(0, "/home/user/tamas/shape-form-proton")
    import envlab

Body, Envelope, BetaDrop = envlab.Body, envlab.Envelope, envlab.BetaDrop
_NEG = -float("inf")


def _mask(ok, val):
    return torch.where(ok, val, torch.full_like(val, _NEG))


def _lgmax(a, b):
    """ln of the maximum of t^a (1-t)^b on [0, 1]."""
    if a + b <= 0:
        return 0.0
    return float(special.xlogy(a, a / (a + b)) + special.xlogy(b, b / (a + b)))


class GenDrop(Body):
    """Generalised drop  r = A g(t),  g = t^alpha (1-t)^beta / g_max,  t = (z' - z0)/L,  0 < t < 1.
    `fixed=(alpha, beta)`: None = free exponent.  `trunc=True`: the body is cut by the plane z' = z_cut (search variable).
    Search variables: ln L, z0, [ln alpha], [ln beta], [z_cut]."""

    def __init__(self, fixed=(None, None), trunc=False, z0_range=(-260.0, 80.0), L_range=(40.0, 9000.0)):
        self.fixed, self.trunc = tuple(fixed), trunc
        fa, fb = self.fixed
        self.names = ["L", "z0"] + (["alpha"] if fa is None else []) + (["beta"] if fb is None else []) + (["z_cut"] if trunc else [])
        self.bounds = [(math.log(L_range[0]), math.log(L_range[1])), z0_range]
        self.steps = [0.2, 20.0]
        ex = []
        if fa is None:
            self.bounds.append((math.log(0.004), math.log(4)))
            self.steps.append(0.3)
            ex.append([0.5, 0.2, 0.05])
        if fb is None:
            self.bounds.append((math.log(0.004), math.log(8)))
            self.steps.append(0.3)
            ex.append([1.5, 0.5, 0.15])
        if trunc:
            self.bounds.append((150.0, 1800.0))
            self.steps.append(60.0)
        grid = [(lL, z0) for lL in np.linspace(math.log(500), math.log(6000), 9) for z0 in (-220, -140, -80, -30)]
        if len(ex) == 2:
            combos = [(0.5, 1.5), (0.5, 0.5), (0.2, 0.5), (0.05, 0.15), (0.2, 1.5), (0.05, 0.5)]
            grid = [g + (math.log(a), math.log(b)) for g in grid for a, b in combos]
        elif len(ex) == 1:
            grid = [g + (math.log(v),) for g in grid for v in ex[0]]
        if trunc:
            grid = [g + (zc,) for g in grid for zc in (1500.0, 1150.0)]
        self.x0_grid = grid

    def _unpack(self, x):
        L, z0 = math.exp(x[0]), x[1]
        i = 2
        fa, fb = self.fixed
        if fa is None:
            a, i = math.exp(x[i]), i + 1
        else:
            a = fa
        if fb is None:
            b, i = math.exp(x[i]), i + 1
        else:
            b = fb
        zc = x[i] if self.trunc else float("inf")
        return L, z0, a, b, zc

    def logg(self, z, x):
        L, z0, a, b, zc = self._unpack(x)
        t = (z - z0) / L
        ok = (t > 0) & (t < 1) & (z <= zc)
        tc = t.clamp(1e-12, 1 - 1e-7)
        return _mask(ok, a * torch.log(tc) + b * torch.log1p(-tc) - _lgmax(a, b))

    def log_i2(self, x):
        L, z0, a, b, zc = self._unpack(x)
        v = math.log(L) + math.lgamma(2 * a + 1) + math.lgamma(2 * b + 1) - math.lgamma(2 * a + 2 * b + 2) - 2 * _lgmax(a, b)
        tcut = (zc - z0) / L
        if tcut < 1:
            frac = float(special.betainc(2 * a + 1, 2 * b + 1, max(tcut, 0.0)))
            v = v + math.log(frac) if frac > 0 else -float("inf")
        return v

    def valid(self, x):
        if not all(lo < v < hi for v, (lo, hi) in zip(x, self.bounds)):
            return False
        L, z0, a, b, zc = self._unpack(x)
        return zc > z0 + 20.0

    def decode(self, x):
        L, z0, a, b, zc = self._unpack(x)
        out = {"L": L, "z0": z0, "alpha": a, "beta": b, "z_thick": z0 + L * (a / (a + b) if a + b > 0 else 0.0)}
        if self.trunc:
            out["z_cut"] = zc
        return out


class TruncEnvelope(Body):
    """The report's gamma envelope r = A u^a exp(-b u), u = z' - z0, cut by the plane z' = z_cut.
    Search variables: ln a, ln z_env (= ln(a/b)), z0, z_cut."""
    names = ["a", "z_env", "z0", "z_cut"]
    bounds = [(math.log(0.02), math.log(10)), (math.log(3), math.log(4000)), (-260, 80), (150.0, 1800.0)]
    steps = [0.3, 0.3, 20.0, 60.0]
    x0_grid = [(la, lz, z0, zc) for la in np.linspace(math.log(0.05), math.log(3), 7)
               for lz in np.linspace(math.log(30), math.log(2000), 8) for z0 in (-200, -100, -30) for zc in (1500.0, 1150.0)]

    def logg(self, z, x):
        a, zenv, z0, zc = math.exp(x[0]), math.exp(x[1]), x[2], x[3]
        u = z - z0
        return _mask((u > 0) & (z <= zc), a * torch.log(u.clamp_min(1e-12)) - (a / zenv) * u)

    def log_i2(self, x):
        a, zenv, z0, zc = math.exp(x[0]), math.exp(x[1]), x[2], x[3]
        b = a / zenv
        frac = float(special.gammainc(2 * a + 1, 2 * b * max(zc - z0, 0.0)))
        if frac <= 0:
            return -float("inf")
        return math.lgamma(2 * a + 1) - (2 * a + 1) * math.log(2 * b) + math.log(frac)

    def valid(self, x):
        return all(lo < v < hi for v, (lo, hi) in zip(x, self.bounds)) and x[3] > x[2] + 20.0

    def decode(self, x):
        a, zenv = math.exp(x[0]), math.exp(x[1])
        return {"a": a, "b": a / zenv, "z_env": zenv, "z0": x[2], "z_cut": x[3]}


class SuperEllipsoid(Body):
    """Super-ellipsoid  r = A (1 - |tau|^n)^(1/m),  tau = (z' - zc)/a.
    two_sided: half length a1 in front of zc and a2 behind it ("egg"); sep: separate exponents for the two halves.
    n, m: None = free, number = fixed (n = m = 2: ellipsoid; n -> inf: cylinder; m = 1, n = 1: double cone).
    Search variables: zc, ln a1, [ln a2], [ln n (x2 if sep)], [ln m (x2 if sep)]."""

    def __init__(self, two_sided=False, n=None, m=None, sep=False):
        self.two, self.n, self.m, self.sep = two_sided, n, m, sep and two_sided
        k = 2 if self.sep else 1
        self.names = ["zc", "a1"] + (["a2"] if two_sided else []) + ([f"n{i + 1}" for i in range(k)] if n is None else []) \
            + ([f"m{i + 1}" for i in range(k)] if m is None else [])
        self.bounds = [(-150.0, 1400.0), (math.log(20), math.log(3000))] + ([(math.log(20), math.log(4000))] if two_sided else [])
        self.steps = [40.0, 0.2] + ([0.2] if two_sided else [])
        nfree = (k if n is None else 0) + (k if m is None else 0)
        self.bounds += [(math.log(0.3), math.log(60))] * nfree
        self.steps += [0.3] * nfree
        if two_sided:
            base = [(zc, math.log(a1), math.log(a2)) for zc in (150, 350, 550) for a1 in (350, 650) for a2 in (900, 1300, 2000)]
        else:
            base = [(zc, math.log(a)) for zc in (400, 550, 700, 850) for a in (700, 900, 1100, 1400)]
        exps = [()]
        if n is None:
            exps = [e + (math.log(v),) * k for e in exps for v in (2.0, 4.0)]
        if m is None:
            exps = [e + (math.log(v),) * k for e in exps for v in (2.0, 1.0, 4.0)]
        self.x0_grid = [b + e for b in base for e in exps]

    def _unpack(self, x):
        zc, a1 = x[0], math.exp(x[1])
        i = 2
        a2 = a1
        if self.two:
            a2, i = math.exp(x[i]), i + 1
        k = 2 if self.sep else 1
        if self.n is None:
            n = [math.exp(v) for v in x[i:i + k]]
            i += k
        else:
            n = [self.n] * k
        if self.m is None:
            m = [math.exp(v) for v in x[i:i + k]]
            i += k
        else:
            m = [self.m] * k
        return zc, a1, a2, n[0], n[-1], m[0], m[-1]

    def logg(self, z, x):
        zc, a1, a2, n1, n2, m1, m2 = self._unpack(x)
        d = z - zc
        front = d < 0
        tau = torch.where(front, -d / a1, d / a2)
        ok = tau < 1
        tc = tau.clamp(1e-12, 1 - 1e-7)
        val = torch.where(front, torch.log1p(-tc ** n1) / m1, torch.log1p(-tc ** n2) / m2)
        return _mask(ok, val)

    @staticmethod
    def _half(a, n, m):
        return a / n * math.exp(math.lgamma(1 / n) + math.lgamma(2 / m + 1) - math.lgamma(1 / n + 2 / m + 1))

    def log_i2(self, x):
        zc, a1, a2, n1, n2, m1, m2 = self._unpack(x)
        return math.log(self._half(a1, n1, m1) + self._half(a2, n2, m2))

    def decode(self, x):
        zc, a1, a2, n1, n2, m1, m2 = self._unpack(x)
        return {"zc": zc, "a1": a1, "a2": a2, "n1": n1, "n2": n2, "m1": m1, "m2": m2, "z_front": zc - a1, "z_back": zc + a2}


class Capsule(Body):
    """Cylinder r = A on [z1, z1 + Lc] with end caps.  Cap types: "round" (half-ellipsoid of length l), "cone", "flat" (none).
    Search variables: z_a (front end of the body), [ln l1], ln Lc, [ln l2]."""
    _VOL = {"round": 2.0 / 3.0, "cone": 1.0 / 3.0}

    def __init__(self, front="round", back="round"):
        self.front, self.back = front, back
        self.names = ["z_a"] + (["l1"] if front != "flat" else []) + ["Lc"] + (["l2"] if back != "flat" else [])
        self.bounds = [(-260.0, 400.0)] + ([(math.log(5), math.log(1500))] if front != "flat" else []) + [(math.log(5), math.log(1900))] \
            + ([(math.log(5), math.log(4000))] if back != "flat" else [])
        self.steps = [20.0] + [0.3] * (len(self.names) - 1)
        g = [(za,) for za in (-220.0, -120.0, -40.0)]
        if front != "flat":
            g = [v + (math.log(l1),) for v in g for l1 in (150, 400, 800)]
        g = [v + (math.log(lc),) for v in g for lc in (200, 600, 1200)]
        if back != "flat":
            g = [v + (math.log(l2),) for v in g for l2 in (300, 800, 1600)]
        self.x0_grid = g

    def _unpack(self, x):
        za, i = x[0], 1
        l1 = l2 = 0.0
        if self.front != "flat":
            l1, i = math.exp(x[i]), i + 1
        lc, i = math.exp(x[i]), i + 1
        if self.back != "flat":
            l2 = math.exp(x[i])
        return za, l1, lc, l2

    @staticmethod
    def _cap(s, kind):
        s = s.clamp(0, 1 - 1e-7)
        return 0.5 * torch.log1p(-s * s) if kind == "round" else torch.log1p(-s)

    def logg(self, z, x):
        za, l1, lc, l2 = self._unpack(x)
        z1, z2 = za + l1, za + l1 + lc
        val = torch.zeros_like(z)
        if l1 > 0:
            val = torch.where(z < z1, self._cap((z1 - z) / l1, self.front), val)
        if l2 > 0:
            val = torch.where(z > z2, self._cap((z - z2) / l2, self.back), val)
        return _mask((z > za) & (z < z2 + l2), val)

    def log_i2(self, x):
        za, l1, lc, l2 = self._unpack(x)
        return math.log(lc + self._VOL.get(self.front, 0.0) * l1 + self._VOL.get(self.back, 0.0) * l2)

    def decode(self, x):
        za, l1, lc, l2 = self._unpack(x)
        return {"z_a": za, "l1": l1, "Lc": lc, "l2": l2, "z_end": za + l1 + lc + l2}


class Frustum(Body):
    """Truncated cone: r linear in z' between the end discs at z0 (radius q A or A) and z0 + L.  q = r(front)/r(back).
    Search variables: ln L, z0, ln q."""
    names = ["L", "z0", "q"]
    bounds = [(math.log(100), math.log(2500)), (-260.0, 300.0), (-7.0, 7.0)]
    steps = [0.15, 20.0, 0.4]
    x0_grid = [(math.log(L), z0, lq) for L in (900, 1300, 1700) for z0 in (-220, -120, -40) for lq in (-2.0, -0.7, 0.0, 0.7, 2.0)]

    def logg(self, z, x):
        L, z0, q = math.exp(x[0]), x[1], math.exp(x[2])
        t = (z - z0) / L
        return _mask((t > 0) & (t < 1), torch.log(((1 - t) * q + t).clamp_min(1e-12)) - math.log(max(q, 1.0)))

    def log_i2(self, x):
        L, q = math.exp(x[0]), math.exp(x[2])
        return math.log(L * (1 + q + q * q) / 3.0) - 2 * math.log(max(q, 1.0))

    def decode(self, x):
        return {"L": math.exp(x[0]), "z0": x[1], "q": math.exp(x[2])}


class TwoPower(Body):
    """Two-piece profile: r = A (t/tm)^p for t <= tm and A ((1-t)/(1-tm))^q for t > tm,  t = (z' - z0)/L.
    Search variables: ln L, z0, logit tm, ln p, ln q."""
    names = ["L", "z0", "tm", "p", "q"]
    bounds = [(math.log(100), math.log(9000)), (-260.0, 80.0), (-6.0, 6.0), (math.log(0.004), math.log(6)), (math.log(0.004), math.log(8))]
    steps = [0.2, 20.0, 0.4, 0.3, 0.3]
    x0_grid = [(math.log(L), z0, ltm, math.log(p), math.log(q)) for L in (1300, 1800, 2600) for z0 in (-220, -120, -40)
               for ltm in (-1.5, -0.5, 0.5) for p in (0.5, 0.15) for q in (1.5, 0.5, 0.15)]

    def _unpack(self, x):
        return math.exp(x[0]), x[1], 1 / (1 + math.exp(-x[2])), math.exp(x[3]), math.exp(x[4])

    def logg(self, z, x):
        L, z0, tm, p, q = self._unpack(x)
        t = (z - z0) / L
        tc = t.clamp(1e-12, 1 - 1e-7)
        val = torch.where(tc <= tm, p * (torch.log(tc) - math.log(tm)), q * (torch.log1p(-tc) - math.log(1 - tm)))
        return _mask((t > 0) & (t < 1), val)

    def log_i2(self, x):
        L, z0, tm, p, q = self._unpack(x)
        return math.log(L * (tm / (2 * p + 1) + (1 - tm) / (2 * q + 1)))

    def decode(self, x):
        L, z0, tm, p, q = self._unpack(x)
        return {"L": L, "z0": z0, "tm": tm, "p": p, "q": q, "z_thick": z0 + tm * L}


class Knots(Body):
    """Non-parametric profile: ln g(z') piecewise linear between the fixed knots `zk`, g = 0 outside [zk[0], zk[-1]]
    (flat end discs).  The value at the knot `ref` is fixed to 0 (the size is carried by A); search variables: the other
    len(zk) - 1 knot values.  Parameters per bin: len(zk) (knot values incl. the size)."""

    def __init__(self, zk, ref=None, lo=-12.0, hi=6.0):
        self.zk = [float(v) for v in zk]
        self.ref = int(np.argmin(np.abs(np.asarray(self.zk) - 350.0))) if ref is None else ref
        n = len(self.zk)
        self.names = [f"k{i}" for i in range(n) if i != self.ref]
        self.bounds = [(lo, hi)] * (n - 1)
        self.steps = [0.25] * (n - 1)
        self.x0_grid = [tuple([0.0] * (n - 1))]

    def values(self, x):
        v = list(x)
        v.insert(self.ref, 0.0)
        return v

    def from_body(self, body, x):
        """Starting point: the knot values of another fitted body."""
        lg = body.logg(torch.tensor(self.zk, dtype=torch.float32), x).double().numpy()
        lg = np.where(np.isfinite(lg), lg, -30.0)
        lg = np.clip(lg - lg[self.ref], self.bounds[0][0] + 0.5, self.bounds[0][1] - 0.5)
        lg[self.ref] = 0.0
        return np.delete(lg, self.ref)

    def logg(self, z, x):
        zk = torch.tensor(self.zk, dtype=z.dtype, device=z.device)
        v = torch.tensor(self.values(x), dtype=z.dtype, device=z.device)
        i = (torch.bucketize(z, zk) - 1).clamp(0, len(self.zk) - 2)
        f = (z - zk[i]) / (zk[i + 1] - zk[i])
        return _mask((z >= zk[0]) & (z <= zk[-1]), v[i] + (v[i + 1] - v[i]) * f)

    def log_i2(self, x):
        v, zk, tot = self.values(x), self.zk, 0.0
        for i in range(len(zk) - 1):
            d, l0, l1 = zk[i + 1] - zk[i], v[i], v[i + 1]
            tot += d * math.exp(2 * l0) if abs(l1 - l0) < 1e-9 else d * (math.exp(2 * l1) - math.exp(2 * l0)) / (2 * (l1 - l0))
        return math.log(tot)

    def decode(self, x):
        return {"knots_z": self.zk, "knots_lng": self.values(x)}


class BodyOfLnP:
    """Closed form in the momentum: every search variable of `body` and ln A are polynomials in ln(p/GeV)
    (`coef[name]` = polynomial coefficients, lowest order first; the last entry "lnA" is the size)."""

    def __init__(self, body, coef):
        self.body, self.coef = body, {k: [float(c) for c in v] for k, v in coef.items()}

    def params(self, p):
        L = math.log(p)
        x = [float(np.polynomial.polynomial.polyval(L, self.coef[n])) for n in self.body.names]
        return x, float(np.polynomial.polynomial.polyval(L, self.coef["lnA"]))

    def radius(self, z, p):
        x, lnA = self.params(p)
        return self.body.radius(z, x, math.exp(lnA))

    def volume_L(self, p):
        x, lnA = self.params(p)
        return math.exp(math.log(math.pi) + 2 * lnA + self.body.log_i2(x)) / 1e6

    def n_params(self):
        return sum(len(v) for v in self.coef.values())


def registry():
    """name -> (body, description).  Parameters per bin = len(body.names) + 1."""
    kn8 = [-235, -100, 50, 250, 500, 800, 1100, 1485]
    kn12 = [-235, -150, -60, 40, 160, 320, 520, 760, 1000, 1220, 1380, 1485]
    kn16 = [-235, -170, -100, -30, 50, 150, 270, 400, 550, 700, 850, 1000, 1150, 1290, 1400, 1485]
    return {
        "envelope": (Envelope(), "report 02: r = A u^a e^{-bu}"),
        "drop": (GenDrop(fixed=(0.5, 1.5)), "report 03: r = A t^1/2 (1-t)^3/2"),
        "gendrop": (GenDrop(), "r = A t^alpha (1-t)^beta, free exponents"),
        "ellipsoid": (GenDrop(fixed=(0.5, 0.5)), "ellipsoid"),
        "cylinder": (GenDrop(fixed=(0.0, 0.0)), "cylinder"),
        "cone": (GenDrop(fixed=(0.0, 1.0)), "cone, base in front"),
        "paraboloid": (GenDrop(fixed=(0.0, 0.5)), "paraboloid, base in front, vertex at the back"),
        "powertip": (GenDrop(fixed=(0.0, None)), "flat front, r = A (1-t)^beta"),
        "powerhorn": (GenDrop(fixed=(None, 0.0)), "r = A t^alpha, flat back"),
        "drop_trunc": (GenDrop(fixed=(0.5, 1.5), trunc=True), "drop cut at z_cut"),
        "gendrop_trunc": (GenDrop(trunc=True), "generalised drop cut at z_cut"),
        "envelope_trunc": (TruncEnvelope(), "gamma envelope cut at z_cut"),
        "superellipsoid": (SuperEllipsoid(), "r = A (1-|tau|^n)^(1/m)"),
        "superellipsoid_n": (SuperEllipsoid(m=2.0), "r = A (1-|tau|^n)^(1/2)"),
        "egg": (SuperEllipsoid(two_sided=True, n=2.0, m=2.0), "two half-ellipsoids"),
        "superegg": (SuperEllipsoid(two_sided=True), "two half super-ellipsoids, common n, m"),
        "superegg_sep": (SuperEllipsoid(two_sided=True, sep=True), "two half super-ellipsoids, separate n, m"),
        "capsule": (Capsule("round", "round"), "cylinder with two rounded ends"),
        "capsule_cone": (Capsule("cone", "cone"), "cylinder with two conical ends"),
        "bullet": (Capsule("round", "flat"), "rounded front, cylinder, flat back"),
        "round_cyl_cone": (Capsule("round", "cone"), "rounded front, cylinder, conical back"),
        "frustum": (Frustum(), "truncated cone"),
        "twopower": (TwoPower(), "two-piece power profile"),
        "knots8": (Knots(kn8), "non-parametric, 8 knots"),
        "knots12": (Knots(kn12), "non-parametric, 12 knots"),
        "knots16": (Knots(kn16), "non-parametric, 16 knots"),
    }
