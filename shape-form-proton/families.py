"""families – zárt formulacsaládok a proton (z', r | p) sűrűségére, a shapelab.PolyFormula ősosztályra építve.

Építőkockák (mind log-sűrűség, torch): hosszanti profilok [1/mm] és laterális profilok [1/mm^2, egységnyi területre normálva].
"""
import math

import torch
import torch.nn.functional as F

from shapelab import PolyFormula

LOG_PI, LOG_2PI = math.log(math.pi), math.log(2 * math.pi)


# ---- laterális építőkockák: log T(r), ∫ T 2 pi r dr = 1 ----
def lat_cauchy(r, R):
    """Cauchy-szerű (GFlash) tag: R^2 / (pi (r^2 + R^2)^2)."""
    return 2 * torch.log(R) - LOG_PI - 2 * torch.log(r * r + R * R)


def lat_gauss(r, s):
    return -r * r / (2 * s * s) - LOG_2PI - 2 * torch.log(s)


def lat_exp(r, lam):
    """Exponenciális sűrűség: e^{-r/lam} / (2 pi lam^2)  (dE/dr ∝ r e^{-r/lam})."""
    return -r / lam - LOG_2PI - 2 * torch.log(lam)


def lat_exp_over_r(r, lam, eps=1.0):
    """„Vonalforrás": e^{-r/lam} / (2 pi lam r)  (dE/dr tisztán exponenciális); eps [mm] a tengelyen regularizál."""
    return -r / lam - LOG_2PI - torch.log(lam) - torch.log(r + eps)


# ---- hosszanti építőkockák: log L(z') (nem normált; a rácson a softmax normál) ----
def long_gamma(z, z0, tmax, am1):
    """Gamma-profil a 00_common `long_gamma` alakjában: s^(alpha-1) e^{-(alpha-1) s / tmax}, s = softplus(z - z0)."""
    s = F.softplus((z - z0) / 2) * 2 + 1e-3
    return am1 * torch.log(s) - am1 / tmax * s


class GammaCoreTail(PolyFormula):
    """A baseline GFlash-formula (gamma × mag+farok) a PolyFormula keretben – ellenőrzéshez és kiindulásnak."""
    SPEC = [("z0", -20.0), ("log_tmax", math.log(100)), ("log_am1", math.log(3)),
            ("log_rc", math.log(10)), ("log_ratio", math.log(3)), ("logit_pc", 0.85),
            ("s_rc", 0.0), ("s_ratio", 0.0), ("s_pc", 0.0)]

    def __init__(self, C, degree=1, **kw):
        super().__init__(C, degree, degrees={"s_rc": 0, "s_ratio": 0, "s_pc": 0}, **kw)

    def log_density(self, th, Z, R):
        zk = Z / 1000
        rc = torch.exp(th["log_rc"] + th["s_rc"] * zk)
        rt = rc * (1 + torch.exp(th["log_ratio"] + th["s_ratio"] * zk))
        lp = th["logit_pc"] + th["s_pc"] * zk
        lat = torch.logaddexp(F.logsigmoid(lp) + lat_cauchy(R, rc), F.logsigmoid(-lp) + lat_cauchy(R, rt))
        return long_gamma(Z, th["z0"], torch.exp(th["log_tmax"]), torch.exp(th["log_am1"])) + lat
