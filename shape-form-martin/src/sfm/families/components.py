"""Component models on the ring grid, built from a ``ParamProvider``.

``ComponentModel`` — the component formula

    rho = w_sh S + w_tr U + w_sp D + w_end E + w_far H            (each component normalised on the truncated grid)
    log P = log_softmax(logsumexp_c(ln w_c + ln P_c) + ln A)      A = acceptance template, floored at 1e-6

      S  shower     depth profile L(z') (swappable, ``longitudinal.py``) x cross-section T(r | z') (swappable mixture,
                    ``lateral.py``), the cross-section renormalised on the disc r < 700 mm at every depth
      U  track      sigma((z'-zt0)/4 mm) sigma((R(T)-z')/wR) exp(-kap z'/R) exp(-z'/lam_att) (1 - z'/R + 0.03)^(-q)
                    x [(1-f) Gauss(r; sigma(z')) + f halo(r; l)],  R(T) = R1 T^k - R0 (soft floor 1 mm),
                    sigma(z')^2 = st^2 + (c (z'/1 m)^k_ms / (beta p))^2
      D  spike      Gauss(z'; sp_z, sz) x Gauss(r; sr)          E  end blob   Gauss(z'; R(T), sez) x Gauss(r; ser)
      H  far halo   sigma(z'/10 mm) exp(-z'/Lf) x halo(r; Rf)
      weights       as ``ProtonShape.log_weights`` / ``ParticleShape.components`` (spike tied to the track start)

    integration="centre"   every component is evaluated at the cell centres (ZC, RC = sqrt(mean r^2 edges)) and
                           multiplied by the cell volume: the legacy rule. With the legacy forms and coefficients
                           this reproduces ``formula(lnp)``.
    integration="exact"    cell masses: Gauss-Legendre nodes in z' (lateral parameters taken at the nodes) x ring
                           masses from survival-function differences; the two Gaussians in z' use normal-CDF
                           differences. Same functional forms, same parameters: the difference to "centre" is the
                           binning error of the legacy rule.
    halos="legacy"         exp(-r/lam) / (2 pi lam (r + eps)) for the track halo (eps = 1 mm) and the far halo
                           (eps = 10 mm); ring masses by radial quadrature
    halos="survival"       regulariser-free exp(-r/lam) / (2 pi lam r), closed-form ring masses exp(-r_j/lam) -
                           exp(-r_{j+1}/lam). A different (slightly more peaked) function: not the legacy map.
    The shower halo follows ``shower_lat`` (default: "exp_reg" for halos="legacy", "exp" for halos="survival").

Named variants (``build_variant``): "legacy_centre", "legacy_exact", "survival_exact" (and "survival_centre").

Measured with the legacy coefficients (tests/test_families_components.py; validation numbers from a one-off run of
``common.test_nll``, K = 64, on the 14,950 validation events)
  * legacy_centre vs the legacy formula: max |d ln P| 6e-5 over cells with P > 1e-9 in float32 (the legacy code's
    own rounding: 1e-13 in float64); validation NLL 18.120061 for both, largest per-event difference 1.3e-5.
  * legacy_exact vs legacy_centre (the binning error of the centre rule): map KL 1.5e-4 (20 GeV) ... 7e-4 (1 GeV)
    ... 9e-2 (0.09 GeV), located at the entrance (spike sigma_z = 3-5 mm and the 4 mm track edge against 12.9 mm
    cells; rings below 12.5 mm). Validation NLL +0.00041 +- 0.00006 at fixed coefficients (+0.0088 below 0.5 GeV,
    +0.00014 above 10 GeV): the coefficients were fitted through the centre rule and have to be refitted.
  * survival_exact vs legacy_exact (dropping the regularisers): map KL 1e-5 ... 7e-4; validation NLL +0.00042 +-
    0.00007 against legacy_centre.
  * exact mode, n_z = 5 coarse and 16 fine z nodes, against a converged reference: max |d ln P| 1e-6 for
    p >= 2 GeV, 3e-4 for 0.3-1.5 GeV, 3e-3 below (kinks of the legacy track profile), KL <= 2e-7.
  * one forward + backward at 256 momenta (RTX 4060 Laptop): legacy_centre 168 MiB / 6 ms, legacy_exact 778 MiB /
    58 ms, survival_exact 703 MiB / 44 ms, mixture K = 5 910 MiB / 52 ms.

Degenerate regimes at p -> 0.0005 GeV (x = -4.1; everything stays finite, values and gradients)
  * the legacy polynomials extrapolate to ell = exp(-28) mm and w2 = 0.05 mm (the plateau becomes a step at z' = 0),
    sigma_core = exp(30) mm (flat), R_Cauchy = 1e-6 mm (a point); the log links are clamped to [-12, 35], which
    does not change the map there;
  * below T ~ 0.06 GeV (p < 0.14 GeV) the range R(T) sits on its 1 mm floor: the track is a sub-cell blob at the
    entrance, its range, Bragg and attenuation parameters are dead and its weight is exchangeable with the spike's;
    the Bragg factor (1 - z'/R + 0.03)^(-q) is then a step inside one z cell (the exact rule resolves it to ~1e-3,
    the centre rule not at all);
  * new families: scales rest on their floors (softplus + floor), shapes on the ends of their ranges.

``MixtureModel`` — K components of [one depth family] x [one radial family], no hand-built physics terms; label
switching is removed by a cumulative ordering of one depth parameter (``LongFamily.ORDER_KEY``) at every momentum.

Counting (``n_params``): the free constants of the provider (unmasked polynomial coefficients, or network weights)
plus trainable scalars of the model itself (detector-frame halo). Not counted, as in the legacy count of 101: the
acceptance template (parameter-free geometry) and the structural constants written into the forms (4 mm, 3 mm,
5 mm, 1 mm, 0.03, the clamps -1 / 1.5, eps = 1 mm and 10 mm, the 10 mm rise of the far halo, the soft-start
softness 2 mm, link floors).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cellint import Grid
from .lateral import ExpLat, ExpRegLat, GaussLat, LateralMixture, LatTerm, legacy_shower_lateral, make_lat
from .longitudinal import EMG, LongFamily, make_long
from .params import M_PION, PolyProvider, lnT

__all__ = ["ComponentModel", "MixtureModel", "build_variant", "load_legacy_core", "kl_div", "default_lnp",
           "init_from_target", "init_shower_long", "VARIANTS"]

# legacy raw names and starting values (ProtonShape / ParticleShape SPEC lists)
_SPIKE = [("logit_sp", -3.0), ("sp_z", 0.0), ("log_sp_sz", math.log(6.0)), ("log_sp_sr", math.log(10.0))]
_TRACK = [("logit_tr", -1.5), ("log_R1", math.log(700.0)), ("k_R", 1.77), ("R0", 15.0), ("log_wR", math.log(0.12)),
          ("kap", 0.3), ("zt0", 10.0), ("log_st", math.log(20.0))]
_TIE = [("log_d1", math.log(22.0))]
_ATT = [("log_latt", math.log(250.0))]
_FAR = [("logit_far", -4.0), ("log_Lf", math.log(600.0)), ("log_Rf", math.log(300.0))]
_END = [("logit_end", -3.0), ("log_sez", math.log(40.0)), ("log_ser", math.log(40.0))]
_THALO = [("logit_th", -2.0), ("log_lh", math.log(30.0))]
_BRAGG = [("q_bk", 0.43)]
_MS = [("log_cms", math.log(50.0)), ("ms_pow", 1.5)]
_DETHALO = [("logit_dh", -5.0)]


def _exp(v, lo: float = -12.0, hi: float = 35.0):
    """exp of a legacy log-parameter, exponent clamped (6e-6 ... 1.6e15; inactive for the fitted coefficients)."""
    return torch.exp(v.clamp(lo, hi))


class _Base(nn.Module):
    """Shared plumbing: provider lookup by name, grid, acceptance, normalisation, counting."""

    def _setup(self, provider, names, defaults, logA, grid, integration, n_z, n_r, mass):
        if integration not in ("centre", "exact"):
            raise ValueError("integration must be 'centre' or 'exact'")
        self.integration = integration
        self.grid = grid if grid is not None else Grid(n_z=n_z, n_r=n_r)
        self.names, self.defaults = list(names), dict(defaults)
        self.register_buffer("mass", torch.tensor(float(mass)))
        if logA is None:
            logA = torch.zeros(self.grid.N_Z, self.grid.N_R)
        self.register_buffer("logA", torch.as_tensor(logA).detach().clone().to(self.grid.zc.dtype))
        self.provider = None
        if provider is not None:
            self.set_provider(provider)

    def set_provider(self, provider):
        missing = [n for n in self.names if n not in provider.names]
        if missing:
            raise KeyError(f"provider lacks the raw parameters {missing}")
        self.provider = provider
        self._idx = {n: provider.names.index(n) for n in self.names}
        return self

    @staticmethod
    def acceptance_log(A, floor: float = 1e-6):
        """ln of an acceptance template floored at 1e-6 (``shapelab.WithAcceptance``)."""
        return torch.log(torch.as_tensor(A).clamp_min(floor))

    def theta(self, lnp):
        raw = self.provider(lnp)
        th = {n: raw[:, i] for n, i in self._idx.items()}
        th["lnT"] = lnT(lnp, self.mass)
        return th

    def n_params(self) -> int:
        """Free constants: the provider's (``n_params()`` if it has one) plus the model's own trainable scalars."""
        prov = self.provider
        n = int(prov.n_params()) if hasattr(prov, "n_params") else sum(p.numel() for p in prov.parameters() if p.requires_grad)
        own = {id(p) for p in prov.parameters()}
        return n + sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in own)

    def _apply_acceptance(self, core):
        B = core.shape[0]
        return F.log_softmax((core + self.logA).reshape(B, -1), 1).reshape(core.shape)

    def _mass_sep(self, lon_z, lat_r):
        """Separable component: ln z-cell masses (B, N_Z) + ln ring masses (B, N_R)."""
        return lon_z[:, :, None] + lat_r[:, None, :]


# ----------------------------------------------------------------------------------------------------------------------
class ComponentModel(_Base):
    """The component formula (see the module docstring).

    ComponentModel(provider, logA=None, integration="exact", halos="legacy",
                   shower_long="plateau", shower_lat=None, lat_norm="mixture", lat_ordered=False,
                   track=True, spike=True, tie=True, att=True, far=True, endblob=True, track_halo=True, bragg=True,
                   curv=True, ms=True, det_halo=None, grid=None, n_z=5, n_r=4, mass=M_PION)

    provider     ``ParamProvider`` whose names include ``self.names`` (looked up by name; order is free)
    logA         ln of the acceptance template (128, 32) (``acceptance_log(abar)``); None: no acceptance
    shower_long  depth family name or instance; "plateau" and "gamma" use the legacy raw names, other families are
                 created with the prefix "sh_". ``shower_long_kw`` is passed to the family (e.g. dict(soft="free")).
                 "emg_tied": EMG whose attenuation length is the track's exp(log_latt) (two own parameters).
    shower_lat   ``LateralMixture`` or a tuple of three radial family names for ``legacy_shower_lateral``
    det_halo     ``halo_det.DetHalo`` (or any module returning a normalised log map (128, 32)); it is mixed in after
                 the acceptance with weight sigma(logit_dh), because it carries its own acceptance.
    fine_shower  exact mode: integrate the shower depth profile on the fine z nodes (``Grid.integrate_z_fine``).
                 Default: off for the plateau (smooth: n_z = 5 nodes are converged to 5e-9), on for every other family.
    ``self.names`` lists the raw parameters the provider must supply, ``self.defaults`` their starting values,
    ``self.legacy_constant`` / ``self.momentum_dependent`` the split of the legacy formula (21 / 22 names for the
    final pion configuration), ``self.new_names`` the names the legacy fit does not have.
    """

    COMPONENTS = ("shower", "track", "spike", "end", "far")

    def __init__(self, provider=None, logA=None, integration: str = "exact", halos: str = "legacy",
                 shower_long="plateau", shower_long_kw: Optional[dict] = None, shower_lat=None,
                 lat_norm: str = "mixture", lat_ordered: bool = False, track: bool = True, spike: bool = True,
                 tie: bool = True, att: bool = True, far: bool = True, endblob: bool = True, track_halo: bool = True,
                 bragg: bool = True, curv: bool = True, ms: bool = True, det_halo: Optional[nn.Module] = None,
                 fine_shower: Optional[bool] = None, grid: Optional[Grid] = None, n_z: int = 5, n_r: int = 4,
                 mass: float = M_PION):
        super().__init__()
        if halos not in ("legacy", "survival"):
            raise ValueError("halos must be 'legacy' or 'survival'")
        self.halos = halos
        self.use_track, self.use_spike = bool(track), bool(spike)
        self.tie = bool(tie and spike and track)
        self.att, self.endblob = bool(att and track), bool(endblob and track)
        self.far = bool(far)
        self.track_halo, self.bragg, self.ms = bool(track_halo and track), bool(bragg and track), bool(ms and track)
        # shower depth profile
        self.tied_tau = shower_long == "emg_tied"
        if self.tied_tau:
            if not self.att:
                raise ValueError("emg_tied needs the track attenuation (att=True)")
            self.sh_long = EMG("sh_", tau="external", **(shower_long_kw or {}))
        elif isinstance(shower_long, LongFamily):
            self.sh_long = shower_long
        else:
            prefix = "" if shower_long in ("plateau", "gamma") else "sh_"
            self.sh_long = make_long(shower_long, prefix, **(shower_long_kw or {}))
        # shower cross-section
        if shower_lat is None:
            shower_lat = ("gauss", "cauchy", "exp_reg" if halos == "legacy" else "exp")
        if isinstance(shower_lat, LateralMixture):
            self.sh_lat = shower_lat
        else:
            self.sh_lat = legacy_shower_lateral(tuple(shower_lat), curv=curv, disc_norm=lat_norm, ordered=lat_ordered)
        self.curv = bool(curv)
        # depth profiles other than the (smooth) plateau get the fine z nodes: soft starts and narrow EMG edges
        self.fine_shower = (self.sh_long.name != "plateau") if fine_shower is None else bool(fine_shower)
        halo = (lambda eps: ExpRegLat(eps)) if halos == "legacy" else (lambda eps: ExpLat())
        self._gauss, self._thalo, self._fhalo = GaussLat(), halo(1.0), halo(10.0)
        self.det_halo = det_halo
        # raw parameter names in the legacy order
        spec = list(self.sh_long.defaults.items()) + list(self.sh_lat.defaults.items())
        spec += (_SPIKE if self.use_spike else []) + (_TRACK if self.use_track else [])
        for flag, block in ((self.tie, _TIE), (self.att, _ATT), (self.far, _FAR), (self.endblob, _END),
                            (self.track_halo, _THALO), (self.bragg, _BRAGG), (self.ms, _MS),
                            (det_halo is not None, _DETHALO)):
            if flag:
                spec += block
        names = [n for n, _ in spec]
        if len(set(names)) != len(names):
            raise ValueError("duplicate raw parameter names")
        self._setup(provider, names, dict(spec), logA, grid, integration, n_z, n_r, mass)

    # names the final pion formula keeps constant in momentum: ProtonShape.CONST, log_latt and
    # ParticleShape.EXTRA_CONST, minus the eight names its ``spike_degrees`` free to degree 2 (sp_z, log_sp_sz,
    # log_sp_sr, s_sig, s_R, s_lam, s_1, s_2)
    LEGACY_CONST = ("log_lam", "log_R1", "k_R", "R0", "log_wR", "kap", "zt0", "log_st", "log_d1", "log_latt", "log_Lf",
                    "log_Rf", "log_sez", "log_ser", "log_lh", "q_bk", "q_sig", "q_1", "q_2", "log_cms", "ms_pow")

    @property
    def legacy_constant(self):
        """Names the final pion formula keeps constant in momentum (polynomial degree 0)."""
        return [n for n in self.names if n in self.LEGACY_CONST]

    @property
    def momentum_dependent(self):
        return [n for n in self.names if n not in self.LEGACY_CONST]

    @property
    def new_names(self):
        """Raw names that the final legacy pion formula does not have (swapped families, detector halo)."""
        legacy = {n for blk in (_SPIKE, _TRACK, _TIE, _ATT, _FAR, _END, _THALO, _BRAGG, _MS) for n, _ in blk}
        legacy |= {"zs", "log_w1", "log_ell", "log_w2", "kappa", "log_sig", "log_R", "log_lam", "logit_1", "logit_2",
                   "s_sig", "s_R", "s_lam", "s_1", "s_2", "q_sig", "q_1", "q_2"}
        return [n for n in self.names if n not in legacy]

    # ---- pieces ----
    def _z(self):
        return self.grid.z_nodes if self.integration == "exact" else self.grid.zc

    def range_mm(self, th):
        """R(T) = R1 T^k - R0 [mm] with the legacy soft floor: 5 softplus(raw/5) + 1, (B, 1)."""
        raw = _exp(th["log_R1"] + th["k_R"] * th["lnT"], -60.0, 60.0) - th["R0"]
        return (F.softplus(raw / 5.0) * 5.0 + 1.0)[:, None]

    def _shower_long(self, th, z):
        raw = self.sh_long.pick(th)
        nat = self.sh_long.natural(raw, tau=_exp(th["log_latt"])[:, None]) if self.tied_tau else self.sh_long.natural(raw)
        return self.sh_long.log_pdf(z[None, :], nat)

    def shower_logmass(self, th):
        g, z = self.grid, self._z()
        if self.integration == "exact":
            lat = self.sh_lat.ring_logmass(th, z / 1000, g)
            if self.fine_shower:
                return g.integrate_z_fine(self._shower_long(th, g.zf_nodes), lat)
            return g.integrate_z(self._shower_long(th, z)[:, :, None] + lat)
        return self._shower_long(th, z)[:, :, None] + self.sh_lat.log_area_density(th, z / 1000, g) + g.log_vol

    def shower_depth_profile(self, lnp, th=None):
        """Normalised ln z-marginal of the shower component (B, N_Z), from the depth profile alone (the cross-section
        is normalised on the disc at every depth, so it does not change the marginal). A cheap 1-D target for
        ``init_from_target`` when only the depth family is swapped."""
        th = self.theta(lnp) if th is None else th
        g = self.grid
        if self.integration != "exact":
            return torch.log_softmax(self._shower_long(th, g.zc), 1)
        if self.fine_shower:
            return torch.log_softmax(g.integrate_zf(self._shower_long(th, g.zf_nodes)), 1)
        return torch.log_softmax(g.integrate_z(self._shower_long(th, g.z_nodes)), 1)

    def _track_long(self, th, z, Rr):
        c = lambda n: th[n][:, None]                                          # noqa: E731
        Z = z[None, :]
        wR = _exp(c("log_wR")) * Rr + 3.0
        lon = F.logsigmoid((Z - c("zt0")) / 4.0) + F.logsigmoid((Rr - Z) / wR) - c("kap") * (Z / Rr).clamp(-1.0, 1.5)
        if self.att:
            lon = lon - Z.clamp_min(0) / _exp(c("log_latt"))
        if self.bragg:
            lon = lon - c("q_bk") * torch.log(1.0 - (Z / Rr).clamp(0.0, 1.0) + 0.03)
        return lon

    def _track_sigma(self, th, z):
        """Track width (B, Nz or 1, 1): st, or sqrt(st^2 + (c (z/1 m)^k / (beta p))^2) with multiple scattering."""
        st = _exp(th["log_st"])[:, None, None]
        if not self.ms:
            return st
        T = torch.exp(th["lnT"])[:, None, None]
        bp = (T * T + 2 * T * self.mass) / (T + self.mass)                    # beta * p [GeV]
        zz = (z.clamp_min(1.0) / 1000)[None, :, None]
        return torch.sqrt(st * st + (_exp(th["log_cms"])[:, None, None] * zz ** th["ms_pow"][:, None, None] / bp) ** 2)

    def track_logmass(self, th, Rr):
        """Exact mode: the depth profile (4 mm entry edge, range edge, Bragg cusp and clamps of the legacy form) is
        integrated on the fine z nodes; the cross-section is computed at the n_z coarse nodes and carried to the
        fine ones by log-polynomial interpolation (``Grid.integrate_z_fine``)."""
        g = self.grid
        exact = self.integration == "exact"
        z = g.z_nodes if exact else g.zc
        sig = self._track_sigma(th, z)
        lat = self._gauss.ring_logmass(g, sig, {}) if exact else self._gauss.log_pdf_area(g.rc, sig, {})
        if self.track_halo:
            lh = _exp(th["log_lh"])[:, None, None]
            hal = self._thalo.ring_logmass(g, lh, {}) if exact else self._thalo.log_pdf_area(g.rc, lh, {})
            lt = th["logit_th"][:, None, None]
            lat = torch.logaddexp(F.logsigmoid(-lt) + lat, F.logsigmoid(lt) + hal)
        if exact:
            return g.integrate_z_fine(self._track_long(th, g.zf_nodes, Rr), lat)
        return self._track_long(th, z, Rr)[:, :, None] + lat + g.log_vol

    def _blob_logmass(self, mu, sz, sr):
        """Gauss(z'; mu, sz) x Gauss(r; sr): mu, sz, sr (B, 1)."""
        g = self.grid
        if self.integration == "exact":
            return self._mass_sep(g.gauss_z(mu, sz), self._gauss.ring_logmass(g, sr, {}))
        return (-0.5 * ((g.zc[None, :] - mu) / sz) ** 2)[:, :, None] + (-0.5 * (g.rc[None, :] / sr) ** 2)[:, None, :] + g.log_vol

    def far_logmass(self, th):
        g = self.grid
        exact = self.integration == "exact"
        z = g.zf_nodes if exact else g.zc
        Lf, Rf = _exp(th["log_Lf"])[:, None], _exp(th["log_Rf"])[:, None]
        lon = F.logsigmoid(z[None, :] / 10.0) - z[None, :].clamp_min(0) / Lf
        if exact:
            return self._mass_sep(g.integrate_zf(lon), self._fhalo.ring_logmass(g, Rf, {}))
        return self._mass_sep(lon, self._fhalo.log_pdf_area(g.rc, Rf, {})) + g.log_vol

    def log_weights(self, th, Rr):
        """ln weights {component: (B,)} (``ProtonShape.log_weights`` + end blob + far halo)."""
        zero = torch.zeros_like(th["lnT"])
        lw = {}
        if self.use_track:
            l_tr, l_ntr = F.logsigmoid(th["logit_tr"]), F.logsigmoid(-th["logit_tr"])
        else:
            l_tr, l_ntr = None, zero
        if not self.use_spike:
            lw["shower"] = l_ntr
            if self.use_track:
                lw["track"] = l_tr
        else:
            l_sp, l_nsp = F.logsigmoid(th["logit_sp"]), F.logsigmoid(-th["logit_sp"])
            if not self.use_track:
                lw["shower"], lw["spike"] = l_nsp, l_sp
            elif not self.tie:
                lw["shower"], lw["track"], lw["spike"] = l_nsp + l_ntr, l_nsp + l_tr, l_sp
            else:                                   # spike of a non-showering particle = first layer of its track
                d1 = _exp(th["log_d1"])
                Lt = F.softplus((Rr[:, 0] - th["zt0"]) / 5.0) * 5.0
                l_f1, l_nf1 = torch.log(d1) - torch.log(d1 + Lt), torch.log(Lt.clamp_min(1e-12)) - torch.log(d1 + Lt)
                lw["shower"], lw["track"] = l_ntr + l_nsp, l_tr + l_nf1
                lw["spike"] = torch.logaddexp(l_tr + l_f1, l_ntr + l_sp)
        if self.endblob:
            le, lne = F.logsigmoid(th["logit_end"]), F.logsigmoid(-th["logit_end"])
            lw = {k: v + lne for k, v in lw.items()}
            lw["end"] = le
        if self.far:
            lf, lnf = F.logsigmoid(th["logit_far"]), F.logsigmoid(-th["logit_far"])
            lw = {k: v + lnf for k, v in lw.items()}
            lw["far"] = lf
        return lw

    # ---- maps ----
    def components(self, lnp, th=None):
        """({component: ln weight (B,)}, {component: normalised ln P (B, N_Z, N_R)}) before the acceptance."""
        th = self.theta(lnp) if th is None else th
        c = lambda n: th[n][:, None]                                          # noqa: E731
        Rr = self.range_mm(th) if self.use_track else None
        norm = self.grid.normalise
        comp = {"shower": norm(self.shower_logmass(th))}
        if self.use_track:
            comp["track"] = norm(self.track_logmass(th, Rr))
        if self.use_spike:
            comp["spike"] = norm(self._blob_logmass(c("sp_z"), _exp(c("log_sp_sz")), _exp(c("log_sp_sr"))))
        if self.endblob:
            comp["end"] = norm(self._blob_logmass(Rr, _exp(c("log_sez")), _exp(c("log_ser"))))
        if self.far:
            comp["far"] = norm(self.far_logmass(th))
        return self.log_weights(th, Rr), comp

    def core_map(self, lnp, th=None):
        """Normalised ln P of the mixture before the acceptance (the legacy ``formula.base(lnp)``)."""
        lw, comp = self.components(lnp, th)
        return torch.logsumexp(torch.stack([lw[k][:, None, None] + comp[k] for k in comp], 0), 0)

    def component_map(self, lnp, name: str):
        """Normalised ln P of one component (before the acceptance)."""
        return self.components(lnp)[1][name]

    def forward(self, lnp):
        th = self.theta(lnp)
        out = self._apply_acceptance(self.core_map(lnp, th))
        if self.det_halo is not None:
            ld = th["logit_dh"][:, None, None]
            out = torch.logaddexp(F.logsigmoid(-ld) + out, F.logsigmoid(ld) + self.det_halo()[None])
        return out


# ----------------------------------------------------------------------------------------------------------------------
class MixtureModel(_Base):
    """Pure mixture: rho = sum_k pi_k L_k(z') T_k(r | z'), K components of one depth family x one radial family.

    MixtureModel(provider, K=3, long="gengamma", lat="burr", logA=None, integration="exact", order=True,
                 depth="linear", long_kw=None, lat_kw=None, fine=False, grid=None)

    fine=True integrates the depth profiles on the fine z nodes (needed when a component may become narrower than
    ~6 mm in depth or has a 2 mm soft start; costs one (B, 128, n_zf, 32) tensor per component).

    Raw names (k = 0..K-1):  c{k}_<depth-family names>;  c{k}_log_r (+ c{k}_s_r, c{k}_q_r) = ln scale of the radial
    term and its depth coefficients; c{k}_lat_<shape names>;  mix_logit{k}, k = 1..K-1 (stick-breaking,
    pi_0 = sigma(-l_1)... see ``log_weights``).
    Every component's cross-section is renormalised on the disc at every depth and the component on the grid.

    Label switching. With identical families the likelihood is invariant under the K! permutations of the components.
    ``order=True`` builds the family's ordering key (EMG: mu; gengamma: mode; invgauss: mean; lognormal: median;
    experlang: l2; burr: lam; gamma: tmax; plateau: ell) cumulatively,
        key_0 = link(raw_0),   key_k = key_{k-1} + U softplus(raw_k)   (U = 100 mm; plateau/gamma: + exp(raw_k)),
    so key_0 < key_1 < ... holds for every momentum and the components cannot exchange roles along a trajectory in
    p. What it does not remove: two components may still approach the same key (softplus -> 0, a flat direction),
    and a component whose weight goes to zero has unconstrained shape parameters.
    """

    def __init__(self, provider=None, K: int = 3, long="gengamma", lat="burr", logA=None, integration: str = "exact",
                 order: bool = True, depth: str = "linear", long_kw: Optional[dict] = None,
                 lat_kw: Optional[dict] = None, fine: bool = False, grid: Optional[Grid] = None, n_z: int = 5,
                 n_r: int = 4, mass: float = M_PION):
        super().__init__()
        if depth not in ("none", "linear", "quadratic"):
            raise ValueError("depth must be 'none', 'linear' or 'quadratic'")
        self.K, self.order, self.fine = int(K), bool(order), bool(fine)
        self.longs = [make_long(long, f"c{k}_", **(long_kw or {})) for k in range(self.K)]
        if self.order and self.longs[0].ORDER_KEY is None:
            raise ValueError(f"family '{self.longs[0].name}' has no ordering key")
        self.lats = []
        spec = []
        for k, fam in enumerate(self.longs):
            d = fam.defaults
            if self.order and k > 0:                                         # raw key of k > 0 is an increment
                d[fam.prefix + fam.ORDER_KEY[1]] = fam.KEY_STEP_RAW
            spec += list(d.items())
            sc = (f"c{k}_log_r", f"c{k}_s_r" if depth != "none" else None, f"c{k}_q_r" if depth == "quadratic" else None)
            term = LatTerm(make_lat(lat, **(lat_kw or {})), sc, f"c{k}_lat_", math.log(25.0) + 0.9 * k)
            mix = LateralMixture([term], [], disc_norm="mixture")
            self.lats.append(mix)
            spec += list(mix.defaults.items())
        spec += [(f"mix_logit{k}", 0.0) for k in range(1, self.K)]
        self._setup(provider, [n for n, _ in spec], dict(spec), logA, grid, integration, n_z, n_r, mass)

    def log_weights(self, th):
        """Stick-breaking: pi_0 = sigma(l_1) ... written so that all-zero logits give pi_k = 1/K."""
        out, rest = [], torch.zeros_like(th["lnT"])
        for k in range(1, self.K):
            v = th[f"mix_logit{k}"] - math.log(self.K - k)
            out.append(rest + F.logsigmoid(v))
            rest = rest + F.logsigmoid(-v)
        return out + [rest]

    def naturals(self, th):
        """Constrained depth parameters of the K components (the ordering applied)."""
        out, prev = [], None
        for fam in self.longs:
            nat = fam.natural(fam.pick(th), key_prev=prev if self.order else None)
            prev = nat[fam.ORDER_KEY[0]] if fam.ORDER_KEY is not None else None
            out.append(nat)
        return out

    def components(self, lnp, th=None):
        th = self.theta(lnp) if th is None else th
        g = self.grid
        exact = self.integration == "exact"
        z = g.z_nodes if exact else g.zc
        comp = []
        for fam, lat, nat in zip(self.longs, self.lats, self.naturals(th)):
            if exact and self.fine:
                m = g.integrate_z_fine(fam.log_pdf(g.zf_nodes[None, :], nat), lat.ring_logmass(th, z / 1000, g))
            elif exact:
                m = g.integrate_z(fam.log_pdf(z[None, :], nat)[:, :, None] + lat.ring_logmass(th, z / 1000, g))
            else:
                m = fam.log_pdf(z[None, :], nat)[:, :, None] + lat.log_area_density(th, z / 1000, g) + g.log_vol
            comp.append(g.normalise(m))
        return self.log_weights(th), comp

    def core_map(self, lnp, th=None):
        lw, comp = self.components(lnp, th)
        return torch.logsumexp(torch.stack([w[:, None, None] + c for w, c in zip(lw, comp)], 0), 0)

    def forward(self, lnp):
        return self._apply_acceptance(self.core_map(lnp))


# ----------------------------------------------------------------------------------------------------------------------
# construction from the legacy fit
# ----------------------------------------------------------------------------------------------------------------------
VARIANTS = {
    # name: (integration, halos, doc)
    "legacy_centre": dict(integration="centre", halos="legacy"),     # = the legacy map (parity)
    "legacy_exact": dict(integration="exact", halos="legacy"),       # same forms and coefficients, exact cell masses
    "survival_exact": dict(integration="exact", halos="survival"),   # regulariser-free halos, closed-form ring masses
    "survival_centre": dict(integration="centre", halos="survival"),
}


def load_legacy_core(device=None):
    """(core, logA): the fitted ``ParticleShape`` of the final pion formula and ln of its acceptance template,
    read from ``shape-form-proton/results/particles/pion`` without touching any data (the checkpoint restores
    lnp0, lnp_s, v0, vs). Read-only; the returned modules are in eval mode with gradients disabled."""
    from sfm import legacy                                            # imported lazily: it imports common (CUDA init)
    model, core, _, _ = legacy.shapelab.build_final(legacy.C, "pion", str(legacy.PION_DIR), 0.0, 1.0, metric="std")
    model = model.eval().requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model.base, model.logA.clone()


def build_variant(variant: str = "legacy_exact", core=None, logA=None, device=None, dtype=None, grid=None,
                  provider=None, degree: Optional[int] = None, **overrides) -> ComponentModel:
    """ComponentModel of a named variant with a ``PolyProvider`` copied from the legacy fit.

    variant    key of ``VARIANTS``; ``overrides`` are further ``ComponentModel`` arguments (shower_long=..., ...)
    core/logA  from ``load_legacy_core()`` when not given
    Raw parameters the legacy fit does not have (swapped families) start at the family defaults with polynomial
    degree ``degree`` (default: the legacy degree 3 for depth-profile parameters, 0 for shape parameters of radial
    terms); use ``init_from_target`` to move them onto the legacy map.
    """
    if core is None:
        core, logA0 = load_legacy_core(device)
        logA = logA0 if logA is None else logA
    cfg = dict(VARIANTS[variant])
    cfg.update(overrides)
    if grid is None:
        C = None
        try:
            from sfm import legacy
            C = legacy.C
        except Exception:                                              # pragma: no cover - common.py not importable
            pass
        grid = Grid.from_common(C, n_z=cfg.pop("n_z", 5), n_r=cfg.pop("n_r", 4)) if C is not None else None
    model = ComponentModel(None, logA, grid=grid, **cfg)
    if provider is None:
        deg = {n: (0 if "_lat" in n or n.startswith("lat") or n in ("logit_dh",) else (int(core.degree) if degree is None else degree))
               for n in model.new_names}
        provider = PolyProvider.from_legacy(core, names=model.names, defaults=model.defaults, degrees=deg)
    model.set_provider(provider)
    model = model.to(device if device is not None else core.coef.device)
    return model.to(dtype) if dtype is not None else model


# ----------------------------------------------------------------------------------------------------------------------
# KL matching (data-free initialisation)
# ----------------------------------------------------------------------------------------------------------------------
def kl_div(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per batch element for normalised log maps (B, N_Z, N_R)."""
    B = logp.shape[0]
    return (logp.exp() * (logp - logq)).reshape(B, -1).sum(1)


def default_lnp(n: int = 48, p_min: float = 0.0895, p_max: float = 20.0, device=None, dtype=torch.float32):
    """n momenta inside the training range: half uniform in ln p, half uniform in p (the events are uniform in p)."""
    a = torch.linspace(math.log(p_min), math.log(p_max), n // 2, dtype=torch.float64)
    b = torch.log(torch.linspace(p_min, p_max, n - n // 2 + 2, dtype=torch.float64)[1:-1])
    return torch.sort(torch.cat([a, b])).values.to(device=device, dtype=dtype)


class _FreeRows(nn.Module):
    """Provider wrapper for stage 1 of ``init_from_target``: the columns ``names`` are free per momentum point."""

    def __init__(self, base, names, start):
        super().__init__()
        self.base, self.names = base, list(base.names)
        self.cols = [self.names.index(n) for n in names]
        self.free = nn.Parameter(start.clone())

    def forward(self, lnp):
        raw = self.base(lnp).detach()
        out = raw.clone()
        out[:, self.cols] = self.free
        return out


def init_from_target(model: nn.Module, target: Callable, names: Sequence[str], lnp: Optional[torch.Tensor] = None,
                     model_fn: Optional[Callable] = None, steps=(300, 200), lr=(0.05, 0.01), n_search: int = 0,
                     search_scale: float = 1.0, seed: int = 0, verbose: bool = False):
    """Data-free initialisation of selected provider rows by KL matching to a target map.

    Minimises  mean_p KL(P_target(. | p) || P_model(. | p))  over the raw parameters ``names`` only; everything else
    in the provider is frozen. ``model.provider`` must be a ``PolyProvider``.
        stage 0   (n_search > 0) random search: n_search candidates around the current raw values (Gaussian, width
                  ``search_scale`` per raw unit, the same candidate for all momenta); the candidate with the smallest
                  mean KL is the common start of all momenta. (One common start, not the best per momentum: weakly
                  identified families have several near-equivalent optima, and independent choices give parameter
                  curves that jump between them and cannot be followed by a polynomial in x.)
        stage 1   the raw values of ``names`` are free numbers at each of the N momenta ``lnp`` (Adam, steps[0])
        stage 2   least-squares polynomials in x of each row's degree through the stage-1 values
        stage 3   joint polish of those polynomial coefficients on the same KL (Adam, steps[1])
    target     callable lnp -> normalised ln P (e.g. the legacy-form model in the same integration mode, one of its
               components, or a 1-D marginal such as ``ComponentModel.shower_depth_profile``)
    model_fn   callable lnp -> ln P of the model to compare, same shape as the target (default ``model``)
    Returns {"kl_start", "kl_search", "kl_stage1", "kl_lstsq", "kl_final"} (means over the momenta, nat).
    """
    prov = model.provider
    if not isinstance(prov, PolyProvider):
        raise TypeError("init_from_target needs a PolyProvider (use its result to warm-start other providers)")
    names = list(names)
    dev, dt = prov.coef.device, prov.coef.dtype
    lnp = default_lnp(device=dev, dtype=dt) if lnp is None else lnp.to(dev, dt)
    fn = model_fn if model_fn is not None else model
    with torch.no_grad():
        tgt = target(lnp).detach()
        out = {"kl_start": float(kl_div(tgt, fn(lnp)).mean())}
        start = prov(lnp)[:, [prov.index(n) for n in names]]
    was_training = model.training
    model.eval()
    free = _FreeRows(prov, names, start).to(dev)
    model.set_provider(free)
    try:
        # stage 0: random search, best candidate per momentum
        if n_search > 0:
            gen = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                best_kl, best = float(kl_div(tgt, fn(lnp)).mean()), start.clone()
                scale = torch.as_tensor(search_scale, dtype=dt).to(dev)
                for _ in range(n_search):
                    cand = (start + scale * torch.randn(1, len(names), generator=gen).to(dev, dt)).contiguous()
                    free.free.copy_(cand)
                    kl = float(kl_div(tgt, fn(lnp)).mean())
                    if math.isfinite(kl) and kl < best_kl:
                        best_kl, best = kl, cand.clone()
                free.free.copy_(best)
                out["kl_search"] = best_kl
        # stage 1: per-momentum free values
        opt = torch.optim.Adam([free.free], lr=lr[0])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps[0], 1), eta_min=lr[0] / 20)
        for it in range(steps[0]):
            loss = kl_div(tgt, fn(lnp)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if verbose and it % 100 == 0:
                print(f"  stage 1 step {it:4d}  KL {float(loss):.3e}")
        with torch.no_grad():
            out["kl_stage1"] = float(kl_div(tgt, fn(lnp)).mean())
            values = free.free.detach().clone()
    finally:
        model.set_provider(prov)
    # stage 2: polynomials through the per-momentum values
    prov.set_rows_lstsq(names, lnp, values)
    with torch.no_grad():
        out["kl_lstsq"] = float(kl_div(tgt, fn(lnp)).mean())
    # stage 3: joint polish of the selected rows
    rmask = prov.row_mask(names)
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    prov.coef.requires_grad_(True)
    opt = torch.optim.Adam([prov.coef], lr=lr[1])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps[1], 1), eta_min=lr[1] / 20)
    for it in range(steps[1]):
        loss = kl_div(tgt, fn(lnp)).mean()
        opt.zero_grad()
        loss.backward()
        prov.coef.grad.mul_(rmask)
        opt.step()
        sched.step()
        if verbose and it % 100 == 0:
            print(f"  stage 3 step {it:4d}  KL {float(loss):.3e}")
    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)
    with torch.no_grad():
        out["kl_final"] = float(kl_div(tgt, fn(lnp)).mean())
    model.train(was_training)
    return out


def init_shower_long(model: "ComponentModel", reference: "ComponentModel", lnp: Optional[torch.Tensor] = None,
                     steps=(1000, 1500), lr=(0.03, 0.01), n_search: int = 400, search_scale: float = 1.0,
                     seed: int = 0, verbose: bool = False):
    """Initialise a swapped shower depth family from a reference model (normally the legacy plateau fit in the same
    integration mode) by matching the shower's depth marginal.

    The shower cross-section is normalised on the disc at every depth, so the depth marginal of the shower component
    is the depth profile alone and  KL(shower maps) = KL(depth marginals) + (a term from the weighting of the
    cross-section inside one z cell, < 1e-6 here).  The match is therefore done in 1-D (``shower_depth_profile``),
    which costs milliseconds per step: random search + per-momentum Adam + polynomial projection + polish
    (``init_from_target``). Only the raw parameters of ``model.sh_long`` are changed.
    Returns the KL trail of ``init_from_target`` plus "kl_map": the mean KL between the two full maps afterwards.
    """
    lnp = default_lnp(device=model.provider.coef.device, dtype=model.provider.coef.dtype) if lnp is None else lnp
    scale = torch.tensor([getattr(model.sh_long, "RAW_SCALE", {}).get(n, 1.0) for n in model.sh_long.local_names])
    out = init_from_target(model, reference.shower_depth_profile, model.sh_long.names, lnp=lnp,
                           model_fn=model.shower_depth_profile, steps=steps, lr=lr, n_search=n_search,
                           search_scale=search_scale * scale, seed=seed, verbose=verbose)
    with torch.no_grad():
        out["kl_shower"] = float(kl_div(reference.component_map(lnp, "shower"), model.component_map(lnp, "shower")).mean())
        out["kl_map"] = float(kl_div(reference(lnp), model(lnp)).mean())
    return out
