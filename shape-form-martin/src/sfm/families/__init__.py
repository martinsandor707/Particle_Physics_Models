"""sfm.families: closed-form density families on the (z', r) ring grid and their exact cell integration.

    params        ParamProvider contract, PolyProvider (legacy-compatible), ConstProvider, ConcatProvider, x_of_lnp, lnT
    cellint       Grid (edges, Gauss-Legendre nodes), log1mexp, logdiffexp, ring masses from ln S, Gaussian z cells
    longitudinal  depth profiles: plateau, gamma, emg, gengamma, invgauss, lognormal, experlang1-3, burr
    lateral       radial families via ln S(r): gauss, cauchy, student, burr, exp, weibull, weibull_core, exp_reg;
                  LateralMixture (stick-breaking weights, depth dependence, optional ordering)
    components    ComponentModel (the component formula; "centre" / "exact" integration; swappable shower families),
                  MixtureModel (pure K-component mixture), build_variant, init_from_target (KL matching)
    halo_det      DetHalo: theta-averaged detector-frame halo template

Every model is an ``nn.Module`` mapping ``lnp (B,) -> log P (B, 128, 32)``, a normalised log-probability over the
ring cells, as required by ``common.event_nll``.

    from sfm.families import build_variant, init_shower_long, load_legacy_core
    core, logA = load_legacy_core()                                   # fitted legacy formula (read-only), ln A-bar
    a0 = build_variant("legacy_centre", core, logA)                   # = the legacy map
    a2 = build_variant("survival_exact", core, logA)                  # exact cell masses, regulariser-free halos
    a4 = build_variant("survival_exact", core, logA, shower_long="emg")
    init_shower_long(a4, a2)                                          # data-free start of the new depth profile
    a7 = build_variant("survival_exact", core, logA, shower_lat=("gauss", "burr", "weibull"))   # starts on a2 exactly
    net = ParamNet(a2.momentum_dependent, a2.legacy_constant, ...)    # any provider with these names
    net.fit_to(a2.provider, lnp_grid); a2.set_provider(net)
"""
from . import cellint, components, halo_det, lateral, longitudinal, params  # noqa: F401
from .cellint import Grid, log1mexp, logdiffexp  # noqa: F401
from .components import (ComponentModel, MixtureModel, build_variant, init_from_target, init_shower_long,  # noqa: F401
                         kl_div, load_legacy_core)
from .halo_det import DetHalo  # noqa: F401
from .lateral import LAT_FAMILIES, LateralMixture, LatTerm, make_lat  # noqa: F401
from .longitudinal import LONG_FAMILIES, make_long  # noqa: F401
from .params import ConcatProvider, ConstProvider, ParamProvider, PolyProvider, lnT, x_of_lnp  # noqa: F401

__all__ = ["cellint", "components", "halo_det", "lateral", "longitudinal", "params", "Grid", "log1mexp", "logdiffexp",
           "ComponentModel", "MixtureModel", "build_variant", "init_from_target", "init_shower_long", "kl_div",
           "load_legacy_core", "DetHalo", "LAT_FAMILIES", "LateralMixture", "LatTerm", "make_lat", "LONG_FAMILIES",
           "make_long", "ConcatProvider", "ConstProvider", "ParamProvider", "PolyProvider", "lnT", "x_of_lnp"]
