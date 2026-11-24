"""Survey of lateral forms at equal polynomial degree, baseline (linear-in-z') depth dependence, slab likelihood.

usage: survey.py <part> <degree>     part: A | B   (two halves, to run as two processes)
"""
import sys

import lab
import fam
import families

part, degree = sys.argv[1], int(sys.argv[2])
C, geo, data, c, s = lab.setup(slab=True)
ref2, ref_val = lab.reference(C, data, c, s, 2)
refd, refd_val = lab.reference(C, data, c, s, degree)
dev = C["DEVICE"]

FORMS = {
    "A": [
        ("a_cauchy_cauchy", ("cauchy", "cauchy")),
        ("b_cauchy_expr", ("cauchy", "expr")),
        ("b_gauss_expr", ("gauss", "expr")),
        ("c_cauchy_exp", ("cauchy", "exp")),
        ("c_gauss_exp", ("gauss", "exp")),
        ("d_cauchy_k0", ("cauchy", "k0")),
        ("d_gauss_k0", ("gauss", "k0")),
        ("e_student", ("student",)),
        ("e_cauchy_student", ("cauchy", "student")),
        ("e_student_student", ("student", "student")),
    ],
    "B": [
        ("f_cauchy_cauchy_expr", ("cauchy", "cauchy", "expr")),
        ("f_gauss_cauchy_expr", ("gauss", "cauchy", "expr")),
        ("f_cauchy_exp_expr", ("cauchy", "exp", "expr")),
        ("f_cauchy_cauchy_cauchy", ("cauchy", "cauchy", "cauchy")),
        ("f_gauss_exp_expr", ("gauss", "exp", "expr")),
        ("g_cauchy_ggamma", ("cauchy", "ggamma")),
        ("g_cauchy_sexp", ("cauchy", "sexp")),
        ("g_cauchy_sexpr", ("cauchy", "sexpr")),
        ("g_expr_expr", ("expr", "expr")),
        ("g_exp_exp", ("exp", "exp")),
        ("g_ggamma", ("ggamma",)),
        ("g_ggamma_ggamma", ("ggamma", "ggamma")),
        ("g_expr", ("expr",)),
        ("g_student_expr", ("student", "expr")),
    ],
}
# reduced list (the forms that matter for the ranked table) - used for degree 1
FORMS["C"] = [f for f in FORMS["A"] + FORMS["B"] if f[0] in (
    "a_cauchy_cauchy", "b_cauchy_expr", "b_gauss_expr", "c_cauchy_exp", "d_cauchy_k0", "e_student", "e_cauchy_student",
    "f_gauss_cauchy_expr", "f_cauchy_cauchy_cauchy", "f_gauss_exp_expr", "g_cauchy_sexp", "g_student_expr")]
for tag, comps in FORMS[part]:
    try:
        m = fam.LatMix(C, degree=degree, comps=comps, depth="lin", lnp0=c, lnp_s=s).to(dev)
        lab.warm_long(m, refd)
        ctor = f'fam.LatMix(C, degree={degree}, comps={comps!r}, depth="lin", lnp0=c, lnp_s=s)'
        lab.run(C, data, m, f"{tag}_n{degree}", ref_val, ctor=ctor, refs={f"gct_n{degree}": refd_val})
    except Exception as e:                      # keep the survey going; the failure is logged
        print(f"{tag}_n{degree} FAILED: {e!r}", flush=True)
