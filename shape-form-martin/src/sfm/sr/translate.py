"""sfm.sr.translate - equation strings <-> expression trees <-> NumPy / torch (PLAN.md section 7: "a string -> torch
translator with a round-trip check <= 1e-5; template results have no sympy export").

torch is imported only by ``TorchExpr`` / ``to_torch`` / ``round_trip`` (on demand), so the parser, the printer and the
NumPy evaluation can be used inside a search process (which must never import torch).

Tree
    ``Expr`` (immutable): kind "const" (``value``; ``fixed=True`` marks a structural literal that is neither trained
    nor counted in k), "var" (``name``), "param" (template parameter ``p[i]``), "op" (``name`` in
    ``operators.OPS``, ``args``), "sub" (a call of a template sub-expression inside a combine string; removed by
    ``parse_template``). ``Expr`` satisfies the duck-typed protocol of ``operators.violations`` /
    ``operators.tree_complexity``.

Grammar read by ``parse`` (PySR's output strings, Julia source of the template ``combine`` strings, and the
sympy-style strings of the earlier study, e.g. ``-1.943752e-5/(r - 1*0.017030673)`` and ``(...)**2``)
    expr   := term (('+' | '-') term)*
    term   := unary (('*' | '/') unary)*
    unary  := ('-' | '+') unary | power                  (so -x^2 = -(x^2), as in Julia and Python)
    power  := atom (('^' | '**') unary)?                 (right-associative; the exponent may carry a sign)
    atom   := number | name | name '(' expr (',' expr)* ')' | '(' expr ')' | '#' digits | name '[' digits ']'
    number := 1.5, 1e-05, 1.5e+3, .5, 2., 0.3f0 (Float32 literal), Inf, NaN
    A sign in front of a numeric literal folds into the constant (``x * -1.5`` has one constant), any other sign is
    a ``neg`` node. ``u^2`` with the literal 2 is ``square(u)``; another literal integer exponent is ``pow`` with a
    *fixed* exponent and integer-power semantics (any base; a negative exponent needs base != 0); every other
    exponent is a real power (base > 0 required - the audit enforces it).

What PySR 2.7.3 prints (confirmed by tiny fits on 25 Nov 2026, see STATUS.md section 9)
    free-form     infix ``+ - * /`` with parentheses around every binary sub-expression that is itself an argument of
                  a binary operator, call syntax for everything else, e.g. ``tanh((zt * 1.3) + -0.4) * 0.5``
    template      ``u = <expr in #1, #2>; v = ...; w = ...`` (``#k`` = the k-th argument of that sub-expression in
                  the combine string, ``TemplateSpec.arguments``)
    constants     ``print_precision`` significant digits (17 for Float64 searches: exact round trip)

Free constants and their order
    ``constants(expr)`` lists the values of all non-fixed constants in depth-first, left-to-right order; the same
    order indexes ``TorchExpr.c``. k = ``n_constants(expr)``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import operators as O

__all__ = ["Expr", "ParseError", "parse", "parse_template", "parse_template_string", "split_template_string",
           "to_string", "to_dict", "from_dict", "evaluate", "constants", "n_constants", "with_constants",
           "variables", "substitute", "complexity", "walk", "const", "var", "op", "TorchExpr", "to_torch",
           "round_trip", "ROUND_TRIP_TOL", "horner_string", "expr_hash"]

ROUND_TRIP_TOL = 1e-5                # PLAN section 7


class ParseError(ValueError):
    """The string is not in the grammar of this module, or names an unknown operator."""


# ----------------------------------------------------------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Expr:
    kind: str                                  # "const" | "var" | "param" | "op" | "sub"
    name: str = ""
    args: Tuple["Expr", ...] = ()
    value: Optional[float] = None              # constants only
    fixed: bool = False                        # structural literal: not trained, not counted in k

    def __post_init__(self):
        if self.kind not in ("const", "var", "param", "op", "sub"):
            raise ValueError(f"unknown node kind {self.kind!r}")
        if self.kind == "op":
            spec = O.OPS.get(self.name)
            if spec is None:
                raise ParseError(f"unknown operator {self.name!r}")
            if len(self.args) != spec.arity:
                raise ParseError(f"operator {self.name!r} takes {spec.arity} argument(s), got {len(self.args)}")
        if self.kind == "const" and self.value is None:
            raise ValueError("a constant needs a value")

    def __str__(self) -> str:
        return to_string(self)


def const(value: float, fixed: bool = False) -> Expr:
    return Expr("const", value=float(value), fixed=bool(fixed))


def var(name: str) -> Expr:
    return Expr("var", name=str(name))


def op(name: str, *args: Expr) -> Expr:
    return Expr("op", name=name, args=tuple(args))


def walk(e: Expr):
    """All nodes, depth first, left to right (the order of ``constants``)."""
    yield e
    for a in e.args:
        yield from walk(a)


def constants(e: Expr) -> List[float]:
    """Values of the free (non-fixed) constants, depth first, left to right."""
    return [float(n.value) for n in walk(e) if n.kind == "const" and not n.fixed]


def n_constants(e: Expr) -> int:
    """k: the number of free constants."""
    return sum(1 for n in walk(e) if n.kind == "const" and not n.fixed)


def variables(e: Expr) -> List[str]:
    """Variable names in order of first appearance."""
    out: List[str] = []
    for n in walk(e):
        if n.kind == "var" and n.name not in out:
            out.append(n.name)
    return out


def with_constants(e: Expr, values: Sequence[float]) -> Expr:
    """The same tree with its free constants replaced (order of ``constants``). Used after a refit, so that the
    refitted equation can be printed, hashed and - mandatory - audited again."""
    vals = [float(v) for v in values]
    if len(vals) != n_constants(e):
        raise ValueError(f"{len(vals)} values for {n_constants(e)} free constants")
    it = iter(vals)

    def rec(n: Expr) -> Expr:
        if n.kind == "const":
            return n if n.fixed else Expr("const", value=next(it))
        if not n.args:
            return n
        return Expr(n.kind, n.name, tuple(rec(a) for a in n.args), n.value, n.fixed)
    return rec(e)


def substitute(e: Expr, mapping: Mapping[str, Expr]) -> Expr:
    """Replace variables (and ``#k`` placeholders, which are variables named "#k") by expressions."""
    def rec(n: Expr) -> Expr:
        if n.kind == "var" and n.name in mapping:
            return mapping[n.name]
        if not n.args:
            return n
        return Expr(n.kind, n.name, tuple(rec(a) for a in n.args), n.value, n.fixed)
    return rec(e)


def complexity(e: Expr) -> int:
    """Operator-weighted size C with the search's weights (``operators.tree_complexity``). Fixed literals count as
    constants here: this is the size of the tree as written."""
    return O.tree_complexity(e)


# ----------------------------------------------------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"""\s*(?:
      (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eEf][+-]?\d+)?)
    | (?P<name>[^\W\d]\w*)
    | (?P<ph>\#\d+)
    | (?P<sym>\*\*|[-+*/^(),\[\]])
    )""", re.VERBOSE)
_ALIASES = {"Abs": "abs", "arctan": "atan", "ln": "log", "pow": "pow", "sqrt_abs": "sqa", "log1p_abs": "l1p"}
_CONST_NAMES = {"Inf": math.inf, "inf": math.inf, "NaN": math.nan, "nan": math.nan, "Inf32": math.inf,
                "NaN32": math.nan}
_INFIX = {"+": "+", "-": "-", "*": "*", "/": "/"}


def _tokenize(s: str) -> List[Tuple[str, str]]:
    out, pos, s = [], 0, s.rstrip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if m is None or m.end() == pos:
            raise ParseError(f"cannot read {s[pos:pos + 20]!r} at position {pos} of {s!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def _number(tok: str) -> float:
    if "f" in tok:                                       # Julia Float32 literal: 0.3f0, 1f-5
        mant, ex = tok.split("f")
        return float(np.float32(float(mant) * 10.0 ** int(ex)))
    return float(tok)


class _Parser:
    def __init__(self, s: str, subs: Iterable[str] = (), fixed_literals: bool = False):
        self.src, self.toks, self.i = s, _tokenize(s), 0
        self.subs, self.fixed_literals = set(subs), fixed_literals

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, sym=None):
        k, v = self.peek()
        if k is None or (sym is not None and v != sym):
            raise ParseError(f"expected {sym!r} at token {self.i} of {self.src!r}, found {v!r}")
        self.i += 1
        return k, v

    def expr(self) -> Expr:
        left = self.term()
        while self.peek()[1] in ("+", "-") and self.peek()[0] == "sym":
            _, s = self.take()
            left = op(s, left, self.term())
        return left

    def term(self) -> Expr:
        left = self.unary()
        while self.peek()[1] in ("*", "/") and self.peek()[0] == "sym":
            _, s = self.take()
            left = op(s, left, self.unary())
        return left

    def unary(self) -> Expr:
        k, v = self.peek()
        if k == "sym" and v in ("-", "+"):
            self.take()
            inner = self.unary()
            if v == "+":
                return inner
            if inner.kind == "const":                    # sign folds into the literal
                return Expr("const", value=-inner.value, fixed=inner.fixed)
            return op("neg", inner)
        return self.power()

    def power(self) -> Expr:
        base = self.atom()
        k, v = self.peek()
        if k == "sym" and v in ("^", "**"):
            self.take()
            ex = self.unary()
            if ex.kind == "const" and float(ex.value).is_integer() and abs(ex.value) < 2 ** 31:
                if ex.value == 2.0:
                    return op("square", base)
                return op("pow", base, Expr("const", value=float(ex.value), fixed=True))
            return op("pow", base, ex)
        return base

    def atom(self) -> Expr:
        k, v = self.take()
        if k == "num":
            return Expr("const", value=_number(v), fixed=self.fixed_literals)
        if k == "ph":
            return var(v)
        if k == "sym" and v == "(":
            e = self.expr()
            self.take(")")
            return e
        if k == "name":
            nk, nv = self.peek()
            if nk == "sym" and nv == "(":
                self.take("(")
                args = [self.expr()]
                while self.peek()[1] == ",":
                    self.take(",")
                    args.append(self.expr())
                self.take(")")
                if v in self.subs:
                    return Expr("sub", name=v, args=tuple(args))
                name = _ALIASES.get(v, v)
                if name not in O.OPS or name in _INFIX:
                    raise ParseError(f"unknown function {v!r} in {self.src!r}")
                if name == "pow" and len(args) == 2:
                    ex = args[1]
                    if ex.kind == "const" and float(ex.value).is_integer():
                        if ex.value == 2.0:
                            return op("square", args[0])
                        return op("pow", args[0], Expr("const", value=float(ex.value), fixed=True))
                return op(name, *args)
            if nk == "sym" and nv == "[":
                self.take("[")
                _, idx = self.take()
                self.take("]")
                return Expr("param", name=f"{v}[{int(idx)}]")
            if v in _CONST_NAMES:
                return Expr("const", value=_CONST_NAMES[v], fixed=self.fixed_literals)
            return var(v)
        raise ParseError(f"unexpected {v!r} in {self.src!r}")


def parse(s: str, variable_names: Optional[Sequence[str]] = None, fixed_literals: bool = False) -> Expr:
    """Parse an equation string into a tree.

    variable_names   if given, any other variable raises (``#k`` placeholders are always allowed: they are the
                     arguments of a template sub-expression)
    fixed_literals   mark every numeric literal as structural (used for template combine strings)
    """
    if not isinstance(s, str) or not s.strip():
        raise ParseError("empty equation string")
    p = _Parser(s, fixed_literals=fixed_literals)
    e = p.expr()
    if p.i != len(p.toks):
        raise ParseError(f"trailing input {p.toks[p.i][1]!r} in {s!r}")
    if variable_names is not None:
        bad = [v for v in variables(e) if v not in variable_names and not v.startswith("#")]
        if bad:
            raise ParseError(f"unknown variable(s) {bad} in {s!r}; known: {list(variable_names)}")
    return e


def split_template_string(eq: str) -> Dict[str, str]:
    """``"u = ...; v = ...; p = [1.0, 2.0]"`` -> {name: text}. Raises if a part has no ``name =``."""
    parts: Dict[str, str] = {}
    for chunk in eq.split(";"):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ParseError(f"template part without 'name =': {chunk!r}")
        k, v = chunk.split("=", 1)
        parts[k.strip()] = v.strip()
    return parts


def parse_template(template: Union[str, "O.TemplateSpec"], subexpressions: Mapping[str, Union[str, Expr]],
                   keep_parts: bool = False):
    """Translate a template result into one tree in the data variables.

    template         a key of ``operators.TEMPLATES`` or a ``TemplateSpec``
    subexpressions   {name: string as PySR prints it (variables ``#1, #2, ...``) or an already parsed tree}

    The combine string is parsed with its numeric literals marked *fixed* (the 0.3 cap of the modulation is
    structural: not trained, not in k); each call ``u(zt, x)`` is replaced by the sub-expression with ``#k`` -> the
    k-th argument of the call. Returns the tree, or (tree, {name: parsed sub-expression}) with ``keep_parts``.
    """
    spec = O.TEMPLATES[template] if isinstance(template, str) else template
    missing = [n for n in spec.expressions if n not in subexpressions]
    if missing:
        raise ParseError(f"template {spec.name}: sub-expression(s) {missing} missing")
    parts: Dict[str, Expr] = {}
    for n in spec.expressions:
        s = subexpressions[n]
        e = s if isinstance(s, Expr) else parse(s)
        nargs = len(spec.arguments[n])
        bad = [v for v in variables(e) if not (v.startswith("#") and 1 <= int(v[1:]) <= nargs)]
        if bad:
            raise ParseError(f"template {spec.name}: sub-expression {n!r} uses {bad}; expected #1..#{nargs}")
        parts[n] = e
    p = _Parser(spec.combine64, subs=spec.expressions, fixed_literals=True)
    comb = p.expr()
    if p.i != len(p.toks):
        raise ParseError(f"template {spec.name}: cannot read the combine string {spec.combine64!r}")

    def rec(n: Expr) -> Expr:
        if n.kind == "sub":
            args = [rec(a) for a in n.args]
            want = spec.arguments[n.name]
            got = tuple(a.name if a.kind == "var" else None for a in args)
            if got != tuple(want):
                raise ParseError(f"template {spec.name}: {n.name} is called with {got}, TemplateSpec.arguments "
                                 f"says {tuple(want)}")
            return substitute(parts[n.name], {f"#{k + 1}": a for k, a in enumerate(args)})
        if not n.args:
            return n
        return Expr(n.kind, n.name, tuple(rec(a) for a in n.args), n.value, n.fixed)
    tree = rec(comb)
    bad = [v for v in variables(tree) if v not in spec.variable_names]
    if bad:
        raise ParseError(f"template {spec.name}: unresolved variable(s) {bad}")
    return (tree, parts) if keep_parts else tree


def parse_template_string(template: Union[str, "O.TemplateSpec"], equation: str, keep_parts: bool = False):
    """``parse_template`` from the single string PySR writes into ``hall_of_fame.csv`` (``u = ...; v = ...``)."""
    spec = O.TEMPLATES[template] if isinstance(template, str) else template
    parts = split_template_string(equation)
    return parse_template(spec, {n: parts[n] for n in spec.expressions if n in parts}, keep_parts=keep_parts)


# ----------------------------------------------------------------------------------------------------------------------
# Printer, lossless (de)serialisation
# ----------------------------------------------------------------------------------------------------------------------
def _fmt(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return repr(float(v))                                # shortest string that round-trips


def to_string(e: Expr) -> str:
    """Julia / PySR-compatible infix string: every binary sub-expression in parentheses, constants with
    ``repr(float)`` (exact round trip). ``parse(to_string(e))`` reproduces the tree except for the ``fixed`` marks
    of template literals - use ``to_dict`` when those matter (hand-over to the refit)."""
    def rec(n: Expr, top: bool) -> str:
        if n.kind == "const":
            return _fmt(n.value)
        if n.kind in ("var", "param"):
            return n.name
        if n.kind == "sub":
            return f"{n.name}({', '.join(rec(a, True) for a in n.args)})"
        if n.name in _INFIX:
            s = f"{rec(n.args[0], False)} {n.name} {rec(n.args[1], False)}"
            return s if top else f"({s})"
        if n.name == "pow":
            b, ex = rec(n.args[0], False), n.args[1]
            if b.startswith("-"):
                b = f"({b})"
            exs = str(int(ex.value)) if (ex.kind == "const" and ex.fixed) else rec(ex, False)
            if exs.startswith("-"):
                exs = f"({exs})"
            s = f"{b} ^ {exs}"
            return s if top else f"({s})"
        return f"{n.name}({', '.join(rec(a, True) for a in n.args)})"
    return rec(e, True)


def to_dict(e: Expr) -> dict:
    """Lossless JSON form (keeps ``fixed``)."""
    if e.kind == "const":
        d = {"k": "const", "v": (e.value if math.isfinite(e.value) else _fmt(e.value))}
        if e.fixed:
            d["fixed"] = True
        return d
    if e.kind in ("var", "param"):
        return {"k": e.kind, "n": e.name}
    return {"k": e.kind, "n": e.name, "a": [to_dict(a) for a in e.args]}


def from_dict(d: Mapping) -> Expr:
    k = d["k"]
    if k == "const":
        v = d["v"]
        if isinstance(v, str):
            v = {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}[v]
        return Expr("const", value=float(v), fixed=bool(d.get("fixed", False)))
    if k in ("var", "param"):
        return Expr(k, name=d["n"])
    return Expr(k, name=d["n"], args=tuple(from_dict(a) for a in d["a"]))


def expr_hash(e: Expr) -> str:
    """SHA-256 (first 16 hex digits) of the lossless form: identifies an equation in the hand-over files."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(to_dict(e), sort_keys=True).encode()).hexdigest()[:16]


def horner_string(coefficients: Sequence[float], variable: str = "x") -> str:
    """A polynomial c0 + c1 x + ... + cn x^n in Horner form, as a ``guesses`` string for PySR
    (a cubic has complexity 13 and k = 4): ``((c3 * x + c2) * x + c1) * x + c0``."""
    c = [float(v) for v in coefficients]
    if not c:
        raise ValueError("no coefficients")
    s = _fmt(c[-1])
    for v in reversed(c[:-1]):
        s = f"({s}) * {variable} + {_fmt(v)}" if " " in s else f"{s} * {variable} + {_fmt(v)}"
    return s


# ----------------------------------------------------------------------------------------------------------------------
# NumPy evaluation
# ----------------------------------------------------------------------------------------------------------------------
def _is_int_pow(n: Expr) -> bool:
    return n.kind == "op" and n.name == "pow" and n.args[1].kind == "const" and n.args[1].fixed


def evaluate(e: Expr, env: Optional[Mapping[str, object]] = None, record: Optional[dict] = None, **named):
    """Float64 NumPy value of the tree. ``env`` maps variable names to arrays (broadcast against each other);
    keyword arguments are added to it. No error state is raised: overflow gives +-Inf, a pole Inf or NaN, exactly
    as the formulas say. With ``record`` (a dict) the largest finite |value| of any node is stored in
    ``record["max_abs"]`` and ``record["nonfinite"]`` says whether any node was not finite."""
    envd: Dict[str, object] = dict(env or {})
    envd.update(named)
    impl = O.numpy_impl()

    def note(v):
        if record is not None:
            a = np.asarray(v, dtype=np.float64)
            fin = np.isfinite(a)
            if not fin.all():
                record["nonfinite"] = True
            if fin.any():
                record["max_abs"] = max(record.get("max_abs", 0.0), float(np.max(np.abs(a[fin]))))
        return v

    def rec(n: Expr):
        if n.kind == "const":
            return note(np.float64(n.value))
        if n.kind == "var":
            if n.name not in envd:
                raise KeyError(f"no value for variable {n.name!r}")
            return note(np.asarray(envd[n.name], dtype=np.float64))
        if n.kind != "op":
            raise ValueError(f"cannot evaluate a {n.kind!r} node ({n.name}); resolve templates / parameters first")
        if _is_int_pow(n):
            return note(np.power(rec(n.args[0]), int(n.args[1].value)) if n.args[1].value >= 0
                        else 1.0 / np.power(rec(n.args[0]), int(-n.args[1].value)))
        return note(impl[n.name](*[rec(a) for a in n.args]))

    if record is not None:
        record.setdefault("max_abs", 0.0)
        record.setdefault("nonfinite", False)
    with np.errstate(all="ignore"):
        out = rec(e)
    shape = np.broadcast_shapes(*[np.shape(envd[v]) for v in variables(e)]) if variables(e) else ()
    return np.broadcast_to(np.asarray(out, dtype=np.float64), shape).copy() if shape else np.float64(out)


# ----------------------------------------------------------------------------------------------------------------------
# torch
# ----------------------------------------------------------------------------------------------------------------------
def _torch_module_class():
    import torch
    import torch.nn as nn

    impl = O.torch_impl()

    class _TorchExpr(nn.Module):
        """An expression tree as a torch module. All k free constants are one ``nn.Parameter`` ``c`` (order of
        ``constants``); fixed literals are plain numbers. No clamp, no ``nan_to_num``: the value is the formula.

            m = TorchExpr(expr, ("zt", "rh", "x"))
            y = m(zt, rh, x)                 # positional, in the order of variable_names; or m(env_dict)
            m.current_expr()                 # the tree with the present values of the constants
        """

        def __init__(self, expr: Expr, variable_names: Sequence[str] = O.VARIABLE_NAMES, dtype=None,
                     trainable: bool = True):
            super().__init__()
            bad = [n.name for n in walk(expr) if n.kind not in ("const", "var", "op")]
            if bad:
                raise ValueError(f"unresolved nodes {bad}: translate the template / parameters first")
            unknown = [v for v in variables(expr) if v not in variable_names]
            if unknown:
                raise ValueError(f"expression uses {unknown}, variable_names are {list(variable_names)}")
            self.expr = expr
            self.variable_names = tuple(variable_names)
            dtype = torch.float64 if dtype is None else dtype
            self.c = nn.Parameter(torch.tensor(constants(expr), dtype=dtype), requires_grad=bool(trainable))
            counter = [0]
            self._fn = self._compile(expr, counter)

        def _compile(self, n: Expr, counter) -> Callable:
            if n.kind == "const":
                if n.fixed:
                    v = float(n.value)
                    return lambda env, c: v
                i = counter[0]
                counter[0] += 1
                return lambda env, c: c[i]
            if n.kind == "var":
                name = n.name
                return lambda env, c: env[name]
            if _is_int_pow(n):
                base, k = self._compile(n.args[0], counter), int(n.args[1].value)
                if k >= 0:
                    return lambda env, c: torch.pow(base(env, c), k)
                return lambda env, c: 1.0 / torch.pow(base(env, c), -k)
            f = impl[n.name]
            subs = [self._compile(a, counter) for a in n.args]
            if len(subs) == 1:
                g = subs[0]
                return lambda env, c: f(_as_tensor(g(env, c), c))
            g, h = subs
            return lambda env, c: f(_as_tensor(g(env, c), c), _as_tensor(h(env, c), c))

        def forward(self, *cols, **named):
            if len(cols) == 1 and isinstance(cols[0], Mapping):
                env = dict(cols[0])
            else:
                if len(cols) > len(self.variable_names):
                    raise ValueError(f"{len(cols)} inputs for variables {self.variable_names}")
                env = dict(zip(self.variable_names, cols))
            env.update(named)
            used = variables(self.expr)
            missing = [v for v in used if v not in env]
            if missing:
                raise KeyError(f"no value for variable(s) {missing}")
            env = {k: (v.to(self.c.dtype) if torch.is_tensor(v) else
                       torch.as_tensor(v, dtype=self.c.dtype, device=self.c.device)) for k, v in env.items()
                   if k in used}
            out = _as_tensor(self._fn(env, self.c), self.c)
            shape = torch.broadcast_shapes(*[env[v].shape for v in used]) if used else ()
            return out.expand(shape) if tuple(out.shape) != tuple(shape) else out

        def n_constants(self) -> int:
            return int(self.c.numel())

        def n_params(self) -> int:
            return int(self.c.numel())

        def current_expr(self) -> Expr:
            return with_constants(self.expr, self.c.detach().cpu().double().tolist())

        def extra_repr(self) -> str:
            return f"k={self.c.numel()}, C={complexity(self.expr)}, {to_string(self.current_expr())[:120]}"

    def _as_tensor(v, like):
        return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=like.dtype, device=like.device)

    return _TorchExpr


_TORCH_CLASS = None


def TorchExpr(expr: Expr, variable_names: Sequence[str] = O.VARIABLE_NAMES, dtype=None, trainable: bool = True):
    """Build the torch module of ``expr`` (see the class docstring inside ``_torch_module_class``): all free
    constants trainable in one parameter vector ``c``; ``forward(*columns)`` or ``forward(env)``;
    ``current_expr()`` returns the tree with the present constants. torch is imported here."""
    global _TORCH_CLASS
    if _TORCH_CLASS is None:
        _TORCH_CLASS = _torch_module_class()
    return _TORCH_CLASS(expr, variable_names, dtype, trainable)


to_torch = TorchExpr


def round_trip(e: Expr, X, variable_names: Sequence[str] = O.VARIABLE_NAMES, reference=None,
               tol: float = ROUND_TRIP_TOL, dtype=None) -> dict:
    """The translator's round-trip check (PLAN section 7, tolerance 1e-5) on the rows ``X`` (N, d):

      1. string round trip: ``parse(to_string(e))`` has the same lossless form up to the fixed marks, and the
         same values on X exactly;
      2. NumPy (float64) against the torch module, max |difference|;
      3. if ``reference`` is given (the search's own prediction on the same rows, exported by the runner),
         max |torch - reference|.

    ``passed`` requires every available difference <= ``tol`` and finite values. With ``dtype=torch.float32`` the
    torch side runs in single precision, as a refit in float32 would.
    """
    import torch

    X = np.asarray(X, dtype=np.float64)
    env = {n: X[:, i] for i, n in enumerate(variable_names)}
    v_np = np.asarray(evaluate(e, env), dtype=np.float64) + np.zeros(len(X))
    e2 = parse(to_string(e))
    v_np2 = np.asarray(evaluate(e2, env), dtype=np.float64) + np.zeros(len(X))
    m = TorchExpr(e, variable_names, dtype=dtype)
    m.eval()
    with torch.inference_mode():
        cols = [torch.as_tensor(X[:, i]) for i in range(X.shape[1])]
        v_t = m(*cols[:len(variable_names)]).double().numpy() + np.zeros(len(X))
    finite = bool(np.all(np.isfinite(v_np)) and np.all(np.isfinite(v_t)))

    def md(a, b):
        with np.errstate(invalid="ignore"):
            d = np.abs(a - b)
        return float(np.nanmax(d)) if np.isfinite(d).any() else math.inf

    out = {"n": int(len(X)), "finite": finite, "string_max_diff": md(v_np, v_np2), "numpy_torch_max_diff": md(v_np, v_t),
           "tol": float(tol)}
    ok = finite and out["string_max_diff"] <= tol and out["numpy_torch_max_diff"] <= tol
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        if ref.shape != v_t.shape:
            raise ValueError(f"reference has shape {ref.shape}, expected {v_t.shape}")
        out["reference_max_diff"] = md(v_t, ref)
        ok = ok and out["reference_max_diff"] <= tol
    out["passed"] = bool(ok)
    return out
