"""Simplification helpers.

`clean` first tries a direct recipe: write sin and cos as variables s, c, cancel the
fraction exactly, and use s² = 1 − c². That is fast and gives a true zero as exactly 0
(e.g. sin(2θ)·tan(θ) + cos(2θ) − 1 → 0). If the recipe can't apply, it falls back to
sympy's general simplify. `is_zero` also falls back to numeric spot checks.
"""
import random

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef


def _trig_rational(e):
    """The direct recipe for fractions in sin, cos and anything else:
    write sin x = s_x, cos x = c_x, cancel (an exact standard form), then use s_x² = 1 − c_x².
    No searching, so it is fast; a true zero always comes out as exactly 0."""
    e = sp.expand_trig(e)
    as_sin_cos = {sp.tan: lambda a: sp.sin(a) / sp.cos(a), sp.cot: lambda a: sp.cos(a) / sp.sin(a),
                  sp.sec: lambda a: 1 / sp.cos(a), sp.csc: lambda a: 1 / sp.sin(a)}
    e = e.replace(lambda x: type(x) in as_sin_cos, lambda x: as_sin_cos[type(x)](x.args[0]))
    args = sorted({f.args[0] for f in e.atoms(sp.sin, sp.cos)}, key=str)
    sub, back, ideal, gens = {}, {}, [], []
    for i, a in enumerate(args):
        S, C = sp.Dummy(f"s{i}"), sp.Dummy(f"c{i}")
        sub[sp.sin(a)], sub[sp.cos(a)] = S, C
        back[S], back[C] = sp.sin(a), sp.cos(a)
        ideal.append(S ** 2 + C ** 2 - 1)
        gens += [S, C]
    e = sp.cancel(sp.together(e.xreplace(sub)))
    if not ideal:
        return sp.factor(e)
    n, d = sp.fraction(e)
    results = []
    for order in (gens, gens[::-1]):            # replace sin² by 1 − cos², or cos² by 1 − sin²
        rn = sp.reduced(sp.expand(n), ideal, *order)[1]
        rd = sp.reduced(sp.expand(d), ideal, *order)[1]
        results.append(sp.factor(sp.cancel(rn / rd)).xreplace(back))
    return min(results, key=sp.count_ops)


def _general(e):
    """The old method: sympy's general simplify, plus trig expansions; keep the shortest."""
    s = sp.simplify(e)
    if s == 0 or not s.has(sp.sin, sp.cos, sp.tan, sp.cot):
        return s
    candidates = [s,
                  sp.simplify(sp.expand_trig(s)),
                  sp.trigsimp(sp.expand_trig(s))]
    return min(candidates, key=sp.count_ops)


def clean(e):
    """Simplify one expression: the direct fraction recipe first; if it can't apply
    (or fails), sympy's general simplify."""
    e = sp.sympify(e)
    if e.is_number:
        return sp.nsimplify(e) if e.is_Float else e
    try:
        return _trig_rational(e)
    except (sp.PolynomialError, sp.GeneratorsNeeded, TypeError, ValueError):
        return _general(e)


def is_zero(e, spot_checks=4):
    """True if the expression is identically zero."""
    c = clean(e)
    if c == 0:
        return True
    c = sp.sympify(c)
    if c.atoms(AppliedUndef):
        return False                       # unknown functions: can't spot check
    syms = sorted(c.free_symbols, key=str)
    if not syms:
        return False
    for _ in range(spot_checks):
        vals = {s: sp.Float(random.uniform(0.3, 1.2)) for s in syms}
        v = complex(c.subs(vals).evalf())
        if abs(v) > 1e-9:
            return False
    return True


clean_array = np.vectorize(clean, otypes=[object])
