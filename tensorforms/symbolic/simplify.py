"""Simplification helpers.

sympy's simplify sometimes misses trig identities such as
sin(2θ)·tan(θ) + cos(2θ) − 1 = 0, so `clean` expands trig functions first,
and `is_zero` falls back to numeric spot checks when every free symbol can
take a value.
"""
import random

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef


def clean(e):
    """Simplify one expression, expanding sin(2θ)-type terms first."""
    e = sp.sympify(e)
    if e.is_number:
        return sp.nsimplify(e) if e.is_Float else e
    s = sp.simplify(e)
    if s == 0 or not s.has(sp.sin, sp.cos, sp.tan, sp.cot):
        return s
    candidates = [s,
                  sp.simplify(sp.expand_trig(s)),
                  sp.trigsimp(sp.expand_trig(s))]
    return min(candidates, key=sp.count_ops)


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
