"""Simplification, done entirely with sympy's own tools.

`clean` (for display) uses sympy's trig algorithm `fu`, then sympy's `simplify` on its result,
keeping the shorter: fu gives readable trig, simplify finishes hyperbolic and exponential parts,
in any mix. `is_zero` (exact) also rewrites everything
as exponentials and cancels: for expressions built from sin, cos, sinh, cosh, exp, … that
decides zero exactly. Every step rewrites an expression into an equal one; nothing is guessed.
"""
import numpy as np
import sympy as sp
from sympy.simplify.fu import fu


def clean(e):
    """Simplify one expression for display, with sympy: fu (readable trig), then simplify
    on its result (hyperbolic and exponential parts, in any mix), keeping the shorter; both
    are equal, and a tie keeps fu's form. Numbers stay as given (a decimal is not turned into
    a fraction)."""
    e = sp.sympify(e)
    if e.is_number:
        return e
    try:
        r = fu(e)
    except Exception:                                    # fu can't handle it: sympy's general simplify
        return sp.simplify(e)
    if r == 0:
        return r
    return min([r, sp.simplify(r)], key=sp.count_ops)


def is_zero(e):
    """Is the expression zero? Sympy decides, exactly:
    True   it simplifies to 0, or written as exponentials it cancels to 0;
    False  sympy proves it is not 0 (a nonzero number, or from the symbols' assumptions);
    None   sympy can't tell.
    Never decided by numerical sampling."""
    c = sp.sympify(clean(e))
    if c == 0:
        return True
    if sp.cancel(sp.powsimp(sp.expand(c.rewrite(sp.exp)))) == 0:
        return True                                      # exact: as exponentials it cancels to 0
    return c.is_zero


clean_array = np.vectorize(clean, otypes=[object])
