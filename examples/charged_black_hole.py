"""A charged black hole (Reissner–Nordström): electromagnetism and gravity together.
Run from the project folder:  python examples/charged_black_hole.py

Mass m, charge q, units G = c = 1 (Gaussian). The field comes from a potential, F = d∧A, with
A = −(q/r) dt. Checks Maxwell's equations outside the charge,

    d∧F = 0          (automatic: d∧d∧ = 0)
    d∧⋆F = 0         (no charges or currents out here)

and Einstein's equation G = 8πT with the electromagnetic stress-energy

    T = (1/4π) (F_μλ F_ν^λ − ¼ g F_αβ F^αβ)
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tensorforms import Manifold

t, th, ph = sp.symbols("t theta phi", real=True)
r = sp.Symbol("r", positive=True)
m, q = sp.symbols("m q", positive=True)

M = Manifold([t, r, th, ph], name="ReissnerNordstrom")
f = 1 - 2 * m / r + q ** 2 / r ** 2
M.metric(sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="-+++")

A = M.one_form([-q / r, 0, 0, 0])               # electric potential q/r
F = A.wedge_d()                                  # field strength (μ ν)
print("E_r = F_rt:", F.data[1, 0])
print("d∧F = 0: ", F.wedge_d().is_zero())
print("d∧⋆F = 0:", F.star().wedge_d().simplify().is_zero())

Fv = F.as_value()                                            # (μ↓ ν↓)
FF = Fv.contract(Fv.raise_(1), 1, 1)                        # F_μλ F_ν^λ
F2 = Fv.raise_(0).raise_(1).contract(Fv, 0, 0).tr(0, 1)     # F^αβ F_αβ
T = (FF - F2 * M.g / 4) / (4 * sp.pi)

print("G = 8πT: ", (M.einstein() - 8 * sp.pi * T).simplify().is_zero())
