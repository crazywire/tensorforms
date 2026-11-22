"""The Friedmann equations: an expanding universe filled with a perfect fluid.
Run from the project folder:  python examples/friedmann.py

Einstein's equation G = 8πT (units G = c = 1), with the stress-energy of a perfect fluid at rest,
T = (ρ + p) u u + p g, on the flat FLRW metric −dt² + a(t)²(dr² + r² dΩ²).
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tensorforms as tf
from tensorforms import otimes

U = tf.Manifold.flrw()                          # scale factor a(t), unknown
t = U.coords[0]
rho = sp.Function("rho")(t)                     # energy density
p = sp.Function("p")(t)                         # pressure

u = U.vector([1, 0, 0, 0]).lower()              # the fluid is at rest: u = e_t, lowered (μ↓ |)
T = (rho + p) * otimes(u, u) + p * U.g          # perfect fluid (μ↓ ν↓ |)

E = (U.einstein() - 8 * sp.pi * T).simplify()   # Einstein's equation: E = 0

print("tt:", sp.simplify(E.data[0, 0]), "= 0")
print("rr:", sp.simplify(E.data[1, 1]), "= 0")

# Solve them for the two Friedmann equations
a = sp.Function("a")(t)
H2 = sp.solve(E.data[0, 0], sp.diff(a, t) ** 2)[0]
print("(ȧ/a)² =", sp.simplify(H2 / a ** 2))
add = sp.solve(E.data[1, 1].subs(sp.diff(a, t) ** 2, H2), sp.diff(a, t, 2))[0]
print("ä/a    =", sp.simplify(add / a))
