"""A short tour: manifold, tensors, metric, the operators, products, presets.
Run from the project folder:  python examples/tour.py
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tensorforms as tf
from tensorforms import Manifold, contract, curvature, nabla, wedge, wedge_d

# 1. A bare manifold: coordinates only
th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")
print(M)

# 2. Tensors; d∧ needs no metric
w = M.one_form([sp.cos(th), sp.sin(th) * ph])
print(wedge_d(w), wedge_d(w).simplify().data.tolist())

# 3. Attach a metric; Γ and R follow
M.metric([[1, 0], [0, sp.sin(th) ** 2]])
G, R = M.levi_civita(), M.riemann()
print(G, R, " R = d∧Γ + Γ∧Γ:", (wedge_d(G) + wedge(G, G)).equals(R))
print("Ric =", M.ricci().data.tolist(), "  R_s =", M.ricci_scalar(), "  K =", M.gaussian_curvature())

# 4. The operators on a vector
v = M.vector([1, 0])                       # e_θ
print(nabla(v), nabla(v).simplify().data.tolist())
print(curvature(v), "= R acting on v")

# 5. Products and traces
print(contract(R, v), "   ", R.tr("ρ", "μ"), R.tr("ρ", "μ").simplify().data.tolist())

# 6. Presets
for P in (tf.geometry.hyperbolic_plane(), tf.geometry.euclidean(coords="spherical")):
    print(P, " scalar curvature:", P.ricci_scalar())
BH = tf.geometry.schwarzschild()
print(BH, " Ricci is zero (vacuum):", BH.ricci().is_zero())
