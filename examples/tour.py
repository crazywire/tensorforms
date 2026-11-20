"""A short tour: manifold, tensors, metric, the four operators, presets.
Run from the project folder:  python examples/tour.py
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tensorforms as tf
from tensorforms import Manifold, contract, curvature, nabla, wedge_nabla

# 1. A bare manifold: coordinates only
th, ph = sp.symbols("theta phi", positive=True)
M = Manifold([th, ph], name="S2")
print(M)

# 2. Indices, then tensors made from them
a, b, c, d = M.indices(4)
w = M.tensor(a.form, data=[sp.cos(th), sp.sin(th) * ph])
print("w =", w, "  d∧w (no metric needed):", wedge_nabla(w).simplify().data.tolist())

# 3. Attach a metric; connection and curvature follow
M.metric([[1, 0], [0, sp.sin(th) ** 2]])
R = M.curvature()
print("R =", R, "  R_θφ block:", R.data[:, :, 0, 1].tolist())
print("Ric =", M.ricci().data.tolist(), "  R_s =", M.scalar_curvature(), "  K =", M.gaussian_curvature())

# 4. The four operators on a vector
v = M.vector([1, 0])                       # e_θ
print("∇v    =", nabla(v), nabla(v).simplify().data.tolist())
print("∇∧∇v  =", curvature(v), "(R acting on v)")

# 5. Explicit pairing by slot name
Ric = R.tr("ρ", "μ")
print("R.tr(ρ, μ) =", Ric, Ric.simplify().data.tolist())

# 6. Presets
for G in (tf.geometry.hyperbolic_plane(), tf.geometry.euclidean(3, "spherical")):
    print(G, " scalar curvature:", G.scalar_curvature())
BH = tf.geometry.schwarzschild()
print(BH, " Ricci is zero (vacuum):", BH.ricci().is_zero())
