"""Checks every result we derived by hand. Run: python -m pytest tests  (or: python tests/test_core.py)."""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorforms as tf
from tensorforms import (Manifold, contract, curvature, curvature_by_R, nabla, product, tr,
                         wedge, wedge_curvature, wedge_nabla)
from tensorforms.symbolic import is_zero


# ---------------------------------------------------------------- sphere
def test_sphere_connection():
    S = tf.geometry.sphere(r=1)
    G = S.connection().Gamma
    th = S.coords[0]
    assert is_zero(G[0, 1, 1] + sp.sin(th) * sp.cos(th))     # Γ^θ_φφ = −sinθ cosθ
    assert is_zero(G[1, 0, 1] - sp.cot(th))                  # Γ^φ_θφ = cotθ
    assert is_zero(G[1, 1, 0] - sp.cot(th))                  # S-symmetric


def test_sphere_curvature():
    S = tf.geometry.sphere(r=1)
    th = S.coords[0]
    R = S.curvature()
    block = R.data[:, :, 0, 1]                               # R_θφ = [0, sin²θ; −1, 0]
    assert is_zero(block[0, 1] - sp.sin(th) ** 2) and is_zero(block[1, 0] + 1)
    assert S.ricci().equals(S.g.rename(dict(zip(S.g.names, S.ricci().names))))
    assert is_zero(S.scalar_curvature() - 2)
    assert is_zero(S.gaussian_curvature() - 1)


def test_sphere_symbolic_radius():
    S = tf.geometry.sphere()                                 # r symbolic
    r = sp.Symbol("r", positive=True)
    assert is_zero(S.gaussian_curvature() - 1 / r ** 2)


def test_nabla_g_is_zero():
    S = tf.geometry.sphere(r=1)
    assert nabla(S.g).is_zero()


# ---------------------------------------------------------------- the four operators
def _sphere_fields():
    S = tf.geometry.sphere(r=1)
    th, ph = S.coords
    f1, f2 = sp.Function("v1")(th, ph), sp.Function("v2")(th, ph)
    return S, S.vector([f1, f2]), S.covector([f1, f2])


def test_curvature_operator_is_R_on_every_slot():
    S, v, w = _sphere_fields()
    assert curvature(v).equals(curvature_by_R(v))
    assert curvature(w).equals(curvature_by_R(w))
    assert not curvature(w).is_zero()                        # −Rᵀw, not zero


def test_curl_needs_no_connection():
    S, v, w = _sphere_fields()
    assert wedge_nabla(w).equals(wedge_nabla(w, conn=False))  # ∇∧w = d∧w


def test_wedge_curvature():
    S, v, w = _sphere_fields()
    assert wedge_curvature(w).is_zero()                       # covector: no value slots
    assert wedge_curvature(v).equals(curvature(v))            # vector: same as the commutator


def test_scalar_has_no_curvature():
    S = tf.geometry.sphere(r=1)
    f = S.scalar(sp.Function("f")(*S.coords))
    assert curvature(f).is_zero()                             # Hessian is symmetric


def test_bianchi_in_3d():
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z])
    M.use_metric(M.metric(sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2)))
    assert wedge_nabla(M.curvature()).is_zero()               # ∇∧R = 0
    u = M.vector([y, x * z, 1])
    assert curvature(u).equals(curvature_by_R(u))


def test_product_rule():
    S = tf.geometry.sphere(r=1)
    th, ph = S.coords
    a, b = S.indices(2)
    A = S.map([[th, ph], [th * ph, 1]], indices=(a, b))
    v = S.vector([sp.sin(th), ph], index=b)
    lhs = nabla(contract(A, v, b))
    rhs = contract(nabla(A), v, b) + contract(A, nabla(v), b)
    assert lhs.equals(rhs)


# ---------------------------------------------------------------- flat space and spacetime
def test_spherical_coordinates_are_flat():
    E = tf.geometry.euclidean(3, coords="spherical")
    assert E.curvature().is_zero()
    assert not np.all(E.connection().Gamma == 0)              # the grid bends, the space doesn't


def test_hyperbolic_plane():
    H = tf.geometry.hyperbolic_plane()
    assert is_zero(H.gaussian_curvature() + 1)


def test_schwarzschild_is_vacuum():
    BH = tf.geometry.schwarzschild()
    assert BH.ricci().is_zero()
    assert BH.einstein().is_zero()


def test_torsion_free():
    S = tf.geometry.sphere(r=1)
    assert S.connection().torsion().is_zero()


# ---------------------------------------------------------------- algebra and types
def test_alt_sym_rules():
    S = tf.geometry.sphere(r=1)
    th, ph = S.coords
    a, b = S.indices(2)
    X = S.tensor(a.form, b.form, data=[[th, ph], [1, th * ph]])
    assert X.alt().alt().equals(X.alt())                      # projections
    assert X.alt().sym().is_zero()                            # Sym Alt = 0
    assert (X.alt() + X.sym()).equals(X)                      # two slots: X = Sym + Alt


def test_wedge_single_factor_is_one_term():
    S = tf.geometry.sphere(r=1)
    R = S.curvature()
    v = S.vector([1, 0])
    assert wedge(R, v).equals(product(R, v))                  # R∧v = Rv


def test_ricci_by_names():
    S = tf.geometry.sphere(r=1)
    R = S.curvature()
    Ric = tr(R, "ρ", "μ").with_roles("value")
    assert Ric.equals(S.ricci())


def test_type_errors():
    S = tf.geometry.sphere(r=1)
    v, w = S.vector([1, 0]), S.covector([1, 0])
    for bad in (lambda: v + w,
                lambda: contract(S.g, S.g, "μ", "μ"),         # down with down
                lambda: nabla(S.connection())):              # Γ is not a tensor
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")


def test_use_metric_rejects_inverse():
    S = tf.geometry.sphere(r=1)
    try:
        S.use_metric(S.ginv)
    except TypeError:
        return
    raise AssertionError("expected a TypeError")


def test_simplified_metric_is_accepted():
    S = tf.geometry.sphere(r=1)
    S.use_metric(S.g.simplify())
    assert is_zero(S.gaussian_curvature() - 1)


def test_simplified_metric_matrix_matches_data():
    S = tf.geometry.sphere(r=1)
    g = S.g.simplify()
    m = g.matrix
    for i, j in np.ndindex(g.data.shape):
        assert m[i, j] == g.data[i, j]


def test_bare_manifold_has_no_metric():
    th, ph = sp.symbols("theta phi")
    M = Manifold([th, ph])
    w = M.covector([sp.cos(th), ph])
    wedge_nabla(w)                                            # d∧ works with no metric
    try:
        M.g
    except AttributeError:
        return
    raise AssertionError("expected no metric")


if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
