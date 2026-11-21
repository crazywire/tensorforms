"""Checks every result we derived by hand. Run: python -m pytest tests  (or: python tests/test_core.py)."""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorforms as tf
from tensorforms import (Manifold, contract, curvature, interior, nabla, otimes,
                         wedge, wedge_curvature, wedge_d, wedge_nabla)
from tensorforms.symbolic import is_zero


# ---- the definitions, used only to check the formulas the package uses ----
def _first_form_position(X):
    return next((i for i, s in enumerate(X.slots) if s.is_form), len(X.slots))


def curvature_by_commutator(X):
    """∇∧∇ by definition: ∇ twice, minus the order swapped."""
    pos = _first_form_position(X)
    Y = nabla(nabla(X))
    return Y - Y.swap_positions(pos, pos + 1)


def wedge_nabla_by_definition(X):
    """∇∧ by definition: ∇ on every slot, then (p+1)·Alt over all form slots."""
    p = len(X.form_slots)
    Y = nabla(X.alt())
    return Y if p == 0 else Y.alt() * (p + 1)


def wedge_curvature_by_definition(X):
    """∇∧∇∧ by definition: the wedge derivative twice."""
    return wedge_nabla_by_definition(wedge_nabla_by_definition(X))


# ---------------------------------------------------------------- sphere
def test_sphere_connection():
    S = tf.geometry.sphere(r=1)
    G = S.levi_civita().data
    th = S.coords[0]
    assert is_zero(G[0, 1, 1] + sp.sin(th) * sp.cos(th))     # Γ^θ_φφ = −sinθ cosθ
    assert is_zero(G[1, 0, 1] - sp.cot(th))                  # Γ^φ_θφ = cotθ
    assert is_zero(G[1, 1, 0] - sp.cot(th))                  # S-symmetric


def test_sphere_curvature():
    S = tf.geometry.sphere(r=1)
    th = S.coords[0]
    R = S.riemann()
    block = R.data[:, :, 0, 1]                               # R_θφ = [0, sin²θ; −1, 0]
    assert is_zero(block[0, 1] - sp.sin(th) ** 2) and is_zero(block[1, 0] + 1)
    assert S.ricci().equals(S.g.rename(dict(zip(S.g.names, S.ricci().names))))
    assert is_zero(S.ricci_scalar() - 2)
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
    return S, S.vector([f1, f2]), S.one_form([f1, f2])


def test_curvature_operator_matches_its_definition():
    S, v, w = _sphere_fields()                                # R on every slot == ∇ twice, antisymmetrized
    assert curvature(v).equals(curvature_by_commutator(v))
    assert curvature(w).equals(curvature_by_commutator(w))
    assert not curvature(w).is_zero()                        # −Rᵀw, not zero


def test_curl_needs_no_connection():
    S, v, w = _sphere_fields()
    assert wedge_nabla(w).equals(wedge_d(w))                  # ∇∧w = d∧w


def test_wedge_nabla_matches_its_definition():
    S, v, w = _sphere_fields()                                # form-slot corrections cancel
    assert wedge_nabla(v).equals(wedge_nabla_by_definition(v))
    assert wedge_nabla(w).equals(wedge_nabla_by_definition(w))
    X = nabla(v)                                              # a vector-valued 1-form
    assert wedge_nabla(X).equals(wedge_nabla_by_definition(X))


def test_wedge_curvature():
    S, v, w = _sphere_fields()
    assert wedge_curvature(w).is_zero()                       # 1-form: no value slots
    assert wedge_curvature(v).equals(curvature(v))            # vector: same as the commutator
    assert wedge_curvature(v).equals(wedge_curvature_by_definition(v))
    assert wedge_curvature(w).equals(wedge_curvature_by_definition(w))


def test_scalar_has_no_curvature():
    S = tf.geometry.sphere(r=1)
    f = S.scalar(sp.Function("f")(*S.coords))
    assert curvature(f).is_zero()                             # Hessian is symmetric


def test_bianchi_in_3d():
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z])
    M.metric(sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2))
    assert wedge_nabla(M.riemann()).is_zero()               # ∇∧R = 0
    u = M.vector([y, x * z, 1])
    assert curvature(u).equals(curvature_by_commutator(u))
    X = nabla(u)                                              # a vector-valued 1-form: R∧ mixes 3 directions
    assert wedge_curvature(X).equals(wedge_curvature_by_definition(X))


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
    assert E.riemann().is_zero()
    assert not np.all(E.levi_civita().data == 0)              # the grid bends, the space doesn't


def test_hyperbolic_plane():
    H = tf.geometry.hyperbolic_plane()
    assert is_zero(H.gaussian_curvature() + 1)


def test_schwarzschild_is_vacuum():
    BH = tf.geometry.schwarzschild()
    assert BH.ricci().is_zero()
    assert BH.einstein().is_zero()


def test_connection_is_s_symmetric():
    S = tf.geometry.sphere(r=1)                               # Γ^S = Γ: no torsion
    G = S.levi_civita().data
    assert all(is_zero(G[a, b, c] - G[a, c, b]) for a in range(2) for b in range(2) for c in range(2))


def test_gamma_and_riemann_match_index_formulas():
    """Γ and R are computed with the operators; check them against the index formulas."""
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z])
    M.metric(sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2))
    X, n = M.coords, 3
    g, gi = M.g.data, M.ginv.data
    G, R = M.levi_civita().data, M.riemann().data
    r = range(n)
    for a in r:
        for b in r:
            for c in r:
                lc = sum(gi[a, s] * (sp.diff(g[s, b], X[c]) + sp.diff(g[s, c], X[b])
                                     - sp.diff(g[b, c], X[s])) for s in r) / 2
                assert is_zero(G[a, b, c] - lc)
                for e in r:
                    ri = (sp.diff(G[a, b, e], X[c]) - sp.diff(G[a, b, c], X[e])
                          + sum(G[a, s, c] * G[s, b, e] - G[a, s, e] * G[s, b, c] for s in r))
                    assert is_zero(R[a, b, c, e] - ri)


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
    R = S.riemann()
    v = S.vector([1, 0])
    assert wedge(R, v).equals(contract(R, v))                 # R∧v = Rv


def test_ricci_by_names():
    S = tf.geometry.sphere(r=1)
    R = S.riemann()
    Ric = R.tr("ρ", "μ")                                     # (σ↓ | ν): same numbers as S.ricci()
    assert all(is_zero(e) for e in (Ric.data - S.ricci().data).flat)


def test_type_errors():
    S = tf.geometry.sphere(r=1)
    v, w = S.vector([1, 0]), S.covector([1, 0])
    for bad in (lambda: v + w,
                lambda: contract(S.g, S.g, "μ", "μ")):        # down with down
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")


def test_metric_checks():
    S = tf.geometry.sphere(r=1)
    for bad in (lambda: S.metric(S.ginv),                       # two up slots: that's g⁻¹
                lambda: S.metric([[1, 0], [0, 0]]),             # determinant 0
                lambda: S.metric([[1, 1], [0, 1]])):            # not symmetric
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError("expected an error")
    S.metric(S.g.simplify())                                    # a tensor with two down slots is fine
    assert is_zero(S.gaussian_curvature() - 1)
    assert tf.geometry.schwarzschild().signature == "-+++"


def test_bare_manifold_has_no_metric():
    th, ph = sp.symbols("theta phi")
    M = Manifold([th, ph])
    w = M.one_form([sp.cos(th), ph])
    wedge_nabla(w)                                            # d∧ works with no metric
    try:
        M.g
    except AttributeError:
        return
    raise AssertionError("expected no metric")


def test_names():
    S = tf.geometry.sphere()
    R = S.riemann()
    assert [i.name for i in S.indices(4)] == ["μ", "ν", "λ", "κ"]   # Greek for the tangent space
    assert nabla(R).names == ["ρ", "σ", "λ", "μ", "ν"]              # new slot: first free Greek name
    assert R.tr("rho", "mu").equals(R.tr("ρ", "μ"))                # spelled-out Greek
    assert R.tr(0, 2).equals(R.tr("ρ", "μ"))                       # slot numbers
    assert R.rename({"mu": "kappa"}).names == ["ρ", "σ", "κ", "ν"]


def test_scalar_multiplication():
    S = tf.geometry.sphere()
    th = S.coords[0]
    v, f = S.vector([1, 0]), S.scalar(sp.sin(th))
    assert (f * v).equals(v * sp.sin(th)) and (v * f).equals(sp.sin(th) * v)
    for bad in ([1, 2], "abc", S.covector([1, 0])):              # only numbers, expressions, scalars
        try:
            v * bad
        except TypeError:
            continue
        raise AssertionError(f"expected TypeError for {bad!r}")


def test_labels():
    S = tf.geometry.sphere()
    R = S.riemann()
    B = R._new(label="B")
    f = S.scalar(sp.sin(S.coords[0]))
    f.label = "f"
    expected = {"Alt(R)": R.alt(), "Sym(R)": R.sym(), "−R": -R, "R": -(-R),
                "R+B": R + B, "R−(−B)": R - (-B), "−(R+B)": -(R + B), "3(R+B)": 3 * (R + B),
                "−2R": -2 * R, "1/2·R": sp.Rational(1, 2) * R, "f·R": f * R,
                "R^U": R.U(), "R^US": R.U().S(), "(R+B)^S": (R + B).S()}
    for label, X in expected.items():
        assert X.label == label, (label, X.label)


def test_contract_default_and_otimes():
    S = tf.geometry.sphere(r=1)
    R, v, u = S.riemann(), S.vector([1, 0]), S.vector([0, 1])
    w, alpha = S.covector([1, 0]), S.one_form([1, 0])
    assert contract(R, v).equals(contract(R, v, "σ", "μ"))       # default: R's column with v's row
    assert contract(w, v).data.item() == 1                       # covector × vector: a number
    assert contract(alpha, v, "μ", "μ").data.item() == 1         # a 1-form is paired by name
    for bad in (lambda: contract(v, u),                          # column × column: mismatch
                lambda: contract(v, w),                          # column × row: use otimes
                lambda: contract(w, w),                          # row × row: mismatch
                lambda: contract(alpha, v)):                     # form slots never pair by default
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")
    T = otimes(v, u)                                             # v^μ u^ν: both slots kept
    assert [s.name for s in T.slots] == ["μ", "ν"] and T.data[0, 1] == 1


def test_raise_lower_undo():
    S = tf.geometry.sphere(r=1)
    th = S.coords[0]
    v, w = S.vector([th, 1]), S.covector([1, th])
    assert v.lower().slots[0].type == w.slots[0].type          # lowering a vector gives a covector
    assert v.lower().raise_().equals(v)                       # exact undo, both ways
    assert w.raise_().lower().equals(w)
    R = S.riemann()
    Rl = R.lower("ρ")                                        # R_ρσμν: any value slot
    assert Rl.slot("ρ").variance == "down" and Rl.raise_("ρ").equals(R)
    assert Rl.alt("ρ", "σ").equals(Rl)                          # pair symmetry shows up
    for bad in (lambda: S.one_form([1, 0]).raise_(),           # form slots are never raised
                lambda: R.lower("μ")):
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")
    F = wedge_nabla(v.lower().as_form())                       # bridge to a 1-form, then d∧
    assert F.form_slots and not F.value_slots
    f = S.scalar(th ** 2)
    df = tf.d(f)                                                # a 1-form (| μ)
    assert contract(df.as_value(), v).data.item() == 2 * th ** 2  # bridge back: df(v) = v^θ ∂_θ f
    assert df.as_value().as_form().types == df.types

def test_manifold_hands_out_copies():
    S = tf.geometry.sphere()
    R = S.riemann()
    R.label = "B"
    assert S.riemann().label == "R"                           # the stored R is untouched
    g = S.g
    g.label = "h"
    assert S.g.label == "g"
    assert isinstance(S.indices(1), list) and S.index().name == "μ"


def test_operator_labels():
    S = tf.geometry.sphere()
    v = S.vector([1, 0])
    assert [X.label for X in (tf.d(v), wedge_d(v), nabla(v), wedge_nabla(v),
                              curvature(v), wedge_curvature(v))] == ["dv", "d∧v", "∇v", "∇∧v", "∇∧∇v", "∇∧∇∧v"]


def test_euclidean_options():
    assert tf.geometry.euclidean().dim == 3 and tf.geometry.euclidean(4).dim == 4
    assert tf.geometry.euclidean(coords="polar").dim == 2                    # n follows the coordinates
    assert tf.geometry.euclidean(3, "Spherical").ricci_scalar() == 0     # any capitalisation
    for bad in (lambda: tf.geometry.euclidean(3, "polar"), lambda: tf.geometry.euclidean(3, "spherial")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")


def test_flrw_scale_factor():
    t = sp.Symbol("t", real=True)
    a1 = tf.geometry.flrw(a=t ** 2).ricci_scalar()               # the preset's t
    a2 = tf.geometry.flrw(a=lambda T: T ** 2).ricci_scalar()     # a function of t
    assert a1 != 0 and is_zero(a1 - a2)
    try:
        tf.geometry.flrw(a=sp.Symbol("t") ** 2)                       # a different t: refused
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_static_spherical_vacuum_gives_schwarzschild():
    """Solve G = 0 for −A(r)dt² + B(r)dr² + r²dΩ²: A = 1 − 2m/r (up to scaling t), B = 1/A."""
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    M = tf.geometry.templates.static_spherical(Manifold([t, r, th, ph]))
    G = M.einstein().data
    A, B = sp.Function("A")(r), sp.Function("B")(r)
    Bsol = sp.solve(G[1, 1], B)                              # G_rr = 0 is algebraic in B
    Bsol = [b for b in Bsol if b != 0][0]
    Aeq = sp.simplify((G[0, 0] * B).subs(B, Bsol).doit())    # then G_tt = 0 is an ODE for A
    m = sp.Symbol("m", positive=True)
    trial = 1 - 2 * m / r
    assert is_zero(Aeq.subs(A, trial).doit())                # A = 1 − 2m/r solves it
    assert is_zero(sp.simplify(Bsol.subs(A, trial).doit() - 1 / trial))   # and B = 1/A
    try:
        tf.geometry.templates.static_spherical(Manifold([t, th, r, ph]))  # wrong order: refused
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_interior():
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z])
    M.metric(sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2))
    f = M.scalar(x * y ** 2 + sp.sin(z))
    v = M.vector([y, x * z, 1])
    df = tf.d(f)
    dv = sum(v.data[i] * sp.diff(f.data.item(), c) for i, c in enumerate(M.coords))
    assert is_zero(interior(df, v).data.item() - dv)                  # df(v): the directional derivative
    a, b = M.one_form([x, y * z, 1]), M.one_form([z, 1, x * y])
    lhs = interior(wedge(a, b), v)                                    # ι(a∧b) = (ι a) b − a (ι b)
    rhs = b * interior(a, v).data.item() - a * interior(b, v).data.item()
    assert lhs.equals(rhs)
    R = M.riemann()
    assert interior(R, v).equals(contract(R, v, "μ", v.names[0]))     # contract with X's first form slot
    for bad in (lambda: interior(v, v), lambda: interior(df, df)):     # no form slot / no row
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")


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
