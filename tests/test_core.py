"""Checks every result we derived by hand. Run: python -m pytest tests  (or: python tests/test_core.py)."""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorforms as tf
from tensorforms import Manifold
from tensorforms.symbolic.simplify import is_zero          # internal: for single components


# ---- the definitions, used only to check the formulas the package uses ----
def _first_form_position(X):
    return next((i for i, s in enumerate(X.indices) if s.is_form), len(X.indices))


def curvature_by_commutator(X):
    """∇∧∇ by definition: ∇ twice, minus the order swapped."""
    pos = _first_form_position(X)
    Y = X.nabla().nabla()
    return Y - Y.swap(pos, pos + 1)


def wedge_nabla_by_definition(X):
    """∇∧ by definition: ∇ on every index, then (p+1)·Alt over all form indices."""
    p = sum(s.is_form for s in X.indices)
    Y = X.alt().nabla()
    return Y if p == 0 else Y.alt() * (p + 1)


def wedge_curvature_by_definition(X):
    """∇∧∇∧ by definition: the wedge derivative twice."""
    return wedge_nabla_by_definition(wedge_nabla_by_definition(X))


# ---------------------------------------------------------------- sphere
def test_sphere_connection():
    S = tf.Manifold.sphere(r=1)
    G = S.levi_civita().data
    th = S.coords[0]
    assert is_zero(G[0, 1, 1] + sp.sin(th) * sp.cos(th))     # Γ^θ_φφ = −sinθ cosθ
    assert is_zero(G[1, 0, 1] - sp.cot(th))                  # Γ^φ_θφ = cotθ
    assert is_zero(G[1, 1, 0] - sp.cot(th))                  # S-symmetric


def test_sphere_curvature():
    S = tf.Manifold.sphere(r=1)
    th = S.coords[0]
    R = S.riemann()
    block = R.data[:, :, 0, 1]                               # R_θφ = [0, sin²θ; −1, 0]
    assert is_zero(block[0, 1] - sp.sin(th) ** 2) and is_zero(block[1, 0] + 1)
    assert S.ricci().equals(S.g)
    assert is_zero(S.ricci_scalar() - 2)
    assert is_zero(S.gaussian_curvature() - 1)


def test_sphere_symbolic_radius():
    S = tf.Manifold.sphere()                                 # r symbolic
    r = sp.Symbol("r", positive=True)
    assert is_zero(S.gaussian_curvature() - 1 / r ** 2)


def test_nabla_g_is_zero():
    S = tf.Manifold.sphere(r=1)
    assert S.g.nabla().is_zero()


# ---------------------------------------------------------------- the derivatives
def _sphere_fields():
    S = tf.Manifold.sphere(r=1)
    th, ph = S.coords
    f1, f2 = sp.Function("v1")(th, ph), sp.Function("v2")(th, ph)
    return S, S.vector([f1, f2]), S.one_form([f1, f2])


def test_curvature_operator_matches_its_definition():
    S, v, w = _sphere_fields()                                # R on every index == ∇ twice, antisymmetrized
    assert v.curvature().equals(curvature_by_commutator(v))
    assert w.curvature().equals(curvature_by_commutator(w))
    assert w.curvature().is_zero() is None                    # w has unknown v1, v2: can't tell
    th, _ = S.coords                                         # a concrete w = dθ: compare with the known −Rᵀw
    C = S.one_form([1, 0]).curvature().simplify()
    assert C[0, 1, 1] == -sp.sin(th) ** 2 and C[1, 0, 1] == sp.sin(th) ** 2


def test_curl_needs_no_connection():
    S, v, w = _sphere_fields()
    assert w.wedge_nabla().equals(w.wedge_d())                  # ∇∧w = d∧w


def test_wedge_nabla_matches_its_definition():
    S, v, w = _sphere_fields()                                # form-index corrections cancel
    assert v.wedge_nabla().equals(wedge_nabla_by_definition(v))
    assert w.wedge_nabla().equals(wedge_nabla_by_definition(w))
    X = v.nabla()                                              # a vector-valued 1-form
    assert X.wedge_nabla().equals(wedge_nabla_by_definition(X))


def test_wedge_curvature():
    S, v, w = _sphere_fields()
    assert w.wedge_curvature().is_zero()                       # 1-form: no value indices
    assert v.wedge_curvature().equals(v.curvature())            # vector: same as the commutator
    assert v.wedge_curvature().equals(wedge_curvature_by_definition(v))
    assert w.wedge_curvature().equals(wedge_curvature_by_definition(w))


def test_scalar_has_no_curvature():
    S = tf.Manifold.sphere(r=1)
    f = S.scalar(sp.Function("f")(*S.coords))
    assert f.curvature().is_zero()                             # Hessian is symmetric


def test_bianchi_in_3d():
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z], metric=sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2))
    assert M.riemann().wedge_nabla().is_zero()               # ∇∧R = 0
    u = M.vector([y, x * z, 1])
    assert u.curvature().equals(curvature_by_commutator(u))
    X = u.nabla()                                              # a vector-valued 1-form: R∧ mixes 3 directions
    assert X.wedge_curvature().equals(wedge_curvature_by_definition(X))


def test_product_rule():
    S = tf.Manifold.sphere(r=1)
    th, ph = S.coords
    a, b = S.indices(2)
    A = S.tensor(a.up, b.down, data=[[th, ph], [th * ph, 1]])
    v = S.vector([sp.sin(th), ph], index=b)
    lhs = A.contract(v, b).nabla()
    rhs = A.nabla().contract(v, b) + A.contract(v.nabla(), b)
    assert lhs.equals(rhs)


# ---------------------------------------------------------------- flat space and spacetime
def test_spherical_coordinates_are_flat():
    E = tf.Manifold.euclidean(3, coords="spherical")
    assert E.riemann().is_zero()
    assert not np.all(E.levi_civita().data == 0)              # the grid bends, the space doesn't


def test_hyperbolic_plane():
    H = tf.Manifold.hyperbolic_plane()
    assert is_zero(H.gaussian_curvature() + 1)


def test_schwarzschild_is_vacuum():
    BH = tf.Manifold.schwarzschild()
    assert BH.ricci().is_zero()
    assert BH.einstein().is_zero()


def test_connection_is_s_symmetric():
    S = tf.Manifold.sphere(r=1)                               # Γ^S = Γ: no torsion
    G = S.levi_civita().data
    assert all(is_zero(G[a, b, c] - G[a, c, b]) for a in range(2) for b in range(2) for c in range(2))


def test_gamma_and_riemann_match_index_formulas():
    """Γ and R are computed with the derivatives; check them against the index formulas."""
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z], metric=sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2))
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
    S = tf.Manifold.sphere(r=1)
    th, ph = S.coords
    a, b = S.indices(2)
    X = S.tensor(a.form, b.form, data=[[th, ph], [1, th * ph]])
    assert X.alt().alt().equals(X.alt())                      # projections
    assert X.alt().sym().is_zero()                            # Sym Alt = 0
    assert (X.alt() + X.sym()).equals(X)                      # two indices: X = Sym + Alt
    assert X.alt(a).equals(X) and X.alt(a) is not X           # one index: same numbers, a new tensor
    try:
        X.alt(a, a)                                           # a repeated index: refused
        assert False
    except tf.IndexMismatchError:
        pass


def test_wedge_single_factor_is_one_term():
    S = tf.Manifold.sphere(r=1)
    R = S.riemann()
    v = S.vector([1, 0])
    assert R.wedge(v).equals(R @ v)                 # R∧v = Rv


def test_ricci_by_names():
    S = tf.Manifold.sphere(r=1)
    R = S.riemann()
    Ric = R.tr("ρ", "μ")                                     # (σ↓ ν): same numbers as S.ricci()
    assert all(is_zero(e) for e in (Ric.data - S.ricci().data).flat)


def test_type_errors():
    S = tf.Manifold.sphere(r=1)
    v, w = S.vector([1, 0]), S.covector([1, 0])
    for bad in (lambda: v + w,
                lambda: S.g.contract(S.g, "μ", "μ")):        # down with down
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")


def test_metric_checks():
    S = tf.Manifold.sphere(r=1)
    th, ph = S.coords
    for bad in (lambda: Manifold([th, ph], metric=S.ginv),               # two up indices: that's g⁻¹
                lambda: Manifold([th, ph], metric=[[1, 0], [0, 0]]),     # determinant 0
                lambda: Manifold([th, ph], metric=[[1, 1], [0, 1]])):    # not symmetric
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError("expected an error")
    S2 = Manifold([th, ph], metric=S.g.simplify())              # a tensor with two down indices is fine
    assert is_zero(S2.gaussian_curvature() - 1)


def test_bare_manifold_has_no_metric():
    th, ph = sp.symbols("theta phi")
    M = Manifold([th, ph])
    w = M.one_form([sp.cos(th), ph])
    w.wedge_nabla()                                            # d∧ works with no metric
    try:
        M.g
    except AttributeError:
        return
    raise AssertionError("expected no metric")


def test_names():
    S = tf.Manifold.sphere()
    R = S.riemann()
    assert [i.label for i in S.indices(4)] == ["μ", "ν", "λ", "κ"]   # Greek for the tangent space
    assert [s.label for s in R.nabla().indices] == ["ρ", "σ", "λ", "μ", "ν"]              # new index: first free Greek name
    assert R.tr("rho", "mu").equals(R.tr("ρ", "μ"))                # spelled-out Greek
    assert R.tr(0, 2).equals(R.tr("ρ", "μ"))                       # index numbers
    K = R.relabel_indices({"mu": "kappa"})
    assert [s.label for s in K.indices] == ["ρ", "σ", "κ", "ν"] and K.equals(R) and K.label == "R"
    assert [s.label for s in R.relabel_indices({"μ": "ν", "ν": "μ"}).indices] == ["ρ", "σ", "ν", "μ"]  # exchange


def test_scalar_multiplication():
    S = tf.Manifold.sphere()
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
    """A label is just a name: given, or the manifold's own (g, Γ, R, …). Computed results have none; changes that
    keep the tensor (index order, index labels, arrows, form/value, simplify) keep the name."""
    S = tf.Manifold.sphere()
    R, g, v = S.riemann(), S.g, S.vector([1, 0], label="v")
    assert S.vector([1, 0]).label is None                       # no made-up label
    assert [S.levi_civita().label, R.label, S.ricci().label, g.label, S.ginv.label] == ["Γ", "R", "Ric", "g", "g⁻¹"]
    computed = (R.alt(), R.sym(), -R, R + R, 3 * R, R / 2, R.tr(0, 1), g.inverse(), v.star(),
                R @ v, v.otimes(v), S.one_form([1, 0]).wedge(S.one_form([0, 1])))
    assert all(X.label is None for X in computed)
    kept = (R.S, R.swap(0, 2), R.reorder(3, 2, 1, 0), R.relabel_indices({"mu": "kappa"}), v.lower(),
            v.lower().as_form(), R.simplify(), R + 0)
    assert all(X.label == X_label for X, X_label in zip(kept, "R R R R v v R R".split()))
    X = R @ v
    X.label = "K"                                              # name a result yourself
    assert repr(X) == "K(ρ↑ μ ν)"

def test_contract_default_and_otimes():
    S = tf.Manifold.sphere(r=1)
    R, v, u = S.riemann(), S.vector([1, 0]), S.vector([0, 1])
    w, alpha = S.covector([1, 0]), S.one_form([1, 0])
    assert (R @ v).equals(R.contract(v, "σ", "μ"))       # default: R's column with v's row
    assert (w @ v).data.item() == 1                       # covector × vector: a number
    assert alpha.contract(v, "μ", "μ").data.item() == 1         # a 1-form is paired by name
    for bad in (lambda: v @ u,                          # column × column: mismatch
                lambda: v @ w,                          # column × row: use otimes
                lambda: w @ w,                          # row × row: mismatch
                lambda: alpha @ v):                     # form indices never pair by default
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")
    T = v.otimes(u)                                             # v^μ u^ν: both indices kept
    assert [s.label for s in T.indices] == ["μ", "ν"] and T.data[0, 1] == 1


def test_raise_lower_undo():
    S = tf.Manifold.sphere(r=1)
    th = S.coords[0]
    v, w = S.vector([th, 1]), S.covector([1, th])
    assert v.lower().indices[0].type == w.indices[0].type          # lowering a vector gives a covector
    assert v.lower().raise_().equals(v)                       # exact undo, both ways
    assert w.raise_().lower().equals(w)
    R = S.riemann()
    Rl = R.lower("ρ")                                        # R_ρσμν: any value index
    assert Rl.index("ρ").variance == "down" and Rl.raise_("ρ").equals(R)
    assert Rl.alt("ρ", "σ").equals(Rl)                          # pair symmetry shows up
    for bad in (lambda: S.one_form([1, 0]).raise_(),           # form indices are never raised
                lambda: R.lower("μ")):
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")
    F = v.lower().as_form().wedge_nabla()                       # bridge to a 1-form, then d∧
    assert all(s.is_form for s in F.indices)
    f = S.scalar(th ** 2)
    df = f.d()                                                # a 1-form (μ)
    assert (df.as_value() @ v).data.item() == 2 * th ** 2  # bridge back: df(v) = v^θ ∂_θ f
    assert all(a.matches(b) for a, b in zip(df.as_value().as_form().indices, df.indices))

def test_manifold_hands_out_copies():
    S = tf.Manifold.sphere()
    R = S.riemann()
    R.label = "B"
    assert S.riemann().label == "R"                           # the stored R is untouched
    g = S.g
    g.label = "h"
    assert S.g.label == "g"
    assert isinstance(S.indices(1), list) and S.index().label == "μ"


def test_operator_results_have_no_label():
    S = tf.Manifold.sphere()
    v = S.vector([1, 0])
    assert all(X.label is None for X in (v.d(), v.wedge_d(), v.nabla(), v.wedge_nabla(),
                                          v.curvature(), v.wedge_curvature()))

def test_euclidean_options():
    assert tf.Manifold.euclidean().dim == 3 and tf.Manifold.euclidean(4).dim == 4
    assert tf.Manifold.euclidean(coords="polar").dim == 2                    # n follows the coordinates
    assert tf.Manifold.euclidean(3, "Spherical").ricci_scalar().is_zero()   # any capitalisation
    for bad in (lambda: tf.Manifold.euclidean(3, "polar"), lambda: tf.Manifold.euclidean(3, "spherial")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")


def test_flrw_scale_factor():
    t = sp.Symbol("t", real=True)
    a1 = tf.Manifold.flrw(a=t ** 2).ricci_scalar()               # the preset's t
    a2 = tf.Manifold.flrw(a=lambda T: T ** 2).ricci_scalar()     # a function of t
    assert a1 != 0 and is_zero(a1 - a2)
    try:
        tf.Manifold.flrw(a=sp.Symbol("t") ** 2)                       # a different t: refused
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_static_spherical_vacuum_gives_schwarzschild():
    """Solve G = 0 for −A(r)dt² + B(r)dr² + r²dΩ²: A = 1 − 2m/r (up to scaling t), B = 1/A."""
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    M = tf.templates.static_spherical([t, r, th, ph])
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
        tf.templates.static_spherical([t, th, r, ph])            # wrong order: refused
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_interior():
    x, y, z = sp.symbols("x y z", real=True)
    M = Manifold([x, y, z], metric=sp.diag(1, 1 + x ** 2, 1 + x ** 2 * y ** 2))
    f = M.scalar(x * y ** 2 + sp.sin(z))
    v = M.vector([y, x * z, 1])
    df = f.d()
    dv = sum(v.data[i] * sp.diff(f.data.item(), c) for i, c in enumerate(M.coords))
    assert is_zero(df.interior(v).data.item() - dv)                  # df(v): the directional derivative
    a, b = M.one_form([x, y * z, 1]), M.one_form([z, 1, x * y])
    lhs = a.wedge(b).interior(v)                                    # ι(a∧b) = (ι a) b − a (ι b)
    rhs = b * a.interior(v).data.item() - a * b.interior(v).data.item()
    assert lhs.equals(rhs)
    R = M.riemann()
    assert R.interior(v).equals(R.contract(v, "μ", v.indices[0].label))     # contract with X's first form index
    for bad in (lambda: v.interior(v), lambda: df.interior(df)):     # no form index / no row
        try:
            bad()
        except TypeError:
            continue
        raise AssertionError("expected a TypeError")


def test_minkowski_options():
    assert tf.Manifold.minkowski().dim == 4
    S = tf.Manifold.minkowski(coords="spherical")                  # −dt² + dr² + r²dΩ²
    assert S.dim == 4 and S.riemann().is_zero() and S.levi_civita().is_zero() is False
    assert tf.Manifold.minkowski(3, "polar").g[0, 0] == -1
    try:
        tf.Manifold.minkowski(5, "spherical")
    except ValueError:
        return
    raise AssertionError("expected a ValueError")




def test_copy_on_write():
    S = tf.Manifold.sphere()
    X = S.tensor(*S.g.indices, data=[[1, 2], [3, 4]])
    Y = X.T                                                  # shares X's numbers
    Y[0, 1] = 99
    assert X.data.tolist() == [[1, 2], [3, 4]] and Y.data.tolist() == [[1, 99], [2, 4]]
    Z = S.tensor(*X.indices, data=X.data)                        # built from X.data: its own copy
    X[1, 1] = 0
    assert Z.data.tolist() == [[1, 2], [3, 4]]
    a = np.array([[1, 2], [3, 4]], dtype=object)
    assert not np.may_share_memory(tf.Tensor(S, a, X.indices)._data, a)              # default: copies
    assert np.may_share_memory(tf.Tensor(S, a, X.indices, copy=False)._data, a)      # copy=False: as is
    g = S.g
    g[0, 0] = 1                                                # your g, not the manifold's
    assert S.g.data[0, 0] != 1
    R = S.riemann()
    R[0, 0, 0, 0] = 5
    assert S.riemann().data[0, 0, 0, 0] == 0                   # the cached R is untouched
    for write in (lambda: X.data.__setitem__((0, 0), 1),       # only X[...] = v writes
                  lambda: X[0].__setitem__(0, 1)):
        try:
            write()
            assert False
        except ValueError:
            pass

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


def test_contract_keeps_form_order_when_names_clash():
    # A's paired index is a form index named ν; B also has a form index named ν.
    # B's form indices must stay in B's order whatever they are called.
    S = tf.Manifold.sphere()
    mu, nu, la = S.indices(3)
    A = S.tensor(mu.up, nu.form, data=[[1, 2], [3, 4]])
    B = S.tensor(la.up, mu.form, nu.form, data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    clash = A.contract(B, 1, 0)
    fresh = A.contract(B.relabel_indices({"μ": "α", "ν": "β"}), 1, 0)
    assert (clash.data == fresh.data).all()


def test_index_lists_are_positions():
    S = tf.Manifold.sphere()
    R = S.riemann()                              # R(ρ↑ σ↓ μ ν)
    assert [s.is_form for s in R.indices] == [False, False, True, True]
    assert [s.label for s in R.indices] == ["ρ", "σ", "μ", "ν"]       # the labels


def test_volume_form():
    P = tf.Manifold.euclidean(coords="polar")
    r, _ = P.coords
    eps = P.volume_form()
    assert all(s.is_form for s in eps.indices) and sp.simplify(eps.data[0, 1] - r) == 0 and eps.data[1, 0] == -eps.data[0, 1]
    S = tf.Manifold.sphere()
    th, _ = S.coords
    rr = sp.Symbol("r", positive=True)
    assert sp.simplify(S.volume_form().data[0, 1] - rr**2 * sp.Abs(sp.sin(th))) == 0


def test_star_signs_and_curl():
    E = tf.Manifold.euclidean()
    x, y, z = E.coords
    alpha = E.one_form([-y, x, 0])
    assert alpha.wedge_d().star().data.tolist() == [0, 0, 2]          # curl of the rotation
    assert alpha.star().star().equals(alpha)                           # 3D Riemannian: ⋆⋆ = 1
    assert E.scalar(1).star().equals(E.volume_form())                  # ⋆1 = ε
    Mk = tf.Manifold.minkowski()
    t, X, Y, Z = Mk.coords
    F = Mk.one_form([t * X, Y, 0, X * Z]).wedge_d()
    assert F.star().star().equals(-F)                                  # 4D Lorentzian 2-form: ⋆⋆ = −1
    S = tf.Manifold.sphere()
    a, b, c = S.indices(3)
    try:
        S.tensor(a.form, b.form, c.form, data=[[[0] * 2] * 2] * 2).star()  # 3 form indices in 2D: refused
        assert False
    except tf.IndexMismatchError:
        pass


def test_star_divergence_matches_trace():
    def div_star(v):
        return v.lower().as_form().star().wedge_d().star().data.item()

    def div_trace(v):
        return v.nabla().tr(0, 1).data.item()

    P = tf.Manifold.euclidean(coords="polar")
    r, ph = P.coords
    v = P.vector([r**2 * sp.cos(ph), sp.sin(ph) / r])
    assert sp.simplify(div_star(v) - div_trace(v)) == 0               # symbolic: no |…| here
    S = tf.Manifold.sphere()                                           # √det g has |sin θ|: sample points
    th, p = S.coords
    rr = sp.Symbol("r", positive=True)
    v = S.vector([sp.sin(th) * p, sp.cos(th) ** 2])
    for pt in ({th: 1, p: 2, rr: 3}, {th: sp.Rational(1, 3), p: -1, rr: sp.Rational(1, 2)}):
        assert abs(sp.N((div_star(v) - div_trace(v)).subs(pt))) < 1e-12


def test_division_is_exact():
    S = tf.Manifold.sphere()
    v = S.vector([1, 3])
    assert (v / 2).data.tolist() == [sp.Rational(1, 2), sp.Rational(3, 2)]
    assert (v / 2).equals(v * sp.Rational(1, 2))
    for bad in (lambda: v / 0, lambda: v / v, lambda: 2 / v):
        try:
            bad()
            assert False, "should refuse"
        except (TypeError, ZeroDivisionError):
            pass


def test_adding_numbers():
    S = tf.Manifold.sphere()
    v = S.vector([1, 2])
    s = v.lower() @ v                                  # a scalar
    assert (v + 0).equals(v) and (0 + v).equals(v) and (v - 0).equals(v) and (0 - v).equals(-v)
    assert sum([v, v, v]).equals(3 * v)                         # sum starts from 0
    assert sp.simplify((s + 1).data.item() - (s.data.item() + 1)) == 0
    assert sp.simplify((1 - s).data.item() - (1 - s.data.item())) == 0
    for bad in (lambda: v + 1, lambda: 1 + v, lambda: v - sp.Symbol("x")):
        try:
            bad()
            assert False, "a nonzero number only adds to a scalar"
        except TypeError:
            pass


def test_sympy_accepts_tensors():
    S = tf.Manifold.sphere()
    th, _ = S.coords
    v = S.vector([1, 2])
    s = v.lower() @ v                                  # a scalar
    assert sp.simplify(sp.diff(s, th) - sp.diff(s.data.item(), th)) == 0   # sympy takes it directly
    assert sp.sympify(v) == sp.Array([1, 2])                    # one axis per index
    for mixed in (th * s, th + s, th - s, sp.sin(th) * v, 2 / s, 0 / s):
        assert isinstance(mixed, tf.Tensor)                     # tensor arithmetic, whichever side
    try:
        2 / v
        assert False, "dividing by a tensor with indices is refused"
    except TypeError:
        pass


def test_value_for_every_tensor():
    S = tf.Manifold.sphere()
    rr = sp.Symbol("r", positive=True)
    assert sp.simplify(S.ricci_scalar().value - 2 / rr**2) == 0      # scalar: its expression
    assert S.vector([1, 0]).value == sp.Array([1, 0])                  # vector: a 1D Array
    assert S.g.value.shape == (2, 2)                                   # g: a 2D Array


def test_is_zero_is_exact():
    # sympy's three answers, never a numerical guess
    x = sp.Symbol("x", positive=True)
    f = sp.Function("f")(x)
    assert is_zero(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1) is True     # shown to be 0
    assert is_zero(sp.Integer(3)) is False                            # shown not to be 0
    assert is_zero(x) is False                                        # x > 0 by assumption
    assert is_zero(f) is None                                         # an unknown function: can't tell
    th = sp.Symbol("theta", real=True)
    assert is_zero(sp.sin(th)) is None                                # zero at θ = 0, so sympy won't say nonzero
    assert is_zero(sp.Rational(1, 10) ** 12 * x) is False             # tiny is not zero
    S = tf.Manifold.sphere()
    assert S.vector([0, 0]).is_zero() is True
    assert S.vector([1, 0]).is_zero() is False
    assert S.vector([f, 0]).is_zero() is None


def test_zeros_of_every_family():
    # circular, hyperbolic, exponential, inverse trig: all exactly 0, all found
    x, t, th = sp.Symbol("x", real=True), sp.Symbol("t", real=True), sp.Symbol("theta", real=True)
    zeros = [sp.tan(x)**2 + 1 - sp.sec(x)**2,
             sp.cosh(x)**2 - sp.sinh(x)**2 - 1,
             sp.tanh(x) - sp.sinh(x) / sp.cosh(x),
             sp.sinh(2*x) - 2*sp.sinh(x)*sp.cosh(x),
             sp.cosh(x) - (sp.exp(x) + sp.exp(-x)) / 2,
             sp.exp(2*t)*sp.exp(-t) - sp.exp(t),
             sp.sin(sp.asin(x)) - x,
             sp.sin(2*th)*sp.tan(th) + sp.cos(2*th) - 1,
             sp.cot(th)*sp.sin(2*th) - 2*sp.cos(th)**2,
             (sp.sin(th)**4 - sp.cos(th)**4) / (sp.sin(th)**2 - sp.cos(th)**2) - 1]
    assert all(is_zero(z) is True for z in zeros)
    S = tf.Manifold.sphere()
    assert S.scalar(sp.cosh(x)**2 - sp.sinh(x)**2).simplify().value == 1   # hyperbolic leftovers simplify too
    th2 = sp.Symbol("theta", real=True)                                  # mixed families simplify together
    mix = sp.sin(th2)**2*sp.cosh(x)**2 + sp.cos(th2)**2*sp.cosh(x)**2 - sp.sinh(x)**2
    assert S.scalar(mix).simplify().value == 1


def test_inverse():
    S = tf.Manifold.sphere()
    th, _ = S.coords
    assert S.g.inverse().equals(S.ginv)                         # g⁻¹ is g's tensor inverse
    assert [s.is_up for s in S.g.inverse().indices] == [True, True]  # arrows flip
    mu, nu, la = S.indices(3)
    A = S.tensor(mu.up, nu.down, data=[[1, sp.sin(th)], [0, 2]])
    assert (A @ A.inverse()).equals(S.tensor(mu.up, nu.down, data=[[1, 0], [0, 1]]))
    F = S.tensor(mu.up, nu.down, la.form, data=[[[1, 0], [2, 1]], [[0, th], [1, 3]]])
    for bad in (lambda: S.tensor(mu.up, nu.down, data=[[1, 1], [1, 1]]).inverse(),  # determinant 0
                lambda: S.vector([1, 0]).inverse(),             # one index
                lambda: F.inverse(),                            # a form index: not linear, so refused
                lambda: S.riemann().inverse(),
                lambda: tf.Manifold(list(sp.symbols("x y", real=True)), metric=[[1, 1], [1, 1]])):  # singular metric
        try:
            bad()
            assert False, "should refuse"
        except (TypeError, ValueError):
            pass


def test_indices_are_immutable():
    S = tf.Manifold.sphere()
    Y = S.g.T
    mu = S.index()
    for change in (lambda: setattr(Y.indices[0], "name", "α"),
                   lambda: setattr(Y.indices[0], "variance", "up"),
                   lambda: setattr(mu, "name", "x")):
        try:
            change()
            assert False
        except AttributeError:                                 # FrozenInstanceError is an AttributeError
            pass
    assert repr(S.g) == "g(μ↓ ν↓)"                             # the manifold's metric is untouched
    assert S.tensor(mu, data=[1, 0]).equals(S.tensor(mu.up, data=[1, 0]))   # default: up value
    assert repr(mu) == "μ↑" and repr(mu.down) == "μ↓" and repr(mu.form) == "μ"
    assert mu.form.up == mu                                    # same label and space, any kind
    nu = S.index("nu")
    assert mu.up.matches(nu.up) and not mu.up.matches(nu.down)     # labels don't matter
    assert mu.up.pairs_with(nu.form) and not mu.down.pairs_with(nu.form)
    assert mu.form.type == "form" and mu.up.type == "value"


def test_arithmetic_edges():
    S = tf.Manifold.sphere()
    th = S.coords[0]
    assert (S.scalar(2) * S.scalar(th)).value == 2 * th                  # scalar × scalar
    hidden_zero = sp.sin(th)**2 + sp.cos(th)**2 - 1
    try:
        S.g / hidden_zero                                                 # sympy shows it is 0
        assert False
    except ZeroDivisionError:
        pass
    assert (S.g + hidden_zero).equals(S.g)                                # a 0 of any type
    assert S.g.matches(S.ricci()) and not S.g.matches(S.vector([1, 0]))   # labels don't matter
    assert not S.g.matches(S.g.as_form(1)) and not S.g.matches(3)
    try:
        S.g + S.vector([1, 0])
        assert False
    except TypeError as e:
        assert "indices don't match" in str(e)


def test_subs_and_at():
    S = tf.Manifold.sphere()
    th, ph = S.coords
    r = next(iter(S.g.data[0, 0].free_symbols))
    assert S.g.subs(r, 1).data.tolist() == [[1, 0], [0, sp.sin(th)**2]]    # a parameter: still a field
    assert S.g.subs({r: 2}).data[0, 0] == 4
    for bad in (lambda: S.g.subs(th, 0), lambda: S.g.subs({"theta": 0}),  # coordinates refused
                lambda: S.g.subs(sp.sin(th), 1),
                lambda: S.g.at({th: sp.pi / 2}),                           # every coordinate needed
                lambda: S.g.at({r: 1, th: 0, ph: 0})):                     # r is not a coordinate
        try:
            bad()
            assert False
        except ValueError:
            pass
    G = S.g.at({th: sp.pi / 2, "phi": 0})                                  # plain values, not a tensor
    assert isinstance(G, sp.Array) and G == sp.Array([[r**2, 0], [0, r**2]])
    assert sp.simplify(S.ricci_scalar().at({th: 1, ph: 0}) - 2 / r**2) == 0


def test_lie_derivative():
    S = tf.Manifold.sphere()
    th, ph = S.coords
    assert S.g.lie(S.vector([0, 1])).is_zero()                           # ∂_φ: a symmetry (Killing)
    assert S.g.lie(S.vector([1, 0])).simplify().data[1, 1] != 0           # ∂_θ: not one
    E = tf.Manifold.euclidean(2)
    x, y = E.coords
    v, w = E.vector([-y, x]), E.vector([1, 0])
    assert w.lie(v).data.tolist() == [0, -1]                               # rotation vs a fixed arrow
    assert w.lie(v).equals(-v.lie(w))                                    # the Lie bracket is antisymmetric
    a = E.one_form([x * y, x**2])
    assert a.lie(v).equals(a.wedge_d().interior(v) + a.interior(v).d())   # Cartan's formula
    f = E.scalar(x**2 * y)
    assert sp.simplify(f.lie(v).value - (-2 * x * y**2 + x**3)) == 0      # on a scalar: v(f)
    # matches the ∇ version (the Γ terms cancel), on a mixed tensor
    i, j, k = S.indices(3)
    T = S.tensor(i.up, j.down, k.form, data=[[[th * ph, sp.sin(th)], [ph**2, 1]],
                                             [[sp.cos(ph), th], [0, th * ph**2]]])
    u = S.vector([sp.sin(ph), th * sp.cos(th)])
    Jn, data = u.nabla().as_value().data, T.nabla().interior(u).data
    for n, s in enumerate(T.indices):
        if s.is_up:
            data = data - np.moveaxis(np.tensordot(Jn, T.data, axes=([1], [n])), 0, n)
        else:
            data = data + np.moveaxis(np.tensordot(T.data, Jn, axes=([n], [0])), -1, n)
    assert T.lie(u).equals(S.tensor(*T.indices, data=data))
    N = tf.Manifold(list(sp.symbols("p q")))                               # no metric needed
    assert N.vector([N.coords[0], 0]).lie(N.vector([1, 0])).data.tolist() == [1, 0]
    try:
        S.g.lie(S.covector([1, 0]))                                       # v must be a vector
        assert False
    except TypeError:
        pass


def test_errors():
    S, S1 = tf.Manifold.sphere(), tf.Manifold.sphere(r=1)
    th = S.coords[0]
    mu, nu = S.indices(2)
    bare = tf.Manifold(list(sp.symbols("x y")))
    cases = [(lambda: S.g + S1.g, tf.IndexMismatchError),                 # different manifolds
             (lambda: S.vector([1, 0]) @ S.vector([0, 1]), tf.IndexMismatchError),
             (lambda: S.g.lie(S.covector([1, 0])), tf.IndexMismatchError),
             (lambda: bare.g, tf.MissingMetricError),
             (lambda: bare.one_form([1, 0]).star(), tf.MissingMetricError),
             (lambda: tf.Manifold(bare.coords, metric=[[1, 2], [3, 4]]), tf.InvalidMetricError),
             (lambda: S.tensor(mu.up, nu.down, data=[[1, 1], [1, 1]]).inverse(), tf.NonInvertibleError),
             (lambda: S.g.subs(th, 0), tf.InvalidCoordinateError),
             (lambda: tf.Manifold.euclidean(coords="polaar"), tf.InvalidCoordinateError)]
    for make, kind in cases:
        try:
            make()
            assert False, kind
        except kind as e:
            assert isinstance(e, tf.TensorformsError)
    assert issubclass(tf.IndexMismatchError, TypeError) and issubclass(tf.NonInvertibleError, ValueError)
    assert not hasattr(bare, "g")                                         # MissingMetricError is an AttributeError
    try:
        S.g + S1.g
    except tf.IndexMismatchError as e:
        assert "different manifolds" in str(e)


def test_input_checks():
    S = tf.Manifold.sphere()
    th = S.coords[0]
    cases = [(lambda: S.tensor("mu", data=[1, 0]), TypeError, "Index objects"),
             (lambda: S.g.subs(), TypeError, "old, new"),
             (lambda: S.g.subs(1, 2, 3), TypeError, "old, new"),
             (lambda: S.g.at([th, 0]), TypeError, "dictionary"),
             (lambda: tf.Manifold.minkowski(n=0), ValueError, "at least 2"),
             (lambda: tf.Manifold.euclidean(n=0), ValueError, "at least 1"),
             (lambda: tf.Manifold.flrw(k="x"), TypeError, "k must be")]
    for make, kind, words in cases:
        try:
            make()
            assert False, words
        except kind as e:
            assert words in str(e), str(e)



def test_templates_make_new_manifolds():
    x, y = sp.symbols("x y", real=True)
    M = tf.templates.general([x, y])
    assert M.g[0, 1] == M.g[1, 0] == sp.Function("g01")(x, y)
    assert tf.templates.diagonal([x, y]).g[1, 0] == 0
    assert tf.templates.conformally_flat([x, y], [-1, 1]).g[0, 0] == -sp.exp(2 * sp.Function("omega")(x, y))


def test_manifold_label_is_optional():
    x, y = sp.symbols("x y", real=True)
    M, N = Manifold([x, y]), Manifold([x, y])
    assert M.label is None and str(M) == "(x, y)" and repr(M) == "Manifold(x, y) (no metric)"
    assert str(tf.Manifold.sphere()) == "S2"
    try:
        M.scalar(1) + N.scalar(1)
    except tf.IndexMismatchError as e:
        assert "(x, y) and (x, y)" in str(e)


def test_submanifold():
    x, y = sp.symbols("x y", real=True)
    u = sp.Symbol("u", real=True)
    P = Manifold([x, y], metric=sp.eye(2))
    C = Manifold([u], ambient=P, map={x: sp.cos(u), y: sp.sin(u)})
    assert C.g.data.tolist() == [[1]]                                    # du²
    assert repr(C.map) == "Tensor(a↑)" and repr(C.map.d()) == "Tensor(a↑ μ)"   # Latin: the ambient's directions
    assert C.map.d().data.tolist() == [[-sp.sin(u)], [sp.cos(u)]]        # J
    assert is_zero(C.pullback(P.one_form([-y, x]))[0] - 1)               # −y dx + x dy → du
    E = tf.Manifold.euclidean(coords="spherical")
    r, th, ph = E.coords
    S = Manifold([th, ph], ambient=E, map={r: 1, th: th, ph: ph})       # a coordinate held fixed
    assert S.g.data.tolist() == [[1, 0], [0, sp.sin(th) ** 2]] and is_zero(S.gaussian_curvature() - 1)
    B = Manifold([x, y])                                                 # no metric: pullback still works
    assert not Manifold([u], ambient=B, map={x: u, y: u ** 2}).has_metric()
    cases = [(lambda: Manifold([u], ambient=P, map={x: 0, y: 0}), tf.InvalidMetricError),    # collapses
             (lambda: Manifold([u], ambient=P, map={x: u}), tf.InvalidCoordinateError),       # y missing
             (lambda: Manifold([u], ambient=P, map={x: u, y: x}), tf.InvalidCoordinateError), # uses x
             (lambda: Manifold([u], ambient=P), TypeError),
             (lambda: Manifold([u], ambient=P, map={x: u, y: u}, metric=[[1]]), TypeError),
             (lambda: C.pullback(P.vector([1, 0])), tf.IndexMismatchError),                  # up index
             (lambda: C.pullback(B.scalar(1)), tf.IndexMismatchError),                       # wrong ambient
             (lambda: P.pullback(P.scalar(1)), TypeError)]                                   # P has no ambient
    for make, kind in cases:
        try:
            make()
            assert False, kind
        except kind:
            pass
def test_describe():
    S = tf.Manifold.sphere()
    assert S.g.describe().splitlines()[0] == "g on S2"
    x, y = sp.symbols("x y", real=True)
    assert Manifold([x, y]).vector([1, 0]).describe().splitlines()[0] == "Tensor on (x, y)"


def test_integrate():
    S = tf.Manifold.sphere()
    th, ph = S.coords
    r = sp.Symbol("r", positive=True)
    assert is_zero(S.volume_form().integrate({th: (0, sp.pi), "phi": (0, 2 * sp.pi)}) - 4 * sp.pi * r ** 2)
    x, y, u = sp.symbols("x y u", real=True)
    P = Manifold([x, y], metric=sp.eye(2))
    w = P.one_form([-y, x])
    C = Manifold([u], ambient=P, map={x: sp.cos(u), y: sp.sin(u)})            # the unit circle
    rho, phi = sp.Symbol("rho", positive=True), sp.Symbol("phi", real=True)
    D = Manifold([rho, phi], ambient=P, map={x: rho * sp.cos(phi), y: rho * sp.sin(phi)})   # the disc
    around = C.pullback(w).integrate({u: (0, 2 * sp.pi)})
    inside = D.pullback(w.wedge_d()).integrate({rho: (0, 1), phi: (0, 2 * sp.pi)})
    assert is_zero(around - 2 * sp.pi) and is_zero(inside - around)        # Stokes
    E = tf.Manifold.euclidean(3)
    X, Y, Z = E.coords
    a, b = sp.symbols("a b", real=True)
    ball = Manifold([a, b], ambient=E, map={X: sp.sin(a) * sp.cos(b), Y: sp.sin(a) * sp.sin(b), Z: sp.cos(a)})
    flux = ball.pullback(E.vector([X, Y, Z]).lower().as_form().star()).integrate({a: (0, sp.pi), b: (0, 2 * sp.pi)})
    assert is_zero(flux - 4 * sp.pi)                                       # flux of (x, y, z) out of the unit sphere
    dx, dy = P.one_form([1, 0]), P.one_form([0, 1])
    cases = [(lambda: P.scalar(1).integrate({x: (0, 1), y: (0, 1)}), tf.IndexMismatchError),        # scalar
             (lambda: w.integrate({x: (0, 1), y: (0, 1)}), tf.IndexMismatchError),                   # degree 1 on 2D
             (lambda: dx.otimes(dy).integrate({x: (0, 1), y: (0, 1)}), tf.IndexMismatchError),       # not antisymmetric
             (lambda: dx.wedge(dy).integrate({x: (0, 1)}), tf.InvalidCoordinateError),               # y missing
             (lambda: dx.wedge(dy).integrate({x: (0, 1), u: (0, 1)}), tf.InvalidCoordinateError),    # not a coordinate
             (lambda: dx.wedge(dy).integrate({x: (0, 1), y: 1}), TypeError)]                        # not a range
    for make, kind in cases:
        try:
            make()
            assert False, kind
        except kind:
            pass
    assert dx.wedge(dy).integrate({x: (0, 2), y: (0, 3)}) == 6


def test_change_of_coordinates():
    x, y = sp.symbols("x y", real=True)
    rho, phi = sp.Symbol("rho", positive=True), sp.Symbol("phi", real=True)
    P = Manifold([x, y], metric=sp.eye(2))
    D = Manifold([rho, phi], ambient=P, map={x: rho * sp.cos(phi), y: rho * sp.sin(phi)})   # the plane, polar
    assert D.g.data.tolist() == [[1, 0], [0, rho ** 2]]
    ex = D.pullback(P.vector([1, 0])).simplify()                       # ∂_x in polar: up index, uses J⁻¹
    assert ex.data.tolist() == [sp.cos(phi), -sp.sin(phi) / rho]
    v = P.vector([x, y ** 2])
    assert is_zero((D.pullback(v).lower() @ D.pullback(v)).value - D.pullback(v.lower() @ v).value)   # length² kept
    u = sp.Symbol("u", real=True)
    C = Manifold([u], ambient=P, map={x: sp.cos(u), y: sp.sin(u)})
    try:
        C.pullback(P.vector([1, 0]))                                   # smaller: up indices still refused
        assert False
    except tf.IndexMismatchError:
        pass
    a, b = sp.symbols("a b", real=True)
    B = Manifold([x, y])                                               # no metric, so the map isn't refused at creation
    flat = Manifold([a, b], ambient=B, map={x: a + b, y: a + b})       # J has determinant 0
    try:
        flat.pullback(B.vector([1, 0]))
        assert False
    except tf.NonInvertibleError:
        pass


def test_det():
    S = tf.Manifold.sphere()
    th, _ = S.coords
    r = sp.Symbol("r", positive=True)
    assert is_zero(S.g.det().value - r ** 4 * sp.sin(th) ** 2)
    assert is_zero((S.g.det() * S.ginv.det()).value - 1)                     # det g⁻¹ = 1/det g
    mu, nu, la = S.indices(3)
    A = S.tensor(mu.up, nu.down, data=[[1, sp.sin(th)], [0, 2]])
    assert A.det().value == 2 and A.det().indices == ()
    F = S.tensor(mu.up, nu.down, la.form, data=[[[1, 0], [2, 1]], [[0, th], [1, 3]]])
    w = S.one_form([1, 0]).wedge(S.one_form([0, 1]))                      # two form indices: fine
    assert w.det().value == 1
    for bad in (lambda: S.vector([1, 0]).det(), lambda: F.det()):          # one index, three indices
        try:
            bad()
            assert False
        except tf.IndexMismatchError:
            pass


def test_swaps_need_two_indices():
    S = tf.Manifold.sphere()
    for X, hint in ((S.vector([1, 0]), "lower"), (S.covector([1, 0]), "raise_"), (S.one_form([1, 0]), None)):
        try:
            X.T
            assert False
        except tf.IndexMismatchError as e:
            assert str(e).startswith("T: swaps positions 0 and 1") and (hint is None or hint in str(e))
    try:
        S.g.S                                                   # two indices: no position 2
        assert False
    except tf.IndexMismatchError as e:
        assert str(e).startswith("S: swaps positions 1 and 2")
