# Manifold

A manifold holds the coordinates, the tangent space and (if you give one) the metric.

```python
import sympy as sp
from tensorforms import Manifold

th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], label="S2")
```

`Manifold(coords, label=None, metric=None)`: a manifold with these coordinates (sympy symbols). The label is only a name for printing.
Tensors, d, d∧ and the Lie derivative work without a metric.

## Making tensors

| call | gives |
|---|---|
| `M.indices(n, names=None)` | a list of n indices: μ, ν, λ, … (or the names you give) |
| `M.index(name=None)` | one index |
| `M.tensor(*indices, data=…, label=None)` | a tensor with the given indices |
| `M.scalar(value, label=None)` | no indices |
| `M.vector(data, index=None, label=None)` | (μ↑) |
| `M.covector(data, index=None, label=None)` | (μ↓), a row |
| `M.one_form(data, index=None, label=None)` | (μ) |

An index from `M.indices` is an up value index (↑) unless you say otherwise: `mu` is `mu.up`; `mu.down` is a down value index (↓); `mu.form` is a form index (always down, so no arrow).

```python
mu, nu = M.indices(2)
A = M.tensor(mu.up, nu.form, data=[[1, 0], [0, sp.sin(th)]])   # A(μ↑ ν)
v = M.vector([1, 0])
```

## The metric

The metric is given when the manifold is made, `Manifold(coords, metric=data)`, and never changes.
It is checked there (right size; shown symmetric; refused if shown to have determinant 0). `data` is a
matrix or a tensor with two down value indices. Without a metric the manifold is bare: forms, d and d∧
still work.


| | |
|---|---|
| `M.g` | g (μ↓ ν↓) |
| `M.ginv` | g⁻¹ (μ↑ ν↑), g's tensor inverse |

## Connection and curvature

Computed from g (Levi-Civita, no torsion):

| tensor | method | description |
|---|---|---|
| g | `M.g` | metric (μ↓ ν↓) |
| Γ = ½ g⁻¹(dg + dg^S − dg^ST) | `M.levi_civita()` | Levi-Civita connection (ρ↑ σ↓ μ) |
| R = d∧Γ + Γ∧Γ | `M.riemann()` | Riemann curvature (ρ↑ σ↓ μ ν) |
| Ric = tr(R^S) | `M.ricci()` | Ricci tensor (μ↓ ν↓) |
| R_s = tr(g⁻¹ Ric) | `M.ricci_scalar()` | Ricci scalar |
| G = Ric − ½ R_s g | `M.einstein()` | Einstein tensor (μ↓ ν↓) |
| ε = √\|det g\| dx¹ ∧ … ∧ dxⁿ | `M.volume_form()` | volume form (μ ν …): length, area, volume, … |
| K = R_s / 2 | `M.gaussian_curvature()` | Gaussian curvature (2D only), a scalar |

### Example: a sphere of radius r

```python
r = sp.Symbol("r", positive=True)
S = Manifold([th, ph], label="S2", metric=[[r**2, 0], [0, r**2 * sp.sin(th)**2]])

G = S.levi_civita()
print(G.data[0])        # Γ^θ
print(G.data[1])        # Γ^φ
print(S.ricci().data)
print(S.ricci_scalar().value, S.gaussian_curvature().value)   # .value: a scalar's expression
```

```text
[[0 0]
 [0 -sin(2*theta)/2]]
[[0 1/tan(theta)]
 [1/tan(theta) 0]]
[[1 0]
 [0 sin(theta)**2]]
2/r**2 r**(-2)
```

Each is computed once and stored; each call gives you your own tensor (changing it never changes
the stored one).

Γ is returned as a tensor so you can multiply, wedge and d it, but it isn't tensorial: ∇Γ is
computed if you ask, and has no meaning.

## Submanifolds

A manifold that sits inside another: give the ambient manifold and the map, every ambient
coordinate written in the new coordinates.

```python
x, y = sp.symbols("x y", real=True)
u = sp.Symbol("u", real=True)
P = Manifold([x, y], label="P", metric=[[1, 0], [0, 1]])
C = Manifold([u], label="C", ambient=P, map={x: sp.cos(u), y: sp.sin(u)})   # the unit circle
C.g                                   # [[1]]: du², from P's metric
C.pullback(P.one_form([-y, x])).simplify()   # [1]: −y dx + x dy on the circle is du
```

A coordinate held fixed is the same thing: the sphere r = 1 in spherical coordinates is
`Manifold([th, ph], ambient=E, map={r: 1, th: th, ph: ph})`.

With the same dimension as the ambient, it is a change of coordinates: the plane in polar
coordinates is `Manifold([rho, phi], ambient=P, map={x: rho*sp.cos(phi), y: rho*sp.sin(phi)})`,
and `pullback` then carries every tensor, up indices included.

| | |
|---|---|
| `C.ambient` | the manifold C sits in |
| `C.map` | the map as a tensor on C, (a↑): its index counts the ambient's directions |
| `C.map.d()` | E (a↑ μ), C's coordinate frame in ambient components: the derivatives of the map |
| `C.pullback(X)` | X, a tensor on the ambient, carried onto C: substitute the map, multiply each down or form index by E, and each up index by E⁻¹ (same dimension only) |

Integrating over a submanifold: see [Integration](integration.md).

The metric comes from the ambient's, so a submanifold is not given one. A map that collapses a
dimension is refused (its metric has determinant 0). Index labels: Greek in a manifold's own
tangent space, Latin in any other, so the map's index is a.

## Ready-made manifolds

Class methods on Manifold that return a manifold with its metric attached, e.g.
`Manifold.sphere()`. Parameters stay symbolic unless you give values.

| | flat | curved |
|---|---|---|
| Riemannian (+ + +) | `euclidean(n, coords)`, `cylinder` | `sphere`, `torus`, `hyperbolic_plane` |
| Lorentzian (− + + +) | `minkowski(n, coords)` | `schwarzschild`, `flrw`, `de_sitter` |

### Flat

R = 0 everywhere.

| call | metric / coordinates |
|---|---|
| `Manifold.euclidean(n=None, coords="cartesian")` | x, y, z (or x1 … xn); `"polar"` (r, φ), `"spherical"` (r, θ, φ), `"cylindrical"` (r, φ, z) |
| `Manifold.minkowski(n=None, coords="cartesian")` | t, then the same spatial coordinates: −dt² plus flat space |
| `Manifold.cylinder(r=None)` | r² dφ² + dz² |

n follows the coordinates (Cartesian default: 3 for Euclidean, 4 for Minkowski).

### Curved

Riemannian surfaces, with their Gaussian curvature K:

| call | metric | K |
|---|---|---|
| `Manifold.sphere(r=None)` | r² dθ² + r² sin²θ dφ² | 1/r² |
| `Manifold.torus(R=None, r=None)` | r² dθ² + (R + r cos θ)² dφ² | > 0 outside, < 0 inside |
| `Manifold.hyperbolic_plane()` | (dx² + dy²)/y² | −1 |

Lorentzian spacetimes, signature −+++, units G = c = 1 (a mass m is a length):

| call | metric |
|---|---|
| `Manifold.schwarzschild(m=None)` | −f dt² + dr²/f + r² dΩ², f = 1 − 2m/r (outside the horizon, r > 2m) |
| `Manifold.flrw(a=None, k=0)` | −dt² + a(t)² [dr²/(1 − kr²) + r² dΩ²] |
| `Manifold.de_sitter(H=None)` | −dt² + e^{2Ht}(dx² + dy² + dz²) |

dΩ² = dθ² + sin²θ dφ². For `Manifold.flrw`, a is an unknown a(t) by default, or a function of t
(`lambda t: t**sp.Rational(2, 3)`), or an expression in `sp.Symbol("t", real=True)`.

### Templates

Metrics with unknown functions, for solving: plain functions in `tf.templates`. Each takes the
coordinates and returns a new manifold with that metric.

| call | metric |
|---|---|
| `tf.templates.general(coords)` | g_ij(x), every component unknown |
| `tf.templates.diagonal(coords)` | diag(f0(x), f1(x), …) |
| `tf.templates.conformally_flat(coords, signs=None)` | e^{2ω(x)} × diag(signs), e.g. signs `[-1, 1, 1, 1]` |
| `tf.templates.static_spherical(coords)` | −A(r)dt² + B(r)dr² + r²dΩ², coordinates (t, r, θ, φ) |

### Example: the Friedmann equation

The Einstein tensor of flat FLRW with scale factor a(t):

```python
import tensorforms as tf
U = tf.Manifold.flrw()
G = U.einstein()
print(G.data[0, 0])
```

```text
3*Derivative(a(t), t)**2/a(t)**2
```
