# Manifold

A manifold is where everything lives: its coordinates, its tangent space, and (once you give
it one) its metric. Γ, R, Ric, R_s and G all come from the metric.

```python
import sympy as sp
from tensorforms import Manifold

th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")
```

`Manifold(coords, name="M")` — a manifold with these coordinates (sympy symbols). It has no
metric yet; tensors, d and d∧ already work.

## Making tensors

| call | gives |
|---|---|
| `M.indices(n, names=None)` | a list of n indices: μ, ν, λ, … (or the names you give) |
| `M.index(name=None)` | one index |
| `M.tensor(*slots, data=…, label=None)` | a tensor with the given slots |
| `M.scalar(value)` | no slots |
| `M.vector(data)` | (μ↑) |
| `M.covector(data)` | (μ↓), a row |
| `M.one_form(data)` | (μ) |
| `M.map(data)` | (μ↑ ν↓), a matrix |

An index gives the three kinds of slot: `mu.up`, `mu.down`, `mu.form`.

```python
mu, nu = M.indices(2)
A = M.tensor(mu.up, nu.form, data=[[1, 0], [0, sp.sin(th)]])   # A(μ↑ ν)
v = M.vector([1, 0])
```

## The metric

`M.metric(data)` makes the metric, checks it (right size, symmetric, invertible) and attaches
it. Calling it again replaces it. With no data, `M.metric(signature="-+++")` makes an unknown
metric of functions g_ij(x), for solving.


| | |
|---|---|
| `M.g` | g (μ↓ ν↓) |
| `M.ginv` | g⁻¹ (μ↑ ν↑) |
| `M.signature` | e.g. `"++"`, `"-+++"` |

## Connection and curvature

Everything follows from g (no torsion):

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

### Example: the curvature of a sphere of radius r, step by step

```python
r = sp.Symbol("r", positive=True)
S = Manifold([th, ph], name="S2")
S.metric([[r**2, 0], [0, r**2 * sp.sin(th)**2]])

G = S.levi_civita()
print(G.data[0])        # Γ^θ: how θ-components change
print(G.data[1])        # Γ^φ: how φ-components change
print(S.ricci().data)
print(S.ricci_scalar().value, S.gaussian_curvature().value)   # scalar tensors; .value gives the expression
```

```text
[[0 0]
 [0 -sin(theta)*cos(theta)]]
[[0 cos(theta)/sin(theta)]
 [cos(theta)/sin(theta) 0]]
[[1 0]
 [0 sin(theta)**2]]
2/r**2 r**(-2)
```

Γ is not zero, because the coordinate grid bends; Ric, R_s and K are not zero because the
surface itself is curved. K = 1/r², the same at every point.

Each is computed once and stored; you get a copy each time.

Γ is stored as a tensor so you can multiply and wedge it, but it is not tensorial (it changes
differently under a change of coordinates), so ∇Γ has no meaning.

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
| `Manifold.cylinder(r=None)` | r² dφ² + dz² (a plane rolled up: flat, though it looks curved from outside) |

Γ ≠ 0 in curved coordinates even though R = 0. n follows the coordinates (Cartesian default:
3 for Euclidean, 4 for Minkowski).

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

Metrics with unknown functions, for solving: plain functions in `tf.templates`. Each takes a
bare manifold, attaches the metric, and returns it.

| call | metric |
|---|---|
| `tf.templates.diagonal(M)` | diag(f0(x), f1(x), …) |
| `tf.templates.conformally_flat(M, signature=None)` | e^{2ω(x)} × the flat metric |
| `tf.templates.static_spherical(M)` | −A(r)dt² + B(r)dr² + r²dΩ², coordinates (t, r, θ, φ) |

Setting the Einstein tensor of `static_spherical` to zero and solving gives A = 1 − 2m/r,
B = 1/A: Schwarzschild.

### Example: the Friedmann equation

The Einstein tensor of the expanding universe (FLRW, flat, scale factor a(t)) gives the
equations for how the universe expands.

```python
import tensorforms as tf
U = tf.Manifold.flrw()
G = U.einstein()
print(G.data[0, 0])
```

```text
3*Derivative(a(t), t)**2/a(t)**2
```

So G_tt = 3(ȧ/a)². Einstein's equation G = 8πT with energy density ρ (T_tt = ρ) gives the
Friedmann equation

(ȧ/a)² = 8πρ/3

the expansion rate set by the density.
