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
| `M.vector(data)` | (μ↑ \|) |
| `M.covector(data)` | (μ↓ \|), a row |
| `M.one_form(data)` | (\| μ) |
| `M.map(data)` | (μ↑ ν↓ \|), a matrix |

An index gives the three kinds of slot: `mu.up`, `mu.down`, `mu.form`.

```python
mu, nu = M.indices(2)
A = M.tensor(mu.up, nu.form, data=[[1, 0], [0, sp.sin(th)]])   # A(μ↑ | ν)
v = M.vector([1, 0])
```

## The metric

`M.metric(data)` makes the metric, checks it (right size, symmetric, invertible) and attaches
it. Calling it again replaces it. With no data, `M.metric(signature="-+++")` makes an unknown
metric of functions g_ij(x), for solving.


| | |
|---|---|
| `M.g` | g (μ↓ ν↓ \|) |
| `M.ginv` | g⁻¹ (μ↑ ν↑ \|) |
| `M.signature` | e.g. `"++"`, `"-+++"` |

## Connection and curvature

Everything follows from g (no torsion):

Γ = ½ g⁻¹(dg + dg^S − dg^ST)

R = d∧Γ + Γ∧Γ

Ric = tr(R^S)

R_s = tr(g⁻¹ Ric)

G = Ric − ½ R_s g

| call | gives |
|---|---|
| `M.levi_civita()` | Γ (ρ↑ σ↓ \| μ) |
| `M.riemann()` | R (ρ↑ σ↓ \| μ ν) |
| `M.ricci()` | Ric (μ↓ ν↓ \|) |
| `M.ricci_scalar()` | R_s |
| `M.einstein()` | G (μ↓ ν↓ \|) |
| `M.gaussian_curvature()` | K = R_s / 2 (2D only) |

### Example: the curvature of a sphere of radius r, step by step

```python
r = sp.Symbol("r", positive=True)
S = Manifold([th, ph], name="S2")
S.metric([[r**2, 0], [0, r**2 * sp.sin(th)**2]])

G = S.levi_civita()
print(G.data[0])        # Γ^θ: how θ-components change
print(G.data[1])        # Γ^φ: how φ-components change
print(S.ricci().data)
print(S.ricci_scalar(), S.gaussian_curvature())
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
