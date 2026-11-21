# tensorforms

Tensor calculus with differential forms, for Riemannian geometry and general relativity,
in a matrix-style notation. Built on Python and sympy.

## What it does

tensorforms treats a tensor as an object whose slots are typed. Each slot lives in the tangent
space, is up (contravariant) or down (covariant), and is either a value slot or a form slot.

With it you can compute the Levi-Civita connection Γ, the Riemann curvature R, the Ricci tensor
Ric, the Ricci scalar R_s and the Einstein tensor G, using differential operators that act
on the tensors.

## The notation

Each slot of a tensor is **up or down** (↑ / ↓: like a matrix's row and column index) and
either a **value** slot (what the tensor is) or a **form** slot (a direction it acts on, like dx).
A bar separates them, value slots first:

| object | slots |
|---|---|
| vector | (μ↑ \|) |
| covector | (μ↓ \|) |
| 1-form | (\| μ) |
| connection Γ | (ρ↑ σ↓ \| μ) |
| curvature R | (ρ↑ σ↓ \| μ ν) |
| metric g | (μ↓ ν↓ \|) |

## The six operators

| operator | function | description |
|---|---|---|
| d | `d(X)` | partial derivative |
| d∧ | `wedge_d(X)` | exterior derivative |
| ∇ = d + Γ | `nabla(X)` | covariant derivative |
| ∇∧ = d∧ + Γ∧ | `wedge_nabla(X)` | exterior covariant derivative |
| ∇∧∇ = R | `curvature(X)` | commutator of covariant derivatives |
| ∇∧∇∧ = R∧ | `wedge_curvature(X)` | exterior covariant derivative squared |

## Manifold: metric, connection and curvature

A manifold M is made from coordinates; `M.metric(...)` gives it a metric. Then it provides:

| tensor | method | description |
|---|---|---|
| g | `M.g` | metric (μ↓ ν↓ \|) |
| Γ = ½ g⁻¹(dg + dg^S − dg^ST) | `M.levi_civita()` | Levi-Civita connection (ρ↑ σ↓ \| μ) |
| R = d∧Γ + Γ∧Γ | `M.riemann()` | Riemann curvature (ρ↑ σ↓ \| μ ν) |
| Ric = tr(R^S) | `M.ricci()` | Ricci tensor (μ↓ ν↓ \|) |
| R_s = tr(g⁻¹ Ric) | `M.ricci_scalar()` | Ricci scalar |
| G = Ric − ½ R_s g | `M.einstein()` | Einstein tensor (μ↓ ν↓ \|) |

## Example: how curved is a sphere?

A sphere of radius r has the metric r² dθ² + r² sin²θ dφ². Its curvature should not depend on
where you are, and should grow as the sphere gets smaller. Let's compute it.

```python
import sympy as sp
from tensorforms import Manifold

r = sp.Symbol("r", positive=True)
th, ph = sp.symbols("theta phi", real=True)

S = Manifold([th, ph], name="S2")
S.metric([[r**2, 0], [0, r**2 * sp.sin(th)**2]])

print(S.ricci().data)
print(S.ricci_scalar())
print(S.gaussian_curvature())
```

```text
[[1 0]
 [0 sin(theta)**2]]
2/r**2
r**(-2)
```

The Gaussian curvature is K = 1/r²: the same everywhere on the sphere, and larger for a
smaller sphere. Behind those three lines the package built Γ from g, then R = d∧Γ + Γ∧Γ, then
took the traces.

## Documentation

| | |
|---|---|
| [Manifold](docs/manifold.md) | coordinates, the metric, Γ, R, Ric, R_s, G |
| [Tensors](docs/tensors.md) | the notation (slots, arrows, the bar), one tensor, products |
| [Operators](docs/operators.md) | the six operators and their product rules |
| [Geometry](docs/geometry.md) | flat space, surfaces, spacetimes, templates |

## Install and test

Needs Python ≥ 3.9, sympy and numpy.

```
pip install .
python tests/test_core.py
```

Schwarzschild's curvature takes about 0.3 s, Kerr's about 30 s.

## Not yet included

Connections with torsion, the interior product, embeddings, several spaces in one tensor.
