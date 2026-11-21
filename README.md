# tensorforms

Tensor calculus with differential forms, for Riemannian geometry and general relativity,
in a matrix-style notation. Built on Python and sympy.

## What it does

tensorforms treats a tensor as an object whose slots are typed. Each slot lives in the tangent
space, is up (contravariant) or down (covariant), and is either a value slot or a form slot.

With it you can compute the Levi-Civita connection Γ, the Riemann curvature R, the Ricci tensor
Ric, the scalar curvature R_s and the Einstein tensor G, using differential operators that act
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
print(S.scalar_curvature())
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
