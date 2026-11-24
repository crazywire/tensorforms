# tensorforms

Tensor calculus with differential forms, for Riemannian geometry and general relativity,
in a matrix-style notation. Built on Python and sympy.

## What it does

tensorforms treats a tensor as an object whose indices are typed. Each index lives in the tangent
space, is up (contravariant) or down (covariant), and is either a value index or a form index.

With it you can compute the Levi-Civita connection Γ, the Riemann curvature R, the Ricci tensor
Ric, the Ricci scalar R_s and the Einstein tensor G, using differential operators that act
on the tensors.

## Notation

Each index of a tensor is **up or down** (↑ / ↓: like a matrix's row and column index) and
either a **value** index (what the tensor is) or a **form** index (a direction it acts on, like dx).
A value index carries ↑ or ↓; a form index carries no arrow (form indices are always down):

| object | indices |
|---|---|
| vector | (μ↑) |
| covector | (μ↓) |
| 1-form | (μ) |
| connection Γ | (ρ↑ σ↓ μ) |
| curvature R | (ρ↑ σ↓ μ ν) |
| metric g | (μ↓ ν↓) |

## Manifold

A manifold M is made from coordinates; `M.metric(...)` gives it a metric. Then it provides:

| tensor | method | description |
|---|---|---|
| g | `M.g` | metric (μ↓ ν↓) |
| Γ = ½ g⁻¹(dg + dg^S − dg^ST) | `M.levi_civita()` | Levi-Civita connection (ρ↑ σ↓ μ) |
| R = d∧Γ + Γ∧Γ | `M.riemann()` | Riemann curvature (ρ↑ σ↓ μ ν) |
| Ric = tr(R^S) | `M.ricci()` | Ricci tensor (μ↓ ν↓) |
| R_s = tr(g⁻¹ Ric) | `M.ricci_scalar()` | Ricci scalar |
| G = Ric − ½ R_s g | `M.einstein()` | Einstein tensor (μ↓ ν↓) |
| ε = √\|det g\| dx¹ ∧ … ∧ dxⁿ | `M.volume_form()` | volume form (μ ν …): length, area, volume, … |

Ready-made manifolds come as class methods, e.g. `Manifold.sphere()` or
`Manifold.minkowski(4, "spherical")`:

| | flat | curved |
|---|---|---|
| Riemannian (+ + +) | `euclidean(n, coords)`, `cylinder` | `sphere`, `torus`, `hyperbolic_plane` |
| Lorentzian (− + + +) | `minkowski(n, coords)` | `schwarzschild`, `flrw`, `de_sitter` |

Metric templates with unknown functions, for solving, are in `tf.templates`.

## Tensors

| operation | function | description |
|---|---|---|
| AB | `contract(A, B)` | product: A's column with B's row |
| | `contract(A, B, a, b)` | pair the indices you name |
| | `interior(X, V)` | X's first form index with V's row |
| A ⊗ B | `otimes(A, B)` | tensor product |
| A∧B, A⊙B | `wedge(A, B)`, `odot(A, B)` | wedge and symmetric products |
| tr X | `X.tr(a, b)` | trace over two indices |
| X^T, X^S, X^U, X^V | `X.T`, `X.S`, `X.U`, `X.V` | swap neighbouring indices |
| Alt X, Sym X | `X.alt()`, `X.sym()` | antisymmetric and symmetric parts |
| | `X.lower(a)`, `X.raise_(a)` | lower or raise a value index with g |
| ⋆X | `X.star()` | Hodge star: k form indices become n − k (direction ↔ plane ↔ volume …) |
| X⁻¹ | `X.inverse()` | inverse of a tensor with exactly two value indices (g⁻¹ = g's inverse) |

## Operators

| operator | function | description |
|---|---|---|
| d | `d(X)` | partial derivative |
| d∧ | `wedge_d(X)` | exterior derivative |
| ∇ = d + Γ | `nabla(X)` | covariant derivative |
| ∇∧ = d∧ + Γ∧ | `wedge_nabla(X)` | exterior covariant derivative |
| ∇∧∇ = R | `curvature(X)` | commutator of covariant derivatives |
| ∇∧∇∧ = R∧ | `wedge_curvature(X)` | exterior covariant derivative squared |
| L_v | `lie(v, X)` | Lie derivative along a vector field v (no metric needed) |

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
print(S.ricci_scalar().value)          # a scalar tensor; .value gives its expression
print(S.gaussian_curvature().value)
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
| [Manifold](docs/manifold.md) | coordinates, the metric, Γ, R, Ric, R_s, G, ready-made manifolds |
| [Tensors](docs/tensors.md) | the notation (indices and arrows), one tensor, products |
| [Operators](docs/operators.md) | the six operators and their product rules |

## Examples

Runnable scripts in `examples/` (run from the project folder, e.g. `python examples/friedmann.py`):

| | |
|---|---|
| [friedmann.py](examples/friedmann.py) | an expanding universe with a perfect fluid: G = 8πT gives the Friedmann equations |
| [charged_black_hole.py](examples/charged_black_hole.py) | Reissner–Nordström: F = d∧A, Maxwell (d∧F = 0, d∧⋆F = 0) and G = 8πT |

## Install and test

Needs Python ≥ 3.9, sympy and numpy.

```
pip install .
python tests/test_core.py
```

Schwarzschild's curvature takes about 0.3 s, Kerr's about 30 s.

## Not yet included

Connections with torsion, embeddings, several spaces in one tensor.
