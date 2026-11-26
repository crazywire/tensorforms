# tensorforms

Exact symbolic tensor calculus with differential forms, for Riemannian geometry and general
relativity, in a matrix-style notation. Built on Python and sympy.

## Notation

Each index of a tensor is a **value** index, up (↑, contravariant) or down (↓, covariant), or a
**form** index (always down, shown with no arrow):

| object | indices |
|---|---|
| vector | (μ↑) |
| covector | (μ↓) |
| 1-form | (μ) |
| connection Γ | (ρ↑ σ↓ μ) |
| curvature R | (ρ↑ σ↓ μ ν) |
| metric g | (μ↓ ν↓) |

## Manifold

A manifold M is made from coordinates and, optionally, a metric: `Manifold(coords, metric=...)`. Then it provides:

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

| operation | call | description |
|---|---|---|
| AB | `A @ B` | product: A's column with B's row |
| | `A.contract(B, a, b)` | pair the indices you name |
| | `X.interior(V)` | X's first form index with V's row |
| A ⊗ B | `A.otimes(B)` | tensor product |
| A∧B, A⊙B | `A.wedge(B)`, `A.odot(B)` | wedge and symmetric products |
| tr X | `X.tr(a, b)` | trace over two indices |
| X^T, X^S, X^U, X^V | `X.T`, `X.S`, `X.U`, `X.V` | swap neighbouring indices |
| Alt X, Sym X | `X.alt()`, `X.sym()` | antisymmetric and symmetric parts |
| | `X.lower(a)`, `X.raise_(a)` | lower or raise a value index with g |
| ⋆X | `X.star()` | Hodge star: k form indices become n − k |
| X⁻¹ | `X.inverse()` | inverse of a tensor with exactly two value indices (g⁻¹ = g's inverse) |

## Derivatives

| derivative | call | description |
|---|---|---|
| d | `X.d()` | partial derivative |
| d∧ | `X.wedge_d()` | exterior derivative |
| ∇ = d + Γ | `X.nabla()` | covariant derivative |
| ∇∧ = d∧ + Γ∧ | `X.wedge_nabla()` | exterior covariant derivative |
| ∇∧∇ = R | `X.curvature()` | commutator of covariant derivatives |
| ∇∧∇∧ = R∧ | `X.wedge_curvature()` | exterior covariant derivative squared |
| L_v | `X.lie(v)` | Lie derivative along a vector field v (no metric needed) |

## Example: the curvature of a sphere of radius r

```python
import sympy as sp
from tensorforms import Manifold

r = sp.Symbol("r", positive=True)
th, ph = sp.symbols("theta phi", real=True)

S = Manifold([th, ph], label="S2", metric=[[r**2, 0], [0, r**2 * sp.sin(th)**2]])

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

## Documentation

| | |
|---|---|
| [Manifold](docs/manifold.md) | coordinates, the metric, Γ, R, Ric, R_s, G, ready-made manifolds |
| [Tensors](docs/tensors.md) | the notation, everything you can do with a tensor, products |
| [Derivatives](docs/derivatives.md) | d, d∧, ∇, ∇∧, the curvature derivatives, the Lie derivative |
| [Integration](docs/integration.md) | what integrates, star for scalars and flux, pullback onto submanifolds, Stokes |

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

## Licence

MIT: free to use, change and share, including in commercial software. See [LICENSE](LICENSE).
