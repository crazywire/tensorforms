# tensorforms

Differential geometry — connections, curvature, Einstein's equations — in a matrix-style
notation, computed with Python and sympy.

## Why

Curvature is usually written in index notation:

R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ

Most of the work is bookkeeping: which letters are summed, in what order, with which sign.
Here tensors are treated like matrices whose entries can be forms, products are "column with
row", and a few operators do all the differentiating. The same tensor becomes

R = d∧Γ + Γ∧Γ

and the code computes in that notation directly.

## The six operators

| operator | function | description |
|---|---|---|
| d | `d(X)` | the partial derivative of every component; adds one form slot |
| d∧ | `wedge_d(X)` | the exterior derivative: d, then antisymmetrized with X's form slots (the curl of a 1-form) |
| ∇ | `nabla(X)` | the covariant derivative: d + Γ on every slot |
| ∇∧ | `wedge_nabla(X)` | the exterior covariant derivative: d∧ + Γ on value slots |
| ∇∧∇ | `curvature(X)` | ∇ twice, antisymmetrized (the commutator of covariant derivatives): R acting on every slot |
| ∇∧∇∧ | `wedge_curvature(X)` | the exterior covariant derivative twice: R acting on value slots |

## Example

```python
import sympy as sp
import tensorforms as tf
from tensorforms import Manifold, nabla, curvature, contract, wedge, wedge_d

th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")
M.metric([[1, 0], [0, sp.sin(th)**2]])        # the unit sphere

G, R = M.connection(), M.curvature()          # Γ(ρ↑ σ↓ | μ), R(ρ↑ σ↓ | μ ν)
(wedge_d(G) + wedge(G, G)).equals(R)          # R = d∧Γ + Γ∧Γ  → True

v = M.vector([1, 0])
nabla(v)                                      # ∇v(μ↑ | ν)
curvature(v).equals(contract(R, v))           # ∇∧∇v = Rv  → True
M.scalar_curvature()                          # 2

tf.geometry.schwarzschild().ricci().is_zero() # vacuum  → True
```

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
