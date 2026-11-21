# tensorforms

Differential geometry — connections, curvature, Einstein's equations — in a matrix-style
notation, computed with Python and sympy.

## What it does

tensorforms computes connections, curvature and Einstein's equations from a metric. Tensors are
written in a matrix-style notation: value slots and form slots separated by a bar, products as
column with row, and six operators for differentiating. Formulas such as

R = d∧Γ + Γ∧Γ

are written in the code the same way they are written on paper.

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
