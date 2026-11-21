# tensorforms

Differential geometry — connections, curvature, Einstein's equations — in a matrix-style
notation, computed with Python and sympy.

## What it does

tensorforms computes connections, curvature and Einstein's equations from a metric. Tensors are
written in a matrix-style notation: value slots and form slots separated by a bar, products as
column with row, and six operators for differentiating. Formulas such as

R = d∧Γ + Γ∧Γ

are written in the code the same way they are written on paper.

## The notation

A tensor is a list of **slots**, one per index. Each slot has two properties:

* **Up or down** (↑ / ↓). As in a matrix, an up slot is a row index and a down slot is a column
  index. A vector (one up slot) is a column, a covector (one down slot) is a row, and a matrix
  has one of each. Pairing always joins an up slot with a down slot.
* **Value or form.** Value slots say what the tensor *is* (a vector, a matrix, …). Form slots are
  the directions it acts on, like dx and dy in a differential form; they are always down.

A bar separates the two kinds, value slots first:

| object | slots | meaning |
|---|---|---|
| vector v | (μ↑ \|) | a column |
| covector w | (μ↓ \|) | a row |
| 1-form α | (\| μ) | one direction, e.g. α = α_μ dx^μ |
| connection Γ | (ρ↑ σ↓ \| μ) | a matrix for each direction μ |
| curvature R | (ρ↑ σ↓ \| μ ν) | a matrix for each pair of directions μ, ν |
| metric g | (μ↓ ν↓ \|) | two down value slots |

The letters are only labels; the arrows and the bar carry the meaning.

**Products.** AB pairs A's column (its last value slot, ↓) with B's row (its first value
slot, ↑), as in matrix multiplication. Everything else is carried along, value slots first,
then form slots. For example, R(ρ↑ σ↓ | μ ν) times v(κ↑ |) pairs σ with κ and gives
Rv(ρ↑ | μ ν).

**Swaps.** Slot order matters, and is changed with named swaps of neighbouring positions,
counted from 0 across all slots:

| swap | exchanges positions |
|---|---|
| T | 0 and 1 |
| S | 1 and 2 |
| U | 2 and 3 |
| V | 3 and 4 |

X^US means apply U, then S. For example, R(ρ↑ σ↓ | μ ν) with S applied is (ρ↑ μ σ↓ ν): the
second and third slots trade places. This is how Ricci is written: Ric = tr(R^S).

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
