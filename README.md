# tensorforms

The calculus of tensor-valued forms, organized around four operators.

|            | every slot              | wedged (form slots antisymmetrized) |
|------------|-------------------------|-------------------------------------|
| once       | `nabla`  = d + Γ        | `wedge_nabla` = d∧ + Γ∧             |
| twice      | `curvature` = R         | `wedge_curvature` = R∧              |

Each operator corrects every slot: + from the left for up slots, − from the right for
down slots. With no connection they become the flat versions (d, d∧, and zero).

## Layout

```
tensorforms/
  manifold/    Manifold (coordinates) + optional metric, connection, curvature
  tensor/      Slot, Index, Tensor, contract, tr, outer, product, Alt/Sym, wedge, odot
  operators/   nabla, wedge_nabla, curvature, wedge_curvature, d
  geometry/    euclidean, surfaces, spacetimes, metric templates
  symbolic/    simplification helpers
tests/         every result derived by hand, checked
examples/      tour.py
```

## Quick start

```python
import sympy as sp
import tensorforms as tf
from tensorforms import Manifold, nabla, wedge_nabla, curvature, contract, tr

th, ph = sp.symbols("theta phi", positive=True)
M = Manifold([th, ph])                         # 1. a bare manifold: coordinates only

a, b, c, d = M.indices(4)                      # 2. indices, then tensors
w = M.tensor(a.form, data=[sp.cos(th), 0])     #    a covector (one form slot)
v = M.vector([1, 0])                           #    shortcut for M.tensor(a.up, ...)

wedge_nabla(w)                                 # d∧w: works with no metric

M.use_metric(M.metric([[1, 0], [0, sp.sin(th)**2]]))   # 3. attach a metric
R = M.curvature()                              # R(ρ↑ σ↓ | μ ν)
M.ricci(), M.scalar_curvature(), M.gaussian_curvature()

nabla(v), curvature(v)                         # 4. the operators
tr(R, "ρ", "μ")                                # 5. pairing is always explicit, by slot name

S  = tf.geometry.sphere()                      # 6. presets (radius symbolic: K = 1/r²)
BH = tf.geometry.schwarzschild()
```

## Conventions

* **Slots** have a name, a space, `up`/`down`, and a role: `value` or `form`.
  Declare them with index objects: `a.up`, `a.down`, `a.form`. Form slots are always down.
* **Types** are the list of (space, up/down, role). Adding needs matching types;
  pairing needs the same space and opposite up/down.
* **d** puts its new slot first among the form slots ("first after the bar").
* **Pairing is explicit**: `contract(A, B, a, b)`, `tr(X, a, b)`, `outer(A, B)`.
  `product(A, B)` is the default matrix-style product (A's last value slot with B's first).
* **Alt / Sym** act on any named slots (default: all form slots).
  `wedge(A, B) = k!/(p! q!) · Alt(AB)`, `odot(A, B) = k!/(p! q!) · Sym(AB)`.
* **Γ is not a tensor** (`Connection` is a separate class), so the operators refuse it.
* A bare `Manifold` has no metric; `M.metric(data)` makes one, `M.metric()` makes an unknown
  one (`signature="-+++"` for Lorentzian), `M.use_metric(g)` attaches it.

## Tests

```
python -m pytest tests        # or: python tests/test_core.py
```

Checks: sphere Γ, R, Ricci and K = 1 (and K = 1/r² with symbolic radius); ∇g = 0;
the curvature operator equals R on every slot; ∇∧w = d∧w (curl needs no Γ); ∇∧∇∧w = 0;
the Bianchi identity ∇∧R = 0 in a lumpy 3D space; the product rule; flat space in
spherical coordinates has R = 0 but Γ ≠ 0; hyperbolic plane K = −1; Schwarzschild is
vacuum (Ric = 0, G = 0); zero torsion; Alt/Sym projection rules; R∧v = Rv; type errors.

## Not yet included

Embeddings (E = dP), several spaces in one tensor beyond declaring them, named-axis data,
declared symmetries on types, the abstract (component-free) level, torsion beyond computing it.
