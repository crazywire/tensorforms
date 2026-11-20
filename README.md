# tensorforms

The calculus of tensor-valued forms, organized around four operators.

|            | every slot              | wedged (form slots antisymmetrized) |
|------------|-------------------------|-------------------------------------|
| once       | `nabla`  = d + Γ        | `wedge_nabla` = d∧ + Γ∧             |
| twice      | `curvature` = R         | `wedge_curvature` = R∧              |

Each operator corrects every slot: + from the left for up slots, − from the right for
down slots. With no connection they become the flat versions (d, d∧, and zero).

Each operator is computed with the formula we proved, assuming no torsion (Γ always
comes from the metric). The two wedged operators correct only value slots, because the
form-slot corrections cancel. The two curvatures use R directly (computed once per
connection) and never differentiate what they act on. The tests check every formula
against its definition (∇ twice, antisymmetrized; ∇∧ twice).

## Layout

```
tensorforms/
  manifold/    Manifold (coordinates) + optional metric, connection, curvature
  tensor/      Slot, Index, Tensor, contract, tr, otimes, Alt/Sym, wedge, odot
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

a, b, c, d = M.indices(4)                      # 2. indices (μ, ν, λ, κ), then tensors
w = M.tensor(a.form, data=[sp.cos(th), 0])     #    a covector (one form slot)
v = M.vector([1, 0])                           #    shortcut for M.tensor(a.up, ...)

wedge_nabla(w)                                 # d∧w: works with no metric

M.use_metric(M.metric([[1, 0], [0, sp.sin(th)**2]]))   # 3. attach a metric
R = M.curvature()                              # R(ρ↑ σ↓ | μ ν)
M.ricci(), M.scalar_curvature(), M.gaussian_curvature()

nabla(v), curvature(v)                         # 4. the operators
tr(R, "ρ", "μ")                                # 5. pairing is explicit: by name ("rho" works too) or number

S  = tf.geometry.sphere()                      # 6. presets (radius symbolic: K = 1/r²)
BH = tf.geometry.schwarzschild()
```

## Conventions

* **Slots** have a name, a space, `up`/`down`, and a role: `value` or `form`.
  Declare them with index objects: `a.up`, `a.down`, `a.form`. Form slots are always down.
* **Types** are the list of (space, up/down, role). Adding needs matching types;
  pairing needs the same space and opposite up/down.
* **d** puts its new slot first among the form slots ("first after the bar").
* **Names**: Greek for tangent-space slots (μ ν λ κ ρ σ τ …, with θ φ π last),
  Latin for other spaces. Spelled-out Greek is accepted
  anywhere a name is: `"mu"` means μ. New slots from d and ∇ get the first free name.
* **Slots can also be given by number** (0, 1, …). Numbers shift when d adds a slot, so prefer names.
* **Pairing**: `contract(A, B)` is your AB: A's column (last value slot, down) with B's row
  (first value slot, up). `contract(A, B, a, b)` pairs the slots you name, `tr(X, a, b)` pairs
  two slots inside X, and `otimes(A, B)` is the tensor product (pairs nothing).
  Nothing is chosen silently: when the default pair doesn't exist, contract refuses.
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
vacuum (Ric = 0, G = 0); Γ is S-symmetric; each operator equals its definition; Alt/Sym projection rules; R∧v = Rv; type errors.

## Not yet included

Embeddings (E = dP), several spaces in one tensor beyond declaring them, named-axis data,
declared symmetries on types, the abstract (component-free) level, connections with torsion.
