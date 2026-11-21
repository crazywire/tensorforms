# tensorforms

The calculus of tensor-valued forms in Python, built on sympy. Every tensor has
**value slots** (what it is) and **form slots** (the steps it takes), separated by a bar:
R(ρ↑ σ↓ | μ ν). Everything is computed from a few operators and the formulas proved for them.

## The operators

|            | every slot              | wedged (form slots antisymmetrized) |
|------------|-------------------------|-------------------------------------|
| plain      | `d`                     | `wedge_d` = d∧                      |
| once       | `nabla` = d + Γ         | `wedge_nabla` = d∧ + Γ∧             |
| twice      | `curvature` = R         | `wedge_curvature` = R∧              |

* Γ and R come from the manifold's metric (no torsion).
* Corrections: + from the left on up slots, − from the right on down slots.
* `nabla` and `curvature` correct every slot. The wedged ones correct only value slots:
  the form-slot corrections cancel because Γ is symmetric in its lower slots.
* New form slots go first after the bar.
* With no metric, ∇ and ∇∧ act as d and d∧, and the curvatures are zero.
* Results are labelled by the operation: ∇v, ∇∧w, ∇∧∇v, …

The geometry is built with the same operations:

Γ = ½ g⁻¹(dg + dg^S − dg^ST)

R = d∧Γ + Γ∧Γ

Ric = tr(R^S)

R_s = tr(g⁻¹ Ric)

G = Ric − ½ R_s g

## Quick start

```python
import sympy as sp
import tensorforms as tf
from tensorforms import Manifold, nabla, wedge_d, wedge, curvature, contract

th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")              # 1. a bare manifold: coordinates only

w = M.one_form([sp.cos(th), 0])                # 2. tensors: a 1-form (| μ)
v = M.vector([1, 0])                           #    a vector (μ↑ |)
wedge_d(w)                                     #    d∧w needs no metric

M.metric([[1, 0], [0, sp.sin(th)**2]])         # 3. make, check and attach the metric
G = M.connection()                             #    Γ(ρ↑ σ↓ | μ)
R = M.curvature()                              #    R(ρ↑ σ↓ | μ ν)
(wedge_d(G) + wedge(G, G)).equals(R)           #    R = d∧Γ + Γ∧Γ, written as on paper
M.ricci(), M.scalar_curvature(), M.einstein()

nabla(v), curvature(v)                         # 4. the operators: ∇v, ∇∧∇v = Rv
contract(R, v)                                 # 5. products: R's column with v's row
R.tr("ρ", "μ")                                 #    a trace, by name ("rho" works) or number

S  = tf.geometry.sphere()                      # 6. presets (radius symbolic: K = 1/r²)
BH = tf.geometry.schwarzschild()               #    BH.ricci().is_zero() → True
```

## Concepts

**Manifold and spaces.** A `Manifold` has coordinates and owns its tangent space. It has no
metric until `M.metric(...)` makes, checks (shape, symmetric, invertible) and attaches one;
calling it again replaces it. `M.metric(signature="-+++")` makes an unknown metric of functions
(its signature is only declared). Γ, R, Ric, R_s and G are computed once and stored; you get copies.

**Slots and indices.** A slot has a name, a space, up or down, and a role (value or form).
Form slots are always down. Declare slots with indices: `mu, nu = M.indices(2)`, then
`mu.up`, `mu.down`, `mu.form`. Names are Greek for the tangent space (μ ν λ κ ρ σ …, θ φ π last)
and Latin for other spaces; `"mu"` means μ anywhere a name is. A slot can also be given by number
(0, 1, …), but numbers shift when d adds a slot, so names are safer in your own work.

**Types.** Each slot has a `type` (space, up/down, role); a tensor's `types` is the list.
Adding needs matching types and goes slot by slot; pairing needs the same space and one up, one down.

**One tensor** (methods): `+`, `−`, × a number or a scalar tensor; `simplify`, `equals`;
swaps `T S U V` (positions 0–1, 1–2, 2–3, 3–4), `swap`, `reorder`, `rename`;
`alt` / `sym` (default: all form slots); `tr(a, b)`; `lower(slot)` / `raise_(slot)` with the
manifold's metric (any value slot; exact inverses; form slots are never raised or lowered);
`as_form` / `as_value` to change a slot's role (same numbers).

**Products of two tensors.**

* `contract(A, B)` is your AB: A's column (last value slot, down) with B's row (first value slot, up).
  If that pair doesn't exist, it refuses: nothing is chosen silently.
* `contract(A, B, a, b)` pairs the slots you name.
* `otimes(A, B)` is the tensor product: pairs nothing.
* `wedge(A, B)` = k!/(p! q!) · Alt(AB), `odot(A, B)` = k!/(p! q!) · Sym(AB).
* Results put value slots first, then A's form slots, then B's.

**Vectors, covectors, 1-forms.** `M.vector` is (μ↑ |). `M.covector` is a row (μ↓ |): it pairs
with vectors by default, and `v.lower()` gives one. `M.one_form` is (| μ): for d∧ and wedge.
`w.as_form()` and `α.as_value()` convert between the two, so `contract(d(f).as_value(), v)` is
the directional derivative.

**Labels.** A result's label is the operation that made it: A+B, 3A, −A, R^US, Rv, A⊗B, A∧B,
Alt(A), tr(R), ∇v. Display only.

**Γ is not tensorial.** It is stored as a plain tensor so you can multiply, wedge and d it, but it
changes differently under a change of coordinates, so ∇Γ has no meaning. The code doesn't stop you.

## Presets (`tf.geometry`)

| | |
|---|---|
| `euclidean(n, coords)` | flat space; coords "cartesian" (default), "polar", "spherical", "cylindrical" |
| `sphere`, `cylinder`, `torus`, `hyperbolic_plane` | surfaces with K > 0, 0, both signs, < 0 |
| `minkowski`, `schwarzschild`, `flrw`, `de_sitter` | spacetimes, −+++, units G = c = 1 |
| `templates.diagonal`, `conformally_flat`, `static_spherical` | metrics with unknown functions, for solving |

Parameters (radius, mass, a(t), H) stay symbolic unless you give values.

## Layout

```
tensorforms/
  manifold/    manifold.py, space.py, metric.py, connection.py (Γ), curvature.py (R, Ric, R_s, G)
  tensor/      tensor.py (one tensor), products.py (two tensors), slot.py, labels.py
  operators/   d, wedge_d, nabla, wedge_nabla, curvature, wedge_curvature
  geometry/    presets and templates
  symbolic/    simplification helpers
tests/         test_core.py
examples/      tour.py
```

## Tests

```
python -m pytest tests        # or: python tests/test_core.py
```

The tests check results derived by hand (sphere Γ, R, Ricci, K; hyperbolic plane K = −1;
flat space in curved coordinates has R = 0 but Γ ≠ 0; Schwarzschild is vacuum), each operator
against its definition, Γ and R against the index formulas, ∇g = 0, the Bianchi identity
∇∧R = 0, the product rule, solving G = 0 for a static spherical metric (Schwarzschild),
Alt/Sym rules, products, raising and lowering, names, labels and error cases.

## Speed

Almost all the time goes into sympy's simplification. Schwarzschild takes about a second;
heavy metrics such as Kerr take many minutes (as in other sympy-based packages).

## Not yet included

Embeddings (E = dP), several spaces in one tensor beyond declaring them, the interior product,
declared symmetries on types, the abstract (component-free) level, connections with torsion.
