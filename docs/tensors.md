# Tensors

## The notation

A tensor has numbers and an ordered tuple of **indices**. Each index is a **value** index, up (↑)
or down (↓), or a **form** index (always down, shown with no arrow):

| object | indices |
|---|---|
| vector | (μ↑) |
| covector | (μ↓) |
| matrix | (μ↑ ν↓): row ↑, column ↓ |
| 1-form | (μ) |
| Γ | (ρ↑ σ↓ μ) |
| R | (ρ↑ σ↓ μ ν) |
| g | (μ↓ ν↓) |

Labels are only for display and for pointing at an index; everything is computed by position.
The package hands out Greek labels (μ ν λ κ ρ σ …); `"mu"` works for μ, and an index can also be
given by number (0, 1, …).

Indices **match** (`a.matches(b)`: same space, up/down and type, any label) to be added or
averaged, and **pair** (`a.pairs_with(b)`: same space, one up and one down; a form index counts as
down) to be contracted.

The examples below use a sphere of radius r:

```python
import sympy as sp
from tensorforms import Manifold
r = sp.Symbol("r", positive=True)
th, ph = sp.symbols("theta phi", real=True)
S = Manifold([th, ph], name="S2")
S.metric([[r**2, 0], [0, r**2 * sp.sin(th)**2]])
```

## One tensor

| call | does |
|---|---|
| `X.data`, `X[i, j]` | the components (numpy, read-only) |
| `X[i, j] = v` | change a component; copy on write, so no other tensor changes |
| `X.value` | the components in sympy form: a scalar's expression, otherwise a sympy Array |
| `X.indices` | the indices, in order (a tuple): each has `label`, `variance`, `type` |
| `X.matches(Y)` | True if X and Y can be added: same number of indices, each matching (space, up/down, type; any label) |
| `X + Y`, `X - Y`, `-X` | index by index (indices must match); a number counts as a scalar, and 0 as the zero tensor of any type |
| `c * X`, `X / c` | c a number, an expression, or a scalar tensor; exact (`X / 2` is ½X, never 0.5X) |
| `X.simplify()` | simplify every component (exact rewrites only) |
| `X.subs(r, 1)`, `X.subs({m: 1})` | substitute for parameters in every component (still a field); coordinates are refused |
| `X.at({θ: π/2, φ: 0})` | the components at a point (every coordinate given): plain values, not a tensor |
| `X.is_zero()`, `X.equals(Y)` | sympy's answer: True (simplifies to 0), False (sympy proves nonzero), None (can't tell); never sampled |
| `sp.diff(s, θ)`, `sp.solve(s, r)`, … | sympy accepts tensors directly: a scalar as its expression, others as a sympy Array (one axis per index) |
| `X.T`, `X.S`, `X.U`, `X.V` | swap positions 0–1, 1–2, 2–3, 3–4 |
| `X.swap(a, b)`, `X.reorder(*indices)` | move indices (by label or position) |
| `X.relabel_indices({"μ": "α"})` | new labels for the indices you name (by label or position); numbers and the tensor's label stay |
| `X.alt()`, `X.sym()` | antisymmetric / symmetric part (default: form indices) |
| `X.tr(a, b)` | pair two indices and sum |
| `X.lower(index)`, `X.raise_(index)` | g or g⁻¹ on a value index; exact inverses |
| `X.as_form()`, `X.as_value()` | change the type: covector ↔ 1-form, same numbers |
| `X.star()` | Hodge star: k form indices become n − k, same amount (uses the metric) |
| `X.inverse()` | for a matrix at each point (exactly two value indices, nothing else): the matrix inverse; the indices swap places and flip arrows, g(μ↓ ν↓) → g⁻¹(ν↑ μ↑) |

**Swaps.** T, S, U and V swap neighbouring indices, counted from 0 across all indices: each
transposes the matrix in those two indices at every value of the others. They are properties
(no brackets) and chain: `X.U.S` is X^US.

| swap | short for | code |
|---|---|---|
| T | swap positions 0 and 1 | `X.T` = `X.swap(0, 1)` |
| S | swap positions 1 and 2 | `X.S` = `X.swap(1, 2)` |
| U | swap positions 2 and 3 | `X.U` = `X.swap(2, 3)` |
| V | swap positions 3 and 4 | `X.V` = `X.swap(3, 4)` |

Any other swap: `X.swap(a, b)` (by label or position) or `X.reorder(...)`.

**Labels.** A tensor's label is a name you give (or the manifold's: g, Γ, R, Ric, …). Swaps,
reorders, index relabelling, raising/lowering, `as_form`/`as_value` and `simplify` keep it;
anything computed has none and prints as `Tensor(…)`. Set one with `X.label = "K"`.

### Example: Ric = tr(R^S)

```python
R = S.riemann()                 # R(ρ↑ σ↓ μ ν)
Ric = R.S.tr(0, 1)
Ric.label = "Ric"               # a computed result has no label until you name it
print(Ric)
print(Ric.simplify().data)
```

```text
Ric(σ↓ ν)
[[1 0]
 [0 sin(theta)**2]]
```

Raising and lowering act on value indices only; use `as_value()` first for a form index.

### The Hodge star

`X.star()` raises X's k form indices, fills them in order into the volume form ε (n form
indices) and divides by k!, leaving n − k form indices. Value indices are carried along. It needs
a metric, and k ≤ n.

### Example: curl and divergence

curl v = ⋆ d∧ v♭ and div v = ⋆ d∧ ⋆ v♭, where v♭ = `v.lower().as_form()`:

```python
import tensorforms as tf
E = tf.Manifold.euclidean()
x, y, z = E.coords

rotation = E.vector([-y, x, 0]).lower().as_form()
print(rotation.wedge_d().star().data.tolist())

outward = E.vector([x, y, z])
flat = outward.lower().as_form()
print(flat.star().wedge_d().star().value, outward.nabla().tr(0, 1).value)
```

```text
[0, 0, 2]
3 3
```

## Products

`A @ B` is AB: A's column (last value index, ↓) with B's row (first value index, ↑). It refuses
if both are up or both down. Every product puts value indices first, then A's form indices, then
B's: R @ v is (ρ↑ σ↓ μ ν)(κ↑) → (ρ↑ μ ν).

The wedge of A (p form indices) and B (q form indices), each first made antisymmetric:

A∧B = k!/(p! q!) · Alt(AB),  k = p + q

| call | is |
|---|---|
| `A @ B` | AB; refuses if the pair doesn't exist |
| `A.contract(B, a, b)` | pair the indices you name |
| `X.interior(V)` | the interior product: like `X @ V`, but X's first form index with V's row |
| `A.otimes(B)` | A ⊗ B: pair nothing |
| `A.wedge(B)`, `A.odot(B)` | A∧B, A⊙B |

### Example: the length² of e_φ

```python
e_phi = S.vector([0, 1])
length2 = e_phi.lower() @ e_phi
print(length2.value)
```

```text
r**2*sin(theta)**2
```

### Example: R times a vector

```python
e_theta = S.vector([1, 0])
Rv = R @ e_theta
Rv.label = "Rv"
print(Rv)
print(Rv.simplify().data[:, 0, 1])     # the θφ part
```

```text
Rv(ρ↑ μ ν)
[0 -1]
```

### Example: df fed a direction (interior product)

```python
f = S.scalar(sp.cos(th))
v = S.vector([1, 0])                  # e_θ
print(f.d().interior(v).value)
```

```text
-sin(theta)
```
