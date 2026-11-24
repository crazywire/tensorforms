# Tensors

## The notation

A tensor is a list of **indices**. Each index is up (↑) or down (↓), and is either a **value**
index (part of what the tensor is) or a **form** index (a step it takes). A value index carries ↑ or
↓; a form index carries no arrow:

| object | indices | reads as |
|---|---|---|
| vector | (μ↑) | a column |
| covector | (μ↓) | a row |
| matrix | (μ↑ ν↓) | row index ↑, column index ↓ |
| 1-form | (μ) | takes one step |
| Γ | (ρ↑ σ↓ μ) | a matrix-valued 1-form |
| R | (ρ↑ σ↓ μ ν) | a matrix-valued 2-form |
| g | (μ↓ ν↓) | |

Form indices are always down, so they need no arrow. The letters are labels; the arrows carry the
meaning. Labels can be anything you choose; the ones the package hands out are Greek
(μ ν λ κ ρ σ …), and `"mu"` works for μ. An index can also be given by number (0, 1, …).

An index's **type** is value or form. Two rules use the rest too: indices **match**
(`a.matches(b)`: same space, up/down and type, any label) to be added or averaged, and **pair**
(`a.pairs_with(b)`: same space, one up and one down; a form index is down) to be contracted.

The examples below use a sphere of radius r:

```python
import sympy as sp
from tensorforms import Manifold, contract
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
| `X + Y`, `X - Y`, `-X` | index by index (indices must match); a number counts as a scalar, and 0 as the zero tensor of any type |
| `c * X`, `X / c` | c a number, an expression, or a scalar tensor; exact (`X / 2` is ½X, never 0.5X) |
| `X.simplify()` | simplify every component (exact rewrites only) |
| `X.is_zero()`, `X.equals(Y)` | sympy's answer: True (simplifies to 0), False (sympy proves nonzero), None (can't tell); never sampled |
| `sp.diff(s, θ)`, `sp.solve(s, r)`, … | sympy accepts tensors directly: a scalar as its expression, others as a sympy Array (one axis per index) |
| `X.T`, `X.S`, `X.U`, `X.V` | swap positions 0–1, 1–2, 2–3, 3–4 |
| `X.swap(a, b)`, `X.reorder(*indices)` | move indices (by label or position) |
| `S.tensor(*(i.relabel(…) …), data=X.data)` | new index labels: relabel the indices you want with `Index.relabel`, then build the tensor |
| `X.alt()`, `X.sym()` | antisymmetric / symmetric part (default: form indices) |
| `X.tr(a, b)` | pair two indices and sum |
| `X.lower(index)`, `X.raise_(index)` | g or g⁻¹ on a value index; exact inverses |
| `X.as_form()`, `X.as_value()` | change the type: covector ↔ 1-form, same numbers |
| `X.star()` | Hodge star: k form indices become n − k, same amount (uses the metric) |
| `X.inverse()` | for a matrix at each point (exactly two value indices, nothing else): the matrix inverse; the indices swap places and flip arrows, g(μ↓ ν↓) → g⁻¹(ν↑ μ↑) |

**Swaps.** Index order matters. T, S, U and V swap neighbouring indices, counted from 0 across
all indices. Each transposes the matrix in those two indices at every value of the others (for
R(ρ↑ σ↓ μ ν), T transposes the (ρ, σ) matrix at each μ, ν). They are properties, so no brackets:

| swap | short for | code |
|---|---|---|
| T | swap positions 0 and 1 | `X.T` = `X.swap(0, 1)` |
| S | swap positions 1 and 2 | `X.S` = `X.swap(1, 2)` |
| U | swap positions 2 and 3 | `X.U` = `X.swap(2, 3)` |
| V | swap positions 3 and 4 | `X.V` = `X.swap(3, 4)` |

Any other swap: `X.swap(a, b)` (by label or position) or `X.reorder(...)`.

X^US means apply U, then S: `X.U.S`. For example, R(ρ↑ σ↓ μ ν) with S applied is
(ρ↑ μ σ↓ ν): the second and third indices trade places.

**Labels.** A tensor's label is just a name: one you give, or the manifold's (g, Γ, R, Ric, …).
Changes that keep the tensor keep the name: swaps and reorders, index relabelling, raising and
lowering, form ↔ value, simplify. Anything computed (sums, products, traces, derivatives, Alt, …)
has no label and prints as `Tensor(…)`; its indices carry the information. Name a result yourself
with `X.label = "K"`.

### Example: Ricci from Riemann by a trace

Ric = tr(R^S): swap R's second and third indices, then pair the first two.

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

The trace leaves one value index and one form index; the numbers are the Ricci tensor.

Raising and lowering never act on form indices; use `as_value()` first.

### The Hodge star

⋆ pairs each set of directions with the directions left over, keeping the amount. In 3D:

number ↔ volume

direction ↔ plane

It raises X's k form indices and fills them, in order, into the volume form ε (n form indices),
then divides by k! (each plane is stored k! times). What is left are n − k form indices. Value
indices are carried along. For a 1-form α:

⋆α = ι(ε, α raised)

### Example: curl and divergence with ⋆

d∧ does the work (circulation per plane, outflow per box); ⋆ turns the result into the shape you
want.

curl v = ⋆ d∧ v♭

div v = ⋆ d∧ ⋆ v♭

where v♭ is v as a 1-form (`v.lower().as_form()`).

```python
import tensorforms as tf
from tensorforms import wedge_d, nabla
E = tf.Manifold.euclidean()
x, y, z = E.coords

rotation = E.vector([-y, x, 0]).lower().as_form()
print(wedge_d(rotation).star().data.tolist())

outward = E.vector([x, y, z])
flat = outward.lower().as_form()
print(wedge_d(flat.star()).star().value, nabla(outward).tr(0, 1).value)
```

```text
[0, 0, 2]
3 3
```

The rotation spins 2 per unit area in the xy plane, so its curl points along z. The outward flow
has divergence 3, the same as tr(∇v).

## Products

**AB is A's column with B's row**: A's last value index (↓) pairs with B's first value index (↑),
as in matrix multiplication. Everything else is carried along: value indices first, then A's
form indices, then B's.

Rv: (ρ↑ σ↓ μ ν)(κ↑) → (ρ↑ μ ν)

If the meeting indices are both up or both down, AB isn't defined (like column × column).

The wedge of A (p form indices) and B (q form indices), each first made antisymmetric:

A∧B = k!/(p! q!) · Alt(AB),  k = p + q

| call | is |
|---|---|
| `contract(A, B)` | AB; refuses if the pair doesn't exist |
| `contract(A, B, a, b)` | pair the indices you name |
| `interior(X, V)` | the interior product: like `contract(X, V)`, but X's first form index with V's row |
| `otimes(A, B)` | A ⊗ B: pair nothing |
| `wedge(A, B)`, `odot(A, B)` | A∧B, A⊙B |

### Example: how long is the φ direction on a sphere?

The length² of a vector v is the row (lowered v) times the column v.

```python
e_phi = S.vector([0, 1])
length2 = contract(e_phi.lower(), e_phi)
print(length2.value)
```

```text
r**2*sin(theta)**2
```

So e_φ has length r sin θ: the radius of the circle of latitude, which shrinks to zero at the poles.

### Example: what does curvature do to a vector?

Multiply R by e_θ (R's column with the vector's row):

```python
e_theta = S.vector([1, 0])
Rv = contract(R, e_theta)
Rv.label = "Rv"
print(Rv)
print(Rv.simplify().data[:, 0, 1])     # the θφ part
```

```text
Rv(ρ↑ μ ν)
[0 -1]
```

Rv is a vector-valued 2-form: for a small loop in the θ and φ directions, it rotates e_θ into
−e_φ. On a flat surface it would be zero.

### Example: how fast does the height change as you move south?

On a sphere, cos θ measures height (1 at the north pole, −1 at the south pole). Feeding the
direction e_θ into its derivative df gives the rate of change along e_θ:

```python
from tensorforms import d, interior
f = S.scalar(sp.cos(th))              # height on the sphere
v = S.vector([1, 0])                  # e_θ: moving south
print(interior(d(f), v).value)
```

```text
-sin(theta)
```

Moving south (increasing θ), the height drops at rate sin θ: fastest at the equator, zero at
the poles.

