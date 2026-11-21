# Tensors

## The notation

A tensor is a list of **slots**. Each slot is up (↑) or down (↓), and is either a **value**
slot (part of what the tensor is) or a **form** slot (a step it takes). A bar separates them,
value slots first:

| object | slots | reads as |
|---|---|---|
| vector | (μ↑ \|) | a column |
| covector | (μ↓ \|) | a row |
| matrix | (μ↑ ν↓ \|) | row index ↑, column index ↓ |
| 1-form | (\| μ) | takes one step |
| Γ | (ρ↑ σ↓ \| μ) | a matrix-valued 1-form |
| R | (ρ↑ σ↓ \| μ ν) | a matrix-valued 2-form |
| g | (μ↓ ν↓ \|) | |

Form slots are always down. The letters are labels; the arrows and the bar carry the meaning.
Names are Greek (μ ν λ κ ρ σ …), and `"mu"` works for μ. A slot can also be given by number
(0, 1, …).

A slot's **type** is (space, up/down, role). Adding needs matching types; pairing needs one
up and one down.

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
| `X.data`, `X[i, j]` | the components |
| `X.names`, `X.types` | the slots' labels and types |
| `X.value_slots`, `X.form_slots` | positions of the value slots and of the form slots |
| `X + Y`, `X - Y`, `-X` | slot by slot (types must match) |
| `c * X` | c a number, an expression, or a scalar tensor |
| `X.simplify()`, `X.equals(Y)`, `X.is_zero()` | simplify, compare |
| `X.T()`, `X.S()`, `X.U()`, `X.V()` | swap positions 0–1, 1–2, 2–3, 3–4 |
| `X.swap(a, b)`, `X.reorder(*names)`, `X.rename({…})` | move or rename slots |
| `X.alt()`, `X.sym()` | antisymmetric / symmetric part (default: form slots) |
| `X.tr(a, b)` | pair two slots and sum |
| `X.lower(slot)`, `X.raise_(slot)` | g or g⁻¹ on a value slot; exact inverses |
| `X.as_form()`, `X.as_value()` | change the role: covector ↔ 1-form, same numbers |

**Swaps.** Slot order matters. T, S, U and V are short for swapping neighbouring slots,
counted from 0 across all slots:

| swap | short for | code |
|---|---|---|
| T | swap positions 0 and 1 | `X.T()` = `X.swap_positions(0, 1)` |
| S | swap positions 1 and 2 | `X.S()` = `X.swap_positions(1, 2)` |
| U | swap positions 2 and 3 | `X.U()` = `X.swap_positions(2, 3)` |
| V | swap positions 3 and 4 | `X.V()` = `X.swap_positions(3, 4)` |

Any other swap: `X.swap(a, b)` (by name or position) or `X.reorder(...)`.

X^US means apply U, then S: `X.U().S()`. For example, R(ρ↑ σ↓ | μ ν) with S applied is
(ρ↑ μ σ↓ ν): the second and third slots trade places. A result's label says how it was made:
R^US, −A, Alt(R), tr(R), …

### Example: Ricci from Riemann by a trace

Ric = tr(R^S): swap R's second and third slots, then pair the first two.

```python
R = S.riemann()                 # R(ρ↑ σ↓ | μ ν)
Ric = R.S().tr(0, 1)
print(Ric)
print(Ric.simplify().data)
```

```text
tr(R^S)(σ↓ | ν)
[[1 0]
 [0 sin(theta)**2]]
```

The trace leaves one value slot and one form slot; the numbers are the Ricci tensor.

Raising and lowering never act on form slots; use `as_value()` first.

## Products

**AB is A's column with B's row**: A's last value slot (↓) pairs with B's first value slot (↑),
as in matrix multiplication. Everything else is carried along: value slots first, then A's
form slots, then B's.

Rv: (ρ↑ σ↓ | μ ν)(κ↑ |) → (ρ↑ | μ ν)

If the meeting slots are both up or both down, AB isn't defined (like column × column).

The wedge of A (p form slots) and B (q form slots), each first made antisymmetric:

A∧B = k!/(p! q!) · Alt(AB),  k = p + q

| call | is |
|---|---|
| `contract(A, B)` | AB; refuses if the pair doesn't exist |
| `contract(A, B, a, b)` | pair the slots you name |
| `interior(X, V)` | the interior product: like `contract(X, V)`, but X's first form slot with V's row |
| `otimes(A, B)` | A ⊗ B: pair nothing |
| `wedge(A, B)`, `odot(A, B)` | A∧B, A⊙B |

### Example: how long is the φ direction on a sphere?

The length² of a vector v is the row (lowered v) times the column v.

```python
e_phi = S.vector([0, 1])
length2 = contract(e_phi.lower(), e_phi)
print(length2.data.item())
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
print(Rv)
print(Rv.simplify().data[:, 0, 1])     # the θφ part
```

```text
Rv(ρ↑ | μ ν)
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
print(interior(d(f), v).data.item())
```

```text
-sin(theta)
```

Moving south (increasing θ), the height drops at rate sin θ: fastest at the equator, zero at
the poles.

