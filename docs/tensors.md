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

The examples below continue from [manifold.md](manifold.md):

```python
import sympy as sp
from tensorforms import Manifold, contract, otimes, wedge
th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")
M.metric([[1, 0], [0, sp.sin(th) ** 2]])
R, v, w = M.curvature(), M.vector([1, 0]), M.covector([0, 1])
alpha = M.one_form([sp.cos(th), 0])
```

## One tensor

| call | does |
|---|---|
| `X.data`, `X[i, j]` | the components |
| `X.names`, `X.value_slots`, `X.form_slots`, `X.types` | the slots |
| `X + Y`, `X - Y`, `-X` | slot by slot (types must match) |
| `c * X` | c a number, an expression, or a scalar tensor |
| `X.simplify()`, `X.equals(Y)`, `X.is_zero()` | simplify, compare |
| `X.T()`, `X.S()`, `X.U()`, `X.V()` | swap positions 0–1, 1–2, 2–3, 3–4 |
| `X.swap(a, b)`, `X.reorder(*names)`, `X.rename({…})` | move or rename slots |
| `X.alt()`, `X.sym()` | antisymmetric / symmetric part (default: form slots) |
| `X.tr(a, b)` | pair two slots and sum |
| `X.lower(slot)`, `X.raise_(slot)` | g or g⁻¹ on a value slot; exact inverses |
| `X.as_form()`, `X.as_value()` | change the role: covector ↔ 1-form, same numbers |

So X^US is `X.U().S()`. A result's label says how it was made: R^US, −A, Alt(R), tr(R), …

```python
R.tr("ρ", "μ")          # tr(R)(σ↓ | ν): the Ricci numbers
R.lower("ρ")            # R with ρ down
v.lower()               # a covector
```

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
| `otimes(A, B)` | A ⊗ B: pair nothing |
| `wedge(A, B)`, `odot(A, B)` | A∧B, A⊙B |

```python
contract(R, v)                  # Rv(ρ↑ | μ ν)
contract(w, v)                  # row × column: a number
contract(alpha, v, "μ", "μ")    # a 1-form on a vector: name the slots
otimes(v, v)                    # v ⊗ v (μ↑ ν↑ |)
```
