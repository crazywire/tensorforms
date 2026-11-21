# tensorforms

Differential geometry — connections, curvature, Einstein's equations — written in a
matrix-style notation, and computed by a small Python package built on sympy.

## What problem this solves

Curvature calculations are usually done in index notation:

R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ

Every term is a sum over repeated letters, and most of the work is bookkeeping: which
index is summed, which order they go in, where the signs come from. It is easy to get wrong,
and the structure (what multiplies what) is hidden behind the letters.

This project treats the same objects the way you treat matrices. A tensor is a matrix whose
entries can be forms; products are "column with row"; and four operators do all the
differentiating. The same Riemann tensor becomes

R = d∧Γ + Γ∧Γ

The package computes in that notation directly: you build tensors, multiply them, apply the
operators, and get components back — with each formula in the code written the way it is
written on paper.

The notation is Abdullahi's own. The formulas themselves are standard results (Levi-Civita,
Cartan, the graded Leibniz rule) rewritten in it.

## The notation

### Slots, arrows and the bar

Every tensor is a list of **slots**. Each slot is up (↑) or down (↓), and is either a
**value** slot (part of what the tensor *is*) or a **form** slot (a step direction it takes).
A bar separates them, value slots first:

| object | slots | reads as |
|---|---|---|
| vector v | (μ↑ \|) | a column |
| covector w | (μ↓ \|) | a row |
| matrix A | (μ↑ ν↓ \|) | a matrix: row index ↑, column index ↓ |
| 1-form α | (\| μ) | takes one step |
| connection Γ | (ρ↑ σ↓ \| μ) | a matrix-valued 1-form |
| curvature R | (ρ↑ σ↓ \| μ ν) | a matrix-valued 2-form |
| metric g | (μ↓ ν↓ \|) | two down value slots |

Form slots are always down. The letters are only labels; the arrows and the bar carry the
information.

### Products

The default product AB is **A's column with B's row**: A's last value slot (↓) pairs with
B's first value slot (↑), as in matrix multiplication. Everything else is carried along:
value slots first, then A's form slots, then B's.

R v: (ρ↑ σ↓ | μ ν)(κ↑ |) → (ρ↑ | μ ν),  σ paired with κ

If the meeting slots are both up or both down, AB is not defined (a shape mismatch), just as
a column times a column isn't. The tensor product A ⊗ B (pair nothing) is written explicitly.

### Swaps

Slot order matters, and is changed with explicit swaps, counted over all slots:

T swaps positions 0,1    S swaps 1,2    U swaps 2,3    V swaps 3,4

so X^US means: apply U, then S.

### Antisymmetrizing and the wedge

Alt (and Sym) average over rearrangements of the chosen slots, signed (or not), divided by k!.
The wedge of A (p form slots) and B (q form slots) is

A∧B = k!/(p! q!) · Alt(AB),  k = p + q

where each factor is made antisymmetric in its own form slots first. ⊙ is the same with Sym.

## The operators

|            | every slot              | wedged (form slots antisymmetrized) |
|------------|-------------------------|-------------------------------------|
| plain      | d                       | d∧                                  |
| once       | ∇ = d + Γ               | ∇∧ = d∧ + Γ∧                        |
| twice      | ∇∧∇ = R                 | ∇∧∇∧ = R∧                           |

* **d** adds one form slot, first after the bar: (dX) has components ∂X.
* **∇** adds Γ for every slot: +Γ from the left on up slots, −Γ from the right on down slots.
* **∇∧** is d∧ plus Γ on **value slots only**. The corrections on form slots cancel, because
  Γ is symmetric in its two lower slots (no torsion).
* **∇∧∇** (applying ∇ twice and antisymmetrizing the two new slots) is **R acting on every
  slot**, with no derivatives left.
* **∇∧∇∧** is **R on value slots**, times C(p+2; 2, p).

### Product rules

For the plain operator, the new slot has to be moved back first after the bar with swaps:

∇(Rv) = (∇R)v + (R∇v)^US

For the wedged operators, swaps become signs (p = number of form slots of A):

∇∧(A∧B) = (∇∧A)∧B + (−1)^p A∧(∇∧B)

∇∧∇∧(A∧B) = (∇∧∇∧A)∧B + A∧(∇∧∇∧B)

For a plain product AB, rewrite it as a wedge first:

AB = p! q!/(p+q)! · A∧B

(the factor is 1 when either has no form slots, as in Rv).

## Geometry from the metric

Given a metric g, everything follows (no torsion):

Γ = ½ g⁻¹(dg + dg^S − dg^ST)

R = d∧Γ + Γ∧Γ

Ric = tr(R^S)

R_s = tr(g⁻¹ Ric)

G = Ric − ½ R_s g

Γ is not itself a tensor (it changes differently under a change of coordinates), so ∇Γ has
no meaning — but the combinations above are tensors.

## In code

Each line of the notation has a direct counterpart:

```python
import sympy as sp
import tensorforms as tf
from tensorforms import Manifold, d, wedge_d, nabla, wedge_nabla, curvature, contract, wedge

th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")             # coordinates only, no metric yet

v = M.vector([1, 0])                          # (μ↑ |)
alpha = M.one_form([sp.cos(th), 0])           # (| μ)
wedge_d(alpha)                                # d∧α: needs no metric

M.metric([[1, 0], [0, sp.sin(th)**2]])        # g for the unit sphere
G = M.connection()                            # Γ  (ρ↑ σ↓ | μ)
R = M.curvature()                             # R  (ρ↑ σ↓ | μ ν)
(wedge_d(G) + wedge(G, G)).equals(R)          # R = d∧Γ + Γ∧Γ  → True

nabla(v)                                      # ∇v  (μ↑ | ν)
curvature(v).equals(contract(R, v))           # ∇∧∇v = R v  → True
M.ricci(), M.scalar_curvature()               # Ric = diag(1, sin²θ),  R_s = 2

tf.geometry.schwarzschild().ricci().is_zero() # vacuum: Ric = 0  → True
```

| notation | code |
|---|---|
| AB (column with row) | `contract(A, B)` |
| A ⊗ B | `otimes(A, B)` |
| A∧B, A⊙B | `wedge(A, B)`, `odot(A, B)` |
| tr over two slots | `X.tr(a, b)` |
| X^T, X^S, X^U, X^V | `X.T()`, `X.S()`, `X.U()`, `X.V()` |
| Alt(X), Sym(X) | `X.alt()`, `X.sym()` |
| raise / lower a value slot with g | `X.raise_(slot)`, `X.lower(slot)` |
| d, d∧, ∇, ∇∧, ∇∧∇, ∇∧∇∧ | `d`, `wedge_d`, `nabla`, `wedge_nabla`, `curvature`, `wedge_curvature` |
| Γ, R, Ric, R_s, G | `M.connection()`, `M.curvature()`, `M.ricci()`, `M.scalar_curvature()`, `M.einstein()` |

Results are labelled by how they were made (∇v, Rv, R^US, Alt(R), …), so a printout shows
where an object came from.

## Details

* **Declaring tensors.** `mu, nu = M.indices(2)`, then `M.tensor(mu.up, nu.form, data=...)`.
  Shortcuts: `M.vector`, `M.covector` (a row), `M.one_form`, `M.scalar`.
* **Names.** Greek for the tangent space (μ ν λ κ ρ σ …); `"mu"` works for μ. Slots can also be
  given by number (0, 1, …), but numbers shift when d adds a slot.
* **Covectors vs 1-forms.** Same numbers, different role: a covector (μ↓ |) multiplies like a
  row; a 1-form (| μ) is wedged and d∧'d. `as_form()` / `as_value()` convert.
* **Adding** goes slot by slot and needs matching types (space, up/down, role).
* **The metric.** `M.metric(data)` makes, checks (symmetric, invertible) and attaches it;
  `M.metric(signature="-+++")` makes an unknown one to solve for.
* **Presets.** `tf.geometry`: flat space (`euclidean`), `sphere`, `cylinder`, `torus`,
  `hyperbolic_plane`, `minkowski`, `schwarzschild`, `flrw`, `de_sitter` (units G = c = 1),
  and templates with unknown functions (`static_spherical`: solving G = 0 gives Schwarzschild).

## Tests and speed

```
python tests/test_core.py
```

The tests check results derived by hand (sphere, hyperbolic plane, flat space in curved
coordinates, Schwarzschild), each operator against its definition, Γ and R against the index
formulas, ∇g = 0, the Bianchi identity ∇∧R = 0, the product rule, and solving G = 0.

Simplification uses a direct recipe (sin and cos as variables, exact cancellation,
sin² + cos² = 1) before sympy's general simplify. Schwarzschild takes about 0.3 s and
Kerr about 30 s.

## Layout

```
tensorforms/
  manifold/    manifold, space, metric, connection (Γ), curvature (R, Ric, R_s, G)
  tensor/      tensor (one tensor), products (two tensors), slot, labels
  operators/   d, wedge_d, nabla, wedge_nabla, curvature, wedge_curvature
  geometry/    presets and templates
  symbolic/    simplification
```

## Not yet included

Connections with torsion, the interior product (feeding a vector into a form slot),
embeddings, several spaces in one tensor, and the component-free level.
