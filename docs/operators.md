# Operators

| operator | function | description |
|---|---|---|
| d | `d(X)` | partial derivative |
| d∧ | `wedge_d(X)` | exterior derivative |
| ∇ | `nabla(X)` | covariant derivative: d + Γ on every slot |
| ∇∧ | `wedge_nabla(X)` | exterior covariant derivative: d∧ + Γ on value slots |
| ∇∧∇ | `curvature(X)` | curvature operator: R on every slot |
| ∇∧∇∧ | `wedge_curvature(X)` | exterior covariant derivative twice: R on value slots |

Each takes one tensor and returns a new one, labelled by the operator (dX, ∇X, …).
New form slots go first after the bar. Γ and R come from the manifold; with no metric, ∇ and
∇∧ act as d and d∧, and the two curvature operators are zero.

Details:

* Γ corrects up slots with +Γ from the left and down slots with −Γ from the right.
* ∇∧ only corrects value slots: the corrections on form slots cancel, because Γ is symmetric in
  its two lower slots (no torsion).
* d∧ multiplies by p + 1 after antisymmetrizing, and ∇∧∇∧ by C(p+2; 2, p), where p is the number
  of form slots of X.
* The two curvature operators never differentiate X; they only multiply by R.

```python
import sympy as sp
from tensorforms import Manifold, d, wedge_d, nabla, wedge_nabla, curvature, wedge_curvature, contract
th, ph = sp.symbols("theta phi", real=True)
M = Manifold([th, ph], name="S2")
M.metric([[1, 0], [0, sp.sin(th) ** 2]])
v, alpha = M.vector([1, 0]), M.one_form([sp.cos(th), 0])

nabla(v)                                # ∇v(μ↑ | ν)
wedge_d(alpha)                          # d∧α, the curl
curvature(v).equals(contract(M.curvature(), v))   # ∇∧∇v = Rv → True
wedge_nabla(alpha).equals(wedge_d(alpha))         # ∇∧ of a 1-form needs no Γ → True
```

## Product rules

A and B are any two tensors. **AB** is their product as defined in [tensors.md](tensors.md):
A's column (last value slot, ↓) paired with B's row (first value slot, ↑), everything else
carried along. **p** and **q** are the numbers of form slots of A and B.

**∇ (every slot).** Each term keeps all slots, but the new slot from ∇ must end up first after
the bar, so terms where it lands later are moved back with swaps. For example, with
R (ρ↑ σ↓ | μ ν) and a vector v:

∇(Rv) = (∇R)v + (R∇v)^US

In R∇v the new slot comes last (after μ ν), and US moves it to the front.

**∇∧ (wedged).** For a wedge product, swaps become signs:

∇∧(A∧B) = (∇∧A)∧B + (−1)^p A∧(∇∧B)

**∇∧∇∧.** No signs and no cross terms:

∇∧∇∧(A∧B) = (∇∧∇∧A)∧B + A∧(∇∧∇∧B)

**Applying ∇∧ or ∇∧∇∧ to a plain product AB.** These operators only see the antisymmetric part
of AB, which is a multiple of A∧B. So rewrite first, then use the rules above:

∇∧(AB) = p! q!/(p+q)! · ∇∧(A∧B)

∇∧∇∧(AB) = p! q!/(p+q)! · ∇∧∇∧(A∧B)

The factor is 1 when A or B has no form slots, so then AB = A∧B. Example, Rv (p = 2, q = 0):

∇∧(Rv) = (∇∧R)v + R∧(∇v) = R∧(∇v)

using ∇∧R = 0 (the Bianchi identity) and ∇∧v = ∇v (v has no form slots). The sign is +
because p = 2 is even.
