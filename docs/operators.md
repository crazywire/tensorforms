# Operators

|            | every slot              | wedged (form slots antisymmetrized) |
|------------|-------------------------|-------------------------------------|
| plain      | `d`                     | `wedge_d` = d∧                      |
| once       | `nabla` = d + Γ         | `wedge_nabla` = d∧ + Γ∧             |
| twice      | `curvature` = R         | `wedge_curvature` = R∧              |

Each takes one tensor and returns a new one, labelled by the operator (dX, ∇X, …).
New form slots go first after the bar. Γ and R come from the manifold; with no metric, ∇ and
∇∧ act as d and d∧, and the curvatures are zero.

* **d** adds one form slot: the derivatives of every component.
* **d∧** makes X antisymmetric in its form slots, takes d, then antisymmetrizes the new slot
  with the old ones, times p + 1. On a 1-form this is the curl.
* **∇ = d + Γ** corrects every slot: +Γ from the left on up slots, −Γ from the right on down slots.
* **∇∧ = d∧ + Γ∧** corrects value slots only. The corrections on form slots cancel, because Γ
  is symmetric in its two lower slots (no torsion).
* **∇∧∇ = R** acts with R on every slot; nothing is differentiated.
* **∇∧∇∧ = R∧** acts with R on value slots, times C(p+2; 2, p).

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

For ∇, the new slot has to be moved back first after the bar with swaps:

∇(Rv) = (∇R)v + (R∇v)^US

For the wedged operators, swaps become signs (p = number of form slots of A):

∇∧(A∧B) = (∇∧A)∧B + (−1)^p A∧(∇∧B)

∇∧∇∧(A∧B) = (∇∧∇∧A)∧B + A∧(∇∧∇∧B)

For a plain product AB, rewrite it as a wedge first:

AB = p! q!/(p+q)! · A∧B

The factor is 1 when either has no form slots, as in Rv.
