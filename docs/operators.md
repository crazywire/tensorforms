# Operators

| operator | function | description |
|---|---|---|
| d | `d(X)` | partial derivative |
| d∧ | `wedge_d(X)` | exterior derivative |
| ∇ = d + Γ | `nabla(X)` | covariant derivative |
| ∇∧ = d∧ + Γ∧ | `wedge_nabla(X)` | exterior covariant derivative |
| ∇∧∇ = R | `curvature(X)` | commutator of covariant derivatives |
| ∇∧∇∧ = R∧ | `wedge_curvature(X)` | exterior covariant derivative squared |


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

### Example: how does e_θ change as you move on a sphere?

e_θ has constant components (1, 0), so its partial derivative is zero. Its covariant derivative
is not, because the direction "θ" itself turns as you move.

```python
import sympy as sp
from tensorforms import Manifold, d, nabla
th, ph = sp.symbols("theta phi", real=True)
S = Manifold([th, ph], name="S2")
S.metric([[1, 0], [0, sp.sin(th)**2]])

e_theta = S.vector([1, 0])
print(d(e_theta).data)
print(nabla(e_theta).simplify().data)
```

```text
[[0 0]
 [0 0]]
[[0 0]
 [0 cos(theta)/sin(theta)]]
```

Moving in the φ direction, e_θ picks up a part cot θ · e_φ. Moving in θ, it doesn't change.

### Example: the curl of a rotation

In the flat plane, the 1-form α = −y dx + x dy describes a rotation. Its exterior derivative is
its curl:

```python
import tensorforms as tf
from tensorforms import wedge_d
P = tf.geometry.euclidean(2)
x, y = P.coords

alpha = P.one_form([-y, x])
print(wedge_d(alpha).data)
```

```text
[[0 2]
 [-2 0]]
```

d∧α = 2 dx∧dy: the rotation has constant curl 2.

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
