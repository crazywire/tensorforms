# Derivatives

| derivative | call | description |
|---|---|---|
| d | `X.d()` | partial derivative |
| d∧ | `X.wedge_d()` | exterior derivative |
| ∇ = d + Γ | `X.nabla()` | covariant derivative |
| ∇∧ = d∧ + Γ∧ | `X.wedge_nabla()` | exterior covariant derivative |
| ∇∧∇ = R | `X.curvature()` | commutator of covariant derivatives |
| ∇∧∇∧ = R∧ | `X.wedge_curvature()` | exterior covariant derivative squared |
| L_v | `X.lie(v)` | Lie derivative along a vector field v |


Each returns a new tensor with no label. Conventions:

* New form indices go first among the form indices.
* Γ and R come from the manifold (Levi-Civita, no torsion). With no metric, ∇ and ∇∧ act as d
  and d∧, and the two curvature derivatives give zero.
* ∇ corrects up indices with +Γ and down indices with −Γ, each term `X.apply(Γ, i)`:
  ∇X = dX + Σ ± X.apply(Γ, i). ∇∧ corrects value indices only.
* d∧ multiplies by p + 1 after antisymmetrizing, and ∇∧∇∧ by C(p+2; 2, p), where p is the number
  of form indices of X.
* The curvature derivatives multiply by R; they don't differentiate X.

### The Lie derivative

`X.lie(v)`: v is a vector field, X any tensor on the same manifold. The result has X's indices.
No metric is needed.

L_v X = dX(v) − J on each up index + J on each down or form index,  with J = (dv).as_value()

dX(v) evaluates d's new form index on v.

For example, checking whether a vector field is a symmetry of the metric (L_v g = 0):

```python
import tensorforms as tf

S = tf.Manifold.sphere()
print(S.g.lie(S.vector([0, 1])).is_zero())         # ∂_φ: a symmetry
print(S.g.lie(S.vector([1, 0])).simplify().data)   # ∂_θ: not one
```

```text
True
[[0 0]
 [0 r**2*sin(2*theta)]]
```

### Example: d and ∇ of e_θ on the unit sphere

```python
import sympy as sp
from tensorforms import Manifold
th, ph = sp.symbols("theta phi", real=True)
S = Manifold([th, ph], label="S2", metric=[[1, 0], [0, sp.sin(th)**2]])

e_theta = S.vector([1, 0])
print(e_theta.d().data)
print(e_theta.nabla().simplify().data)
```

```text
[[0 0]
 [0 0]]
[[0 0]
 [0 1/tan(theta)]]
```

### Example: d∧ of a 1-form in the plane

```python
import tensorforms as tf
P = tf.Manifold.euclidean(2)
x, y = P.coords

alpha = P.one_form([-y, x])
print(alpha.wedge_d().data)
```

```text
[[0 2]
 [-2 0]]
```
