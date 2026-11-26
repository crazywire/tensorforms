# Integration

| call | description |
|---|---|
| `X.integrate({θ: (0, π), φ: (0, 2π)})` | ∫ X over the coordinate ranges |
| `C.pullback(X)` | X, a tensor on C's ambient manifold, carried onto the submanifold C |

* `integrate`: X has one form index per coordinate (μ ν …), antisymmetric. Every coordinate
  needs a range, by symbol or name; the orientation is the coordinate order. Returns a plain
  value, or sympy's unevaluated `Integral` if sympy can't do it.
* `pullback`: X has down and form indices only. Each index keeps its label and type; its space
  becomes C's tangent space.

To integrate over a region smaller than the manifold, make the region a
[submanifold](manifold.md#submanifolds) C, pull X back onto it, and integrate over C's
coordinates. Over the whole manifold, skip the pullback.

## Getting form indices

`integrate` needs as many form indices as C has coordinates. Other tensors get them first
(`star` needs a metric):

| X | over C | call |
|---|---|---|
| (μ ν …), one form index per coordinate of C | any | `C.pullback(X).integrate(...)` |
| () | any | `C.pullback(X).star().integrate(...)` |
| (μ↑) | a curve | `C.pullback(X.lower().as_form()).integrate(...)` |
| (μ↑) | dimension n − 1 | `C.pullback(X.lower().as_form().star()).integrate(...)` |

n is the dimension of X's manifold. For (), `star` is applied on C, after the pullback; for
(μ↑) through dimension n − 1, on X's manifold, before it.

### Example: the area of a sphere

```python
import sympy as sp
import tensorforms as tf

S = tf.Manifold.sphere()
th, ph = S.coords
S.scalar(1).star().integrate({th: (0, sp.pi), ph: (0, 2*sp.pi)})
```

```text
4*pi*r**2
```

### Example: around the unit circle and over the disc

```python
from tensorforms import Manifold

x, y = sp.symbols("x y", real=True)
u, phi = sp.symbols("u phi", real=True)
rho = sp.Symbol("rho", positive=True)
P = Manifold([x, y], metric=[[1, 0], [0, 1]])
C = Manifold([u], ambient=P, map={x: sp.cos(u), y: sp.sin(u)})                          # the circle
D = Manifold([rho, phi], ambient=P, map={x: rho * sp.cos(phi), y: rho * sp.sin(phi)})   # the disc

w = P.one_form([-y, x])                                                 # (μ)
C.pullback(w).integrate({u: (0, 2*sp.pi)})
D.pullback(w.wedge_d()).integrate({rho: (0, 1), phi: (0, 2*sp.pi)})     # d∧w: (μ ν)
```

```text
2*pi
2*pi
```

### Example: (μ↑) through the unit sphere

```python
E = tf.Manifold.euclidean(3)
X, Y, Z = E.coords
a, b = sp.symbols("a b", real=True)
sphere = Manifold([a, b], ambient=E, map={X: sp.sin(a) * sp.cos(b), Y: sp.sin(a) * sp.sin(b), Z: sp.cos(a)})
v = E.vector([X, Y, Z])                                                 # (μ↑)
sphere.pullback(v.lower().as_form().star()).integrate({a: (0, sp.pi), b: (0, 2*sp.pi)})
```

```text
4*pi
```
