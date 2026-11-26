# Integration

| call | description |
|---|---|
| `X.integrate({θ: (0, π), φ: (0, 2π)})` | ∫ X over the coordinate ranges |
| `C.pullback(X)` | X carried onto the submanifold C |

Conventions:

* `integrate` takes a form with one form index per coordinate, antisymmetric. Every coordinate
  needs a range (by symbol or name); the orientation is the coordinate order.
* `integrate` returns a plain value, or sympy's unevaluated `Integral` if sympy can't do it.
* `pullback` takes a tensor on C's ambient manifold with down and form indices only. Each index
  keeps its label and type.

## What to integrate

A form is integrated over a region of its degree: a 1-form along a curve, a 2-form over a
surface. A region smaller than the manifold is a [submanifold](manifold.md#submanifolds): pull
back onto it, then integrate over its coordinates. Star (which needs a metric) gives the form
for a scalar or a vector field:

| you have | integrate | over |
|---|---|---|
| a form | the form | a region of its degree |
| a scalar f (length, area, mass, …) | `f.star()`, or `C.pullback(f).star()` on a submanifold C | the manifold, or C |
| a vector field v, along a curve (work) | `v.lower().as_form()` | the curve |
| a vector field v, through a surface (flux) | `v.lower().as_form().star()` | the surface |

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

### Example: Stokes, around the unit circle and over the disc

```python
from tensorforms import Manifold

x, y = sp.symbols("x y", real=True)
u, phi = sp.symbols("u phi", real=True)
rho = sp.Symbol("rho", positive=True)
P = Manifold([x, y], metric=[[1, 0], [0, 1]])
C = Manifold([u], ambient=P, map={x: sp.cos(u), y: sp.sin(u)})                    # the circle
D = Manifold([rho, phi], ambient=P, map={x: rho * sp.cos(phi), y: rho * sp.sin(phi)})   # the disc

w = P.one_form([-y, x])
C.pullback(w).integrate({u: (0, 2*sp.pi)})                          # around the circle
D.pullback(w.wedge_d()).integrate({rho: (0, 1), phi: (0, 2*sp.pi)}) # d∧w over the disc
```

```text
2*pi
2*pi
```

### Example: flux out of the unit sphere

```python
E = tf.Manifold.euclidean(3)
X, Y, Z = E.coords
a, b = sp.symbols("a b", real=True)
ball = Manifold([a, b], ambient=E, map={X: sp.sin(a) * sp.cos(b), Y: sp.sin(a) * sp.sin(b), Z: sp.cos(a)})
v = E.vector([X, Y, Z])
ball.pullback(v.lower().as_form().star()).integrate({a: (0, sp.pi), b: (0, 2*sp.pi)})
```

```text
4*pi
```
