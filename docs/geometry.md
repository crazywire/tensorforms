# Geometry

Ready-made manifolds with their metric attached: `tf.geometry.<name>()`. Parameters stay
symbolic unless you give values.

## Flat space

| call | coordinates |
|---|---|
| `euclidean(n=None, coords="cartesian")` | x, y, z (or x1 … xn); `"polar"` (r, φ), `"spherical"` (r, θ, φ), `"cylindrical"` (r, φ, z) |

R = 0 always, but Γ ≠ 0 in curved coordinates. n follows the coordinates (Cartesian default 3).

## Surfaces

| call | metric | K |
|---|---|---|
| `sphere(r=None)` | r² dθ² + r² sin²θ dφ² | 1/r² |
| `cylinder(r=None)` | r² dφ² + dz² | 0 |
| `torus(R=None, r=None)` | r² dθ² + (R + r cos θ)² dφ² | > 0 outside, < 0 inside |
| `hyperbolic_plane()` | (dx² + dy²)/y² | −1 |

## Spacetimes

Signature −+++, units G = c = 1 (a mass m is a length).

| call | metric |
|---|---|
| `minkowski()` | −dt² + dx² + dy² + dz² |
| `schwarzschild(m=None)` | −f dt² + dr²/f + r² dΩ², f = 1 − 2m/r (outside the horizon, r > 2m) |
| `flrw(a=None, k=0)` | −dt² + a(t)² [dr²/(1 − kr²) + r² dΩ²] |
| `de_sitter(H=None)` | −dt² + e^{2Ht}(dx² + dy² + dz²) |

dΩ² = dθ² + sin²θ dφ². For `flrw`, a is an unknown a(t) by default, or a function of t
(`lambda t: t**sp.Rational(2, 3)`), or an expression in `sp.Symbol("t", real=True)`.

## Templates

Metrics with unknown functions, for solving. Each takes a bare manifold, attaches the metric,
and returns it.

| call | metric |
|---|---|
| `templates.diagonal(M)` | diag(f0(x), f1(x), …) |
| `templates.conformally_flat(M, signature=None)` | e^{2ω(x)} × the flat metric |
| `templates.static_spherical(M)` | −A(r)dt² + B(r)dr² + r²dΩ², coordinates (t, r, θ, φ) |

Setting the Einstein tensor of `static_spherical` to zero and solving gives A = 1 − 2m/r,
B = 1/A: Schwarzschild.

```python
import tensorforms as tf
S = tf.geometry.sphere()
S.gaussian_curvature()                           # 1/r**2
tf.geometry.schwarzschild().ricci().is_zero()    # True
tf.geometry.de_sitter().scalar_curvature()       # 12*H**2
```
