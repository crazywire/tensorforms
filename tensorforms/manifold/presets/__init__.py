"""Ready-made manifolds, reached as Manifold.<name>(...)."""
from .euclidean import euclidean
from .surfaces import sphere, cylinder, torus, hyperbolic_plane
from .spacetimes import minkowski, schwarzschild, flrw, de_sitter

__all__ = ["euclidean", "sphere", "cylinder", "torus", "hyperbolic_plane",
           "minkowski", "schwarzschild", "flrw", "de_sitter"]
