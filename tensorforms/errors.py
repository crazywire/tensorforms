"""The package's exceptions. Each also inherits the matching built-in, so `except TypeError`,
`except ValueError` and `hasattr(M, "g")` keep working; `except TensorformsError` catches them all."""


class TensorformsError(Exception):
    """Any error raised by tensorforms itself."""


class IndexMismatchError(TensorformsError, TypeError):
    """The indices don't fit the operation: adding or pairing indices that don't match, raising a
    form index, a v that isn't a vector, tensors on different manifolds, …"""


class MissingMetricError(TensorformsError, AttributeError):
    """The operation needs a metric and the manifold has none yet (call M.metric(...))."""


class InvalidMetricError(TensorformsError, ValueError):
    """The metric given is rejected: wrong size, not shown symmetric, determinant 0, signature."""


class InvalidCoordinateError(TensorformsError, ValueError):
    """A coordinate used where it can't be: subs on a coordinate, at() with a missing or unknown
    coordinate, an unknown coordinate system, coordinates in the wrong order."""


class NonInvertibleError(TensorformsError, ValueError):
    """inverse() of a matrix sympy shows is singular, or that isn't square."""
