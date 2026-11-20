"""tensorforms: the calculus of tensor-valued forms, with four operators.

    manifold/    Manifold (coordinates), its tangent Space, and its optional metric, connection, curvature
    tensor/      Slot, Index, Tensor (with tr, raise/lower, Alt/Sym); products: contract, otimes, wedge, odot
    operators/   nabla, wedge_nabla, curvature, wedge_curvature (and d)
    geometry/    ready-made geometries and metric templates
    symbolic/    simplification helpers
"""
from .manifold import Manifold, Space, Connection
from .tensor import Slot, Index, Tensor, contract, otimes, wedge, odot
from .operators import d, nabla, wedge_nabla, curvature, wedge_curvature
from . import geometry

__all__ = ["Manifold", "Connection", "Slot", "Space", "Index", "Tensor",
           "contract", "otimes", "wedge", "odot",
           "d", "nabla", "wedge_nabla", "curvature", "wedge_curvature",
           "geometry"]
__version__ = "0.1.0"
