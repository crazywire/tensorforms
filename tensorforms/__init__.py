"""tensorforms: the calculus of tensor-valued forms, with four operators.

    manifold/    Manifold (coordinates) and its optional metric, connection, curvature
    tensor/      Slot, Index, Tensor, contract, tr, outer, product, Alt/Sym, wedge, odot
    operators/   nabla, wedge_nabla, curvature, wedge_curvature (and d)
    geometry/    ready-made geometries and metric templates
    symbolic/    simplification helpers
"""
from .manifold import Manifold, Connection
from .tensor import Slot, Space, Index, Tensor, contract, tr, outer, product, wedge, odot
from .operators import d, nabla, wedge_nabla, curvature, wedge_curvature
from . import geometry

__all__ = ["Manifold", "Connection", "Slot", "Space", "Index", "Tensor",
           "contract", "tr", "outer", "product", "wedge", "odot",
           "d", "nabla", "wedge_nabla", "curvature", "wedge_curvature",
           "geometry"]
__version__ = "0.1.0"
