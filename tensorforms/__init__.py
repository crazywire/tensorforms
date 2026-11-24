"""tensorforms: the calculus of tensor-valued forms, with six operators (d, d∧, ∇, ∇∧, ∇∧∇, ∇∧∇∧).

    manifold/    Manifold (coordinates), its tangent Space, metric, connection, curvature;
                 ready-made manifolds (Manifold.sphere(), …) and metric templates
    tensor/      Index, Tensor (with tr, raise/lower, Alt/Sym); products: contract, otimes, wedge, odot
    operators/   d, wedge_d, nabla, wedge_nabla, curvature, wedge_curvature; lie (Lie derivative)
    symbolic/    simplification helpers
"""
from .manifold import Manifold, Space, templates
from .tensor import Index, Tensor, contract, interior, otimes, wedge, odot
from .operators import d, wedge_d, nabla, wedge_nabla, curvature, wedge_curvature, lie

__all__ = ["Manifold", "Index", "Space", "Tensor",
           "contract", "interior", "otimes", "wedge", "odot",
           "d", "wedge_d", "nabla", "wedge_nabla", "curvature", "wedge_curvature", "lie",
           "templates"]
__version__ = "0.1.0"
