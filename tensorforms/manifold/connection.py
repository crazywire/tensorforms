"""The connection Γ, built from the metric (so no torsion: Γ is S-symmetric).

Γ is stored as a plain tensor (ρ↑ σ↓ | μ) so you can multiply, wedge and d it,
e.g. R = d∧Γ + Γ∧Γ. It is not tensorial (it doesn't change like a tensor under a
change of coordinates), so ∇Γ has no meaning; nothing stops you writing it.
"""
import sympy as sp

from ..tensor.products import contract


def levi_civita(g, ginv):
    """Γ = ½ g⁻¹(dg + dg^S − dg^{ST}), computed with the operations themselves."""
    from ..operators.operators import d                        # here, to avoid a circular import
    D = d(g).as_value()                                        # dg, with all three slots as values
    X = D + D.S() - D.S().T()                                  # dg + dg^S − dg^ST
    G = sp.Rational(1, 2) * contract(ginv, X, 1, 0)
    G = G.as_form(2).simplify()                                # the step slot becomes a form slot
    G = G.rename({0: "ρ", 1: "σ", 2: "μ"})
    G.label = "Γ"
    return G
