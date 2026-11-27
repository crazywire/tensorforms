"""The connection Γ, built from the metric (so no torsion: Γ is S-symmetric).

Γ is stored as a plain tensor (ρ↑ σ↓ μ) so you can multiply, wedge and d it,
e.g. R = d∧Γ + Γ∧Γ. It is not tensorial (it doesn't change like a tensor under a
change of coordinates), so ∇Γ has no meaning; nothing stops you writing it.
"""
import sympy as sp



def levi_civita(g):
    """Γ = ½ (dg + dg^S − dg^{ST}) with its first index raised, computed with the operations
    themselves."""
    D = g.d().as_value()                                        # dg, with all three indices as values
    X = D + D.S - D.S.T                                  # dg + dg^S − dg^ST
    G = sp.Rational(1, 2) * X.raise_(0)                        # raise the first index: g⁻¹ on the left
    G = G.as_form(2).simplify()                                # the step index becomes a form index
    G = G.relabel_indices({0: "ρ", 1: "σ", 2: "μ"})
    G.label = "Γ"
    return G
