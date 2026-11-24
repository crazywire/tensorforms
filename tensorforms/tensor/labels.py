"""Labels: display only, never used in calculations.

A tensor's label is just a name: one you give (S.tensor(..., label="F"), X.label = "K") or the
manifold's (g, Γ, R, Ric, …). A computed result has no label; its indices carry the information.
Index labels: whatever you choose; spelled-out Greek becomes the letter ("mu" → μ)."""

# Spelled-out Greek is accepted wherever a name is: "mu" means μ.
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ",
         "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "lamda": "λ",
         "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ",
         "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω"}


def index_label(x):
    """An index label as stored, from a string ("mu" → "μ"; anything else is kept), or an Index."""
    from .index import Index                # here, to avoid a circular import
    if isinstance(x, Index):
        return x.label
    name = str(x)
    return GREEK.get(name, name)
