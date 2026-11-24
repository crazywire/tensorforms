"""Labels, for tensors and for indices. Display only; never used in calculations.

Tensor labels: a result's label is the operation that made it (A+B, 3A, −A, R^US, AB, A∧B, …);
if an input has no label, the result has none.
Index labels: whatever you choose; spelled-out Greek becomes the letter ("mu" → μ)."""
import re

import sympy as sp

# Spelled-out Greek is accepted wherever a name is: "mu" means μ.
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ",
         "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "lamda": "λ",
         "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ",
         "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω"}


def index_label(x):
    """An index label as stored, from a string ("mu" → "μ"; anything else is kept), an Index or a Index."""
    from .index import Index                # here, to avoid a circular import
    if isinstance(x, Index):
        return x.name
    name = str(x)
    return GREEK.get(name, name)


def wrap(label):
    """Brackets around a label that has + or − in it, so it can sit inside a bigger expression."""
    if label and any(ch in label for ch in "+−"):
        return f"({label})"
    return label


def join(a, op, b):
    return f"{a}{op}{b}" if a and b else None


def negated(label):
    if not label:
        return None
    if label.startswith("−") and not any(ch in label[1:] for ch in "+−"):
        return label[1:]                                         # −(−A) = A
    return "−" + wrap(label)


def scaled(c, c_label, label):
    """3A, ½·A, sin(θ)·A, f·A. Unlabelled tensor: no label."""
    if not label:
        return None
    if c == 1:
        return label
    if c == -1:
        return negated(label)
    if c_label:
        return f"{c_label}·{wrap(label)}"
    if isinstance(c, sp.Integer):
        return f"{c}{wrap(label)}" if c > 0 else f"−{-c}{wrap(label)}"
    return f"{wrap(sp.pretty(c, use_unicode=True))}·{wrap(label)}"


def swapped(label, letter):
    """R^S, then R^ST: the swap letters in the order they were applied."""
    if not label:
        return None
    if re.search(r"\^[TSUV]+$", label) and wrap(label.split("^")[0]) == label.split("^")[0]:
        return label + letter
    return f"{wrap(label)}^{letter}"
