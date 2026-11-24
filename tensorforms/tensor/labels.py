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
        return x.label
    name = str(x)
    return GREEK.get(name, name)


# ---------------------------------------------------------------- tensor labels
# A label built by an operation remembers what kind of expression it is, so an operand gets
# brackets exactly when reading it without them would be wrong or ambiguous:
#   atom    R, Ric, tr(R), Alt(R), (A+B)        a name, a function form, or bracketed
#   post    R^S, A⁻¹                            a postfix on an atom
#   pre     ∇v, dg, ⋆α, ∇∇v                     a prefix operator (they chain: ∇∇v)
#   prod    Rv, v⊗w, α∧β, 3A, f·A, A/2          products and scaling
#   sum     A+B, A−B, −A
# A plain string (a name you give, or the manifold's) is an atom.

class Label(str):
    """A label built by an operation: the text plus what kind of expression it is."""

    def __new__(cls, text, kind, **extra):
        obj = super().__new__(cls, text)
        obj.kind = kind
        obj.extra = extra
        return obj


def kind(label):
    return getattr(label, "kind", "atom")


def _operand(label, allowed):
    """The label as an operand: bracketed unless its kind is allowed in that place."""
    return label if kind(label) in allowed else f"({label})"


def prefix(op, label):
    """∇v, dg, ⋆α. The operand is an atom or another prefix (∇∇v); anything else is bracketed:
    ∇(Rv), ∇(R^S)."""
    if not label:
        return None
    return Label(f"{op}{_operand(label, {'atom', 'pre'})}", "pre")


def postfix(label, mark):
    """A⁻¹, (Rv)⁻¹: the operand must be an atom."""
    if not label:
        return None
    return Label(f"{_operand(label, {'atom'})}{mark}", "post")


def swapped(label, letter):
    """R^S, then R^ST: the swap letters in the order applied. The same letter twice in a row
    cancels (a swap undone): R^SS is R. The base must be an atom: (Rv)^S."""
    if not label:
        return None
    letters = label.extra.get("letters") if isinstance(label, Label) else None
    if letters is not None:
        base, letters = label.extra["base"], letters + letter
    else:
        base, letters = label, letter
    while any(a == b for a, b in zip(letters, letters[1:])):        # TT, SS, … cancel
        k = next(i for i, (a, b) in enumerate(zip(letters, letters[1:])) if a == b)
        letters = letters[:k] + letters[k + 2:]
    if not letters:
        return base
    return Label(f"{_operand(base, {'atom'})}^{letters}", "post", base=base, letters=letters)


def product(a, op, b):
    """AB (op ""), A⊗B, A∧B. The left factor is an atom or postfix; the right may also be a
    prefix (R∇v). Products nest with brackets: (AB)C, A(BC)."""
    if not (a and b):
        return None
    return Label(f"{_operand(a, {'atom', 'post'})}{op}{_operand(b, {'atom', 'post', 'pre'})}", "prod")


def func(name, *labels):
    """tr(R), Alt(R), ι(X,V): the brackets are the function's own."""
    if not all(labels):
        return None
    return Label(f"{name}({','.join(labels)})", "atom")


def summed(a, op, b):
    """A+B, A−B; the right side is bracketed if it is itself a sum: A−(B+C)."""
    if not (a and b):
        return None
    return Label(f"{a}{op}{_operand(b, {'atom', 'post', 'pre', 'prod'})}", "sum")


def negated(label):
    """−A, −(A+B); and −(−A) is A."""
    if not label:
        return None
    if isinstance(label, Label) and "negates" in label.extra:
        return label.extra["negates"]
    return Label(f"−{_operand(label, {'atom', 'post', 'pre', 'prod'})}", "sum", negates=label)


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def number(c):
    """A number or expression as one line of text: r/2, sin(θ), θ², 2·θ, −1/2."""
    text = sp.sstr(c)
    text = re.sub(r"\*\*(\d+)", lambda m: m.group(1).translate(_SUPERSCRIPT), text)   # θ**2 → θ²
    text = text.replace("**", "^").replace("*", "·").replace("-", "−")
    return re.sub(r"[A-Za-z]+", lambda m: GREEK.get(m.group(0), m.group(0)), text)     # theta → θ


def _top_level(text):
    """The characters of text that are not inside brackets."""
    depth, out = 0, []
    for ch in text:
        depth += ch == "("
        if depth == 0:
            out.append(ch)
        depth -= ch == ")"
    return "".join(out)


def _number(c):
    """A number as a factor: bracketed only if it has an operator at the top level, (r/2)·A,
    but sin(θ)·A and 3A."""
    text = number(c)
    return f"({text})" if any(ch in _top_level(text) for ch in "+−·/^") else text


def scaled(c, c_label, label):
    """3A, ½·A, sin(θ)·A, f·A; 1·A is A and −1·A is −A. Unlabelled tensor: no label."""
    if not label:
        return None
    if c == 1:
        return label
    if c == -1:
        return negated(label)
    body = _operand(label, {"atom", "post", "pre", "prod"})
    if c_label:
        return Label(f"{_operand(c_label, {'atom', 'post'})}·{body}", "prod")
    if isinstance(c, sp.Integer):
        return Label(f"{c}{body}", "prod") if c > 0 else negated(Label(f"{-c}{body}", "prod"))
    return Label(f"{_number(c)}·{body}", "prod")


def divided(label, c, c_label):
    """A/2, A/f, (A+B)/2."""
    if not label:
        return None
    by = _operand(c_label, {"atom", "post"}) if c_label else _number(c)
    return Label(f"{_operand(label, {'atom', 'post', 'pre', 'prod'})}/{by}", "prod")
