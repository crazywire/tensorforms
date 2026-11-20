"""Labels: a result's label is the operation that made it (A+B, 3A, −A, R^US, AB, A∧B, …).
If an input has no label, the result has none. Display only; never used in calculations."""
import re

import sympy as sp


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
