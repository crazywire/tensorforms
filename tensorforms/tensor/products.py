"""Products of two tensors (no Einstein summation; nothing is chosen silently).

    contract(A, B)         your AB: A's column (last value slot, down) with B's row (first, up)
    contract(A, B, a, b)   the paired product over the slots you name
    interior(X, V)         X's first form slot with V's row (the interior product)
    otimes(A, B)           tensor product: pairs nothing
    wedge(A, B)            A ∧ B = k!/(p! q!) · Alt(AB) over all form slots
    odot(A, B)             A ⊙ B = k!/(p! q!) · Sym(AB)

Results put value slots first, then A's form slots, then B's.
"""
import math

import numpy as np
import sympy as sp

from .labels import join, wrap
from .tensor import Tensor


def _rename_clashes(keep_names, slots):
    """B's slots, with any name already used by A replaced by a fresh one from the same space."""
    used, out = set(keep_names) | {s.name for s in slots}, []
    taken = set(keep_names)
    for s in slots:
        n = s.name
        if n in taken:
            n = s.space.fresh_name(used)
            used.add(n)
        taken.add(n)
        out.append(s.renamed(n))
    return out


def _check_pair(sa, sb):
    if sa.space is not sb.space:
        raise TypeError(f"can't pair {sa.name} ({sa.space.name}) with {sb.name} ({sb.space.name}): different spaces")
    if sa.variance == sb.variance:
        raise TypeError(f"can't pair {sa.name} and {sb.name}: both {sa.variance}; raise or lower one first")


def _values_first(slots, n_a):
    """Order: value slots, then A's form slots, then B's form slots.
    The first n_a slots came from A (told apart by position, not by name)."""
    vals = [k for k, s in enumerate(slots) if not s.is_form]
    fa = [k for k, s in enumerate(slots) if s.is_form and k < n_a]
    fb = [k for k, s in enumerate(slots) if s.is_form and k >= n_a]
    return vals + fa + fb


def _joined(A, data, slots, n_a, label):
    order = _values_first(slots, n_a)
    return Tensor(A.manifold, np.transpose(data, order), [slots[k] for k in order], label)


def _default_pair(A, B):
    """A's column (last value slot, down) with B's row (first value slot, up)."""
    av, bv = A.value_slots, B.value_slots
    if not (av and bv):
        raise TypeError("contract(A, B) needs value slots on both: one factor has none, "
                        "so nothing pairs; use otimes(A, B)")
    sa, sb = A.slot(av[-1]), B.slot(bv[0])
    if sa.space is sb.space and not sa.is_up and sb.is_up:
        return av[-1], bv[0]
    if sa.space is sb.space and sa.is_up and not sb.is_up:
        raise TypeError(f"{A.declaration()} has no column and {B.declaration()} has no row: "
                        "nothing pairs (column × row); use otimes(A, B)")
    raise TypeError(f"can't pair {sa.name} with {sb.name} (shape mismatch); "
                    "name the slots, use otimes(A, B), or raise or lower one first")


def contract(A, B, a=None, b=None):
    """Pair one slot of A with one slot of B and sum over it.
    contract(A, B): the default, A's column with B's row (your AB).
    contract(A, B, a, b): the slots you name (b defaults to a).
    Result: value slots first, then A's form slots, then B's."""
    if a is None:
        a, b = _default_pair(A, B)
    b = a if b is None else b
    i, j = A.position(a), B.position(b)
    _check_pair(A.slots[i], B.slots[j])
    data = np.tensordot(A.data, B.data, axes=([i], [j]))
    keep_a = [s for k, s in enumerate(A.slots) if k != i]
    keep_b = _rename_clashes([s.name for s in keep_a], [s for k, s in enumerate(B.slots) if k != j])
    return _joined(A, data, keep_a + keep_b, len(keep_a), join(wrap(A.label), "", wrap(B.label)))


def interior(X, V):
    """The interior product: X's first form slot with V's row
    (its first value slot, up). Like contract(X, V), but using X's first form slot instead of
    its column. Everything else is carried along: value slots first, then X's form slots, then V's."""
    if not X.form_slots:
        raise TypeError(f"interior needs a form slot in X; {X.declaration()} has none")
    if not V.value_slots or not V.slot(V.value_slots[0]).is_up:
        raise TypeError(f"interior needs a row (first value slot, up) in V; got {V.declaration()}")
    P = contract(X, V, X.form_slots[0], V.value_slots[0])
    label = f"ι({X.label},{V.label})" if X.label and V.label else None
    return P._new(label=label)


def otimes(A, B):
    """Tensor product A ⊗ B: pair nothing, keep every slot.
    Result: value slots first, then A's form slots, then B's."""
    keep_b = _rename_clashes(A.names, B.slots)
    return _joined(A, np.multiply.outer(A.data, B.data), A.slots + keep_b, len(A.slots),
                   join(wrap(A.label), "⊗", wrap(B.label)))


def _wedge_like(A, B, pair, signed):
    p, q = len(A.form_slots), len(B.form_slots)
    project = Tensor.alt if signed else Tensor.sym
    A2, B2 = project(A), project(B)              # each factor's own form slots first
    if pair is not None:
        P = contract(A2, B2, *pair)
    elif not (A.value_slots and B.value_slots):
        P = otimes(A2, B2)                       # a scalar-valued factor: nothing can pair
    else:
        P = contract(A2, B2)
    k = p + q
    if k >= 2:
        P = project(P) * sp.Rational(math.factorial(k), math.factorial(p) * math.factorial(q))
    sym = "∧" if signed else "⊙"
    return P._new(label=join(wrap(A.label), sym, wrap(B.label)))


def wedge(A, B, pair=None):
    """A ∧ B = k!/(p! q!) · Alt(AB) over all form slots (each interleaving once).
    AB pairs A's column with B's row (or the slots in pair); with a scalar-valued factor it is A ⊗ B."""
    return _wedge_like(A, B, pair, signed=True)


def odot(A, B, pair=None):
    """A ⊙ B = k!/(p! q!) · Sym(AB) over all form slots."""
    return _wedge_like(A, B, pair, signed=False)
