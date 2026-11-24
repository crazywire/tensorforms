"""Products of two tensors (no Einstein summation; nothing is chosen silently).

    contract(A, B)         your AB: A's column (last value index, down) with B's row (first, up)
    contract(A, B, a, b)   the paired product over the indices you name
    interior(X, V)         X's first form index with V's row (the interior product)
    otimes(A, B)           tensor product: pairs nothing
    wedge(A, B)            A ∧ B = k!/(p! q!) · Alt(AB) over all form indices
    odot(A, B)             A ⊙ B = k!/(p! q!) · Sym(AB)

Results put value indices first, then A's form indices, then B's.
"""
import math

import numpy as np
import sympy as sp

from .labels import func, product
from .tensor import Tensor


def _rename_clashes(keep_names, indices):
    """B's indices, with any name already used by A replaced by a fresh one from the same space."""
    used, out = set(keep_names) | {s.label for s in indices}, []
    taken = set(keep_names)
    for s in indices:
        n = s.label
        if n in taken:
            n = s.space.fresh_name(used)
            used.add(n)
        taken.add(n)
        out.append(s.relabel(n))
    return out


def _check_pair(sa, sb):
    if sa.pairs_with(sb):
        return
    if sa.space is not sb.space:
        raise TypeError(f"can't pair {sa.label} ({sa.space.name}) with {sb.label} ({sb.space.name}): different spaces")
    raise TypeError(f"can't pair {sa.label} and {sb.label}: both {sa.variance}; raise or lower one first")


def _values_first(indices, n_a):
    """Order: value indices, then A's form indices, then B's form indices.
    The first n_a indices came from A (told apart by position, not by name)."""
    vals = [k for k, s in enumerate(indices) if not s.is_form]
    fa = [k for k, s in enumerate(indices) if s.is_form and k < n_a]
    fb = [k for k, s in enumerate(indices) if s.is_form and k >= n_a]
    return vals + fa + fb


def _joined(A, data, indices, n_a, label):
    order = _values_first(indices, n_a)
    return Tensor(A.manifold, np.transpose(data, order), [indices[k] for k in order], label, copy=False)


def _default_pair(A, B):
    """A's column (last value index, down) with B's row (first value index, up)."""
    av, bv = A._value_positions(), B._value_positions()
    if not (av and bv):
        raise TypeError("contract(A, B) needs value indices on both: one factor has none, "
                        "so nothing pairs; use otimes(A, B)")
    sa, sb = A.index(av[-1]), B.index(bv[0])
    if sa.space is sb.space and not sa.is_up and sb.is_up:
        return av[-1], bv[0]
    if sa.space is sb.space and sa.is_up and not sb.is_up:
        raise TypeError(f"{A.declaration()} has no column and {B.declaration()} has no row: "
                        "nothing pairs (column × row); use otimes(A, B)")
    raise TypeError(f"can't pair {sa.label} with {sb.label} (shape mismatch); "
                    "name the indices, use otimes(A, B), or raise or lower one first")


def contract(A, B, a=None, b=None):
    """Pair one index of A with one index of B and sum over it.
    contract(A, B): the default, A's column with B's row (your AB).
    contract(A, B, a, b): the indices you name (b defaults to a).
    Result: value indices first, then A's form indices, then B's."""
    if a is None:
        a, b = _default_pair(A, B)
    b = a if b is None else b
    i, j = A.position(a), B.position(b)
    _check_pair(A.indices[i], B.indices[j])
    data = np.tensordot(A.data, B.data, axes=([i], [j]))
    keep_a = [s for k, s in enumerate(A.indices) if k != i]
    keep_b = _rename_clashes([s.label for s in keep_a], [s for k, s in enumerate(B.indices) if k != j])
    return _joined(A, data, keep_a + keep_b, len(keep_a), product(A.label, "", B.label))


def interior(X, V):
    """The interior product: X's first form index with V's row
    (its first value index, up). Like contract(X, V), but using X's first form index instead of
    its column. Everything else is carried along: value indices first, then X's form indices, then V's."""
    if not X._form_positions():
        raise TypeError(f"interior needs a form index in X; {X.declaration()} has none")
    if not V._value_positions() or not V.index(V._value_positions()[0]).is_up:
        raise TypeError(f"interior needs a row (first value index, up) in V; got {V.declaration()}")
    P = contract(X, V, X._form_positions()[0], V._value_positions()[0])
    label = func("ι", X.label, V.label)
    return P._new(label=label)


def otimes(A, B):
    """Tensor product A ⊗ B: pair nothing, keep every index.
    Result: value indices first, then A's form indices, then B's."""
    keep_b = _rename_clashes(A._index_labels(), B.indices)
    return _joined(A, np.multiply.outer(A.data, B.data), (*A.indices, *keep_b), len(A.indices),
                   product(A.label, "⊗", B.label))


def _wedge_like(A, B, pair, signed):
    p, q = len(A._form_positions()), len(B._form_positions())
    project = Tensor.alt if signed else Tensor.sym
    A2, B2 = project(A), project(B)              # each factor's own form indices first
    if pair is not None:
        P = contract(A2, B2, *pair)
    elif not (A._value_positions() and B._value_positions()):
        P = otimes(A2, B2)                       # a scalar-valued factor: nothing can pair
    else:
        P = contract(A2, B2)
    k = p + q
    if k >= 2:
        P = project(P) * sp.Rational(math.factorial(k), math.factorial(p) * math.factorial(q))
    sym = "∧" if signed else "⊙"
    return P._new(label=product(A.label, sym, B.label))


def wedge(A, B, pair=None):
    """A ∧ B = k!/(p! q!) · Alt(AB) over all form indices (each interleaving once).
    AB pairs A's column with B's row (or the indices in pair); with a scalar-valued factor it is A ⊗ B."""
    return _wedge_like(A, B, pair, signed=True)


def odot(A, B, pair=None):
    """A ⊙ B = k!/(p! q!) · Sym(AB) over all form indices."""
    return _wedge_like(A, B, pair, signed=False)
