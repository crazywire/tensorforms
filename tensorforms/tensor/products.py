"""Products of two tensors (no Einstein summation; nothing is chosen silently), as tensor methods
(Tensor inherits Products):

    A @ B                  your AB: A's column (last value index, down) with B's row (first, up)
    A.contract(B, a, b)    the paired product over the indices you name
    X(u, w, …)             evaluate X on vectors: each fills the next form index
    A.otimes(B)            tensor product: pairs nothing
    A.wedge(B)             A ∧ B = k!/(p! q!) · Alt(AB) over all form indices
    A.odot(B)              A ⊙ B = k!/(p! q!) · Sym(AB)
    X.apply(A, i)          the matrix A acting on X's index i, which stays in place

Results put value indices first, then A's form indices, then B's.
"""
import math

import numpy as np
import sympy as sp
from .index import Index
from .labels import _new_label
from ..errors import IndexMismatchError, ManifoldMismatchError


def _rename_clashes(manifold, keep_names, indices):
    """B's indices, with any label already used by A replaced by a new one (labels._new_label)."""
    used, out = set(keep_names) | {s.label for s in indices}, []
    taken = set(keep_names)
    for s in indices:
        n = s.label
        if n in taken:
            n = _new_label(s.space, manifold, used)
            used.add(n)
        taken.add(n)
        out.append(s.relabel(n))
    return out


def _check_pair(sa, sb):
    if sa.pairs_with(sb):
        return
    if sa.space is not sb.space:
        raise IndexMismatchError(f"pair: {sa.label} and {sb.label} are in different spaces "
                                 f"(different manifolds, or not the tangent space)")
    raise IndexMismatchError(f"pair: {sa.label} and {sb.label} are both {sa.variance}; raise or lower one first")


def _values_first(indices, n_a):
    """Order: value indices, then A's form indices, then B's form indices.
    The first n_a indices came from A (told apart by position, not by name)."""
    vals = [k for k, s in enumerate(indices) if not s.is_form]
    fa = [k for k, s in enumerate(indices) if s.is_form and k < n_a]
    fb = [k for k, s in enumerate(indices) if s.is_form and k >= n_a]
    return vals + fa + fb


def _joined(A, data, indices, n_a, label):
    order = _values_first(indices, n_a)
    return type(A)(A.manifold, np.transpose(data, order), [indices[k] for k in order], label, copy=False)


def _default_pair(A, B):
    """A's column (last value index, down) with B's row (first value index, up)."""
    av, bv = A._value_positions(), B._value_positions()
    if not (av and bv):
        raise IndexMismatchError("@: a factor has no value indices, so nothing pairs; "
                                 "use A.otimes(B)")
    sa, sb = A.index(av[-1]), B.index(bv[0])
    if sa.space is sb.space and not sa.is_up and sb.is_up:
        return av[-1], bv[0]
    if sa.space is sb.space and sa.is_up and not sb.is_up:
        raise IndexMismatchError(f"@: pairs A's last value index (↓) with B's first (↑); here {sa!r} "
                                 f"meets {sb!r}. To pair them, A.contract(B, a, b); to keep both, A.otimes(B)")
    raise IndexMismatchError(f"@: {sa.label} and {sb.label} don't pair; name the indices "
                             "(A.contract(B, a, b)), use A.otimes(B), or raise or lower one")


def _contract(A, B, a, b):
    """Pair index a of A with index b of B and sum. Result: value indices first, then A's form
    indices, then B's."""
    i, j = A.position(a), B.position(b)
    _check_pair(A.indices[i], B.indices[j])
    data = np.tensordot(A.data, B.data, axes=([i], [j]))
    keep_a = [s for k, s in enumerate(A.indices) if k != i]
    keep_b = _rename_clashes(A.manifold, [s.label for s in keep_a], [s for k, s in enumerate(B.indices) if k != j])
    return _joined(A, data, keep_a + keep_b, len(keep_a), None)


def _assert_tensor(x):
    if not isinstance(x, Products):                  # every tensor inherits Products
        raise TypeError(f"the argument must be a tensor, not {type(x).__name__}")


def _assert_same_manifold(A, B):
    if A.manifold is not B.manifold:
        raise ManifoldMismatchError("the tensors must be on the same manifold")


def _wedge_like(A, B, pair, signed):
    p, q = len(A._form_positions()), len(B._form_positions())
    project = (lambda X: X.alt()) if signed else (lambda X: X.sym())
    A2, B2 = project(A), project(B)              # each factor's own form indices first
    if pair is not None:
        P = _contract(A2, B2, *pair)
    elif not (A._value_positions() and B._value_positions()):
        P = A2.otimes(B2)                        # a scalar-valued factor: nothing can pair
    else:
        P = A2 @ B2
    k = p + q
    if k >= 2:
        P = project(P) * sp.Rational(math.factorial(k), math.factorial(p) * math.factorial(q))
    return P._new(label=None)


class Products:
    """Products with another tensor, as tensor methods (Tensor inherits them)."""

    def __matmul__(self, other):
        """A @ B: A's column (last value index, down) with B's row (first value index, up), as in AB."""
        _assert_tensor(other)
        _assert_same_manifold(self, other)
        return _contract(self, other, *_default_pair(self, other))

    def contract(self, other, a, b):
        """Pair index a of this tensor with index b of other (each by label or position) and sum.
        For the usual AB (column with row), write A @ B."""
        _assert_tensor(other)
        _assert_same_manifold(self, other)
        return _contract(self, other, a, b)

    def __call__(self, *vectors):
        """X(u, w, …): evaluate X on vectors; each vector fills the next form index, in order.
        With fewer vectors than form indices, the remaining form indices stay (ω(u) for a 2-form
        is the interior product). Value indices are paired with @."""
        forms = len(self._form_positions())
        if not forms:
            raise IndexMismatchError(f"call: {self.declaration()} has no form index to evaluate on a vector; "
                                     "pair value indices with @, or evaluate at a point with .at({...})")
        if not vectors:
            raise TypeError("call: give at least one vector")
        if len(vectors) > forms:
            raise IndexMismatchError(f"call: {self.declaration()} has {forms} form "
                                     f"{'index' if forms == 1 else 'indices'}; got {len(vectors)} vectors")
        X = self
        for u in vectors:
            if not isinstance(u, type(self)):
                raise TypeError(f"call: arguments must be vectors, not {type(u).__name__}; "
                                "to evaluate at a point, use .at({...})")
            _assert_same_manifold(self, u)
            if len(u.indices) != 1 or not u.indices[0].is_up:
                raise IndexMismatchError(f"call: arguments must be vectors (one up index); got {u.declaration()}; "
                                         "to pair with a larger tensor, use contract")
            X = _contract(X, u, X._form_positions()[0], 0)
        return X

    def apply(self, A, i):
        """X.apply(A, i): the matrix A acting on X's index i, which stays in place. A has exactly two
        value indices (and any form indices). The one that pairs with index i is summed against it
        (A's second if both could); the other moves into i's place, renamed to i's label (a new
        label if it is in another space), keeping i's type and taking its own arrow and space. A's form indices go first among X's form
        indices. No sign: v.apply(g, "μ") is v lowered; the derivatives add ± X.apply(Γ, i)."""
        _assert_tensor(A)
        _assert_same_manifold(self, A)
        i = self.position(i)
        s = self.indices[i]
        vals, forms = A._value_positions(), A._form_positions()
        if len(vals) != 2:
            raise IndexMismatchError(f"apply: needs exactly two value indices; got {A.declaration()}")
        first, second = vals
        eat, put = (second, first) if A.index(second).pairs_with(s) else (first, second)
        if not A.index(eat).pairs_with(s):
            raise IndexMismatchError(f"apply: {A.declaration()} has no index that pairs with {s!r}")
        p = A.index(put)
        if s.is_form and p.is_up:
            raise IndexMismatchError(f"apply: {s.label} is a form index, always down; "
                                     f"{A.declaration()} would make it up")
        data = np.tensordot(A.data, self.data, axes=([eat], [i]))   # [A's put and forms, in A's order; X's others]
        a_rest = [m for m in range(len(A.indices)) if m != eat]
        n, k = len(self.indices), len(forms)
        x_axis = {m: k + 1 + pos for pos, m in enumerate(m for m in range(n) if m != i)}
        source = [a_rest.index(put) if m == i else x_axis[m] for m in range(n)]
        at = next((m for m, t in enumerate(self.indices) if t.is_form), n)    # first among the forms
        data = np.transpose(data, source[:at] + [a_rest.index(f) for f in forms] + source[at:])
        others = [t.label for m, t in enumerate(self.indices) if m != i]
        label = s.label if p.space is s.space else _new_label(p.space, self.manifold, others)  # a new space: its own letters
        rest = [Index(label, p.space, p.variance, s.type) if m == i else t for m, t in enumerate(self.indices)]
        new = _rename_clashes(self.manifold, [t.label for t in rest], [A.indices[f] for f in forms])
        return type(self)(self.manifold, data, rest[:at] + new + rest[at:], None, copy=False)

    def otimes(self, other):
        """Tensor product A ⊗ B: pair nothing, keep every index.
        Result: value indices first, then A's form indices, then B's."""
        _assert_tensor(other)
        _assert_same_manifold(self, other)
        keep_b = _rename_clashes(self.manifold, self._index_labels(), other.indices)
        return _joined(self, np.multiply.outer(self.data, other.data), (*self.indices, *keep_b),
                       len(self.indices), None)

    def wedge(self, other, pair=None):
        """A ∧ B = k!/(p! q!) · Alt(AB) over all form indices (each interleaving once).
        AB pairs A's column with B's row (or the indices in pair); with a scalar-valued factor it is A ⊗ B."""
        _assert_tensor(other)
        _assert_same_manifold(self, other)
        return _wedge_like(self, other, pair, signed=True)

    def odot(self, other, pair=None):
        """A ⊙ B = k!/(p! q!) · Sym(AB) over all form indices."""
        _assert_tensor(other)
        _assert_same_manifold(self, other)
        return _wedge_like(self, other, pair, signed=False)
