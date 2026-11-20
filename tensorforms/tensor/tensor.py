"""Tensors: data plus an ordered list of slots.

Pairing is always explicit (no Einstein summation):
    contract(A, B, a, b)   pair slot a of A with slot b of B
    tr(X, a, b)            pair two slots inside X
    outer(A, B)            pair nothing
    product(A, B)          the default product: A's last value slot with B's first
"""
import itertools
import math

import numpy as np
import sympy as sp

from ..symbolic.simplify import clean_array, is_zero as _is_zero
from .slot import clean_name


def _name(x):
    return x.name if hasattr(x, "name") else clean_name(x)


def _perm_sign(perm):
    sign, seen = 1, [False] * len(perm)
    for i in range(len(perm)):
        if not seen[i]:
            j, n = i, 0
            while not seen[j]:
                seen[j], j, n = True, perm[j], n + 1
            if n % 2 == 0:
                sign = -sign
    return sign


def _as_object_array(data):
    if isinstance(data, np.ndarray):
        return data.astype(object)
    return np.array(data, dtype=object)


class Tensor:
    def __init__(self, manifold, data, slots, label=None):
        self.manifold = manifold
        self.data = _as_object_array(data)
        self.slots = list(slots)
        self.label = label
        if self.data.ndim != len(self.slots):
            raise ValueError(f"data has {self.data.ndim} axes but {len(self.slots)} slots were declared")
        for n, s in zip(self.data.shape, self.slots):
            if n != s.space.dim:
                raise ValueError(f"slot {s.name} lives in {s.space.name} (size {s.space.dim}) but its axis has size {n}")
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"slot names must be unique within a tensor: {names}")

    def _new(self, data=None, slots=None, label=None):
        """A tensor on the same manifold; data and slots default to this tensor's."""
        return Tensor(self.manifold,
                      self.data if data is None else data,
                      self.slots if slots is None else slots,
                      label)

    # ---------- description ----------
    @property
    def names(self):
        return [s.name for s in self.slots]

    @property
    def signature(self):
        return [s.type for s in self.slots]

    @property
    def form_slots(self):
        return [s.name for s in self.slots if s.is_form]

    @property
    def value_slots(self):
        return [s.name for s in self.slots if not s.is_form]

    def position(self, x):
        """A slot by name ("μ", "mu", an Index or Slot) or by number (0, 1, …)."""
        if isinstance(x, int) and not isinstance(x, bool):
            if not 0 <= x < len(self.slots):
                raise IndexError(f"no slot number {x}; this tensor has {len(self.slots)} slots")
            return x
        n = _name(x)
        if n not in self.names:
            raise KeyError(f"no slot named {n}; slots are {self.names}")
        return self.names.index(n)

    def slot(self, x):
        return self.slots[self.position(x)]

    def declaration(self):
        flags = [s.is_form for s in self.slots]
        if flags == sorted(flags):                            # value slots, then form slots
            vals = " ".join(repr(s) for s in self.slots if not s.is_form)
            forms = " ".join(repr(s) for s in self.slots if s.is_form)
            return f"({vals} | {forms})".replace("( ", "(").replace(" )", ")")
        return "(" + " ".join(repr(s) + ("ᶠ" if s.is_form else "") for s in self.slots) + ")"

    def describe(self):
        head = f"{self.label or 'Tensor'} on {self.manifold.name}"
        return "\n".join([head] + ["  " + s.describe() for s in self.slots])

    def __repr__(self):
        return f"{self.label or 'Tensor'}{self.declaration()}"

    def __getitem__(self, key):
        return self.data[key]

    # ---------- arithmetic (needs matching types) ----------
    def _check_same_type(self, other):
        if not isinstance(other, Tensor) or self.signature != other.signature:
            raise TypeError(f"can only add matching types: {self.declaration()} vs "
                            f"{getattr(other, 'declaration', lambda: other)()}")

    def __add__(self, other):
        self._check_same_type(other)
        return self._new(self.data + other.data)

    def __sub__(self, other):
        self._check_same_type(other)
        return self._new(self.data - other.data)

    def __neg__(self):
        return self._new(-self.data)

    def __mul__(self, c):
        if isinstance(c, Tensor):
            raise TypeError("use contract, outer, product or wedge to multiply tensors")
        return self._new(self.data * sp.sympify(c))

    __rmul__ = __mul__

    def simplify(self):
        return self._new(clean_array(self.data), label=self.label)

    def is_zero(self):
        return all(_is_zero(e) for e in self.data.flat)

    def equals(self, other):
        return (self - other).is_zero()

    # ---------- moving slots ----------
    def _transposed(self, order):
        return self._new(np.transpose(self.data, order), [self.slots[i] for i in order], self.label)

    def reorder(self, *names):
        order = [self.position(n) for n in names]
        if sorted(order) != list(range(len(self.slots))):
            raise ValueError("reorder needs every slot exactly once")
        return self._transposed(order)

    def swap(self, a, b):
        return self.swap_positions(self.position(a), self.position(b))

    def swap_positions(self, i, j):
        order = list(range(len(self.slots)))
        order[i], order[j] = order[j], order[i]
        return self._transposed(order)

    # positional swaps from the notation: T, S, U, V
    def T(self): return self.swap_positions(0, 1)
    def S(self): return self.swap_positions(1, 2)
    def U(self): return self.swap_positions(2, 3)
    def V(self): return self.swap_positions(3, 4)

    def rename(self, mapping):
        mapping = {self.names[self.position(k)]: _name(v) for k, v in mapping.items()}
        return self._new(slots=[s.renamed(mapping.get(s.name, s.name)) for s in self.slots], label=self.label)

    def with_roles(self, role, *names):
        """Relabel slots as value or form (same numbers, different role)."""
        targets = {self.names[self.position(n)] for n in names} or set(self.names)
        return self._new(slots=[s.with_role(role) if s.name in targets else s for s in self.slots],
                         label=self.label)

    # ---------- symmetric and antisymmetric parts ----------
    def _average(self, names, signed):
        pos = [self.position(n) for n in (names or self.form_slots)]
        types = {self.slots[p].type for p in pos}
        if len(types) > 1:
            raise TypeError("Alt/Sym need slots of the same type (space and up/down)")
        k = len(pos)
        if k < 2:
            return self
        out = 0
        for perm in itertools.permutations(range(k)):
            axes = list(range(len(self.slots)))
            for src, dst in zip(perm, range(k)):
                axes[pos[dst]] = pos[src]
            term = np.transpose(self.data, axes)
            out = out + (_perm_sign(perm) if signed else 1) * term
        return self._new(out * sp.Rational(1, math.factorial(k)), label=self.label)

    def alt(self, *names):
        """Antisymmetric part over the named slots (default: all form slots)."""
        return self._average(names, signed=True)

    def sym(self, *names):
        """Symmetric part over the named slots (default: all form slots)."""
        return self._average(names, signed=False)


# ---------------------------------------------------------------- pairing
def _rename_clashes(keep_names, slots):
    used, out = set(keep_names), []
    for s in slots:
        n = s.name
        while n in used:
            n += "′"
        used.add(n)
        out.append(s.renamed(n))
    return out


def _check_pair(sa, sb):
    if sa.space is not sb.space:
        raise TypeError(f"can't pair {sa.name} ({sa.space.name}) with {sb.name} ({sb.space.name}): different spaces")
    if sa.variance == sb.variance:
        raise TypeError(f"can't pair {sa.name} and {sb.name}: both {sa.variance}; raise or lower one first")


def contract(A, B, a, b=None):
    """Pair slot a of A with slot b of B (b defaults to the same name)."""
    b = a if b is None else b
    i, j = A.position(a), B.position(b)
    _check_pair(A.slots[i], B.slots[j])
    data = np.tensordot(A.data, B.data, axes=([i], [j]))
    keep_a = [s for k, s in enumerate(A.slots) if k != i]
    keep_b = _rename_clashes([s.name for s in keep_a], [s for k, s in enumerate(B.slots) if k != j])
    return Tensor(A.manifold, data, keep_a + keep_b)


def tr(X, a, b):
    """Pair two slots inside X."""
    i, j = X.position(a), X.position(b)
    _check_pair(X.slots[i], X.slots[j])
    data = np.asarray(np.trace(X.data, axis1=i, axis2=j), dtype=object)
    return Tensor(X.manifold, data, [s for k, s in enumerate(X.slots) if k not in (i, j)])


def outer(A, B):
    keep_b = _rename_clashes(A.names, B.slots)
    return Tensor(A.manifold, np.multiply.outer(A.data, B.data), A.slots + keep_b)


def _values_first(X, a_forms):
    """Put value slots first, then A's forms, then B's forms."""
    vals = X.value_slots
    forms = [n for n in X.form_slots if n in a_forms] + [n for n in X.form_slots if n not in a_forms]
    return X.reorder(*(vals + forms))


def _default_pair(A, B):
    """(A's last value slot, B's first value slot) if they can pair, else None."""
    av, bv = A.value_slots, B.value_slots
    if not (av and bv):
        return None
    sa, sb = A.slot(av[-1]), B.slot(bv[0])
    if sa.space is sb.space and not sa.is_up and sb.is_up:
        return av[-1], bv[0]
    return None


def product(A, B, pair=None):
    """Default product: A's last value slot with B's first value slot, if they can pair.
    pair=(a, b) chooses the slots explicitly; pair=False means outer product."""
    if pair is None:
        pair = _default_pair(A, B)
    P = contract(A, B, *pair) if pair else outer(A, B)
    return _values_first(P, set(A.form_slots))


def _wedge_like(A, B, pair, signed):
    p, q = len(A.form_slots), len(B.form_slots)
    project = Tensor.alt if signed else Tensor.sym
    P = product(project(A), project(B), pair)    # each factor's own form slots, then the product
    k = p + q
    if k < 2:
        return P
    return project(P) * sp.Rational(math.factorial(k), math.factorial(p) * math.factorial(q))


def wedge(A, B, pair=None):
    """A ∧ B = k!/(p! q!) · Alt(AB) over all form slots (each interleaving once)."""
    return _wedge_like(A, B, pair, signed=True)


def odot(A, B, pair=None):
    """A ⊙ B = k!/(p! q!) · Sym(AB) over all form slots."""
    return _wedge_like(A, B, pair, signed=False)
