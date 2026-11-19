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
from .slot import Slot


def _name(x):
    return x.name if hasattr(x, "name") else str(x)


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

    # ---------- description ----------
    @property
    def names(self):
        return [s.name for s in self.slots]

    @property
    def signature(self):
        return [s.type for s in self.slots]

    @property
    def form_slots(self):
        return [s.name for s in self.slots if s.role == "form"]

    @property
    def value_slots(self):
        return [s.name for s in self.slots if s.role == "value"]

    def position(self, x):
        n = _name(x)
        if n not in self.names:
            raise KeyError(f"no slot named {n}; slots are {self.names}")
        return self.names.index(n)

    def slot(self, x):
        return self.slots[self.position(x)]

    def declaration(self):
        roles = [s.role for s in self.slots]
        if roles == sorted(roles, key=lambda r: r == "form"):
            vals = " ".join(repr(s) for s in self.slots if s.role == "value")
            forms = " ".join(repr(s) for s in self.slots if s.role == "form")
            return f"({vals} | {forms})".replace("( ", "(").replace(" )", ")")
        return "(" + " ".join(repr(s) + ("ᶠ" if s.role == "form" else "") for s in self.slots) + ")"

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
        return Tensor(self.manifold, self.data + other.data, self.slots)

    def __sub__(self, other):
        self._check_same_type(other)
        return Tensor(self.manifold, self.data - other.data, self.slots)

    def __neg__(self):
        return Tensor(self.manifold, -self.data, self.slots)

    def __mul__(self, c):
        if isinstance(c, Tensor):
            raise TypeError("use contract, outer, product or wedge to multiply tensors")
        return Tensor(self.manifold, self.data * sp.sympify(c), self.slots)

    __rmul__ = __mul__

    def simplify(self):
        return Tensor(self.manifold, clean_array(self.data), self.slots, self.label)

    def is_zero(self):
        return all(_is_zero(e) for e in np.asarray(self.data).flat)

    def equals(self, other):
        return (self - other).is_zero()

    # ---------- moving slots ----------
    def reorder(self, *names):
        order = [self.position(n) for n in names]
        if sorted(order) != list(range(len(self.slots))):
            raise ValueError("reorder needs every slot exactly once")
        return Tensor(self.manifold, np.transpose(self.data, order), [self.slots[i] for i in order], self.label)

    def swap(self, a, b):
        i, j = self.position(a), self.position(b)
        return self.swap_positions(i, j)

    def swap_positions(self, i, j):
        order = list(range(len(self.slots)))
        order[i], order[j] = order[j], order[i]
        return Tensor(self.manifold, np.transpose(self.data, order), [self.slots[k] for k in order], self.label)

    # positional swaps from the notation: T, S, U, V
    def T(self): return self.swap_positions(0, 1)
    def S(self): return self.swap_positions(1, 2)
    def U(self): return self.swap_positions(2, 3)
    def V(self): return self.swap_positions(3, 4)

    def rename(self, mapping):
        mapping = {_name(k): _name(v) for k, v in mapping.items()}
        return Tensor(self.manifold, self.data, [s.renamed(mapping.get(s.name, s.name)) for s in self.slots], self.label)

    def with_roles(self, role, *names):
        """Relabel slots as value or form (same numbers, different role)."""
        targets = {_name(n) for n in names} or set(self.names)
        return Tensor(self.manifold, self.data,
                      [s.with_role(role) if s.name in targets else s for s in self.slots], self.label)

    # ---------- symmetric and antisymmetric parts ----------
    def _average(self, names, signed):
        names = [_name(n) for n in names] or self.form_slots
        pos = [self.position(n) for n in names]
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
        return Tensor(self.manifold, out * sp.Rational(1, math.factorial(k)), self.slots, self.label)

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
    data = np.trace(X.data, axis1=i, axis2=j)
    data = np.asarray(data, dtype=object)
    return Tensor(X.manifold, data, [s for k, s in enumerate(X.slots) if k not in (i, j)])


def outer(A, B):
    keep_b = _rename_clashes(A.names, B.slots)
    return Tensor(A.manifold, np.multiply.outer(A.data, B.data), A.slots + keep_b)


def _values_first(X, a_forms, b_forms):
    """Put value slots first, then A's forms, then B's forms."""
    vals = [s.name for s in X.slots if s.role == "value"]
    forms = [n for n in X.names if n in a_forms] + [n for n in X.names if n not in a_forms and n not in vals]
    return X.reorder(*(vals + forms))


def product(A, B, pair=None):
    """Default product: A's last value slot with B's first value slot, if they can pair.
    pair=(a, b) chooses the slots explicitly; pair=False means outer product."""
    a_forms = set(A.form_slots)
    if pair is False:
        P = outer(A, B)
    elif pair is not None:
        P = contract(A, B, *pair)
    else:
        av, bv = A.value_slots, B.value_slots
        if av and bv:
            sa, sb = A.slot(av[-1]), B.slot(bv[0])
            if sa.space is sb.space and sa.variance == "down" and sb.variance == "up":
                P = contract(A, B, av[-1], bv[0])
            else:
                P = outer(A, B)
        else:
            P = outer(A, B)
    b_forms = [n for n in P.form_slots if n not in a_forms]
    return _values_first(P, a_forms, b_forms)


def _wedge_like(A, B, pair, signed):
    p, q = len(A.form_slots), len(B.form_slots)
    A2 = A.alt() if signed else A.sym()      # step 1: each factor's own form slots
    B2 = B.alt() if signed else B.sym()
    P = product(A2, B2, pair)                # step 2: the product
    k = p + q
    if k < 2:
        return P
    part = P.alt(*P.form_slots) if signed else P.sym(*P.form_slots)
    return part * sp.Rational(math.factorial(k), math.factorial(p) * math.factorial(q))


def wedge(A, B, pair=None):
    """A ∧ B = k!/(p! q!) · Alt(AB) over all form slots (each interleaving once)."""
    return _wedge_like(A, B, pair, signed=True)


def odot(A, B, pair=None):
    """A ⊙ B = k!/(p! q!) · Sym(AB) over all form slots."""
    return _wedge_like(A, B, pair, signed=False)
