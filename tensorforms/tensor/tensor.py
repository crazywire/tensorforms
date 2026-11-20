"""Tensors: data plus an ordered list of slots.

Pairing is always explicit (no Einstein summation):
    contract(A, B)         your AB: A's column (last value slot, down) with B's row (first, up)
    contract(A, B, a, b)   pair slot a of A with slot b of B
    tr(X, a, b)            pair two slots inside X
    otimes(A, B)           tensor product: pair nothing
Nothing is chosen silently: if the default pair doesn't exist, contract refuses.
"""
import itertools
import math
import re

import numpy as np
import sympy as sp

from ..symbolic.simplify import clean_array, is_zero as _is_zero
from .slot import clean_name


def _name(x):
    return x.name if hasattr(x, "name") else clean_name(x)


# ---------- labels: the result's label is the operation that made it ----------
def _wrap(label):
    """Brackets around a label that has + or − in it, so it can sit inside a bigger expression."""
    if label and any(ch in label for ch in "+−"):
        return f"({label})"
    return label


def _join(a, op, b):
    return f"{a}{op}{b}" if a and b else None


def _negated(label):
    if not label:
        return None
    if label.startswith("−") and not any(ch in label[1:] for ch in "+−"):
        return label[1:]                                         # −(−A) = A
    return "−" + _wrap(label)


def _scaled(c, c_label, label):
    """3A, ½·A, sin(θ)·A, f·A. Unlabelled tensor: no label."""
    if not label:
        return None
    if c == 1:
        return label
    if c == -1:
        return _negated(label)
    if c_label:
        return f"{c_label}·{_wrap(label)}"
    if isinstance(c, sp.Integer):
        return f"{c}{_wrap(label)}" if c > 0 else f"−{-c}{_wrap(label)}"
    return f"{_wrap(sp.pretty(c, use_unicode=True))}·{_wrap(label)}"


def _swapped(label, letter):
    """R^S, then R^ST: the swap letters in the order they were applied."""
    if not label:
        return None
    if re.search(r"\^[TSUV]+$", label) and _wrap(label.split("^")[0]) == label.split("^")[0]:
        return label + letter
    return f"{_wrap(label)}^{letter}"


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
        return self._new(self.data + other.data, label=_join(self.label, "+", _wrap(other.label)
                                                              if other.label and other.label[0] == "−" else other.label))

    def __sub__(self, other):
        self._check_same_type(other)
        return self._new(self.data - other.data, label=_join(self.label, "−", _wrap(other.label)))

    def __neg__(self):
        return self._new(-self.data, label=_negated(self.label))

    def __mul__(self, c):
        """Multiply by a number, a sympy expression, or a scalar tensor (no slots)."""
        if not self.slots and isinstance(c, Tensor):         # scalar × tensor
            return c * self
        c_label = None
        if isinstance(c, Tensor):
            if c.slots:
                raise TypeError("use contract, otimes or wedge to multiply tensors")
            c_label, c = c.label, c.data.item()
        try:
            c = sp.sympify(c, strict=True)
        except sp.SympifyError:
            c = None
        if not isinstance(c, sp.Expr):
            raise TypeError("can only multiply a tensor by a number, an expression or a scalar tensor")
        return self._new(self.data * c, label=_scaled(c, c_label, self.label))

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
    def _letter_swap(self, i, letter):
        X = self.swap_positions(i, i + 1)
        X.label = _swapped(self.label, letter)
        return X

    def T(self): return self._letter_swap(0, "T")
    def S(self): return self._letter_swap(1, "S")
    def U(self): return self._letter_swap(2, "U")
    def V(self): return self._letter_swap(3, "V")

    def rename(self, mapping):
        mapping = {self.names[self.position(k)]: _name(v) for k, v in mapping.items()}
        return self._new(slots=[s.renamed(mapping.get(s.name, s.name)) for s in self.slots], label=self.label)

    def as_form(self, *names):
        """The same numbers with down value slots relabelled as form slots (default: all of them).
        The bridge from a covector (μ↓ |) to a 1-form (| μ), e.g. before d∧."""
        if names:
            targets = {self.names[self.position(n)] for n in names}
            bad = [s for s in self.slots if s.name in targets and (s.is_up or s.is_form)]
            if bad:
                raise TypeError(f"as_form needs down value slots; got {bad}")
        else:
            targets = {s.name for s in self.slots if not s.is_up and not s.is_form}
        return self._new(slots=[s.with_role("form") if s.name in targets else s for s in self.slots],
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
        label = f"{'Alt' if signed else 'Sym'}({self.label})" if self.label else None   # brackets already
        return self._new(out * sp.Rational(1, math.factorial(k)), label=label)

    def alt(self, *names):
        """Antisymmetric part over the named slots (default: all form slots)."""
        return self._average(names, signed=True)

    def sym(self, *names):
        """Symmetric part over the named slots (default: all form slots)."""
        return self._average(names, signed=False)


# ---------------------------------------------------------------- pairing
def _fresh(used, space):
    """The first name from the space's list not in `used`."""
    for n in space.names:
        if n not in used:
            return n
    k = 1
    while f"{space.names[0]}{k}" in used:
        k += 1
    return f"{space.names[0]}{k}"


def _rename_clashes(keep_names, slots):
    """B's slots, with any name already used by A replaced by a fresh one from the same space."""
    used, out = set(keep_names) | {s.name for s in slots}, []
    taken = set(keep_names)
    for s in slots:
        n = s.name
        if n in taken:
            n = _fresh(used, s.space)
            used.add(n)
        taken.add(n)
        out.append(s.renamed(n))
    return out


def _check_pair(sa, sb):
    if sa.space is not sb.space:
        raise TypeError(f"can't pair {sa.name} ({sa.space.name}) with {sb.name} ({sb.space.name}): different spaces")
    if sa.variance == sb.variance:
        raise TypeError(f"can't pair {sa.name} and {sb.name}: both {sa.variance}; raise or lower one first")


def _values_first(slots, a_forms):
    """Order: value slots, then A's form slots, then B's form slots."""
    vals = [k for k, s in enumerate(slots) if not s.is_form]
    fa = [k for k, s in enumerate(slots) if s.is_form and s.name in a_forms]
    fb = [k for k, s in enumerate(slots) if s.is_form and s.name not in a_forms]
    return vals + fa + fb


def _joined(A, B, data, slots, label):
    order = _values_first(slots, set(A.form_slots))
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
    return _joined(A, B, data, keep_a + keep_b, _join(_wrap(A.label), "", _wrap(B.label)))


def tr(X, a, b):
    """Pair two slots inside X."""
    i, j = X.position(a), X.position(b)
    _check_pair(X.slots[i], X.slots[j])
    data = np.asarray(np.trace(X.data, axis1=i, axis2=j), dtype=object)
    label = f"tr({X.label})" if X.label else None
    return Tensor(X.manifold, data, [s for k, s in enumerate(X.slots) if k not in (i, j)], label)


def otimes(A, B):
    """Tensor product A ⊗ B: pair nothing, keep every slot.
    Result: value slots first, then A's form slots, then B's."""
    keep_b = _rename_clashes(A.names, B.slots)
    return _joined(A, B, np.multiply.outer(A.data, B.data), A.slots + keep_b,
                   _join(_wrap(A.label), "⊗", _wrap(B.label)))


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
    return P._new(label=_join(_wrap(A.label), sym, _wrap(B.label)))


def wedge(A, B, pair=None):
    """A ∧ B = k!/(p! q!) · Alt(AB) over all form slots (each interleaving once).
    AB pairs A's column with B's row (or the slots in pair); with a scalar-valued factor it is A ⊗ B."""
    return _wedge_like(A, B, pair, signed=True)


def odot(A, B, pair=None):
    """A ⊙ B = k!/(p! q!) · Sym(AB) over all form slots."""
    return _wedge_like(A, B, pair, signed=False)
