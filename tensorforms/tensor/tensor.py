"""Tensors: data plus an ordered list of slots.

Everything here acts on one tensor: describing it, adding and scaling, moving slots,
changing roles, Alt/Sym, trace, and raising/lowering with the manifold's metric.
Products of two tensors are in products.py.
"""
import itertools
import math

import numpy as np
import sympy as sp
from sympy.combinatorics import Permutation

from ..symbolic.simplify import clean_array, is_zero as _is_zero
from .labels import join, negated, scaled, slot_label, swapped, wrap
from .slot import Slot


class Tensor:
    def __init__(self, manifold, data, slots, label=None):
        self.manifold = manifold
        self.data = np.asarray(data, dtype=object)
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

    def _new(self, data=None, slots=None, *, label):
        """A tensor on the same manifold; data and slots default to this tensor's.
        The label must be given (label=self.label to keep it, label=None for none)."""
        return Tensor(self.manifold,
                      self.data if data is None else data,
                      self.slots if slots is None else slots,
                      label)

    # ---------- description ----------
    @property
    def names(self):
        return [s.name for s in self.slots]

    @property
    def types(self):
        return [s.type for s in self.slots]

    @property
    def form_slots(self):
        """Positions of the form slots."""
        return [k for k, s in enumerate(self.slots) if s.is_form]

    @property
    def value_slots(self):
        """Positions of the value slots."""
        return [k for k, s in enumerate(self.slots) if not s.is_form]

    def position(self, x):
        """A slot by name ("μ", "mu", an Index or Slot) or by number (0, 1, …)."""
        if isinstance(x, int) and not isinstance(x, bool):
            if not 0 <= x < len(self.slots):
                raise IndexError(f"no slot number {x}; this tensor has {len(self.slots)} slots")
            return x
        n = slot_label(x)
        if n not in self.names:
            raise KeyError(f"no slot named {n}; slots are {self.names}")
        return self.names.index(n)

    def slot(self, x):
        return self.slots[self.position(x)]

    def declaration(self):
        """The slots in order: value slots carry ↑ or ↓, form slots carry no arrow."""
        return "(" + " ".join(repr(s) for s in self.slots) + ")"

    def describe(self):
        head = f"{self.label or 'Tensor'} on {self.manifold.name}"
        return "\n".join([head] + ["  " + s.describe() for s in self.slots])

    def __repr__(self):
        return f"{self.label or 'Tensor'}{self.declaration()}"

    def __getitem__(self, key):
        return self.data[key]

    # ---------- arithmetic (needs matching types) ----------
    def _check_same_type(self, other):
        if not isinstance(other, Tensor) or self.types != other.types:
            raise TypeError(f"can only add matching types: {self.declaration()} vs "
                            f"{getattr(other, 'declaration', lambda: other)()}")

    def __add__(self, other):
        self._check_same_type(other)
        return self._new(self.data + other.data, label=join(self.label, "+", wrap(other.label)
                                                              if other.label and other.label[0] == "−" else other.label))

    def __sub__(self, other):
        self._check_same_type(other)
        return self._new(self.data - other.data, label=join(self.label, "−", wrap(other.label)))

    def __neg__(self):
        return self._new(-self.data, label=negated(self.label))

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
        return self._new(self.data * c, label=scaled(c, c_label, self.label))

    __rmul__ = __mul__

    def simplify(self):
        return self._new(clean_array(self.data), label=self.label)

    def is_zero(self):
        return all(_is_zero(e) for e in self.data.flat)

    def equals(self, other):
        return (self - other).is_zero()

    # ---------- moving slots ----------
    def _transposed(self, order):
        return self._new(np.transpose(self.data, order), [self.slots[i] for i in order], label=self.label)

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
        X.label = swapped(self.label, letter)
        return X

    # T, S, U, V: short for swap_positions of neighbouring slots
    def T(self):
        """Short for swap_positions(0, 1)."""
        return self._letter_swap(0, "T")

    def S(self):
        """Short for swap_positions(1, 2)."""
        return self._letter_swap(1, "S")

    def U(self):
        """Short for swap_positions(2, 3)."""
        return self._letter_swap(2, "U")

    def V(self):
        """Short for swap_positions(3, 4)."""
        return self._letter_swap(3, "V")

    def rename(self, mapping):
        """New labels: {slot (name or number): new name}."""
        mapping = {self.position(k): slot_label(v) for k, v in mapping.items()}
        return self._new(slots=[s.renamed(mapping[k]) if k in mapping else s for k, s in enumerate(self.slots)],
                         label=self.label)

    def _relabel(self, which, new_role, eligible, need):
        if which:
            targets = {self.position(x) for x in which}
            bad = [self.slots[k] for k in sorted(targets) if not eligible(self.slots[k])]
            if bad:
                raise TypeError(f"needs {need}; got {bad}")
        else:
            targets = {k for k, s in enumerate(self.slots) if eligible(s)}
        return self._new(slots=[s.with_role(new_role) if k in targets else s for k, s in enumerate(self.slots)],
                         label=self.label)

    def as_form(self, *names):
        """Down value slots → form slots (default: all). Covector (μ↓) → 1-form (μ). Same numbers."""
        return self._relabel(names, "form", lambda s: not s.is_up and not s.is_form, "down value slots")

    def as_value(self, *names):
        """Form slots → down value slots (default: all). 1-form (μ) → covector (μ↓). Same numbers."""
        return self._relabel(names, "value", lambda s: s.is_form, "form slots")

    # ---------- trace ----------
    def tr(self, a, b):
        """Pair two slots inside this tensor (one up, one down) and sum over them."""
        i, j = self.position(a), self.position(b)
        si, sj = self.slots[i], self.slots[j]
        if si.space is not sj.space or si.variance == sj.variance:
            raise TypeError(f"can't pair {si.name} and {sj.name}: need the same space, one up and one down")
        data = np.asarray(np.trace(self.data, axis1=i, axis2=j), dtype=object)
        label = f"tr({self.label})" if self.label else None
        return Tensor(self.manifold, data, [s for k, s in enumerate(self.slots) if k not in (i, j)], label)

    # ---------- raising and lowering (needs the manifold's metric) ----------
    def _move_index(self, slot, want_up, verb):
        if slot is None:
            options = [k for k, s in enumerate(self.slots) if not s.is_form and s.is_up == want_up]
            if len(options) != 1:
                raise TypeError(f"{verb}: say which slot (candidates: {[self.names[k] for k in options]})")
            slot = options[0]
        i = self.position(slot)
        s = self.slots[i]
        if s.is_form:
            raise TypeError(f"{verb}: {s.name} is a form slot; form slots are never raised or lowered "
                            "(use as_value() first)")
        if s.is_up != want_up:
            raise TypeError(f"{verb}: {s.name} is already {'down' if want_up else 'up'}")
        M = self.manifold
        if s.space is not M.tangent:
            raise TypeError(f"{verb}: {s.name} is not in the tangent space")
        metric = M.g if want_up else M.ginv
        data = np.moveaxis(np.tensordot(metric.data, self.data, axes=([1], [i])), 0, i)
        flipped = Slot(s.name, s.space, "down" if want_up else "up", "value")
        return self._new(data, self.slots[:i] + [flipped] + self.slots[i + 1:], label=self.label)

    def lower(self, slot=None):
        """g on one up value slot: ↑ → ↓. Same physics; raise_ undoes it exactly.
        v.lower() for a vector; R.lower("ρ") for a named slot."""
        return self._move_index(slot, True, "lower")

    def raise_(self, slot=None):
        """g⁻¹ on one down value slot: ↓ → ↑. Same physics; lower undoes it exactly."""
        return self._move_index(slot, False, "raise_")

    # ---------- Hodge star (needs the manifold's metric) ----------
    def star(self):
        """⋆X: the k form slots become n − k, with the same amount (direction ↔ plane ↔ volume …).
        Raise the form slots, fill them into the volume form ε in order, divide by k!.
        Value slots are carried along. Uses the metric and the coordinate order (orientation)."""
        from .products import otimes                 # here, to avoid a circular import
        M = self.manifold
        if not M.has_metric():
            raise TypeError("star needs a metric: the volume form and raising both use it")
        forms = self.form_slots
        k = len(forms)
        X = self.as_value(*forms) if k else self
        for p in forms:
            X = X.raise_(p)
        T = otimes(X, M.volume_form())                # X's value slots, then ε's n form slots
        nv = len(X.slots)
        for j, p in enumerate(forms):                 # the j-th raised slot fills ε's next slot
            T = T.tr(p - j, nv - j)
        T = T * sp.Rational(1, math.factorial(k)) if k > 1 else T
        return T._new(label=f"⋆{wrap(self.label)}" if self.label else None)

    # ---------- symmetric and antisymmetric parts ----------
    def _average(self, names, signed):
        pos = [self.position(n) for n in names] if names else self.form_slots
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
            out = out + (Permutation(perm).signature() if signed else 1) * term
        label = f"{'Alt' if signed else 'Sym'}({self.label})" if self.label else None   # brackets already
        return self._new(out * sp.Rational(1, math.factorial(k)), label=label)

    def alt(self, *names):
        """Antisymmetric part over the named slots (default: all form slots)."""
        return self._average(names, signed=True)

    def sym(self, *names):
        """Symmetric part over the named slots (default: all form slots)."""
        return self._average(names, signed=False)
