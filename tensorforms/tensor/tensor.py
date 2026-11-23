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

from ..symbolic.simplify import is_zero as _is_zero, simplify_expr
from .labels import join, negated, scaled, slot_label, swapped, wrap
from .slot import Slot


def _as_expr(x):
    """x as an exact sympy expression (2 → Integer(2), never a float), or None if it isn't one."""
    try:
        x = sp.sympify(x, strict=True)
    except sp.SympifyError:
        return None
    return x if isinstance(x, sp.Expr) else None


def _as_scalar(c):
    """c as an exact scalar, with its label: a number, an expression or a scalar tensor."""
    not_scalar = ("a tensor can only be scaled by a number, an expression or a scalar tensor "
                  "(for products of tensors: contract, otimes, wedge)")
    c_label = None
    if isinstance(c, Tensor):
        if c.slots:
            raise TypeError(not_scalar)
        c_label, c = c.label, c.data.item()
    c = _as_expr(c)
    if c is None:
        raise TypeError(not_scalar)
    return c, c_label


class Tensor:
    # sympy defers mixed operations to objects with a higher _op_priority (its expressions have 10),
    # so θ * X and θ + s stay tensor arithmetic instead of sympy taking over.
    _op_priority = 20.0

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
    def value(self):
        """The components in sympy form: a scalar's expression, otherwise a sympy Array
        (one axis per slot). This is also what sympy sees when it is handed a tensor."""
        return self.data.item() if not self.slots else sp.Array(self.data.tolist())

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

    def _operand(self, other):
        """The other side of + or −, as a tensor. A number means a scalar (no slots), so the
        type check decides; 0 is the zero tensor of any type, returned as None (nothing to add)."""
        if isinstance(other, Tensor):
            return other
        c = _as_expr(other)
        if c is None:
            raise TypeError(f"can only add tensors and numbers, not {type(other).__name__}")
        if c == 0:
            return None
        return Tensor(self.manifold, c, [], label=sp.pretty(c, use_unicode=True))

    def __add__(self, other):
        other = self._operand(other)
        if other is None:                                     # X + 0 is X
            return self._new(label=self.label)
        self._check_same_type(other)
        # brackets only where needed: A+(−B); A+B+C needs none
        return self._new(self.data + other.data, label=join(self.label, "+", wrap(other.label)
                                                              if other.label and other.label[0] == "−" else other.label))

    __radd__ = __add__                                        # 0 + X, 1 + s

    def __sub__(self, other):
        other = self._operand(other)
        if other is None:                                     # X − 0 is X
            return self._new(label=self.label)
        self._check_same_type(other)
        return self._new(self.data - other.data, label=join(self.label, "−", wrap(other.label)))

    def __rsub__(self, other):                                # 0 − X is −X; 1 − s
        other = self._operand(other)
        return -self if other is None else other - self

    def __neg__(self):
        return self._new(-self.data, label=negated(self.label))

    def __mul__(self, c):
        """Multiply by a number, a sympy expression, or a scalar tensor (no slots)."""
        if not self.slots and isinstance(c, Tensor):         # scalar × tensor
            return c * self
        c, c_label = _as_scalar(c)
        return self._new(self.data * c, label=scaled(c, c_label, self.label))

    __rmul__ = __mul__

    def __truediv__(self, c):
        """Divide by a number, a sympy expression, or a scalar tensor: exact (X / 2 is ½X, never 0.5X)."""
        c, c_label = _as_scalar(c)
        if c == 0:
            raise ZeroDivisionError("division of a tensor by zero")
        label = f"{wrap(self.label)}/{c_label or wrap(sp.pretty(c, use_unicode=True))}" if self.label else None
        return self._new(self.data / c, label=label)

    def __rtruediv__(self, c):
        """A number divided by a scalar is a scalar (2 / s); dividing by a tensor with slots is refused."""
        if self.slots:
            raise TypeError("can't divide by a tensor with slots (an inverse is a different operation)")
        n = _as_expr(c)
        if n is None:
            raise TypeError(f"can only divide a number or expression by a scalar, not {type(c).__name__}")
        return Tensor(self.manifold, n, [], label=sp.pretty(n, use_unicode=True)) / self

    def _sympy_(self):
        """What sympy sees when it is handed a tensor: its value. So sp.simplify(s), sp.diff(s, θ) work."""
        return self.value

    def simplify(self):
        """Every component simplified (sympy: fu, then simplify; exact rewrites only)."""
        return self._new(np.vectorize(simplify_expr, otypes=[object])(self.data), label=self.label)

    def is_zero(self):
        """True if every component is shown to be 0; False if one is shown not to be;
        None if that can't be decided exactly."""
        answers = [_is_zero(e) for e in self.data.flat]
        if all(a is True for a in answers):
            return True
        return False if any(a is False for a in answers) else None

    def equals(self, other):
        """Same three answers as is_zero, for X − Y."""
        return (self - other).is_zero()

    # ---------- moving slots ----------
    def _transposed(self, order):
        return self._new(np.transpose(self.data, order), [self.slots[i] for i in order], label=self.label)

    def reorder(self, *slots):
        """Put every slot in the given order (labels or positions)."""
        order = [self.position(s) for s in slots]
        if sorted(order) != list(range(len(self.slots))):
            raise ValueError("reorder needs every slot exactly once")
        return self._transposed(order)

    def swap(self, a, b):
        """Exchange two slots, given by label or position."""
        i, j = self.position(a), self.position(b)
        order = list(range(len(self.slots)))
        order[i], order[j] = order[j], order[i]
        return self._transposed(order)

    # T, S, U, V: the notation's swaps of neighbouring slots
    def _letter_swap(self, i, letter):
        X = self.swap(i, i + 1)
        X.label = swapped(self.label, letter)
        return X

    def T(self):
        """Short for swap(0, 1)."""
        return self._letter_swap(0, "T")

    def S(self):
        """Short for swap(1, 2)."""
        return self._letter_swap(1, "S")

    def U(self):
        """Short for swap(2, 3)."""
        return self._letter_swap(2, "U")

    def V(self):
        """Short for swap(3, 4)."""
        return self._letter_swap(3, "V")

    def rename(self, mapping):
        """New labels: {slot (name or number): new name}."""
        mapping = {self.position(k): slot_label(v) for k, v in mapping.items()}
        return self._new(slots=[s.renamed(mapping[k]) if k in mapping else s for k, s in enumerate(self.slots)],
                         label=self.label)

    def _change_role(self, which, new_role, eligible, need):
        """The chosen slots (default: every eligible one) get the new role; same numbers."""
        if which:
            targets = {self.position(x) for x in which}
            bad = [self.slots[k] for k in sorted(targets) if not eligible(self.slots[k])]
            if bad:
                raise TypeError(f"needs {need}; got {bad}")
        else:
            targets = {k for k, s in enumerate(self.slots) if eligible(s)}
        return self._new(slots=[s.with_role(new_role) if k in targets else s for k, s in enumerate(self.slots)],
                         label=self.label)

    def as_form(self, *slots):
        """Down value slots → form slots (default: all). Covector (μ↓) → 1-form (μ). Same numbers."""
        return self._change_role(slots, "form", lambda s: not s.is_up and not s.is_form, "down value slots")

    def as_value(self, *slots):
        """Form slots → down value slots (default: all). 1-form (μ) → covector (μ↓). Same numbers."""
        return self._change_role(slots, "value", lambda s: s.is_form, "form slots")

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
    def _raise_or_lower(self, slot, verb):
        """Flip one value slot with the metric. verb "lower": up → down with g;
        verb "raise_": down → up with g⁻¹."""
        if verb not in ("lower", "raise_"):
            raise ValueError(f"verb must be 'lower' or 'raise_', not {verb!r}")
        from_up = verb == "lower"                            # lowering acts on an up slot
        if slot is None:
            options = [k for k, s in enumerate(self.slots) if not s.is_form and s.is_up == from_up]
            if len(options) != 1:
                raise TypeError(f"{verb}: say which slot (candidates: {[self.names[k] for k in options]})")
            slot = options[0]
        i = self.position(slot)
        s = self.slots[i]
        if s.is_form:
            raise TypeError(f"{verb}: {s.name} is a form slot; form slots are never raised or lowered "
                            "(use as_value() first)")
        if s.is_up != from_up:
            raise TypeError(f"{verb}: {s.name} is already {'down' if from_up else 'up'}")
        M = self.manifold
        if s.space is not M.tangent:
            raise TypeError(f"{verb}: {s.name} is not in the tangent space")
        metric = M.g if from_up else M.ginv
        data = np.moveaxis(np.tensordot(metric.data, self.data, axes=([1], [i])), 0, i)
        flipped = Slot(s.name, s.space, "down" if from_up else "up", "value")
        return self._new(data, self.slots[:i] + [flipped] + self.slots[i + 1:], label=self.label)

    def lower(self, slot=None):
        """g on one up value slot: ↑ → ↓. Same physics; raise_ undoes it exactly.
        v.lower() for a vector; R.lower("ρ") for a named slot."""
        return self._raise_or_lower(slot, "lower")

    def raise_(self, slot=None):
        """g⁻¹ on one down value slot: ↓ → ↑. Same physics; lower undoes it exactly."""
        return self._raise_or_lower(slot, "raise_")

    # ---------- inverse ----------
    def inverse(self):
        """The inverse of a tensor that is a matrix at each point: exactly two value slots and
        nothing else (a linear map like A(ρ↑ σ↓), or a bilinear form like g(μ↓ ν↓)).
        Numbers: the matrix inverse. Slots: the two swap roles and both arrows flip,
        g(μ↓ ν↓) → g⁻¹(ν↑ μ↑), A(ρ↑ σ↓) → A⁻¹(σ↑ ρ↓).
        Refused for anything else: inverting is not linear, so it can't be done slot by slot
        over other (e.g. form) slots without depending on the coordinates. Also refused if the
        sizes differ or sympy shows the determinant is 0."""
        if len(self.slots) != 2 or any(s.is_form for s in self.slots):
            raise TypeError(f"inverse needs exactly two value slots and no others; got {self.declaration()}")
        so, si = self.slots
        if so.space.dim != si.space.dim:
            raise ValueError(f"inverse: slots {so.name} and {si.name} have different sizes "
                             f"({so.space.dim}, {si.space.dim}): not square")
        m = sp.Matrix(self.data.tolist())                     # rows: slot 0 (out), columns: slot 1 (in)
        if _is_zero(m.det()) is True:
            raise ValueError("inverse: the determinant is 0 (not invertible)")
        flip = lambda s: Slot(s.name, s.space, "down" if s.is_up else "up", "value")
        return self._new(np.array(m.inv().tolist(), dtype=object), [flip(si), flip(so)],
                         label=f"{wrap(self.label)}⁻¹" if self.label else None)

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
