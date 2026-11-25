"""Tensors: data plus an ordered list of indices.

Everything here acts on one tensor: describing it, adding and scaling, moving indices,
changing value / form type, Alt/Sym, trace, and raising/lowering with the manifold's metric.
Products of two tensors are in products.py, derivatives in derivatives.py (both mixed in).
"""
import itertools
import math

import numpy as np
import weakref

import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError, NonSquareMatrixError
from sympy.combinatorics import Permutation

from ..symbolic.simplify import is_zero as _is_zero, simplify_expr
from .labels import index_label
from .index import Index
from .products import Products
from .derivatives import Derivatives
from ..errors import (IndexMismatchError, InvalidCoordinateError, MissingMetricError,
                      NonInvertibleError)


def _as_expr(x):
    """x as an exact sympy expression (2 → Integer(2), never a float), or None if it isn't one."""
    try:
        x = sp.sympify(x, strict=True)
    except sp.SympifyError:
        return None
    return x if isinstance(x, sp.Expr) else None


def _as_scalar(c):
    """c as an exact scalar: a number, an expression or a scalar tensor."""
    not_scalar = ("scale: only by a number, an expression or a scalar tensor; "
                  "for products of tensors use @, contract, otimes or wedge")
    if isinstance(c, Tensor):
        if c.indices:
            raise TypeError(not_scalar)
        c = c.data.item()
    c = _as_expr(c)
    if c is None:
        raise TypeError(not_scalar)
    return c


class Tensor(Derivatives, Products):
    """Data plus an ordered tuple of indices. The derivatives (X.d(), X.nabla(), X.lie(v), …) come
    from Derivatives in derivatives.py; the products (A @ B, A.otimes(B), …) from Products
    in products.py."""
    # sympy defers mixed operations to objects with a higher _op_priority (its expressions have 10),
    # so θ * X and θ + s stay tensor arithmetic instead of sympy taking over.
    _op_priority = 20.0

    def __init__(self, manifold, data, indices, label=None, *, copy=True, _refs=None):
        """copy=True (default): the tensor takes its own copy of an array it is given, so it
        never shares numbers behind our back. copy=False: use the array as is; only for an
        array just made that nothing else holds (e.g. a freshly computed result)."""
        self.manifold = manifold
        if _refs is None:                                    # a new tensor: its own numbers
            self._data = np.array(data, dtype=object) if copy else np.asarray(data, dtype=object)
            self._refs = weakref.WeakSet()
        else:                                                # made by _new: shares the numbers
            self._data = data
            self._refs = _refs
        self._refs.add(self)                                 # every tensor using these numbers
        self.indices = tuple(indices)
        self.label = label
        if self._data.ndim != len(self.indices):
            raise ValueError(f"Tensor: data has {self._data.ndim} axes but {len(self.indices)} indices were declared")
        for n, s in zip(self._data.shape, self.indices):
            if n != s.space.dim:
                raise ValueError(f"Tensor: index {s.label} lives in {s.space.name} (size {s.space.dim}) but its axis has size {n}")
        labels = self._index_labels()
        if len(set(labels)) != len(labels):
            raise ValueError(f"Tensor: index labels must be unique; got {labels}")

    def _new(self, data=None, indices=None, *, label):
        """A tensor on the same manifold; data and indices default to this tensor's.
        The label must be given (label=self.label to keep it, label=None for none).
        If the data is ours or a view of it, the new tensor shares it (copy on write)."""
        if data is None:
            data = self._data
        shared = np.may_share_memory(data, self._data)       # "maybe" only costs a copy later
        return Tensor(self.manifold, data,
                      self.indices if indices is None else indices,
                      label, _refs=self._refs if shared else None)

    def __copy__(self):
        return self._new(label=self.label)

    def __deepcopy__(self, memo):
        return Tensor(self.manifold, self._data, self.indices, self.label)

    # ---------- description ----------
    @property
    def value(self):
        """The components in sympy form: a scalar's expression, otherwise a sympy Array
        (one axis per index). This is also what sympy sees when it is handed a tensor."""
        return self._data.item() if not self.indices else sp.Array(self._data.tolist())

    # private helpers: each rule written once (users read X.indices)
    def _index_labels(self):
        return [s.label for s in self.indices]

    def _form_positions(self):
        return [k for k, s in enumerate(self.indices) if s.is_form]

    def _value_positions(self):
        return [k for k, s in enumerate(self.indices) if not s.is_form]

    def position(self, x):
        """Where an index is: given by label ("μ", "mu", or an Index) or by number (0, 1, …)."""
        if isinstance(x, int) and not isinstance(x, bool):
            if not 0 <= x < len(self.indices):
                raise IndexError(f"index: no number {x}; this tensor has {len(self.indices)} indices")
            return x
        n, labels = index_label(x), self._index_labels()
        if n not in labels:
            raise KeyError(f"index: no label {n}; labels are {labels}")
        return labels.index(n)

    def index(self, x):
        """One index, by label or number (X.indices has them all)."""
        return self.indices[self.position(x)]

    def declaration(self):
        """The indices in order: value indices carry ↑ or ↓, form indices carry no arrow."""
        return "(" + " ".join(repr(s) for s in self.indices) + ")"

    def describe(self):
        head = f"{self.label or 'Tensor'} on {self.manifold.name}"
        return "\n".join([head] + ["  " + s.describe() for s in self.indices])

    def __repr__(self):
        return f"{self.label or 'Tensor'}{self.declaration()}"

    @property
    def data(self):
        """The components: a read-only view (no copy). Change numbers with X[...] = v."""
        view = self._data.view()
        view.flags.writeable = False
        return view

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        """Change components: X[0, 1] = 5. Copy on write: if other tensors use these numbers,
        this tensor takes its own copy first, so no other tensor changes."""
        if len(self._refs) > 1:
            self._refs.discard(self)
            self._data = self._data.copy()
            self._refs = weakref.WeakSet([self])
        self._data[key] = value

    # ---------- arithmetic (+ and − need matching indices) ----------
    def matches(self, other):
        """Can be added: the same number of indices, each matching the one at the same position
        (same space, up/down and type; labels don't matter). g(μ↓ ν↓) matches h(α↓ β↓)."""
        return (isinstance(other, Tensor) and len(self.indices) == len(other.indices)
                and all(a.matches(b) for a, b in zip(self.indices, other.indices)))


    def _operand(self, other):
        """The other side of + or −, as a tensor. A number means a scalar (no indices), so the
        index check decides; something sympy shows is 0 is the zero tensor of any type, returned
        as None (nothing to add)."""
        if isinstance(other, Tensor):
            return other
        c = _as_expr(other)
        if c is None:
            raise TypeError(f"+/−: needs a tensor or a number, not {type(other).__name__}")
        if _is_zero(c) is True:
            return None
        return Tensor(self.manifold, c, [], label=None)

    def __add__(self, other):
        other = self._operand(other)
        if other is None:                                     # X + 0 is X
            return self._new(label=self.label)
        if not self.matches(other):
            if other.manifold is not self.manifold:
                raise IndexMismatchError(f"+: the tensors are on different manifolds "
                                         f"({self.manifold.name} and {other.manifold.name})")
            raise IndexMismatchError(f"+: indices don't match ({self.declaration()} vs {other.declaration()})")
        return self._new(self._data + other._data, label=None)

    __radd__ = __add__                                        # 0 + X, 1 + s

    def __sub__(self, other):
        other = self._operand(other)
        if other is None:                                     # X − 0 is X
            return self._new(label=self.label)
        if not self.matches(other):
            if other.manifold is not self.manifold:
                raise IndexMismatchError(f"−: the tensors are on different manifolds "
                                         f"({self.manifold.name} and {other.manifold.name})")
            raise IndexMismatchError(f"−: indices don't match ({self.declaration()} vs {other.declaration()})")
        return self._new(self._data - other._data, label=None)

    def __rsub__(self, other):                                # 0 − X is −X; 1 − s
        other = self._operand(other)
        return -self if other is None else other - self

    def __neg__(self):
        return self._new(-self._data, label=None)

    def __mul__(self, c):
        """Multiply by a number, a sympy expression, or a scalar tensor (no indices)."""
        if not self.indices and isinstance(c, Tensor) and c.indices:   # scalar × tensor
            return c * self
        c = _as_scalar(c)
        return self._new(self._data * c, label=None)

    __rmul__ = __mul__

    def __truediv__(self, c):
        """Divide by a number, a sympy expression, or a scalar tensor: exact (X / 2 is ½X, never 0.5X).
        Refused if sympy shows the divisor is 0."""
        c = _as_scalar(c)
        if _is_zero(c) is True:
            raise ZeroDivisionError("/: division by zero")
        return self._new(self._data / c, label=None)

    def __rtruediv__(self, c):
        """A number divided by a scalar is a scalar (2 / s); dividing by a tensor with indices is refused."""
        if self.indices:
            raise TypeError("/: can't divide by a tensor with indices; for an inverse use X.inverse()")
        n = _as_expr(c)
        if n is None:
            raise TypeError(f"/: only a number or an expression can be divided by a scalar, not {type(c).__name__}")
        return Tensor(self.manifold, n, [], label=None) / self

    def _sympy_(self):
        """What sympy sees when it is handed a tensor: its value. So sp.simplify(s), sp.diff(s, θ) work."""
        return self.value

    def simplify(self):
        """Every component simplified (sympy: fu, then simplify; exact rewrites only)."""
        return self._new(np.vectorize(simplify_expr, otypes=[object])(self._data), label=self.label)

    def is_zero(self):
        """True if every component is shown to be 0; False if one is shown not to be;
        None if that can't be decided exactly."""
        answers = [_is_zero(e) for e in self._data.flat]
        if all(a is True for a in answers):
            return True
        return False if any(a is False for a in answers) else None

    def equals(self, other):
        """Same three answers as is_zero, for X − Y."""
        return (self - other).is_zero()

    # ---------- substituting and evaluating ----------
    def _coordinate(self, x):
        """A coordinate given as its symbol or its name ("theta" or θ), or None if x isn't one."""
        for c in self.manifold.coords:
            if x == c or (isinstance(x, str) and index_label(x) in (c.name, index_label(c.name))):
                return c
        return None

    def subs(self, *args):
        """Substitute for parameters in every component, as sympy's subs: X.subs(r, 1) or
        X.subs({m: 1, q: 0}). The result is still a field. Coordinates are refused: replacing θ by a
        number gives values at one point (use at), and by an expression is not a change of
        coordinates (the components would also have to mix)."""
        pairs = args[0].items() if len(args) == 1 and isinstance(args[0], dict) else [args]
        coords = set(self.manifold.coords)
        for old, _ in pairs:
            if self._coordinate(old) or (not isinstance(old, str) and coords & sp.sympify(old).free_symbols):
                raise InvalidCoordinateError(f"subs: {old} involves a coordinate; subs is for parameters "
                                             f"(for values at a point use X.at({{...}}))")
        f = lambda e: sp.sympify(e).subs(*args)
        return self._new(np.vectorize(f, otypes=[object])(self._data), label=None)

    def at(self, point):
        """The components at one point: X.at({θ: π/2, φ: 0}), every coordinate given (by symbol or
        name). Plain values, not a tensor: a scalar's expression, otherwise a sympy Array (one axis
        per index), so no derivative can be taken of it by mistake."""
        values = {}
        for k, v in point.items():
            c = self._coordinate(k)
            if c is None:
                raise InvalidCoordinateError(f"at: {k} is not a coordinate of {self.manifold.name} {tuple(self.manifold.coords)}")
            values[c] = v
        missing = [c for c in self.manifold.coords if c not in values]
        if missing:
            raise InvalidCoordinateError(f"at: every coordinate is needed; missing {missing}")
        f = lambda e: sp.sympify(e).subs(values)
        data = np.vectorize(f, otypes=[object])(self._data)
        return data.item() if not self.indices else sp.Array(data.tolist())

    # ---------- moving indices ----------
    def _transposed(self, order):
        return self._new(np.transpose(self._data, order), [self.indices[i] for i in order], label=self.label)

    def reorder(self, *indices):
        """Put every index in the given order (labels or positions)."""
        order = [self.position(s) for s in indices]
        if sorted(order) != list(range(len(self.indices))):
            raise ValueError("reorder: needs every index exactly once")
        return self._transposed(order)

    def swap(self, a, b):
        """Exchange two indices, given by label or position."""
        i, j = self.position(a), self.position(b)
        order = list(range(len(self.indices)))
        order[i], order[j] = order[j], order[i]
        return self._transposed(order)

    # T, S, U, V: the notation's swaps of neighbouring indices. Each transposes the matrix in two
    # neighbouring indices at every value of the others; they chain: R.U.S is R^US.
    @property
    def T(self):
        """Transpose the matrix in indices 0 and 1, at every value of the others: swap(0, 1)."""
        return self.swap(0, 1)

    @property
    def S(self):
        """Transpose the matrix in indices 1 and 2, at every value of the others: swap(1, 2)."""
        return self.swap(1, 2)

    @property
    def U(self):
        """Transpose the matrix in indices 2 and 3, at every value of the others: swap(2, 3)."""
        return self.swap(2, 3)

    @property
    def V(self):
        """Transpose the matrix in indices 3 and 4, at every value of the others: swap(3, 4)."""
        return self.swap(3, 4)

    def relabel_indices(self, mapping):
        """New index labels: {index (label or number): new label}, for the indices you name; the
        others, the numbers (shared) and the tensor's own label stay. R.relabel_indices({"μ": "α"})."""
        mapping = {self.position(k): index_label(v) for k, v in mapping.items()}
        return self._new(indices=[s.relabel(mapping[k]) if k in mapping else s for k, s in enumerate(self.indices)],
                         label=self.label)

    def _change_type(self, which, change, eligible, need):
        """The given indices (default: every eligible one) are changed by `change`; same numbers."""
        if which:
            targets = {self.position(x) for x in which}
            bad = [self.indices[k] for k in sorted(targets) if not eligible(self.indices[k])]
            if bad:
                raise IndexMismatchError(f"{need}; got {bad}")
        else:
            targets = {k for k, s in enumerate(self.indices) if eligible(s)}
        return self._new(indices=[change(s) if k in targets else s for k, s in enumerate(self.indices)],
                         label=self.label)

    def as_form(self, *indices):
        """Down value indices → form indices (default: all). Covector (μ↓) → 1-form (μ). Same numbers."""
        return self._change_type(indices, lambda s: s.form, lambda s: s.is_down and not s.is_form, "as_form: needs down value indices")

    def as_value(self, *indices):
        """Form indices → down value indices (default: all). 1-form (μ) → covector (μ↓). Same numbers."""
        return self._change_type(indices, lambda s: s.down, lambda s: s.is_form, "as_value: needs form indices")

    # ---------- trace ----------
    def tr(self, a, b):
        """Pair two indices inside this tensor (one up, one down) and sum over them."""
        i, j = self.position(a), self.position(b)
        si, sj = self.indices[i], self.indices[j]
        if not si.pairs_with(sj):
            raise IndexMismatchError(f"tr: {si.label} and {sj.label} don't pair (need the same space, one up and one down)")
        data = np.asarray(np.trace(self._data, axis1=i, axis2=j), dtype=object)
        label = None
        return Tensor(self.manifold, data, [s for k, s in enumerate(self.indices) if k not in (i, j)], label,
                      copy=False)

    # ---------- raising and lowering (needs the manifold's metric) ----------
    def _flip(self, index, to, metric):
        """Make one value index point `to` (up or down), multiplying by `metric` along it."""
        name = "lower" if to == "down" else "raise_"
        if index is None:                                   # the only index that can be flipped
            options = [k for k, s in enumerate(self.indices) if not s.is_form and s.variance != to]
            if not options:
                frm = "up" if to == "down" else "down"
                raise IndexMismatchError(f"{name}: no {frm} value index in {self.declaration()}")
            if len(options) > 1:
                raise IndexMismatchError(f"{name}: say which index; candidates "
                                         f"{[self._index_labels()[k] for k in options]}")
            index = options[0]
        i = self.position(index)
        s = self.indices[i]
        if s.is_form or s.variance == to:
            raise IndexMismatchError(f"{name}: can't make {s.label} {to}; it is a form index or already {to}")
        data = np.moveaxis(np.tensordot(metric._data, self._data, axes=([1], [i])), 0, i)
        flipped = Index(s.label, s.space, to, "value")
        return self._new(data, (*self.indices[:i], flipped, *self.indices[i + 1:]), label=self.label)

    def lower(self, index=None):
        """g on one up value index: ↑ → ↓. Same physics; raise_ undoes it exactly.
        v.lower() for a vector; R.lower("ρ") for a named index."""
        return self._flip(index, "down", self.manifold.g)

    def raise_(self, index=None):
        """g⁻¹ on one down value index: ↓ → ↑. Same physics; lower undoes it exactly."""
        return self._flip(index, "up", self.manifold.ginv)

    # ---------- inverse ----------
    def inverse(self):
        """The inverse of a tensor that is a matrix at each point: exactly two value indices and
        nothing else (a linear map like A(ρ↑ σ↓), or a bilinear form like g(μ↓ ν↓)).
        Numbers: the matrix inverse. Indices: the two swap places and both arrows flip,
        g(μ↓ ν↓) → g⁻¹(ν↑ μ↑), A(ρ↑ σ↓) → A⁻¹(σ↑ ρ↓).
        Refused for anything else: inverting is not linear, so it can't be done index by index
        over other (e.g. form) indices without depending on the coordinates. Also refused if it
        isn't square or the determinant is 0 (sympy checks; the messages are ours)."""
        if len(self.indices) != 2 or any(s.is_form for s in self.indices):
            raise IndexMismatchError(f"inverse: needs exactly two value indices and no others; got {self.declaration()}")
        so, si = self.indices
        m = sp.Matrix(self._data.tolist())                     # rows: index 0 (out), columns: index 1 (in)
        try:                                                  # sympy does the checks, with our zero test
            inv = m.inv(iszerofunc=_is_zero)
        except NonSquareMatrixError:
            raise NonInvertibleError(f"inverse: not square ({so.label} has size {so.space.dim}, "
                                     f"{si.label} has size {si.space.dim})") from None
        except NonInvertibleMatrixError:
            raise NonInvertibleError("inverse: not invertible (the determinant is 0)") from None
        flip = lambda s: Index(s.label, s.space, "down" if s.is_up else "up", "value")
        return self._new(np.array(inv.tolist(), dtype=object), [flip(si), flip(so)],
                         label=None)

    # ---------- Hodge star (needs the manifold's metric) ----------
    def star(self):
        """⋆X: the k form indices become n − k, with the same amount (direction ↔ plane ↔ volume …).
        Raise the form indices, fill them into the volume form ε in order, divide by k!.
        Value indices are carried along. Uses the metric and the coordinate order (orientation)."""
        M = self.manifold
        if not M.has_metric():
            raise MissingMetricError("star: needs a metric (for the volume form and raising)")
        forms = self._form_positions()
        k = len(forms)
        if k > M.dim:
            raise ValueError(f"star: {k} form indices but the space has only {M.dim} dimensions")
        X = self.as_value(*forms) if k else self
        for p in forms:
            X = X.raise_(p)
        T = X.otimes(M.volume_form())                # X's value indices, then ε's n form indices
        nv = len(X.indices)
        for j, p in enumerate(forms):                 # the j-th raised index fills ε's next index
            T = T.tr(p - j, nv - j)
        T = T / math.factorial(k)
        return T._new(label=None)

    # ---------- symmetric and antisymmetric parts ----------
    def _average(self, indices, signed):
        pos = [self.position(s) for s in indices] if indices else self._form_positions()
        if len(set(pos)) != len(pos):
            raise ValueError(f"alt/sym: an index is repeated in {list(indices)}")
        if not all(self.indices[p].matches(self.indices[pos[0]]) for p in pos):
            raise IndexMismatchError("alt/sym: the indices must match (same space, up/down and type)")
        k = len(pos)
        out = 0
        for perm in itertools.permutations(range(k)):
            axes = list(range(len(self.indices)))
            for src, dst in zip(perm, range(k)):
                axes[pos[dst]] = pos[src]
            term = np.transpose(self._data, axes)
            out = out + (Permutation(perm).signature() if signed else 1) * term
        label = None
        return self._new(out * sp.Rational(1, math.factorial(k)), label=label)

    def alt(self, *indices):
        """Antisymmetric part over the given indices, by label or position (default: all form indices)."""
        return self._average(indices, signed=True)

    def sym(self, *indices):
        """Symmetric part over the given indices, by label or position (default: all form indices)."""
        return self._average(indices, signed=False)
