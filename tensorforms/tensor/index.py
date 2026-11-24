"""Indices: one per axis of a tensor.

An index has a label and a space (owned by the manifold). Up, down or form is chosen once, when it
is used: mu.up, mu.down (value indices), mu.form (a form index: always down, so no arrow).
A tensor only holds chosen indices.
"""

from dataclasses import dataclass

from .labels import index_label

ARROW = {"up": "↑", "down": "↓"}


@dataclass(frozen=True, repr=False)
class Index:
    """A label in a space; once chosen, also up/down and value/form. Immutable: mu.up, renamed
    and with_type give new indices."""
    label: str
    space: object
    variance: str = None   # "up" or "down"; None until chosen
    type: str = None       # "value" or "form"; None until chosen

    def __post_init__(self):
        if (self.variance is None) != (self.type is None):
            raise ValueError("choose both up/down and value/form, or neither")
        if self.chosen:
            if self.variance not in ("up", "down"):
                raise ValueError("variance must be 'up' or 'down'")
            if self.type not in ("value", "form"):
                raise ValueError("type must be 'value' or 'form'")
            if self.type == "form" and self.variance != "down":
                raise ValueError("form indices are always down")
        object.__setattr__(self, "label", index_label(self.label))   # "mu" → "μ", once, at creation

    # ---------- choosing (once) ----------
    def _choose(self, variance, type):
        if self.chosen:
            raise TypeError(f"{self!r} is already chosen; start from the label's index")
        return Index(self.label, self.space, variance, type)

    @property
    def up(self):
        """An up value index (↑)."""
        return self._choose("up", "value")

    @property
    def down(self):
        """A down value index (↓)."""
        return self._choose("down", "value")

    @property
    def form(self):
        """A form index: always down, so it carries no arrow."""
        return self._choose("down", "form")

    @property
    def chosen(self):
        """True once up, down or form has been chosen (a tensor only holds chosen indices)."""
        return self.variance is not None

    # ---------- reading ----------
    @property
    def is_form(self):
        return self.type == "form"

    @property
    def is_up(self):
        return self.variance == "up"

    @property
    def is_down(self):
        return self.variance == "down"

    # ---------- the two rules ----------
    def matches(self, other):
        """Same kind of index: same space, up/down and type; the label doesn't matter.
        Indices that match can be added or averaged (Alt/Sym)."""
        return (self.space, self.variance, self.type) == (other.space, other.variance, other.type)

    def pairs_with(self, other):
        """Can be contracted: same space, one up and the other down (a form index is down)."""
        return self.space is other.space and {self.variance, other.variance} == {"up", "down"}

    # ---------- new indices ----------
    def renamed(self, label):
        return Index(label, self.space, self.variance, self.type)

    def with_type(self, type):
        return Index(self.label, self.space, self.variance, type)

    def describe(self):
        if not self.chosen:
            return f"{self.label}: index in {self.space.name} (up, down or form not chosen)"
        return f"{self.label}: {self.variance} {self.type} index in {self.space.name}"

    def __repr__(self):
        if not self.chosen:
            return f"Index({self.label})"
        return self.label if self.is_form else f"{self.label}{ARROW[self.variance]}"
