"""Indices: one per axis of a tensor.

An index has a label, a space (owned by the manifold), up/down and a type (value or form).
It is an up value index unless you say otherwise: mu is mu.up; mu.down is a down value index;
mu.form is a form index (always down, so no arrow).
"""

from dataclasses import dataclass

from .labels import index_label

ARROW = {"up": "↑", "down": "↓"}


@dataclass(frozen=True, repr=False)
class Index:
    """A label in a space, up or down, value or form (default: up value). Immutable: .up, .down,
    .form, renamed and with_type give new indices."""
    label: str
    space: object
    variance: str = "up"   # "up" or "down"
    type: str = "value"    # "value" or "form"

    def __post_init__(self):
        if self.variance not in ("up", "down"):
            raise ValueError("variance must be 'up' or 'down'")
        if self.type not in ("value", "form"):
            raise ValueError("type must be 'value' or 'form'")
        if self.type == "form" and self.variance != "down":
            raise ValueError("form indices are always down")
        object.__setattr__(self, "label", index_label(self.label))   # "mu" → "μ", once, at creation

    # ---------- the three kinds, same label and space ----------
    @property
    def up(self):
        """An up value index (↑): the default."""
        return Index(self.label, self.space, "up", "value")

    @property
    def down(self):
        """A down value index (↓)."""
        return Index(self.label, self.space, "down", "value")

    @property
    def form(self):
        """A form index: always down, so it carries no arrow."""
        return Index(self.label, self.space, "down", "form")

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
        return f"{self.label}: {self.variance} {self.type} index in {self.space.name}"

    def __repr__(self):
        return self.label if self.is_form else f"{self.label}{ARROW[self.variance]}"
