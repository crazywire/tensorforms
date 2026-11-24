"""Indices: one per axis of a tensor.

An index has a name and a space (owned by the manifold). Up, down or form is chosen when it is
used: mu.up, mu.down, mu.form. A tensor only holds chosen indices. Form indices are always down:
they take a step.
"""

from dataclasses import dataclass

from .labels import index_label

ARROW = {"up": "↑", "down": "↓"}


@dataclass(frozen=True, repr=False)
class Index:
    """A name in a space; once chosen, also up/down and value/form. Immutable: mu.up, renamed and
    with_role give new indices."""
    name: str
    space: object
    variance: str = None   # "up" or "down"; None until chosen
    role: str = None       # "value" or "form"; None until chosen

    def __post_init__(self):
        if (self.variance is None) != (self.role is None):
            raise ValueError("choose both up/down and value/form, or neither")
        if self.chosen:
            if self.variance not in ("up", "down"):
                raise ValueError("variance must be 'up' or 'down'")
            if self.role not in ("value", "form"):
                raise ValueError("role must be 'value' or 'form'")
            if self.role == "form" and self.variance != "down":
                raise ValueError("form indices are always down")
        object.__setattr__(self, "name", index_label(self.name))     # "mu" → "μ", once, at creation

    # ---------- choosing ----------
    @property
    def up(self):
        return Index(self.name, self.space, "up", "value")

    @property
    def down(self):
        return Index(self.name, self.space, "down", "value")

    @property
    def form(self):
        return Index(self.name, self.space, "down", "form")

    @property
    def chosen(self):
        """True once up, down or form has been chosen (a tensor only holds chosen indices)."""
        return self.variance is not None

    # ---------- reading ----------
    @property
    def type(self):
        """What decides pairing and adding: (space, up/down, role). The name is not part of it."""
        return (self.space, self.variance, self.role)

    @property
    def is_form(self):
        return self.role == "form"

    @property
    def is_up(self):
        return self.variance == "up"

    # ---------- new indices ----------
    def renamed(self, name):
        return Index(name, self.space, self.variance, self.role)

    def with_role(self, role):
        return Index(self.name, self.space, self.variance, role)

    def describe(self):
        if not self.chosen:
            return f"{self.name}: index in {self.space.name} (up, down or form not chosen)"
        return f"{self.name}: {self.variance} {self.role} index in {self.space.name}"

    def __repr__(self):
        if not self.chosen:
            return f"Index({self.name})"
        return self.name if self.is_form else f"{self.name}{ARROW[self.variance]}"
