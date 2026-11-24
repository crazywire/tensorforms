"""Slots: one per axis of a tensor.

A slot has a name, a space (owned by the manifold), up or down, and a role: "value" or "form".
Form slots are always down: they take a step.
"""

from dataclasses import dataclass

from .labels import slot_label

ARROW = {"up": "↑", "down": "↓"}


@dataclass(frozen=True, repr=False)
class Slot:
    """Immutable: a slot never changes after it is made; renamed / with_role give new slots."""
    name: str
    space: object
    variance: str          # "up" or "down"
    role: str              # "value" or "form"

    def __post_init__(self):
        if self.variance not in ("up", "down"):
            raise ValueError("variance must be 'up' or 'down'")
        if self.role not in ("value", "form"):
            raise ValueError("role must be 'value' or 'form'")
        if self.role == "form" and self.variance != "down":
            raise ValueError("form slots are always down")
        object.__setattr__(self, "name", slot_label(self.name))     # "mu" → "μ", once, at creation

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

    def renamed(self, name):
        return Slot(name, self.space, self.variance, self.role)

    def with_role(self, role):
        return Slot(self.name, self.space, self.variance, role)

    def describe(self):
        return f"{self.name}: {self.variance} {self.role} slot in {self.space.name}"

    def __repr__(self):
        return self.name if self.is_form else f"{self.name}{ARROW[self.variance]}"


@dataclass(frozen=True, repr=False)
class Index:
    """A slot name tied to a space. Choose up, down or form when you use it. Immutable."""
    name: str
    space: object

    def __post_init__(self):
        object.__setattr__(self, "name", slot_label(self.name))

    @property
    def up(self):
        return Slot(self.name, self.space, "up", "value")

    @property
    def down(self):
        return Slot(self.name, self.space, "down", "value")

    @property
    def form(self):
        return Slot(self.name, self.space, "down", "form")

    def __repr__(self):
        return f"Index({self.name})"
