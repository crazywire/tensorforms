"""Slots: one per axis of a tensor.

A slot has a name, a space, up or down, and a role: "value" or "form".
Form slots are always down: they take a step.
"""

ARROW = {"up": "↑", "down": "↓"}


class Space:
    """A space slots can live in, e.g. a manifold's tangent bundle."""

    def __init__(self, name, dim, manifold=None):
        self.name, self.dim, self.manifold = name, dim, manifold

    def __repr__(self):
        return f"Space({self.name}, dim={self.dim})"


class Slot:
    def __init__(self, name, space, variance="up", role="value"):
        if variance not in ("up", "down"):
            raise ValueError("variance must be 'up' or 'down'")
        if role not in ("value", "form"):
            raise ValueError("role must be 'value' or 'form'")
        if role == "form" and variance != "down":
            raise ValueError("form slots are always down")
        self.name, self.space, self.variance, self.role = name, space, variance, role

    @property
    def type(self):
        """What decides pairing and adding: (space, up/down, role). The name is not part of it."""
        return (self.space, self.variance, self.role)

    def renamed(self, name):
        return Slot(name, self.space, self.variance, self.role)

    def with_role(self, role):
        return Slot(self.name, self.space, self.variance, role)

    def describe(self):
        return f"{self.name}: {self.variance} {self.role} slot in {self.space.name}"

    def __repr__(self):
        return f"{self.name}{ARROW[self.variance]}" if self.role == "value" else f"{self.name}"


class Index:
    """A slot name tied to a space. Choose up, down or form when you use it."""

    def __init__(self, name, space):
        self.name, self.space = name, space

    @property
    def up(self):
        return Slot(self.name, self.space, "up", "value")

    @property
    def down(self):
        return Slot(self.name, self.space, "down", "value")

    @property
    def form(self):
        return Slot(self.name, self.space, "down", "form")

    @property
    def dim(self):
        return self.space.dim

    def __repr__(self):
        return f"Index({self.name})"
