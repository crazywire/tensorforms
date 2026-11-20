"""Slots: one per axis of a tensor.

A slot has a name, a space (owned by the manifold), up or down, and a role: "value" or "form".
Form slots are always down: they take a step.
"""

ARROW = {"up": "↑", "down": "↓"}

# Spelled-out Greek is accepted wherever a name is: "mu" means μ.
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ",
         "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "lamda": "λ",
         "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ",
         "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω"}


def clean_name(name):
    """A slot name as stored: spelled-out Greek becomes the letter."""
    name = str(name)
    return GREEK.get(name, name)


class Slot:
    def __init__(self, name, space, variance, role):
        if variance not in ("up", "down"):
            raise ValueError("variance must be 'up' or 'down'")
        if role not in ("value", "form"):
            raise ValueError("role must be 'value' or 'form'")
        if role == "form" and variance != "down":
            raise ValueError("form slots are always down")
        self.name, self.space, self.variance, self.role = clean_name(name), space, variance, role

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


class Index:
    """A slot name tied to a space. Choose up, down or form when you use it."""

    def __init__(self, name, space):
        self.name, self.space = clean_name(name), space

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
