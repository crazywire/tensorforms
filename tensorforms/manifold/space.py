"""Spaces: where slots live. Each manifold owns its tangent space (M.tangent).

A space has a name, a dimension, and the list of names it hands out to new slots:
Greek for the tangent space, Latin for any other space.
"""

# θ, φ, π come last: they are often coordinate names or the number π.
TANGENT_NAMES = list("μνλκρσταβγδεζηιουξχψωθφπ")
OTHER_NAMES = list("abcdefghijklmnopqrstuvwxyz")


class Space:
    def __init__(self, name, dim, names=None):
        self.name, self.dim = name, dim
        self.names = list(names) if names else OTHER_NAMES     # names handed out automatically

    def fresh_name(self, used):
        """The first name from this space's list not in `used`."""
        used = set(used)
        for n in self.names:
            if n not in used:
                return n
        k = 1
        while f"{self.names[0]}{k}" in used:
            k += 1
        return f"{self.names[0]}{k}"

    def __repr__(self):
        return f"Space({self.name}, dim={self.dim})"
