"""Spaces: where indices live. Each manifold owns its tangent space (M.tangent).

A space has a label (just a name for printing) and a dimension. Index labels are picked by
the tensor, not the space: see labels._new_label.
"""


class Space:
    def __init__(self, label, dim):
        self.label, self.dim = label, dim

    def __repr__(self):
        return f"Space({self.label}, dim={self.dim})"
