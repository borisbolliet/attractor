import copy
import numpy as np
class Field:
    def __init__(self, x, phi):
        self.x, self.phi = x, phi
bodies = {
    "(a) self.phi": lambda s: s.phi,
    "(b) self.phi[::-1]": lambda s: s.phi[::-1],
    "(c) self.phi[self.x > 1]": lambda s: s.phi[s.x > 1],
    "(d) self.phi.copy()[::-1]": lambda s: s.phi.copy()[::-1],
}
for label, body in bodies.items():
    F = Field(np.arange(5.0), np.arange(5.0) * 10)
    G = copy.copy(F)
    G.phi = body(F)
    G.phi[0] = -1.0
    print(f"{label:26s} shares {np.shares_memory(G.phi, F.phi)!s:6s} base is {G.phi.base is F.phi!s:6s} F.phi changed {not np.array_equal(F.phi, np.arange(5.0) * 10)}")
