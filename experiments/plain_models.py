"""The plain classes of the chapter's first section: attribute lookup, bound methods, a shared
class attribute.

Run from the project root: python -m experiments.plain_models
(importing the module prints nothing)
"""
import numpy as np


class DynamicalSystem:
    """A right-hand side with named states."""
    names = ("x", "z")

    def __call__(self, t, y):
        return self.rhs(t, y)


class LotkaVolterra(DynamicalSystem):
    """dx/dt = a x - b x z, dz/dt = c x z - d z."""
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def rhs(self, t, y):
        x, z = y
        return np.array([self.a * x - self.b * x * z, self.c * x * z - self.d * z])


class Trajectory:
    """A first attempt with a list in the class body."""
    labels = []

    def __init__(self, t, y):
        self.t, self.y = t, y

    def tag(self, label):
        self.labels.append(label)


if __name__ == "__main__":
    lv = LotkaVolterra(0.8, 0.08, 0.77, 0.77)
    y = np.array([1.5, 8.0])
    print(vars(lv))
    print(type(lv), lv.names)
    print([c.__name__ for c in LotkaVolterra.__mro__])
    print(LotkaVolterra.rhs)
    print(lv.rhs)
    print(np.array_equal(lv.rhs(0.0, y), LotkaVolterra.rhs(lv, 0.0, y)), lv.rhs.__func__ is LotkaVolterra.rhs)
    print(np.array_equal(lv(0.0, y), lv.rhs(0.0, y)))
    lv.names = ("hare", "lynx")
    print(lv.names, LotkaVolterra.names, "names" in vars(lv))

    a, b = Trajectory(0, 0), Trajectory(0, 0)
    a.tag("run 1")
    print(b.labels, "labels" in vars(a))
    a.labels = ["own"]
    print(a.labels, b.labels, "labels" in vars(a))
