"""A trajectory: times, states and state names, with methods that return modified copies."""
import copy
from dataclasses import dataclass, field

import numpy as np

from .cycles import cycle_period


@dataclass(eq=False)
class Trajectory:
    """Times t of shape (n,), states y of shape (n, k), one name per state, free-text labels."""
    t: np.ndarray
    y: np.ndarray
    names: tuple = ("x", "z")
    labels: list = field(default_factory=list)

    def __post_init__(self):
        self.t = np.array(self.t, dtype=float)  # owning arrays, whatever was passed in
        self.y = np.array(self.y, dtype=float)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        """The i-th point as (t_i, y_i); also what `for t, y in tr` iterates over."""
        return float(self.t[i]), self.y[i]

    def __repr__(self):
        return f"Trajectory({len(self)} points, t in [{self.t[0]:g}, {self.t[-1]:g}])"

    def __eq__(self, other):
        """Same shapes and values within the tolerance of np.allclose."""
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (self.t.shape == other.t.shape and self.y.shape == other.y.shape
                and np.allclose(self.t, other.t) and np.allclose(self.y, other.y))

    def shifted(self, dt):
        """The same states, dt later."""
        new = copy.copy(self)
        new.t = self.t + dt  # a new array; y, names and labels are shared
        return new

    def time_reversed(self):
        """The states in reverse order, with time running forward from 0."""
        new = copy.copy(self)
        new.t = self.t[-1] - self.t[::-1]
        new.y = self.y[::-1]  # a view of the original states
        return new

    def every(self, k):
        """Every k-th point."""
        new = copy.copy(self)
        new.t, new.y = self.t[::k], self.y[::k]  # two strided views
        return new

    def window(self, t0, t1):
        """The points with t0 <= t <= t1."""
        new = copy.copy(self)
        keep = (self.t >= t0) & (self.t <= t1)
        new.t, new.y = self.t[keep], self.y[keep]  # a boolean mask copies
        return new

    def copy(self):
        """An independent copy: new arrays, new list."""
        return copy.deepcopy(self)

    def period(self, column=0, min_gap=3.0):
        """Mean peak-to-peak spacing of one state, in the units of t."""
        return cycle_period(self.t, self.y[:, column], min_gap)[0]
