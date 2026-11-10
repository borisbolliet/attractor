"""A trajectory: times, states and state names, with methods that return modified copies."""
import copy
from dataclasses import dataclass, field

import numpy as np

from .cycles import cycle_period


@dataclass(eq=False)
class Trajectory:
    """Store sample times, state values and their names.

    Parameters
    ----------
    t : array_like, shape (n,)
        Sample times.
    y : array_like, shape (n, k)
        State values, with one row per sample and one column per state.
    names : tuple of str, optional
        State names in column order (default ("x", "z")).
    labels : list, optional
        Free-text labels. A new empty list is created when omitted.

    Notes
    -----
    The constructor copies t and y into floating-point arrays. Supplied
    names and labels are retained without copying. Methods such as shifted
    and every return new Trajectory objects that share some data with the
    original; their docstrings specify which. Use copy for an independent copy.
    """
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
        """Return the i-th sample as a pair (time, state).

        The time is a float. The state is a row view of y, so modifying its
        elements also modifies the trajectory. Negative integer indices count
        from the end. Iteration yields these pairs in sample order.
        """
        return float(self.t[i]), self.y[i]

    def __repr__(self):
        return f"Trajectory({len(self)} points, t in [{self.t[0]:g}, {self.t[-1]:g}])"

    def __eq__(self, other):
        """Compare time and state arrays using numpy.allclose's default tolerances.

        Both array shapes must match. State names and labels are not compared.
        """
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (self.t.shape == other.t.shape and self.y.shape == other.y.shape
                and np.allclose(self.t, other.t) and np.allclose(self.y, other.y))

    def shifted(self, dt):
        """Return a trajectory whose sample times are shifted by dt.

        Parameters
        ----------
        dt : float
            Time shift, in the units of t.

        Returns
        -------
        Trajectory
            New time array t + dt. The state array, names and labels are shared
            with the original trajectory.
        """
        new = copy.copy(self)
        new.t = self.t + dt  # a new array; y, names and labels are shared
        return new

    def time_reversed(self):
        """Return states in reverse order, with time starting at zero.

        Returns
        -------
        Trajectory
            Times t[-1] - t[::-1] in a new array and states y[::-1] as a view
            of the original state array. Names and labels are shared. For
            increasing input times, the returned times are also increasing.
        """
        new = copy.copy(self)
        new.t = self.t[-1] - self.t[::-1]
        new.y = self.y[::-1]  # a view of the original states
        return new

    def every(self, k):
        """Return a trajectory selected with the slice [::k].

        Parameters
        ----------
        k : int
            Non-zero slice stride. For positive k, retain samples 0, k, 2*k,
            and so on. A negative stride traverses the samples in reverse.

        Returns
        -------
        Trajectory
            Time and state arrays that are views of the original arrays.
            Names and labels are also shared.

        Raises
        ------
        ValueError
            If k is zero.
        """
        new = copy.copy(self)
        new.t, new.y = self.t[::k], self.y[::k]  # two strided views
        return new

    def window(self, t0, t1):
        """Return the samples in the inclusive time interval [t0, t1].

        Parameters
        ----------
        t0, t1 : float
            Lower and upper time bounds, in the units of t.

        Returns
        -------
        Trajectory
            Copies of the selected time and state values, with names and
            labels shared with the original. The arrays can be empty.
        """
        new = copy.copy(self)
        keep = (self.t >= t0) & (self.t <= t1)
        new.t, new.y = self.t[keep], self.y[keep]  # a boolean mask copies
        return new

    def copy(self):
        """Return a deep copy with independent arrays and labels."""
        return copy.deepcopy(self)

    def period(self, column=0, min_gap=3.0):
        """Estimate the period of one state from its peak spacing.

        Parameters
        ----------
        column : int, optional
            State column to analyse (default 0).
        min_gap : float, optional
            Minimum separation of retained peaks, in the units of t (default 3.0).

        Returns
        -------
        float
            Mean spacing between retained peaks, in the units of t, or numpy.nan
            if fewer than three peaks are found.

        See Also
        --------
        attractor.cycles.cycle_period : Peak-based period estimate.
        """
        return cycle_period(self.t, self.y[:, column], min_gap)[0]
