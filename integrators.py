"""Fixed-step explicit integrators; `integrate` returns the times and the states as arrays,
`Euler(h).run` and `RK4(h).run` return a `Trajectory`."""
import numpy as np

from trajectory import Trajectory


def euler_step(rhs, t, y, h):
    """One forward-Euler step."""
    return y + h * rhs(t, y)


def rk4_step(rhs, t, y, h):
    """One classical Runge-Kutta step of order four."""
    k1 = rhs(t, y)
    k2 = rhs(t + h / 2, y + h / 2 * k1)
    k3 = rhs(t + h / 2, y + h / 2 * k2)
    k4 = rhs(t + h, y + h * k3)
    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def integrate(step, rhs, y0, t0, t1, h):
    """Integrate from t0 to t1 with a fixed step; returns (t, y) with y of shape (n+1, len(y0))."""
    n = round((t1 - t0) / h)
    y = np.array(y0, dtype=float)
    ts, ys = [t0], [y]
    for i in range(n):
        y = step(rhs, t0 + i * h, y, h)
        ts.append(t0 + (i + 1) * h)
        ys.append(y)
    return np.array(ts), np.array(ys)


class Euler:
    """Forward Euler with a fixed step h; runs any callable system."""

    def __init__(self, h):
        self.h = h

    def step(self, f, t, y):
        return euler_step(f, t, y, self.h)

    def run(self, system, y0, t0, t1):
        """Integrate from t0 to t1; returns a Trajectory named after the system's states."""
        t, y = integrate(euler_step, system, y0, t0, t1, self.h)
        return Trajectory(t, y, getattr(system, "names", ("x", "z")))


class RK4:
    """Classical Runge-Kutta of order four with a fixed step h; runs any callable system."""

    def __init__(self, h):
        self.h = h

    def step(self, f, t, y):
        return rk4_step(f, t, y, self.h)

    def run(self, system, y0, t0, t1):
        """Integrate from t0 to t1; returns a Trajectory named after the system's states."""
        t, y = integrate(rk4_step, system, y0, t0, t1, self.h)
        return Trajectory(t, y, getattr(system, "names", ("x", "z")))
