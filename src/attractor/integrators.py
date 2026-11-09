"""Fixed-step explicit integrators.

`integrate` returns the times and the states as arrays; `Euler(h).run` and
`RK4(h).run` return a `Trajectory`.
"""
import numpy as np

from .trajectory import Trajectory


def euler_step(rhs, t, y, h):
    """One forward-Euler step, a method of order one.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(t, y)`` of the equation dy/dt = rhs(t, y).
    t : float
        Time at the start of the step.
    y : float or numpy.ndarray
        State at time ``t``.
    h : float
        Step size.

    Returns
    -------
    float or numpy.ndarray
        Approximate state at time ``t + h``.

    Examples
    --------
    >>> euler_step(lambda t, y: -y, 0.0, 1.0, 0.1)
    0.9
    """
    return y + h * rhs(t, y)


def rk4_step(rhs, t, y, h):
    """One classical Runge-Kutta step, a method of order four.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(t, y)`` of the equation dy/dt = rhs(t, y).
    t : float
        Time at the start of the step.
    y : float or numpy.ndarray
        State at time ``t``.
    h : float
        Step size.

    Returns
    -------
    float or numpy.ndarray
        Approximate state at time ``t + h``.

    Examples
    --------
    One step of dy/dt = -y from y = 1, against the exact value exp(-0.1):

    >>> import math
    >>> y = rk4_step(lambda t, y: -y, 0.0, 1.0, 0.1)
    >>> print(f"{y:.8f}  {math.exp(-0.1):.8f}")
    0.90483750  0.90483742
    """
    k1 = rhs(t, y)
    k2 = rhs(t + h / 2, y + h / 2 * k1)
    k3 = rhs(t + h / 2, y + h / 2 * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def check_step(h, t0=0.0, t1=None):
    """Return the number of steps of size h from t0 to t1.

    Raises
    ------
    ValueError
        If h is not a positive number, or if t1 - t0 is not a whole number of
        steps.

    Examples
    --------
    >>> check_step(0.1, 0.0, 1.0)
    10
    >>> check_step(0.3, 0.0, 1.0)
    Traceback (most recent call last):
    ...
    ValueError: (t1 - t0) / h = 3.33333 is not a whole number of steps
    """
    if not h > 0:  # also rejects nan
        raise ValueError(f"step size h must be positive, got {h}")
    if t1 is None:
        return None
    n = round((t1 - t0) / h)
    if not np.isclose(n * h, t1 - t0, rtol=1e-9, atol=0.0):
        raise ValueError(f"(t1 - t0) / h = {(t1 - t0) / h:g} is not a whole number of steps")
    return n


def integrate(step, rhs, y0, t0, t1, h):
    """Integrate dy/dt = rhs(t, y) from t0 to t1 with a fixed step.

    Parameters
    ----------
    step : callable
        One step of the method, such as `euler_step` or `rk4_step`.
    rhs : callable
        Right-hand side ``rhs(t, y)``.
    y0 : sequence of float
        State at time ``t0``.
    t0, t1 : float
        Start and end times.
    h : float
        Step size; ``(t1 - t0) / h`` must be a whole number.

    Returns
    -------
    t : numpy.ndarray
        The n + 1 times, of shape (n + 1,).
    y : numpy.ndarray
        The states, of shape (n + 1, len(y0)).

    Raises
    ------
    ValueError
        If h is not positive, or if t1 - t0 is not a whole number of steps.
    """
    n = check_step(h, t0, t1)
    y = np.array(y0, dtype=float)
    ts, ys = [t0], [y]
    for i in range(n):
        y = step(rhs, t0 + i * h, y, h)
        ts.append(t0 + (i + 1) * h)
        ys.append(y)
    return np.array(ts), np.array(ys)


class Euler:
    """Forward Euler with a fixed step h; runs any callable system.

    Raises
    ------
    ValueError
        If h is not positive.
    """

    def __init__(self, h):
        check_step(h)
        self.h = h

    def step(self, f, t, y):
        """One step of size h from (t, y)."""
        return euler_step(f, t, y, self.h)

    def run(self, system, y0, t0, t1):
        """Integrate from t0 to t1; returns a Trajectory named after the system's states."""
        t, y = integrate(euler_step, system, y0, t0, t1, self.h)
        return Trajectory(t, y, getattr(system, "names", ("x", "z")))


class RK4:
    """Classical Runge-Kutta of order four with a fixed step h; runs any callable system.

    Raises
    ------
    ValueError
        If h is not positive.
    """

    def __init__(self, h):
        check_step(h)
        self.h = h

    def step(self, f, t, y):
        """One step of size h from (t, y)."""
        return rk4_step(f, t, y, self.h)

    def run(self, system, y0, t0, t1):
        """Integrate from t0 to t1; returns a Trajectory named after the system's states."""
        t, y = integrate(rk4_step, system, y0, t0, t1, self.h)
        return Trajectory(t, y, getattr(system, "names", ("x", "z")))
