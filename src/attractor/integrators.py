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
    """Check a positive step size and, if given, its fit to an interval.

    Parameters
    ----------
    h : float
        Positive step size.
    t0 : float, optional
        Start time (default 0.0), used only when t1 is given.
    t1 : float or None, optional
        End time. If None, only check h and return None.

    Returns
    -------
    int or None
        Number of steps when t1 is supplied, otherwise None. The interval
        check allows a relative tolerance of 1e-9 in its length.

    Raises
    ------
    ValueError
        If h is non-positive or NaN, or the interval cannot be divided into
        whole steps within the tolerance.

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
    """Integrate dy/dt = rhs(t, y) over a forward interval with a fixed step.

    Parameters
    ----------
    step : callable
        Function step(rhs, t, y, h) returning the next state, such as
        euler_step or rk4_step. It must not modify y in place.
    rhs : callable
        Function rhs(t, y) returning the derivative with the same shape as y.
    y0 : array_like, shape (k,)
        Initial state at t0, copied into a floating-point array.
    t0, t1 : float
        Start and end times, with t1 >= t0.
    h : float
        Positive step size. The interval must contain a whole number of
        steps, allowing the tolerance used by check_step.

    Returns
    -------
    t : numpy.ndarray, shape (n + 1,)
        Sample times, including the initial time. Here n is the number
        of steps, round((t1 - t0) / h).
    y : numpy.ndarray, shape (n + 1, k)
        State at each sample time. The first row is the initial state.

    Raises
    ------
    ValueError
        If h is non-positive or NaN, or the interval cannot be divided into
        whole steps within the tolerance.
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
    """Forward Euler integration with a fixed step size.

    Parameters
    ----------
    h : float
        Positive step size, in the same time units as the system.

    Attributes
    ----------
    h : float
        Step size used by step and run.

    Raises
    ------
    ValueError
        If h is non-positive or NaN.
    """

    def __init__(self, h):
        check_step(h)
        self.h = h

    def step(self, f, t, y):
        """Advance the state by one step of size self.h.

        Parameters
        ----------
        f : callable
            Function f(t, y) returning the time derivative of y.
        t : float
            Time at the start of the step.
        y : float or numpy.ndarray
            State at time t.

        Returns
        -------
        float or numpy.ndarray
            Approximate state at t + self.h.
        """
        return euler_step(f, t, y, self.h)

    def run(self, system, y0, t0, t1):
        """Integrate a system over a forward interval and return its trajectory.

        Parameters
        ----------
        system : callable
            Function or object called as system(t, y), returning the derivative
            with the same shape as y. If it has a names attribute, that attribute
            supplies the state names. Otherwise the names default to ("x", "z").
        y0 : array_like, shape (k,)
            Initial state at t0.
        t0, t1 : float
            Start and end times, with t1 >= t0. The interval must contain a
            whole number of steps of size self.h, within check_step's tolerance.

        Returns
        -------
        Trajectory
            Sample times, state values and state names, including the initial
            state and one row for each integration step.

        Raises
        ------
        ValueError
            If the interval cannot be divided into whole steps within the
            tolerance, or self.h has been changed to a non-positive value or NaN.
        """
        t, y = integrate(euler_step, system, y0, t0, t1, self.h)
        return Trajectory(t, y, getattr(system, "names", ("x", "z")))


class RK4:
    """Classical fourth-order Runge-Kutta integration with a fixed step size.

    Parameters
    ----------
    h : float
        Positive step size, in the same time units as the system.

    Attributes
    ----------
    h : float
        Step size used by step and run.

    Raises
    ------
    ValueError
        If h is non-positive or NaN.
    """

    def __init__(self, h):
        check_step(h)
        self.h = h

    def step(self, f, t, y):
        """Advance the state by one step of size self.h.

        Parameters
        ----------
        f : callable
            Function f(t, y) returning the time derivative of y.
        t : float
            Time at the start of the step.
        y : float or numpy.ndarray
            State at time t.

        Returns
        -------
        float or numpy.ndarray
            Approximate state at t + self.h.
        """
        return rk4_step(f, t, y, self.h)

    def run(self, system, y0, t0, t1):
        """Integrate a system over a forward interval and return its trajectory.

        Parameters
        ----------
        system : callable
            Function or object called as system(t, y), returning the derivative
            with the same shape as y. If it has a names attribute, that attribute
            supplies the state names. Otherwise the names default to ("x", "z").
        y0 : array_like, shape (k,)
            Initial state at t0.
        t0, t1 : float
            Start and end times, with t1 >= t0. The interval must contain a
            whole number of steps of size self.h, within check_step's tolerance.

        Returns
        -------
        Trajectory
            Sample times, state values and state names, including the initial
            state and one row for each integration step.

        Raises
        ------
        ValueError
            If the interval cannot be divided into whole steps within the
            tolerance, or self.h has been changed to a non-positive value or NaN.
        """
        t, y = integrate(rk4_step, system, y0, t0, t1, self.h)
        return Trajectory(t, y, getattr(system, "names", ("x", "z")))
