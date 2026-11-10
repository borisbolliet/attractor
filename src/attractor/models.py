"""Dynamical systems of the project as classes implementing `DynamicalSystem`."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Protocol, runtime_checkable

import numpy as np

from .tools import REGISTRY, register

# Chemostat scales: algae x in 10^6 cells/ml, rotifers z in animals/ml, rates per day.
# Equilibrium (d/c, a/b) = (1, 10); linearised period 2*pi/sqrt(a*d) = 8.0 days.
CHEMOSTAT = dict(a=0.8, b=0.08, c=0.77, d=0.77)


class DynamicalSystem(ABC):
    """Define a differential equation with named states and parameters.

    Subclasses implement rhs(t, y). An instance can also be called directly
    as system(t, y), which calls rhs. The default names are ("x", "z").
    The default parameters method reads dataclass fields; other subclasses
    should supply their own parameters method.
    """
    names = ("x", "z")

    @abstractmethod
    def rhs(self, t, y):
        """Return the time derivative of the state.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        y : array_like, shape (k,)
            State in the order given by names.

        Returns
        -------
        numpy.ndarray, shape (k,)
            Time derivative in the same state order.
        """

    def __call__(self, t, y):
        """Evaluate rhs(t, y) so that the system can be used as a callable."""
        return self.rhs(t, y)  # an instance is the callable the integrators expect

    def parameters(self):
        """Return a dictionary of parameter names and values from dataclass fields."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class LotkaVolterra(DynamicalSystem):
    """Model prey x and predator z with the Lotka-Volterra equations.

    The equations are dx/dt = a*x - b*x*z and dz/dt = c*x*z - d*z.
    The default state names are "algae" and "rotifers".

    Parameters
    ----------
    a : float, optional
        Prey growth rate, in inverse time (default 0.8).
    b : float, optional
        Prey loss coefficient per predator density and per unit time
        (default 0.08).
    c : float, optional
        Predator growth coefficient per prey density and per unit time
        (default 0.77).
    d : float, optional
        Predator loss rate, in inverse time (default 0.77).

    Notes
    -----
    The default scale uses time in days, x in millions of algae cells per
    millilitre, and z in rotifers per millilitre. These parameters provide
    an illustrative model, not a fit to the experimental data. Parameters
    are stored without validation.
    """
    a: float = 0.8
    b: float = 0.08
    c: float = 0.77
    d: float = 0.77
    names = ("algae", "rotifers")  # no annotation: a class attribute, not a field

    def rhs(self, t, y):
        """Return (dx/dt, dz/dt) for y = (x, z); the equations do not depend on t.

        The state and return value follow the rhs interface of DynamicalSystem.
        """
        x, z = y
        return np.array([self.a * x - self.b * x * z, self.c * x * z - self.d * z])


@dataclass(frozen=True)
class RosenzweigMacArthur(DynamicalSystem):
    """Model prey growth with a carrying capacity and saturating predation.

    For prey x and predator z, define f(x) = a*x / (1 + a*h*x).
    Then dx/dt = r*x*(1 - x/K) - f(x)*z and dz/dt = e*f(x)*z - m*z.
    The state names are "prey" and "predator".

    Parameters
    ----------
    r : float, optional
        Prey growth rate at low density (default 1.2).
    K : float, optional
        Prey carrying capacity, in the units of x (default 4.0).
    a : float, optional
        Attack coefficient in f(x) (default 1.0).
    h : float, optional
        Handling-time parameter that limits the intake rate (default 0.5).
    e : float, optional
        Conversion factor from prey intake to predator growth (default 0.6).
    m : float, optional
        Predator mortality rate (default 0.35).

    Notes
    -----
    With dimensionless densities, r, a and m have units of inverse time,
    h has units of time, and K and e are dimensionless. Parameters are
    stored without validation. The behaviour depends on the parameters;
    a limit cycle is not guaranteed for arbitrary values.
    """
    r: float = 1.2
    K: float = 4.0
    a: float = 1.0
    h: float = 0.5
    e: float = 0.6
    m: float = 0.35
    names = ("prey", "predator")

    def rhs(self, t, y):
        """Return (dx/dt, dz/dt) for y = (prey, predator).

        The state and return value follow the rhs interface of DynamicalSystem.
        The equations do not depend explicitly on t.
        """
        x, z = y
        f = self.a * x / (1 + self.a * self.h * x)
        return np.array([self.r * x * (1 - x / self.K) - f * z, self.e * f * z - self.m * z])


class Goodwin(DynamicalSystem):
    """Represent employment rate v and wage share u by a Lotka-Volterra system.

    The equations are dv/dt = alpha*v - beta*v*u and
    du/dt = rho*v*u - gamma*u. The state order is (v, u).

    Parameters
    ----------
    alpha : float
        Employment growth coefficient in the absence of the wage-share term.
    beta : float
        Coefficient of the wage share in employment growth.
    rho : float
        Coefficient of employment in wage-share growth.
    gamma : float
        Wage-share loss coefficient in the absence of employment.

    Notes
    -----
    Both state variables are dimensionless fractions. All four coefficients
    have units of inverse time. This implementation delegates to LotkaVolterra
    and does not enforce the interval [0, 1] for either state.
    """
    names = ("employment rate", "wage share")

    def __init__(self, alpha, beta, rho, gamma):
        self.alpha, self.beta, self.rho, self.gamma = alpha, beta, rho, gamma
        self.core = LotkaVolterra(alpha, beta, rho, gamma)  # has-a, not is-a

    def rhs(self, t, y):
        """Return (dv/dt, du/dt) for y = (employment rate, wage share).

        The state and return value follow the rhs interface of DynamicalSystem.
        The equations do not depend explicitly on t.
        """
        return self.core.rhs(t, y)

    def parameters(self):
        """Return the named coefficients alpha, beta, rho and gamma as a dictionary."""
        return dict(alpha=self.alpha, beta=self.beta, rho=self.rho, gamma=self.gamma)

    def __repr__(self):
        return f"Goodwin(alpha={self.alpha}, beta={self.beta}, rho={self.rho}, gamma={self.gamma})"


@runtime_checkable
class HasRHS(Protocol):
    """Anything with an rhs(t, y) method, inheritance or not."""
    def rhs(self, t, y): ...


@register("lotka_volterra")
def make_lotka_volterra(a, b, c, d):
    """Return a LotkaVolterra instance using its four named parameters.

    See Also
    --------
    LotkaVolterra : Parameter meanings, units and equations.
    """
    return LotkaVolterra(a, b, c, d)


@register("rosenzweig_macarthur")
def make_rosenzweig_macarthur(r, K, a, h, e, m):
    """Return a RosenzweigMacArthur instance using its six named parameters.

    See Also
    --------
    RosenzweigMacArthur : Parameter meanings, units and equations.
    """
    return RosenzweigMacArthur(r, K, a, h, e, m)
