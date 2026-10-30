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
    """A right-hand side rhs(t, y) with named states and named parameters."""
    names = ("x", "z")

    @abstractmethod
    def rhs(self, t, y):
        """Time derivative of the state y at time t."""

    def __call__(self, t, y):
        return self.rhs(t, y)  # an instance is the callable the integrators expect

    def parameters(self):
        """Parameter names and values of a dataclass system."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class LotkaVolterra(DynamicalSystem):
    """dx/dt = a x - b x z, dz/dt = c x z - d z; defaults on chemostat scales."""
    a: float = 0.8
    b: float = 0.08
    c: float = 0.77
    d: float = 0.77
    names = ("algae", "rotifers")  # no annotation: a class attribute, not a field

    def rhs(self, t, y):
        x, z = y
        return np.array([self.a * x - self.b * x * z, self.c * x * z - self.d * z])


@dataclass(frozen=True)
class RosenzweigMacArthur(DynamicalSystem):
    """Logistic prey, type-II functional response f(x) = a x / (1 + a h x); a limit cycle."""
    r: float = 1.2
    K: float = 4.0
    a: float = 1.0
    h: float = 0.5
    e: float = 0.6
    m: float = 0.35
    names = ("prey", "predator")

    def rhs(self, t, y):
        x, z = y
        f = self.a * x / (1 + self.a * self.h * x)
        return np.array([self.r * x * (1 - x / self.K) - f * z, self.e * f * z - self.m * z])


class Goodwin(DynamicalSystem):
    """Goodwin's growth cycle: employment rate v and wage share u obey the Lotka-Volterra
    equations with (a, b, c, d) = (alpha, beta, rho, gamma)."""
    names = ("employment rate", "wage share")

    def __init__(self, alpha, beta, rho, gamma):
        self.alpha, self.beta, self.rho, self.gamma = alpha, beta, rho, gamma
        self.core = LotkaVolterra(alpha, beta, rho, gamma)  # has-a, not is-a

    def rhs(self, t, y):
        return self.core.rhs(t, y)

    def parameters(self):
        return dict(alpha=self.alpha, beta=self.beta, rho=self.rho, gamma=self.gamma)

    def __repr__(self):
        return f"Goodwin(alpha={self.alpha}, beta={self.beta}, rho={self.rho}, gamma={self.gamma})"


@runtime_checkable
class HasRHS(Protocol):
    """Anything with an rhs(t, y) method, inheritance or not."""
    def rhs(self, t, y): ...


@register("lotka_volterra")
def make_lotka_volterra(a, b, c, d):
    """Factory kept from the earlier chapters; the system object is itself callable as rhs(t, y)."""
    return LotkaVolterra(a, b, c, d)


@register("rosenzweig_macarthur")
def make_rosenzweig_macarthur(r, K, a, h, e, m):
    """Factory kept from the earlier chapters."""
    return RosenzweigMacArthur(r, K, a, h, e, m)
