"""attractor: predator-prey and chemical oscillators, integrators and cycle periods.

The names in __all__ form the public interface; the rest may change.
"""
from importlib.metadata import version as _version

from .cycles import cycle_period, peaks
from .datasets import load_run, lynx_hare
from .integrators import RK4, Adaptive, Euler, integrate
from .models import (CHEMOSTAT, DynamicalSystem, Goodwin, LotkaVolterra,
                     Oregonator, RosenzweigMacArthur)
from .trajectory import Trajectory

__version__ = _version("attractor")  # from the metadata written at install time

__all__ = ["DynamicalSystem", "LotkaVolterra", "RosenzweigMacArthur", "Goodwin",
           "Oregonator", "CHEMOSTAT", "Euler", "RK4", "Adaptive", "integrate",
           "Trajectory", "peaks", "cycle_period", "load_run", "lynx_hare"]
