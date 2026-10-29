"""A timing decorator on `integrate`, and a wrapper seen by the profiler.

Run from the project root: python -m experiments.timing
"""
import cProfile

import numpy as np

from integrators import integrate, rk4_step
from models import CHEMOSTAT, make_lotka_volterra
from tools import counted, timed

y0 = np.array([1.5, 8.0])
timed_integrate = timed(integrate)
timed_integrate(rk4_step, make_lotka_volterra(**CHEMOSTAT), y0, 0.0, 40.0, 0.05)
print(timed_integrate.__name__, timed_integrate.__wrapped__ is integrate)
print(f"{timed_integrate.calls} call, {timed_integrate.seconds:.3f} s")

rhs = counted(make_lotka_volterra(**CHEMOSTAT))
cProfile.run("integrate(rk4_step, rhs, y0, 0.0, 40.0, 0.05)", sort="tottime")
