"""Profile of the stability interval of RK4, computed on four million points of the real axis.

Run from the project root: python experiments/stability_profile.py
"""
import cProfile
import pstats

import numpy as np

from attractor.integrators import rk4_factor


def stability_limit(n=4_000_001):
    """Most negative z in [-4, 0] at which one RK4 step does not amplify."""
    z = np.linspace(-4.0, 0.0, n)
    stable = np.abs(rk4_factor(z)) <= 1
    return z[stable].min()


profiler = cProfile.Profile()
limit = profiler.runcall(stability_limit)
print(f"|R(z)| <= 1 for {limit:.4f} <= z <= 0")
pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(4)
