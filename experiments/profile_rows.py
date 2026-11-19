"""Which NumPy operations have a row in a cProfile table.

Run from the project root: python experiments/profile_rows.py
"""
import cProfile
import pstats

import numpy as np


def mean_abs_cube(x):
    cubes = x ** 3                      # an operator
    sizes = np.abs(cubes)               # a ufunc, called directly
    return np.sum(sizes) / x.size       # np.sum, a function written in Python


x = np.linspace(-1.0, 1.0, 4_000_001)
profiler = cProfile.Profile()
profiler.runcall(mean_abs_cube, x)
pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats()
