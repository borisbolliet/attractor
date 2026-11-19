"""Fixed-step RK4 for the Oregonator, compiled by Numba, against the same loop in Python.

Run from the project root: python experiments/bz_numba.py
"""
import time

import numpy as np
from numba import njit

from attractor import RK4, Oregonator
from attractor.tools import median_time


@njit
def rhs(y, s, q, w, f):
    a, e, r = y[0], y[1], y[2]
    return np.array([s * (e - e * a + a - q * a * a), (-e - e * a + f * r) / s, w * (a - r)])


@njit
def rk4_run(y0, h, n, s, q, w, f):
    """n RK4 steps of size h from y0; returns the n + 1 states."""
    ys = np.empty((n + 1, 3))
    y = y0
    ys[0] = y
    for i in range(n):
        k1 = rhs(y, s, q, w, f)
        k2 = rhs(y + h / 2 * k1, s, q, w, f)
        k3 = rhs(y + h / 2 * k2, s, q, w, f)
        k4 = rhs(y + h * k3, s, q, w, f)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i + 1] = y
    return ys


m, y0, h = Oregonator(), np.array([1.0, 2.0, 3.0]), 2e-5
n = round(30.0 / h)
args = (y0, h, n, m.s, m.q, m.w, m.f)
start = time.perf_counter()
ys = rk4_run(*args)                      # the first call compiles rk4_run and rhs
first = time.perf_counter() - start
compiled = median_time(rk4_run, *args, repeat=5)
python = median_time(RK4(h).run, m, y0, 0.0, 30.0, repeat=3)
print(f"{n} steps of size {h:g}")
print(f"Numba, first call: {first:.3f} s")
print(f"Numba, median of 5 later calls: {compiled:.3f} s")
print(f"Python, median of 3 calls: {python:.2f} s")
print(f"same last state: {np.allclose(ys[-1], RK4(h).run(m, y0, 0.0, 30.0).y[-1], rtol=1e-12)}")
