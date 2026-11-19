"""Profile of the RK45 run of the Oregonator, sorted by the time in each function's own lines.

Run from the project root: python experiments/bz_profile.py
"""
import cProfile
import pstats

import scipy.integrate  # noqa: F401  (imported here, so that the profile holds the run only)

from attractor import Adaptive, Oregonator

profiler = cProfile.Profile()
profiler.runcall(Adaptive("RK45").run, Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)
stats = pstats.Stats(profiler).strip_dirs().sort_stats("tottime")
stats.print_stats(8)
stats.print_callees("rk_step")
