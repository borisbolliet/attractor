"""Exercise 3 of Lecture 19: the profile of a fixed-step RK4 run of the Oregonator.

Run from the project root: python exercises/lecture19/e3_profile.py
"""
import cProfile
import pstats

from attractor import RK4, Oregonator

profiler = cProfile.Profile()
profiler.runcall(RK4(0.001).run, Oregonator(), [1.0, 2.0, 3.0], 0.0, 20.0)
pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(7)
