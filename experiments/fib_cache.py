"""Memoisation with lru_cache: Fibonacci, an unhashable argument, and a cached data loader.

Run from the project root: python -i -m experiments.fib_cache
"""
import functools
import time

import numpy as np

from attractor import datasets  # the reader of one run, formerly in analyse.py


@functools.lru_cache(maxsize=None)
def fib(n):
    """The n-th Fibonacci number, each value computed once."""
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def fib_plain(n):
    """The same recursion without a cache."""
    return n if n < 2 else fib_plain(n - 1) + fib_plain(n - 2)


@functools.lru_cache
def load_run(path):
    """Time and algae density of one run, read from disk once per process."""
    return datasets.load_run(str(path))


print(fib(80), fib.cache_info())
start = time.perf_counter()
fib_plain(28)
print(f"fib_plain(28): {time.perf_counter() - start:.2f} s")
try:
    fib(np.array([3]))
except TypeError as e:
    print("TypeError:", e)

t1, a1 = load_run("data/C9.csv")
t2, a2 = load_run("data/C9.csv")
print(a1 is a2, load_run.cache_info())
a1[0] = -1.0
print(a2[0], datasets.load_run("data/C9.csv")[1][0])
