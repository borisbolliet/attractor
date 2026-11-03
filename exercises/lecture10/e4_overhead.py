"""Run from the project root: python -m exercises.lecture10.e4_overhead"""
import time
from tools import counted, timed
def empty(): pass
N = 10**6
PAIRS = (("bare", empty), ("counted", counted(empty)), ("timed", timed(empty)))
for name, f in PAIRS:
    start = time.perf_counter()
    for _ in range(N):
        f()
    dt = time.perf_counter() - start
    print(f"{name:8s} {dt:.3f} s  {dt / N * 1e9:.0f} ns per call")
