"""Run from the project root: python -m exercises.lecture11.e4_interface_cost"""
import time
import numpy as np
from models import CHEMOSTAT, LotkaVolterra
def make_closure(a, b, c, d):
    def rhs(t, y):
        x, z = y
        return np.array([a * x - b * x * z, c * x * z - d * z])
    return rhs
closure, lv = make_closure(**CHEMOSTAT), LotkaVolterra()
y = np.array([1.5, 8.0])
times = {}
for name, f in (("closure", closure), ("lv.rhs", lv.rhs), ("lv()", lv)):
    start = time.perf_counter()
    for _ in range(10**5):
        f(0.0, y)
    times[name] = time.perf_counter() - start
    print(f"{name:8s} {times[name]:.3f} s")
print(f"lv() / closure = {times['lv()'] / times['closure']:.2f}")
