"""Where the RK4 run of the Oregonator stops being finite, for several step sizes.

Run from the project root: python experiments/bz_steps.py
"""
import logging
import time

import numpy as np

from attractor import RK4, Oregonator

logging.getLogger("attractor").setLevel(logging.ERROR)  # the table reports the failures
np.seterr(over="ignore", invalid="ignore")              # no warnings: values become inf and nan

print(f"{'h':>7} {'steps':>9} {'first non-finite t':>19} {'seconds':>8}")
for h in (1e-2, 1e-3, 1e-4, 2e-5):
    start = time.perf_counter()
    tr = RK4(h).run(Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)
    seconds = time.perf_counter() - start
    bad = ~np.isfinite(tr.y).all(axis=1)
    where = f"{tr.t[bad.argmax()]:.4g}" if bad.any() else "none"
    print(f"{h:>7g} {len(tr) - 1:>9} {where:>19} {seconds:>8.2f}")
