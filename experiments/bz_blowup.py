"""The Oregonator integrated with RK4 at a fixed step: the state overflows.

Run from the project root: python experiments/bz_blowup.py [h]
"""
import sys

from attractor import RK4, Oregonator

h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
tr = RK4(h).run(Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)
print(f"{len(tr) - 1} steps of size {h:g}; last state {tr.y[-1]}")
