"""Shallow copies of an object that holds arrays: what is shared, what is a view, what a write
through the copy changes.

Run from the project root: python -i -m experiments.copies
"""
import numpy as np

from attractor.trajectory import Trajectory

tr = Trajectory(np.arange(6.0), np.arange(12.0).reshape(6, 2).copy())  # y must own its memory
sh, ev, rv, dp = tr.shifted(1.0), tr.every(2), tr.time_reversed(), tr.copy()

print(repr(tr), len(tr), tr[1])
print(f"{'method':<17}{'attribute':<10}{'is':<7}{'shares':<8}{'base is'}")
for method, other in (("shifted", sh), ("every", ev), ("time_reversed", rv), ("copy", dp)):
    for name, value in vars(tr).items():
        o = getattr(other, name)
        shares = np.shares_memory(o, value) if isinstance(value, np.ndarray) else "-"
        base = (o.base is value) if isinstance(value, np.ndarray) else "-"
        print(f"{method:<17}{name:<10}{o is value!s:<7}{shares!s:<8}{base}")
print("vars(sh) is vars(tr):", vars(sh) is vars(tr))

sh.y[0, 0] = -1.0
print("after sh.y[0, 0] = -1.0:  tr.y[0] =", tr.y[0], " sh.y is tr.y:", sh.y is tr.y)
ev.y[1, 0] = 0.0
print("after ev.y[1, 0] = 0.0:   tr.y[2] =", tr.y[2])
rv.y[0, 0] = 99.0
print("after rv.y[0, 0] = 99.0:  tr.y[-1] =", tr.y[-1])
sh.labels.append("shifted")
print("after sh.labels.append:   tr.labels =", tr.labels, " dp.labels =", dp.labels)
dp.y[0, 0] = 7.0
print("after dp.y[0, 0] = 7.0:   tr.y[0] =", tr.y[0], " shares:", np.shares_memory(dp.y, tr.y))
