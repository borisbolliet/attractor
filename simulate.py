"""Simulate the chemostat model and measure its period.

The stepping loop is written out, so that `python -i simulate.py` leaves the list
`rows` in the session. Step size from ATTRACTOR_H (default 0.05 days).
"""
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from attractor.cycles import cycle_period
from attractor.integrators import rk4_step
from attractor.models import CHEMOSTAT, make_lotka_volterra

load_dotenv()
H = float(os.environ.get("ATTRACTOR_H", "0.05"))
RESULTS = Path(os.environ.get("RESULTS_PATH", "results"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is set)

rhs = make_lotka_volterra(**CHEMOSTAT)
y0, t0, t1, h = np.array([1.5, 8.0]), 0.0, 40.0, H
n = round((t1 - t0) / h)
y = y0.copy()
rows = [y]
for i in range(n):
    y = rk4_step(rhs, t0 + i * h, y, h)
    rows.append(y)
t = t0 + h * np.arange(n + 1)
y = np.array(rows)

period, _ = cycle_period(t, y[:, 0])
print(f"first row  {y[0, 0]:.4f} {y[0, 1]:.4f}")
print(f"last row   {y[-1, 0]:.4f} {y[-1, 1]:.4f}")
print(f"{len(y)} rows, step {H}, algae period {period:.2f} days")

RESULTS.mkdir(parents=True, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
ax1.plot(t, y[:, 0], color="C2")
ax1.set_ylabel("algae ($10^6$ cells/ml)")
ax2.plot(t, y[:, 1], color="C1")
ax2.set_ylabel("rotifers (animals/ml)")
ax2.set_xlabel("time (days)")
fig.savefig(RESULTS / "simulated.png", dpi=150, bbox_inches="tight")
print(f"written: {RESULTS / 'simulated.png'}")
