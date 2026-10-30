"""Goodwin's growth cycle under the project's RK4: the same equations as the chemostat model.

Run from the project root: python -m experiments.goodwin_cycle
"""
import os
from pathlib import Path

import numpy as np

from attractor.integrators import RK4
from attractor.models import Goodwin

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

system = Goodwin(alpha=0.03, beta=0.05, rho=0.5, gamma=0.45)
tr = RK4(0.05).run(system, (0.95, 0.65), 0.0, 300.0)
print(system)
print(system.parameters())
print(tr, tr.names)
print(f"equilibrium (gamma/rho, alpha/beta) = ({system.gamma / system.rho:.2f}, {system.alpha / system.beta:.2f})")
print(f"linearised period 2*pi/sqrt(alpha*gamma) = {2 * np.pi / np.sqrt(system.alpha * system.gamma):.1f}")
print(f"measured period {tr.period(0):.1f} time units")
print(f"employment rate {tr.y[:, 0].min():.3f}-{tr.y[:, 0].max():.3f}, wage share {tr.y[:, 1].min():.3f}-{tr.y[:, 1].max():.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(tr.y[:, 0], tr.y[:, 1], color="C0", lw=1)
ax.plot(system.gamma / system.rho, system.alpha / system.beta, "k+", ms=10)
ax.set_xlabel("employment rate v")
ax.set_ylabel("wage share u")
Path("results").mkdir(exist_ok=True)
fig.savefig("results/goodwin.png", dpi=150, bbox_inches="tight")
print("written: results/goodwin.png")
