"""Fit LotkaVolterra to the lynx-hare pelts by least squares on the project's RK4.

Run from the project root: python -m experiments.fit_lynx_hare
"""
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from attractor.cycles import peaks
from attractor.datasets import lynx_hare
from attractor.integrators import RK4
from attractor.models import LotkaVolterra

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

year, lynx, hare = lynx_hare()  # thousands of pelts, 1900-1920, from the package data
P0 = (0.5, 0.02, 0.01, 0.8, 30.0, 4.0)  # a, b, c, d, H0, L0


def residuals(p):
    """Model minus data for hare and lynx, stacked; p = (a, b, c, d, H0, L0)."""
    system = LotkaVolterra(*p[:4])
    tr = RK4(0.02).run(system, p[4:], 0.0, 20.0)
    model = tr.y[::50]  # rows at integer years, since h = 0.02 year
    return np.concatenate([model[:, 0] - hare, model[:, 1] - lynx])


def fit(p0):
    """Bounded least squares from p0; returns the result and the wall time."""
    start = time.perf_counter()
    result = least_squares(residuals, p0, bounds=(0, np.inf))
    return result, time.perf_counter() - start


def main():
    result, seconds = fit(P0)
    a, b, c, d_, H0, L0 = result.x
    system = LotkaVolterra(a, b, c, d_)
    rms = np.sqrt(np.mean(result.fun**2))
    print(f"LotkaVolterra(a={a:.3f}, b={b:.4f}, c={c:.4f}, d={d_:.3f})")
    print(f"H0 = {H0:.2f}, L0 = {L0:.2f} thousand pelts")
    print(f"rms residual {rms:.2f} thousand pelts, {result.nfev} evaluations, {seconds:.1f} s")
    print(f"linearised period 2*pi/sqrt(a*d) = {2 * np.pi / np.sqrt(a * d_):.2f} years")
    for p0 in ((1.0, 0.05, 0.02, 0.5, 20, 10), (0.3, 0.01, 0.03, 1.2, 40, 2)):
        other, s = fit(p0)
        print(f"other start {p0}: max diff {np.abs(other.x - result.x).max():.1e}, {s:.1f} s")
    print("hare peak years:", year[peaks(year, hare, min_gap=3.0)].astype(int).tolist())
    print("lynx peak years:", year[peaks(year, lynx, min_gap=3.0)].astype(int).tolist())

    tr = RK4(0.02).run(system, (H0, L0), 0.0, 20.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5.5), height_ratios=(3, 1))
    ax1.plot(year, hare, "o", color="C2", label="hare pelts")
    ax1.plot(year, lynx, "s", color="C1", label="lynx pelts")
    ax1.plot(1900 + tr.t, tr.y[:, 0], color="C2", label="model, hare")
    ax1.plot(1900 + tr.t, tr.y[:, 1], color="C1", label="model, lynx")
    ax1.set_ylabel("thousands of pelts")
    ax1.legend(ncol=2)
    ax2.plot(year, tr.y[::50, 0] - hare, "o", color="C2")
    ax2.plot(year, tr.y[::50, 1] - lynx, "s", color="C1")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("model $-$ data")
    ax2.set_xticks(range(1900, 1921, 5))
    ax2.set_xlabel("year")
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/lynx_hare_fit.png", dpi=150, bbox_inches="tight")
    print("written: results/lynx_hare_fit.png")


if __name__ == "__main__":
    main()
