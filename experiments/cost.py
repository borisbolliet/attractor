"""Right-hand-side calls and end-state error of Euler and RK4 for a range of steps.

Run from the project root: python -m experiments.cost
"""
import csv
from pathlib import Path

import numpy as np

from integrators import euler_step, integrate, rk4_step
from models import CHEMOSTAT, make_lotka_volterra
from tools import counted

Y0, T1, TARGET = np.array([1.5, 8.0]), 40.0, 1e-5
STEPS = {"euler": (0.1, 0.05, 0.01, 0.005, 0.001), "rk4": (0.4, 0.2, 0.1, 0.05, 0.01)}
METHODS = {"euler": euler_step, "rk4": rk4_step}


def end_state_error(step, h, reference):
    """Calls to the right-hand side and max abs error at T1 for one run."""
    rhs = counted(make_lotka_volterra(**CHEMOSTAT))
    _, y = integrate(step, rhs, Y0, 0.0, T1, h)
    return rhs.calls, np.abs(y[-1] - reference).max()


def extrapolated_calls(rows):
    """Calls for TARGET from the slope of the last two rows (h, calls, error)."""
    (h1, _, e1), (h2, c2, e2) = rows[-2:]
    order = np.log(e1 / e2) / np.log(h1 / h2)
    h_target = h2 * (TARGET / e2) ** (1 / order)
    return c2 * h2 / h_target


def main():
    _, y_ref = integrate(rk4_step, make_lotka_volterra(**CHEMOSTAT), Y0, 0.0, T1, 1e-3)
    reference = y_ref[-1]
    table = {}
    print(f"{'method':<8}{'h':>7}{'steps':>8}{'calls':>8}{'error at t=40':>16}")
    for name, step in METHODS.items():
        table[name] = []
        for h in STEPS[name]:
            calls, err = end_state_error(step, h, reference)
            table[name].append((h, calls, err))
            print(f"{name:<8}{h:>7g}{round(T1 / h):>8}{calls:>8}{err:>16.1e}")
    below = [(h, c) for h, c, e in table["rk4"] if e < TARGET]
    print(f"RK4: {below[0][1]} calls for error < {TARGET:g} (h = {below[0][0]:g})")
    print(f"Euler: ~{extrapolated_calls(table['euler']):.0e} calls (extrapolated from the slope of the table)")
    Path("results").mkdir(exist_ok=True)
    with open("results/cost.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("method", "h", "calls", "error"))
        w.writerows((name, h, c, f"{e:.3e}") for name in table for h, c, e in table[name])


if __name__ == "__main__":
    main()
