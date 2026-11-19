"""Steps, evaluations and time of five solvers on the Oregonator, 0 <= t <= 30.

Run from the project root: python experiments/bz_solvers.py
"""
from scipy.integrate import odeint

from attractor import Adaptive, Oregonator
from attractor.tools import median_time

system, y0 = Oregonator(), [1.0, 2.0, 3.0]


def lsoda_loop():
    """LSODA through odeint: the loop over steps runs in compiled code."""
    _, info = odeint(system.rhs, y0, [0.0, 30.0], Dfun=system.jacobian, tfirst=True,
                     rtol=1e-6, atol=1e-6, mxstep=10_000, full_output=True)
    return dict(steps=info["nst"][-1], nfev=info["nfe"][-1], njev=info["nje"][-1], nlu=info["nje"][-1])


print(f"{'method':<15}{'steps':>8}{'nfev':>9}{'njev':>6}{'nlu':>6}{'seconds':>9}")
for method, repeat in [("RK45", 3), ("Radau", 5), ("BDF", 5), ("LSODA", 5)]:
    solver = Adaptive(method)
    seconds = median_time(solver.run, system, y0, 0.0, 30.0, repeat=repeat)
    s = solver.stats
    print(f"{method:<15}{s['steps']:>8}{s['nfev']:>9}{s['njev']:>6}{s['nlu']:>6}{seconds:>9.4f}")
s = lsoda_loop()
print(f"{'LSODA (odeint)':<15}{s['steps']:>8}{s['nfev']:>9}{s['njev']:>6}{s['nlu']:>6}"
      f"{median_time(lsoda_loop, repeat=5):>9.4f}")
