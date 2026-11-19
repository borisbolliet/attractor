"""Eigenvalues of the Oregonator's Jacobian along a run: one state at a time, then the whole batch.

Run from the project root: python experiments/bz_eigenvalues.py
"""
import numpy as np

from attractor import Adaptive, Oregonator
from attractor.tools import median_time

m = Oregonator()
tr = Adaptive("LSODA", rtol=1e-8, atol=1e-8).run(m, [1.0, 2.0, 3.0], 0.0, 30.0)
Y = tr.y.T                                      # shape (3, n): one column per state


def one_at_a_time(Y):
    return np.array([np.linalg.eigvals(m.jacobian(0.0, y)) for y in Y.T])


def batch(Y):
    return np.linalg.eigvals(m.jacobian(0.0, Y))   # (n, 3, 3) -> (n, 3)


assert np.allclose(np.sort_complex(one_at_a_time(Y)), np.sort_complex(batch(Y)))
lam = batch(Y)
fastest = np.abs(lam).max(axis=1)
print(f"{Y.shape[1]} states; largest |eigenvalue| {fastest.max():.4g} at t = {tr.t[fastest.argmax()]:.2f}")
for h in (1e-2, 1e-3, 1e-4, 2e-5):
    over = h * fastest > 2.785
    print(f"h = {h:g}: h*|lambda| > 2.785 from t = {tr.t[over.argmax()]:.2f}" if over.any()
          else f"h = {h:g}: h*|lambda| <= 2.785 up to t = 30")
print(f"one at a time: {1e3 * median_time(one_at_a_time, Y):.1f} ms")
print(f"batch: {1e3 * median_time(batch, Y):.2f} ms")
