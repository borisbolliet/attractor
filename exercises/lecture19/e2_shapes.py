"""Exercise 2 of Lecture 19: the shapes along the batched eigenvalue computation.

Run from the project root: python exercises/lecture19/e2_shapes.py
"""
import numpy as np

from attractor import Oregonator

m = Oregonator()
Y = np.random.default_rng(0).uniform(1.0, 100.0, (3, 6))   # six states, one per column
J = m.jacobian(0.0, Y)
lam = np.linalg.eigvals(J)
fastest = np.abs(lam).max(axis=1)
k = fastest.argmax()
print(Y.shape, J.shape, lam.shape, fastest.shape, np.shape(k), Y[:, k].shape)
