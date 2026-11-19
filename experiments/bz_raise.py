"""The same run with NumPy floating-point errors raised as exceptions.

Run from the project root: python experiments/bz_raise.py
"""
import numpy as np

from attractor import RK4, Oregonator

np.seterr(over="raise", invalid="raise")  # overflow and invalid results raise FloatingPointError
tr = RK4(0.01).run(Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)
