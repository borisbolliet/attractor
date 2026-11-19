"""Exercise 1 of Lecture 19: a state of three values passed to a model of two.

Run from the project root: python exercises/lecture19/e1_traceback.py
"""
from attractor import RK4, LotkaVolterra

tr = RK4(0.1).run(LotkaVolterra(), [1.5, 8.0, 0.0], 0.0, 1.0)
