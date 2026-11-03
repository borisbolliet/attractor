"""Run from the project root: python -m exercises.lecture11.e5_tolerance"""
import numpy as np
from attractor.trajectory import Trajectory
t = np.arange(3.0)
A = Trajectory(t, np.full((3, 2), 1.0))
B = Trajectory(t, np.full((3, 2), 1.0 + 0.8e-5))
C = Trajectory(t, np.full((3, 2), 1.0 + 1.6e-5))
print(A == B, B == C, A == C)
print(Trajectory.__hash__)
