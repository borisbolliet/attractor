"""Step through the Jacobian of a batch of four states in the debugger.

Run from the project root: python experiments/bz_shapes.py
"""
import numpy as np

from attractor import Oregonator

m = Oregonator()
Y = np.ones((3, 4))                     # four states, one per column
breakpoint()                            # pdb stops at this line
J = m.jacobian(0.0, Y)
print(J.shape)
