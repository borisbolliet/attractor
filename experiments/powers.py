"""Time of ** against * on an array of four million float64 values.

Run from the project root: python experiments/powers.py
"""
import numpy as np

from attractor.tools import median_time

z = np.linspace(-4.0, 0.0, 4_000_001)
for expression in ["z * z", "z ** 2", "z * z * z", "z ** 3", "z ** 4"]:
    f = eval("lambda: " + expression, {"z": z})
    print(f"{expression:10} {1e3 * median_time(f, repeat=7):6.2f} ms")
