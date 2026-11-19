"""Exercise 4 of Lecture 19: which powers of an array run as fast as one multiplication, and x ** 1.5 rewritten.

Run from the project root: python exercises/lecture19/e4_powers.py
"""
import numpy as np

from attractor.tools import median_time

x = np.linspace(1.0, 2.0, 4_000_001)
expressions = ["x * x", "x ** 2", "x ** 2.0", "np.power(x, 2)", "x ** 0.5", "x ** -1", "x ** 1.5", "x * np.sqrt(x)"]
for expression in expressions:
    f = eval("lambda: " + expression, {"x": x, "np": np})
    print(f"{expression:15} {1e3 * median_time(f, repeat=7):6.2f} ms")

a, b = x ** 1.5, x * np.sqrt(x)
spacings = np.abs(a - b) / np.spacing(a)  # each difference in units of the spacing of floats at a
print(f"the two forms of x ** 1.5 differ at {np.mean(a != b):.0%} of the values, by at most {spacings.max():.0f} spacing")
