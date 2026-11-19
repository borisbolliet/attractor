"""Line profile of rk4_factor on four million points.

Run from the project root: python experiments/stability_lines.py
"""
import numpy as np
from line_profiler import LineProfiler

from attractor.integrators import rk4_factor

z = np.linspace(-4.0, 0.0, 4_000_001)
profiler = LineProfiler(rk4_factor)        # time each line of rk4_factor
for _ in range(5):
    profiler.runcall(rk4_factor, z)
profiler.print_stats(output_unit=1e-3)     # times in milliseconds
