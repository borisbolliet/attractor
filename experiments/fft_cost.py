"""The cost of Fourier transforms: the DFT as a matrix-vector product against the FFT,
a direct cross-correlation against the FFT one, and FFT lengths with and without
a large prime factor.

Run from the project root: python experiments/fft_cost.py
Each time is the median of five calls after one untimed call; times vary between machines.
"""
import time

import numpy as np

from attractor.spectrum import cross_correlation


def seconds(function, *args, repeat=5):
    """Median elapsed time of function(*args), after one untimed call."""
    function(*args)
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(*args)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


rng = np.random.default_rng(0)

print("     N   matrix (ms)   fft (ms)   ratio")
for N in (512, 1024, 2048, 4096):
    k = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(k, k) / N)  # N x N complex128, built once
    x = rng.standard_normal(N)
    t_matrix, t_fft = seconds(W.__matmul__, x), seconds(np.fft.fft, x)
    print(f"{N:6d} {1e3 * t_matrix:12.3f} {1e3 * t_fft:10.4f} {t_matrix / t_fft:7.0f}")

print("      n   correlate (ms)   fft (ms)   ratio")
for n in (5_000, 10_000, 20_000, 40_000):
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    t_direct = seconds(np.correlate, y, x, "full")  # every lag by direct sums
    t_fft = seconds(cross_correlation, x, y)
    print(f"{n:7d} {1e3 * t_direct:15.2f} {1e3 * t_fft:10.3f} {t_direct / t_fft:7.0f}")

print("        N   factors        fft (ms)")
for N, factors in ((2**20, "2**20"), (2**20 - 3, "prime")):
    x = rng.standard_normal(N)
    print(f"{N:9d}   {factors:8s} {1e3 * seconds(np.fft.fft, x):12.2f}")
