"""The first call of a run against later calls; the median against the mean, with and without other programs running.

Run from the project root: python experiments/timing_spread.py
"""
import multiprocessing
import os
import statistics
import time

from attractor import Adaptive, Oregonator


def busy(seconds):
    """Keep one processor busy, as another program would."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def call_times(f, n):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return times


def report(label, times):
    ms = [1e3 * t for t in times]
    print(f"{label}: median {statistics.median(ms):.2f} ms, mean {statistics.mean(ms):.2f} ms, "
          f"max {max(ms):.2f} ms")


if __name__ == "__main__":
    run = lambda: Adaptive("LSODA").run(Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)  # noqa: E731
    first = call_times(run, 1)[0]
    print(f"first call: {1e3 * first:.1f} ms")
    report("100 later calls", call_times(run, 100))
    others = [multiprocessing.Process(target=busy, args=(10.0,)) for _ in range(os.cpu_count() + 8)]
    for p in others:
        p.start()
    report(f"100 calls beside {len(others)} busy processes", call_times(run, 100))
    for p in others:
        p.terminate()
