"""Peaks and cycle period of a sampled oscillation."""
import numpy as np


def peaks(t, x, min_gap=3.0):
    """Indices of the local maxima of x above its mean, at least min_gap apart (units of t)."""
    level = x.mean()
    found = []
    for i in range(1, len(x) - 1):
        if not (x[i - 1] < x[i] >= x[i + 1] and x[i] > level):
            continue
        if found and t[i] - t[found[-1]] < min_gap:
            if x[i] > x[found[-1]]:
                found[-1] = i  # the higher of two close maxima is kept
            continue
        found.append(i)
    return np.array(found, dtype=int)


def cycle_period(t, x, min_gap=3.0):
    """Mean peak-to-peak spacing (units of t), and the peak times."""
    p = peaks(t, x, min_gap)
    if len(p) < 3:
        return np.nan, t[p]
    return (t[p[-1]] - t[p[0]]) / (len(p) - 1), t[p]
