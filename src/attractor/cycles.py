"""Peaks and cycle period of a sampled oscillation."""
import numpy as np


def peaks(t, x, min_gap=3.0):
    """Return indices of peaks above the mean, separated by at least min_gap.

    Parameters
    ----------
    t : numpy.ndarray, shape (n,)
        Sample times in increasing order.
    x : numpy.ndarray, shape (n,)
        Finite sample values corresponding to t.
    min_gap : float, optional
        Minimum time between retained peaks, in the units of t (default 3.0).

    Returns
    -------
    numpy.ndarray of int
        Indices of the retained peaks, in time order. The array is empty
        when no peaks are found.

    Notes
    -----
    An interior sample is a candidate when it is above the mean of x,
    strictly above its left neighbour, and at least as high as its right
    neighbour. If it is too close to the last retained peak, keep the
    higher of the two; an equal height leaves the earlier peak in place.
    The first and last samples are not candidates.
    """
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
    """Estimate a period from the mean spacing of retained peaks.

    Parameters
    ----------
    t : numpy.ndarray, shape (n,)
        Sample times in increasing order.
    x : numpy.ndarray, shape (n,)
        Finite sample values corresponding to t.
    min_gap : float, optional
        Minimum separation of retained peaks, in the units of t (default 3.0).

    Returns
    -------
    period : float
        Mean spacing between consecutive retained peaks, in the units of t.
        Returns numpy.nan when fewer than three peaks are found.
    peak_times : numpy.ndarray
        Times of all retained peaks, including when period is NaN.

    See Also
    --------
    peaks : Peak selection and treatment of nearby maxima.
    """
    p = peaks(t, x, min_gap)
    if len(p) < 3:
        return np.nan, t[p]
    return (t[p[-1]] - t[p[0]]) / (len(p) - 1), t[p]
