"""Spectra of sampled time series, computed with the fast Fourier transform.

The discrete Fourier transform assumes samples taken at equal intervals:
`uniform` puts irregular samples on such a grid. The other functions act
along one axis of their array arguments, the last by default, so that
several series of equal length, such as the algae and rotifer densities of
one run, are transformed in one call.
"""
import math

import numpy as np


def uniform(t, x, dt=1.0):
    """Interpolate samples linearly onto equally spaced times.

    Parameters
    ----------
    t : numpy.ndarray, shape (n,)
        Sample times, strictly increasing.
    x : numpy.ndarray, shape (n,)
        Sample values at the times t.
    dt : float, optional
        Spacing of the new times, in the units of t (default 1.0).

    Returns
    -------
    tu : numpy.ndarray
        The times t[0], t[0] + dt, t[0] + 2 dt, ..., up to t[-1].
    xu : numpy.ndarray
        x interpolated linearly at the times tu.

    Raises
    ------
    ValueError
        If t does not increase strictly; np.interp would then return
        meaningless values without raising an error.

    Examples
    --------
    >>> tu, xu = uniform([0.0, 0.9, 2.2, 3.1], [0.0, 1.8, 4.4, 6.2])
    >>> tu, xu
    (array([0., 1., 2., 3.]), array([0., 2., 4., 6.]))
    """
    t = np.asarray(t, dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("sample times must increase strictly")
    m = int((t[-1] - t[0]) // dt) + 1
    tu = t[0] + dt * np.arange(m)
    return tu, np.interp(tu, t, x)


def fourier(x, dt=1.0, n=None, window=True, axis=-1):
    """Frequencies and Fourier coefficients of real samples taken every dt.

    The mean of each series is removed first. With window=True the samples
    are multiplied by a Hann window, which reduces leakage between
    frequencies. An n larger than the number of samples m pads them with
    zeros: the frequencies are then k / (n dt), a finer grid.

    Returns
    -------
    f : numpy.ndarray, shape (n // 2 + 1,)
        Frequencies in cycles per unit of t, from 0 to the Nyquist frequency.
    X : numpy.ndarray of complex
        The coefficients of numpy.fft.rfft, with n // 2 + 1 values along axis.
    """
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    m = x.shape[-1]
    n = m if n is None else n
    x = x - x.mean(axis=-1, keepdims=True)
    if window:
        x = x * np.hanning(m)  # shape (m,), broadcast over the leading axes
    X = np.fft.rfft(x, n=n)
    return np.fft.rfftfreq(n, d=dt), np.moveaxis(X, -1, axis)


def periodogram(x, dt=1.0, n=None, window=True, axis=-1):
    """Frequencies and power |X_k|**2 of real samples; arguments as for `fourier`.

    Examples
    --------
    >>> x = np.cos(2 * np.pi * np.arange(16) / 4)     # period of 4 samples
    >>> f, power = periodogram(x, window=False)
    >>> float(f[np.argmax(power)])
    0.25
    """
    f, X = fourier(x, dt, n, window, axis)
    return f, np.abs(X) ** 2


def dominant_period(f, power, shortest, longest, axis=-1):
    """Period 1 / f of the largest power among periods from shortest to longest.

    Examples
    --------
    >>> f = np.array([0.0, 0.1, 0.2, 0.3])
    >>> float(dominant_period(f, np.array([9.0, 1.0, 3.0, 2.0]), 3, 20))
    5.0
    """
    power = np.moveaxis(power, axis, -1)
    band = (f >= 1 / longest) & (f <= 1 / shortest)
    return 1 / f[np.argmax(np.where(band, power, -np.inf), axis=-1)]


def moving_average(x, w, axis=-1):
    """Centred moving average over w samples, computed as a product of FFTs.

    The result equals np.convolve(x, np.ones(w) / w, mode="same") along
    axis: values beyond the two ends of the record count as zero. The
    transforms have length N >= n + w - 1, so that the end of the series does
    not wrap around onto its start.

    Examples
    --------
    >>> moving_average(np.array([3.0, 6.0, 9.0, 12.0]), 3)
    array([3., 6., 9., 7.])
    """
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    n = x.shape[-1]
    N = 2 ** math.ceil(math.log2(n + w - 1))  # room for all n + w - 1 sums
    kernel = np.fft.rfft(np.ones(w) / w, N)
    y = np.fft.irfft(np.fft.rfft(x, N) * kernel, N)
    lo = (w - 1) // 2  # w // 2 samples before the centre, (w - 1) // 2 after
    return np.moveaxis(y[..., lo:lo + n], -1, axis)


def cross_correlation(x, y, axis=-1):
    """Correlation of y against x at every lag, computed by FFT.

    Both series are standardised (mean removed, divided by the standard
    deviation). Then c[tau] is the sum of x[k] * y[k + tau] over the pairs
    inside the record, divided by n. A maximum at tau > 0 means that y
    follows x by tau samples.

    Returns
    -------
    lags : numpy.ndarray of int, shape (2 n - 1,)
        The lags -(n - 1), ..., n - 1.
    c : numpy.ndarray
        The correlation at each lag, along axis.

    Examples
    --------
    >>> lags, c = cross_correlation([1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0])
    >>> int(lags[np.argmax(c)])
    1
    """
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    y = np.moveaxis(np.asarray(y, dtype=float), axis, -1)
    n = x.shape[-1]
    xs = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    ys = (y - y.mean(axis=-1, keepdims=True)) / y.std(axis=-1, keepdims=True)
    N = 2 ** math.ceil(math.log2(2 * n - 1))  # room for all 2n - 1 lags
    c = np.fft.irfft(np.fft.rfft(xs, N).conj() * np.fft.rfft(ys, N), N) / n
    # the circular result holds lags 0 .. n - 1, then the negative lags
    c = np.concatenate([c[..., N - n + 1:], c[..., :n]], axis=-1)
    return np.arange(-(n - 1), n), np.moveaxis(c, -1, axis)
