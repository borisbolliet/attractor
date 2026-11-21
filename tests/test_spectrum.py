"""Tests of the spectrum module: each FFT result against a direct computation, and known spectra."""
import numpy as np
import pytest

from attractor.spectrum import (cross_correlation, dominant_period, fourier,
                                moving_average, periodogram, uniform)


def test_uniform_rejects_times_out_of_order():
    with pytest.raises(ValueError, match="increase strictly"):
        uniform([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])


def test_uniform_reproduces_a_straight_line():
    """Linear interpolation is exact for a straight line, whatever the sample times."""
    t = np.cumsum(np.random.default_rng(3).uniform(0.5, 1.5, 40))
    tu, xu = uniform(t, 2 * t + 1)
    np.testing.assert_allclose(np.diff(tu), 1.0)
    np.testing.assert_allclose(xu, 2 * tu + 1, rtol=1e-12)


def test_fourier_halves_the_full_transform():
    """For real samples, rfft keeps X_0 .. X_{n//2} of fft; the rest are conjugates."""
    x = np.random.default_rng(4).standard_normal(15)
    f, X = fourier(x, window=False)
    full = np.fft.fft(x - x.mean())
    np.testing.assert_allclose(X, full[:8], atol=1e-12)
    np.testing.assert_allclose(full[1:][::-1], full[1:].conj(), atol=1e-12)
    np.testing.assert_allclose(f, np.arange(8) / 15)


@pytest.mark.parametrize("n", [None, 4096])
def test_period_of_a_sampled_cosine(n):
    """An 8-day cosine sampled daily for 141 days: 17.6 cycles, between two frequencies of the grid.

    Without padding the peak lies on the grid point k = 18, period 141/18 = 7.83 days;
    padding to 4096 samples puts a grid point within 0.02 days of 8.
    """
    x = np.cos(2 * np.pi * np.arange(141) / 8.0)
    period = dominant_period(*periodogram(x, n=n), 3, 20)
    assert period == pytest.approx(141 / 18 if n is None else 8.0, abs=0.02)


def test_periodogram_along_an_axis():
    """Two series in the columns of one array give the spectra of each column."""
    x = np.random.default_rng(5).standard_normal((50, 2))
    f, power = periodogram(x, n=64, axis=0)
    assert power.shape == (33, 2)
    np.testing.assert_allclose(power[:, 1], periodogram(x[:, 1], n=64)[1])


@pytest.mark.parametrize("w", [3, 4, 7, 8])
def test_moving_average_matches_convolve(w):
    """Odd and even w: the slice start (w - 1) // 2 matters for even w only."""
    x = np.random.default_rng(w).standard_normal(50)
    expected = np.convolve(x, np.ones(w) / w, "same")  # direct sums
    np.testing.assert_allclose(moving_average(x, w), expected,
                               rtol=0, atol=1e-12)


def test_moving_average_along_an_axis():
    x = np.random.default_rng(6).standard_normal((40, 3))
    y = moving_average(x, 5, axis=0)
    assert y.shape == (40, 3)
    np.testing.assert_allclose(y[:, 2], moving_average(x[:, 2], 5), atol=1e-12)


def test_moving_average_over_one_period_removes_the_cycle():
    """The average of w samples of a cycle of w samples is zero, away from the ends."""
    x = np.cos(2 * np.pi * np.arange(80) / 8)
    np.testing.assert_allclose(moving_average(x, 8)[8:-8], 0.0, atol=1e-12)


def test_cross_correlation_matches_correlate():
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal(30), rng.standard_normal(30)
    lags, c = cross_correlation(x, y)
    xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    np.testing.assert_array_equal(lags, np.arange(-29, 30))
    np.testing.assert_allclose(c, np.correlate(ys, xs, "full") / 30, atol=1e-12)
    assert c[lags == 0][0] == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_cross_correlation_finds_a_delay():
    """y repeats x three samples later, so the correlation peaks at lag 3."""
    x = np.random.default_rng(8).standard_normal(103)
    lags, c = cross_correlation(x[3:], x[:-3])
    assert lags[np.argmax(c)] == 3
