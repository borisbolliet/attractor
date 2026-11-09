"""Unit tests of the integrators."""
import math

import numpy as np
import pytest

from attractor.integrators import RK4, Euler, euler_step, integrate, rk4_step


def error_at_1(step, problem, h):
    """Absolute error at t = 1 of a fixed-step run of size h."""
    rhs, exact = problem
    t, y = integrate(step, rhs, [1.0], 0.0, 1.0, h)
    return abs(y[-1, 0] - exact)


@pytest.mark.parametrize("step, order", [(euler_step, 1), (rk4_step, 4)])
def test_convergence_order(step, order, problem):
    """Halving h divides the error by about 2**order."""
    ratio = error_at_1(step, problem, 0.1) / error_at_1(step, problem, 0.05)
    assert math.log2(ratio) == pytest.approx(order, abs=0.2)


@pytest.mark.parametrize("step", [euler_step, rk4_step])
def test_constant_rhs(step):
    """dy/dt = 2 is integrated exactly by both methods."""
    assert step(lambda t, y: 2.0, 0.0, 1.0, 0.5) == 2.0


def test_shapes():
    t, y = integrate(rk4_step, lambda t, y: -y, [1.0, 2.0], 0.0, 1.0, 0.1)
    assert t.shape == (11,)
    assert y.shape == (11, 2)


def test_last_time_is_t1():
    t, y = integrate(rk4_step, lambda t, y: -y, [1.0], 0.0, 1.0, 0.1)
    assert t[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(y[-1], np.exp(-1.0), rtol=1e-6)


@pytest.mark.parametrize("method", [Euler, RK4])
@pytest.mark.parametrize("h", [0.0, -0.1, float("nan")])
def test_step_must_be_positive(method, h):
    with pytest.raises(ValueError, match="must be positive"):
        method(h)


def test_step_must_divide_the_interval():
    with pytest.raises(ValueError, match="whole number of steps"):
        integrate(rk4_step, lambda t, y: -y, [1.0], 0.0, 1.0, 0.3)
