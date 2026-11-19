"""Unit tests of the integrators."""
import logging
import math

import numpy as np
import pytest

from attractor.integrators import RK4, Euler, euler_step, integrate, rk4_factor, rk4_step


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


def test_integrate_logs_its_steps(caplog):
    with caplog.at_level(logging.DEBUG, logger="attractor.integrators"):
        integrate(rk4_step, lambda t, y: -y, [1.0], 0.0, 1.0, 0.1)
    assert caplog.messages == ["10 steps of size 0.1 from t = 0"]


def rk4_factor_powers(z):
    """The reference: the polynomial written with powers, as at lecture-19-start."""
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def test_rk4_factor_matches_the_powers():
    rng = np.random.default_rng(4)
    z = rng.uniform(-4, 1, 1000) + 1j * rng.uniform(-3, 3, 1000)
    np.testing.assert_allclose(rk4_factor(z), rk4_factor_powers(z), rtol=1e-13)


def test_rk4_factor_equals_one_step_for_a_linear_equation():
    """R(h*lam) is the factor of one rk4_step of dy/dt = lam*y."""
    lam, h = -3.0, 0.5
    assert rk4_step(lambda t, y: lam * y, 0.0, 1.0, h) == pytest.approx(rk4_factor(h * lam), rel=1e-15)


def test_rk4_stability_interval_ends_at_minus_2_785():
    assert abs(rk4_factor(-2.78)) < 1 < abs(rk4_factor(-2.79))
    assert np.all(np.abs(rk4_factor(np.linspace(-2.785, 0, 10001))) <= 1)
