"""Tests of the integrators, before Lecture 15."""
import math

from attractor.integrators import euler_step, integrate, rk4_step


def test_constant_rhs():
    assert euler_step(lambda t, y: 2.0, 0.0, 1.0, 0.5) == 2.0
    assert rk4_step(lambda t, y: 2.0, 0.0, 1.0, 0.5) == 2.0


def test_shapes():
    t, y = integrate(rk4_step, lambda t, y: -y, [1.0, 2.0], 0.0, 1.0, 0.1)
    assert t.shape == (11,)
    assert y.shape == (11, 2)


def test_decay():
    t, y = integrate(rk4_step, lambda t, y: -y, [1.0], 0.0, 1.0, 0.1)
    assert math.isclose(y[-1, 0], math.exp(-1), rel_tol=1e-3)
