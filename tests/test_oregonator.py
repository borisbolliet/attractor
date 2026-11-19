"""Tests of the Oregonator: the model, its Jacobian, the blow-up of RK4 and the stiff solvers."""
import logging

import numpy as np
import pytest

from attractor import RK4, Adaptive, Oregonator

Y0 = [1.0, 2.0, 3.0]


def steady_state(m):
    """The positive steady state: r = a, e = f*a/(1 + a), and a root of q a^2 + (f + q - 1) a - (f + 1)."""
    b = m.f + m.q - 1
    a = (-b + np.sqrt(b * b + 4 * m.q * (m.f + 1))) / (2 * m.q)
    return np.array([a, m.f * a / (1 + a), a])


def test_rhs_vanishes_at_the_steady_state():
    m = Oregonator()
    np.testing.assert_allclose(m.rhs(0.0, steady_state(m)), 0.0, atol=1e-9)


@pytest.mark.parametrize("y", [Y0, [1.1e5, 0.01, 1500.0], [1.0, 1500.0, 400.0]])
def test_jacobian_matches_central_differences(y):
    """Column j of the Jacobian against (rhs(y + d e_j) - rhs(y - d e_j)) / (2 d)."""
    m, y = Oregonator(), np.array(y)
    J = m.jacobian(0.0, y)
    for j in range(3):
        d = 1e-6 * abs(y[j])
        dy = np.zeros(3)
        dy[j] = d
        column = (m.rhs(0.0, y + dy) - m.rhs(0.0, y - dy)) / (2 * d)
        np.testing.assert_allclose(J[:, j], column, rtol=1e-6, atol=1e-9 * np.abs(J).max())


def test_batch_of_states_matches_one_state_at_a_time():
    m = Oregonator()
    Y = np.random.default_rng(3).uniform(1.0, 100.0, (3, 5))
    J, F = m.jacobian(0.0, Y), m.rhs(0.0, Y)
    for k in range(5):
        np.testing.assert_array_equal(J[k], m.jacobian(0.0, Y[:, k]))
        np.testing.assert_array_equal(F[:, k], m.rhs(0.0, Y[:, k]))


def test_rk4_blow_up_is_logged(caplog):
    with np.errstate(over="ignore", invalid="ignore"), caplog.at_level(logging.WARNING):
        tr = RK4(0.01).run(Oregonator(), Y0, 0.0, 30.0)
    assert not np.isfinite(tr.y[-1]).any()
    assert "state not finite from step 2045 of 3000, t = 20.45" in caplog.text


def test_floating_point_errors_can_raise():
    with np.errstate(over="raise", invalid="raise"), pytest.raises(FloatingPointError):
        RK4(0.01).run(Oregonator(), Y0, 0.0, 30.0)


@pytest.mark.parametrize("method", ["Radau", "BDF", "LSODA"])
def test_stiff_solvers_agree(method):
    pytest.importorskip("scipy")
    reference = Adaptive("Radau", rtol=1e-10, atol=1e-10).run(Oregonator(), Y0, 0.0, 30.0)
    solver = Adaptive(method)
    tr = solver.run(Oregonator(), Y0, 0.0, 30.0)
    np.testing.assert_allclose(tr.y[-1], reference.y[-1], rtol=1e-4)
    assert solver.stats["steps"] < 1000 and solver.stats["njev"] > 0


def test_period_is_about_303():
    pytest.importorskip("scipy")
    tr = Adaptive().run(Oregonator(), Y0, 0.0, 1000.0)
    assert tr.period(column=0, min_gap=50.0) == pytest.approx(302.86, abs=0.05)
