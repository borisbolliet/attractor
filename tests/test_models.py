"""Properties of the models: equilibrium and a conserved quantity."""
import numpy as np

from attractor import RK4, LotkaVolterra


def invariant(m, x, z):
    """V = c x - d ln x + b z - a ln z, constant along every Lotka-Volterra orbit."""
    return m.c * x - m.d * np.log(x) + m.b * z - m.a * np.log(z)


def test_equilibrium():
    m = LotkaVolterra()
    np.testing.assert_allclose(m.rhs(0.0, (m.d / m.c, m.a / m.b)), 0.0, atol=1e-12)


def test_rk4_conserves_the_invariant():
    m = LotkaVolterra()
    tr = RK4(0.05).run(m, (1.5, 8.0), 0.0, 40.0)
    v = invariant(m, tr.y[:, 0], tr.y[:, 1])
    assert np.ptp(v) < 1e-6 * abs(v[0])
