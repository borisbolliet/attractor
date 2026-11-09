import math
import pytest
from attractor.integrators import integrate, rk4_step


def error(h):
    t, y = integrate(rk4_step, lambda t, y: -y,
                     [1.0], 0.0, 1.0, h)
    return abs(y[-1, 0] - math.exp(-1))


def test_rk4_order():
    order = math.log2(error(0.1) / error(0.05))
    assert order == pytest.approx(4, abs=0.2)
