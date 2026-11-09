"""Fixtures shared by the test files."""
import numpy as np
import pytest


@pytest.fixture(params=["decay", "forced"])
def problem(request):
    """An equation dy/dt = rhs(t, y) with y(0) = 1, and its exact solution at t = 1.

    The fixture has two values, so every test that uses it runs twice.
    "forced" depends on t explicitly; "decay" does not.
    """
    if request.param == "decay":
        return (lambda t, y: -y), np.exp(-1.0)
    return (lambda t, y: np.cos(t) * y), np.exp(np.sin(1.0))
