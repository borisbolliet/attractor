"""Right-hand sides of the predator-prey models, built as closures over their parameters."""
import numpy as np

from tools import REGISTRY, register

# Chemostat scales: algae x in 10^6 cells/ml, rotifers z in animals/ml, rates per day.
# Equilibrium (d/c, a/b) = (1, 10); linearised period 2*pi/sqrt(a*d) = 8.0 days.
CHEMOSTAT = dict(a=0.8, b=0.08, c=0.77, d=0.77)


@register("lotka_volterra")
def make_lotka_volterra(a, b, c, d):
    """Return rhs(t, y) for dx/dt = a x - b x z, dz/dt = c x z - d z."""
    def rhs(t, y):
        x, z = y
        return np.array([a * x - b * x * z, c * x * z - d * z])
    return rhs


@register("rosenzweig_macarthur")
def make_rosenzweig_macarthur(r, K, a, h, e, m):
    """Return rhs(t, y) for logistic prey and a type-II functional response."""
    def rhs(t, y):
        x, z = y
        f = a * x / (1 + a * h * x)  # prey eaten per predator per unit time
        return np.array([r * x * (1 - x / K) - f * z, e * f * z - m * z])
    return rhs
