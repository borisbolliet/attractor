"""Gray-Scott reaction-diffusion on a periodic n x n grid, written with loops.

u and v are the concentrations of two chemicals. The grid spacing is 1 and the
boundaries are periodic: the right neighbour of the last column is the first.
"""
import numpy as np

SPOTS = (0.030, 0.062)    # (F, k) of a pattern of spots
STRIPES = (0.040, 0.060)  # (F, k) of a pattern of stripes


def laplacian(a):
    """Five-point Laplacian of the 2-D array a, with periodic boundaries.

    Examples
    --------
    >>> a = np.zeros((3, 3)); a[1, 1] = 1.0
    >>> print(laplacian(a))
    [[ 0.  1.  0.]
     [ 1. -4.  1.]
     [ 0.  1.  0.]]
    """
    n0, n1 = a.shape
    out = np.empty_like(a)
    for i in range(n0):
        for j in range(n1):
            out[i, j] = (a[(i - 1) % n0, j] + a[(i + 1) % n0, j]
                         + a[i, (j - 1) % n1] + a[i, (j + 1) % n1] - 4 * a[i, j])
    return out


def initial(n, seed=0):
    """u = 1 and v = 0, except a central square of u = 0.5 and v = 0.25, with bounded noise in that square."""
    rng = np.random.default_rng(seed)
    u, v = np.ones((n, n)), np.zeros((n, n))
    c = slice(n // 2 - 8, n // 2 + 8)
    u[c, c], v[c, c] = 0.50, 0.25
    u[c, c] += rng.uniform(-0.01, 0.01, u[c, c].shape)
    v[c, c] += rng.uniform(-0.01, 0.01, v[c, c].shape)
    return u, v


def step(u, v, F, k, Du=0.16, Dv=0.08, dt=1.0):
    """One forward Euler step of the Gray-Scott equations; returns new arrays."""
    uvv = u * v * v
    return (u + dt * (Du * laplacian(u) - uvv + F * (1 - u)),
            v + dt * (Dv * laplacian(v) + uvv - (F + k) * v))


def run(F, k, n=256, steps=10_000, dt=1.0, seed=0):
    """u and v after `steps` steps from `initial(n, seed)`."""
    u, v = initial(n, seed)
    for _ in range(steps):
        u, v = step(u, v, F, k, dt=dt)
    return u, v
