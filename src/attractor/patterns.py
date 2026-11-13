"""Gray-Scott reaction-diffusion on periodic n x n grids, written with whole-array operations.

u and v are the concentrations of two chemicals. The grid spacing is 1 and the
boundaries are periodic: the right neighbour of the last column is the first.
The last two axes of an array are the grid; any axes before them index a batch
of grids, one per pair of parameters (F, k).
"""
import numpy as np

SPOTS = (0.030, 0.062)    # (F, k) of a pattern of spots
STRIPES = (0.040, 0.060)  # (F, k) of a pattern of stripes


def laplacian(a):
    """Five-point Laplacian over the last two axes of a, with periodic boundaries.

    Parameters
    ----------
    a : numpy.ndarray
        Array of shape (..., n0, n1); the axes before the last two are a batch.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``a``.

    Examples
    --------
    >>> a = np.zeros((3, 3)); a[1, 1] = 1.0
    >>> print(laplacian(a))
    [[ 0.  1.  0.]
     [ 1. -4.  1.]
     [ 0.  1.  0.]]
    """
    pad = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(a, pad, mode="wrap")   # one layer of periodic copies around each grid
    return (p[..., :-2, 1:-1] + p[..., 2:, 1:-1]
            + p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4 * a)


def initial(n, seed=0, batch=()):
    """u = 1 and v = 0, except a central square of u = 0.5 and v = 0.25, with bounded noise in that square.

    ``batch`` is the shape of the batch axes; every grid of the batch starts
    from different noise.
    """
    rng = np.random.default_rng(seed)
    u, v = np.ones(batch + (n, n)), np.zeros(batch + (n, n))
    c = slice(n // 2 - 8, n // 2 + 8)
    u[..., c, c], v[..., c, c] = 0.50, 0.25
    u[..., c, c] += rng.uniform(-0.01, 0.01, u[..., c, c].shape)
    v[..., c, c] += rng.uniform(-0.01, 0.01, v[..., c, c].shape)
    return u, v


def step(u, v, F, k, Du=0.16, Dv=0.08, dt=1.0):
    """One forward Euler step of the Gray-Scott equations; returns new arrays.

    F and k are numbers, or arrays that broadcast against u and v.
    The pure-diffusion Euler bound is max(Du, Dv) * dt <= 1/4.
    Reactions can impose an additional restriction.
    """
    uvv = u * v * v
    return (u + dt * (Du * laplacian(u) - uvv + F * (1 - u)),
            v + dt * (Dv * laplacian(v) + uvv - (F + k) * v))


def run(F, k, n=256, steps=10_000, dt=1.0, seed=0):
    """u and v after `steps` steps, for one pair (F, k) or arrays of them.

    Examples
    --------
    >>> u, v = run([0.030, 0.040], [0.062, 0.060], n=32, steps=10)
    >>> v.shape
    (2, 32, 32)
    """
    F, k = np.asarray(F, dtype=float), np.asarray(k, dtype=float)
    batch = np.broadcast_shapes(F.shape, k.shape)
    u, v = initial(n, seed, batch)
    F, k = F[..., None, None], k[..., None, None]   # one value per grid, broadcast over its n x n points
    for _ in range(steps):
        u, v = step(u, v, F, k, dt=dt)
    return u, v
