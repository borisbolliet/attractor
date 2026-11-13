"""Tests of the Gray-Scott module: the vectorised stencil against loops, and its properties."""
import numpy as np
import pytest

from attractor.patterns import initial, laplacian, step


def laplacian_loops(a):
    """The reference: the Laplacian of lecture-17-start, one grid point at a time."""
    n0, n1 = a.shape
    out = np.empty_like(a)
    for i in range(n0):
        for j in range(n1):
            out[i, j] = (a[(i - 1) % n0, j] + a[(i + 1) % n0, j]
                         + a[i, (j - 1) % n1] + a[i, (j + 1) % n1] - 4 * a[i, j])
    return out


def test_laplacian_matches_loops():
    """Same sums in the same order: equal to rounding. A 5 x 7 grid tells the two axes apart."""
    a = np.random.default_rng(1).random((5, 7))
    np.testing.assert_allclose(laplacian(a), laplacian_loops(a), rtol=0, atol=1e-15)


def test_laplacian_of_a_constant_is_zero():
    assert np.all(laplacian(np.full((4, 6), 2.5)) == 0.0)


def test_laplacian_sums_to_zero():
    """Periodic boundaries: diffusion moves material around the grid but creates none."""
    a = np.random.default_rng(2).random((8, 8))
    assert laplacian(a).sum() == pytest.approx(0.0, abs=1e-12)


def test_batch_matches_one_grid_at_a_time():
    """A step of two grids with F, k of shape (2, 1, 1) equals two separate steps."""
    u, v = initial(16, batch=(2,))
    F, k = np.array([0.030, 0.040]), np.array([0.062, 0.060])
    ub, vb = step(u, v, F[:, None, None], k[:, None, None])
    for i in range(2):
        ui, vi = step(u[i], v[i], F[i], k[i])
        np.testing.assert_array_equal(ub[i], ui)
        np.testing.assert_array_equal(vb[i], vi)


def test_uniform_state_is_steady():
    """u = 1, v = 0 is a steady state: no reaction, and the Laplacian of a constant is 0."""
    u, v = np.ones((8, 8)), np.zeros((8, 8))
    u1, v1 = step(u, v, 0.030, 0.062)
    assert np.all(u1 == 1.0) and np.all(v1 == 0.0)


def test_unit_impulse_has_five_point_stencil():
    a = np.zeros((3, 3))
    a[1, 1] = 1
    np.testing.assert_array_equal(laplacian(a), [[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def test_initial_patch_is_nonnegative_and_reproducible():
    u, v = initial(32, seed=7)
    u_again, v_again = initial(32, seed=7)
    np.testing.assert_array_equal(u, u_again)
    np.testing.assert_array_equal(v, v_again)
    assert u.min() >= 0 and v.min() >= 0
    assert np.all(u[:8] == 1) and np.all(v[:8] == 0)
    assert np.all((u[8:24, 8:24] >= .49) & (u[8:24, 8:24] <= .51))
    assert np.all((v[8:24, 8:24] >= .24) & (v[8:24, 8:24] <= .26))


def test_checkerboard_at_diffusion_limit_changes_sign_without_decay():
    i, j = np.indices((8, 8))
    a = (-1.0) ** (i + j)
    np.testing.assert_array_equal(laplacian(a), -8 * a)
    # D * dt = 1/4 at the unit-grid Euler limit.
    np.testing.assert_array_equal(a + 0.25 * laplacian(a), -a)
