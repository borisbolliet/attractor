"""Tests of attractor.modes: the randomised eigensolver against eigh, with fixed seeds."""
import numpy as np
import pytest

from attractor.modes import leading_eigh, principal_angles, randomised_eigh


def planted(n, eigenvalues, seed=0):
    """Symmetric, with these eigenvalues and random orthonormal eigenvectors."""
    U, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return (U * eigenvalues) @ U.T


# Eigenvalues 1, 1/2, 1/4, ...: the 10th and 11th largest differ by 2**-10.
LAM = 0.5 ** np.arange(300)


def test_leading_eigh_puts_the_largest_first():
    A = planted(6, [4.0, 1.0, 3.0, 2.0, 6.0, 5.0])
    w, U = leading_eigh(A, 3)
    np.testing.assert_allclose(w, [6.0, 5.0, 4.0], rtol=1e-12)
    np.testing.assert_allclose(A @ U, U * w, atol=1e-12)


def test_principal_angles_ignore_signs_and_rotations():
    U, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((50, 3)))
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, -1.0]])   # rotate, flip
    assert not np.allclose(U, U @ R)
    np.testing.assert_allclose(principal_angles(U, U @ R), 0.0, atol=1e-14)


def test_principal_angles_agree_with_scipy():
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(3)
    U, _ = np.linalg.qr(rng.standard_normal((40, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((40, 4)))
    np.testing.assert_allclose(principal_angles(U, V),
                               np.sort(linalg.subspace_angles(U, V)), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_randomised_matches_eigh(seed):
    """Rounding: eps ||A|| / gap = 2.2e-16 / 2**-10 = 2.3e-13 rad."""
    A = planted(300, LAM)
    w, U = randomised_eigh(A, k=10, q=2, seed=seed)
    w_ref, U_ref = leading_eigh(A, 10)
    assert principal_angles(U, U_ref).max() < 2.3e-11
    np.testing.assert_allclose(w, w_ref, rtol=1e-12)


def test_float32_keeps_the_subspace():
    """In float32: eps ||A|| / gap = 1.2e-7 / 2**-10 = 1.2e-4 rad."""
    A = planted(300, LAM)
    _, U = randomised_eigh(A.astype(np.float32), k=10, q=2, seed=0)
    _, U_ref = leading_eigh(A, 10)
    assert principal_angles(U.astype(np.float64), U_ref).max() < 1.2e-2
