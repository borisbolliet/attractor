"""Leading eigenvectors of a large symmetric matrix, exact and randomised.

The rows of an (n, p) data matrix X are n samples of p values. Here each row is
a snapshot of one Gray-Scott run: the grid of v at one time, flattened. The
n x n matrix A = Xc Xc^T / p, where Xc is X minus its mean row, holds the inner
products of all pairs of centred snapshots. The leading eigenvectors of A give
the time dependence of the dominant modes of the pattern (method of snapshots).
"""
import numpy as np

from .patterns import initial, step


def snapshots(F, k, count=4096, every=2, grid=128, seed=0):
    """The field v of one Gray-Scott run, recorded every `every` steps.

    Returns an array of shape (count, grid * grid). Row i holds v after
    (i + 1) * every steps, flattened in row-major order.

    Examples
    --------
    >>> snapshots(0.030, 0.062, count=3, every=1, grid=16).shape
    (3, 256)
    """
    u, v = initial(grid, seed)
    X = np.empty((count, grid * grid))
    for i in range(count):
        for _ in range(every):
            u, v = step(u, v, F, k)
        X[i] = v.ravel()
    return X


def gram(X):
    """The (n, n) matrix Xc Xc^T / p of the (n, p) array X, where Xc is X minus its mean row.

    Examples
    --------
    >>> gram(np.array([[1.0, 2.0], [3.0, 2.0]]))
    array([[ 0.5, -0.5],
           [-0.5,  0.5]])
    """
    Xc = X - X.mean(axis=0)
    return Xc @ Xc.T / X.shape[1]


def leading_eigh(A, k):
    """The k largest eigenvalues of the symmetric matrix A, largest first, and their eigenvectors.

    np.linalg.eigh returns the eigenvalues in ascending order, with the
    eigenvector of w[i] in column U[:, i]. Reversing both puts the largest first.

    Examples
    --------
    >>> w, U = leading_eigh(np.array([[2.0, 1.0], [1.0, 2.0]]), 1)
    >>> w
    array([3.])
    >>> np.abs(U).round(4)
    array([[0.7071],
           [0.7071]])
    """
    w, U = np.linalg.eigh(A)           # ascending order
    return w[::-1][:k], U[:, ::-1][:, :k]


def range_finder(A, l, q, rng):
    """An (n, l) matrix Q with orthonormal columns that nearly span the leading eigenvectors of A.

    The columns of the Gaussian sketch A @ Omega are random combinations of the
    columns of A. Each of the q power iterations multiplies by A twice. A QR
    factorisation after every product keeps the columns orthonormal.
    A is symmetric positive-semidefinite, float32 or float64; Omega has the same dtype.
    """
    Omega = rng.standard_normal((A.shape[0], l), dtype=A.dtype)
    Q, _ = np.linalg.qr(A @ Omega)     # the sketch, orthonormalised
    for _ in range(2 * q):
        Q, _ = np.linalg.qr(A @ Q)     # orthonormalise after every product
    return Q


def randomised_eigh(A, k, l=None, q=2, seed=0):
    """Approximate the k largest eigenpairs of a symmetric positive-semidefinite matrix A.

    The range finder gives Q, of shape (n, l), with l = k + 10 by default. The
    eigenvectors V of the small l x l matrix Q^T A Q give the approximate
    eigenvectors Q V of A: the Rayleigh-Ritz method. With q power iterations,
    the method multiplies A by an (n, l) matrix 2 (q + 1) times.
    `seed` is an int (restarts the draw) or a Generator (advances its state).
    """
    l = k + 10 if l is None else l
    Q = range_finder(A, l, q, np.random.default_rng(seed))
    w, V = leading_eigh(Q.T @ (A @ Q), k)    # the l x l eigenproblem
    return w, Q @ V


def rsvd(A, k, q, Omega):
    """Approximate k leading eigenvectors of a positive-semidefinite matrix.

    A has shape (n, n); the supplied Gaussian sketch Omega has shape (n, l),
    with k <= l <= n. Perform q pairs of power products, with QR between
    pairs, then an SVD of Q.T @ A. Return an (n, k) orthonormal basis.
    This is the NumPy counterpart of the coursework recurrence.
    """
    Q, _ = np.linalg.qr(A @ Omega)
    for _ in range(q):
        Q, _ = np.linalg.qr(A @ (A @ Q))
    Uh, _, _ = np.linalg.svd(Q.T @ A, full_matrices=False)
    return Q @ Uh[:, :k]


def principal_angles(U, V):
    """The principal angles between the column spans of U and V, in radians, smallest first.

    U and V have orthonormal columns and the same shape (n, k). The cosines of
    the angles are the singular values of U^T V. Their sines are the singular
    values of V - U U^T V, which resolve small angles more accurately.

    Examples
    --------
    >>> U = np.eye(3)[:, :2]
    >>> principal_angles(U, -U[:, ::-1])         # other signs and order, same plane
    array([0., 0.])
    >>> v = np.array([[0.5], [np.sqrt(3) / 2], [0.0]])
    >>> np.degrees(principal_angles(U[:, :1], v)).round(6)
    array([60.])
    """
    M = U.T @ V
    cos = np.linalg.svd(M, compute_uv=False)                  # largest first
    sin = np.linalg.svd(V - U @ M, compute_uv=False)[::-1]    # smallest first
    return np.where(cos**2 < 0.5, np.arccos(np.minimum(cos, 1.0)),
                    np.arcsin(np.minimum(sin, 1.0)))
