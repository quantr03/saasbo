"""Centered, normalized Matern-5/2 kernel on [0,1] and a cached PSD eigen-factor.

`matern52`, `centered`, `v`, and the 64-node Gauss-Legendre quadrature grid (`GL_NODES`,
`GL_WEIGHTS`) are copied verbatim from
`Research Context/research direction/Kernel normalization check.py`, the thesis's Gate 1
check script; their numbers must reproduce that script bit-for-bit. `centered` projects a
unit-amplitude Matern-5/2 kernel onto the orthogonal complement of constants under the
uniform reference measure U[0,1] (Lu et al. 2022, eq. 8) so each component integrates to
zero; `v` does not center anything itself -- it reports the average marginal variance of
that already-centered kernel. `normalized` rescales by `v(ell)` so that average marginal
variance is exactly 1 regardless of lengthscale.

`eigen_factor` draws an exact PSD square-root factor of `normalized(Z, Z, ell)` via `eigh`
with tiny/negative eigenvalues clipped to zero — no jitter is ever added to a covariance
matrix here (plan decision D3): jitter's white-noise floor would corrupt the slope-energy
labels of smooth components.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

# 64-node Gauss-Legendre quadrature on [0,1], realizing the uniform reference measure U[0,1].
GL_NODES, GL_WEIGHTS = leggauss(64)
GL_NODES, GL_WEIGHTS = 0.5 * (GL_NODES + 1), 0.5 * GL_WEIGHTS


def matern52(r):
    """Unit-variance Matern-5/2 covariance in scaled distance r = |x - y| / ell.

    Copied verbatim from `Research Context/research direction/Kernel normalization check.py`.
    """
    s = np.sqrt(5.0) * r
    return (1 + s + s * s / 3.0) * np.exp(-s)


def centered(X, Y, ell):
    """k_perp(x,y) = k - E_z k(x,z) - E_z k(z,y) + E_{z,z'} k(z,z').

    Projection-form centering of the Matern-5/2 kernel under the reference measure U[0,1]
    (Lu et al. 2022, eq. 8), realized at the 64-node Gauss-Legendre quadrature (GL_NODES,
    GL_WEIGHTS). Copied verbatim from
    `Research Context/research direction/Kernel normalization check.py`.
    """
    K = matern52(np.abs(X[:, None] - Y[None, :]) / ell)
    kx = (matern52(np.abs(X[:, None] - GL_NODES[None, :]) / ell) * GL_WEIGHTS).sum(1)
    ky = (matern52(np.abs(Y[:, None] - GL_NODES[None, :]) / ell) * GL_WEIGHTS).sum(1)
    kk = (GL_WEIGHTS[:, None] * matern52(np.abs(GL_NODES[:, None] - GL_NODES[None, :]) / ell) * GL_WEIGHTS[None, :]).sum()
    return K - kx[:, None] - ky[None, :] + kk


def v(ell):
    """Average marginal variance of the centered unit-amplitude component under U[0,1].

    Copied verbatim from `Research Context/research direction/Kernel normalization check.py`.
    """
    kk = (GL_WEIGHTS[:, None] * matern52(np.abs(GL_NODES[:, None] - GL_NODES[None, :]) / ell) * GL_WEIGHTS[None, :]).sum()
    return 1.0 - kk


def normalized(X: np.ndarray, Y: np.ndarray, ell: float) -> np.ndarray:
    """centered(X, Y, ell) / v(ell); E_nu[k(x,x)] = 1 under U[0,1]."""
    return centered(X, Y, ell) / v(ell)


def make_grid(lo: float, hi: float, n: int) -> np.ndarray:
    """n evenly spaced points on [lo, hi]."""
    return np.linspace(lo, hi, n)


def joint_points(grid: np.ndarray) -> np.ndarray:
    """Concatenate a draw grid with the 64 GL quadrature nodes, nodes last."""
    return np.concatenate([grid, GL_NODES])


_eigen_cache = {}


def eigen_factor(lo: float, hi: float, n: int, ell: float) -> np.ndarray:
    """Cached PSD square-root factor of normalized(Z, Z, ell), Z = joint_points(make_grid(lo, hi, n)).

    Computed via eigh, with eigenvalues below 1e-12 * lambda_max clipped to zero (exact PSD
    sampling without jitter). The returned F satisfies F @ F.T == normalized(Z, Z, ell) up to
    that clip and has shape (n + 64, n + 64): row order matches Z, i.e. the n grid rows come
    first, followed by the 64 quadrature-node rows last -- downstream code (e.g. `Component`,
    which reads the trailing 64 rows as `raw_joint_values[-GL_NODES.size:]`) slices on that
    ordering. Results are cached in a module-level dict keyed by (lo, hi, n, ell); a repeated
    call with the same key returns the identical cached array, so callers must not mutate it.
    """
    key = (lo, hi, n, ell)
    if key in _eigen_cache:
        return _eigen_cache[key]
    Z = joint_points(make_grid(lo, hi, n))
    K = normalized(Z, Z, ell)
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.where(eigvals < 1e-12 * eigvals.max(), 0.0, eigvals)
    F = eigvecs * np.sqrt(eigvals)
    _eigen_cache[key] = F
    return F


def nu_mean(values_at_nodes: np.ndarray) -> float:
    """Quadrature mean Sum_q w_q * value_q of the trailing 64 node values under U[0,1]."""
    return float((GL_WEIGHTS * values_at_nodes).sum())


def nu_var(values_at_nodes: np.ndarray) -> float:
    """Quadrature variance Sum_q w_q * value_q^2 - (Sum_q w_q * value_q)^2 under U[0,1]."""
    return float((GL_WEIGHTS * values_at_nodes**2).sum() - nu_mean(values_at_nodes) ** 2)
