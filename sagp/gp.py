"""sagp.gp: the four surrogate cells -- kernels, models, NUTS inference, prediction and readouts.

The thesis compares four Gaussian-process cells ({additive, product} kernel structure x
{amplitude, lengthscale} sparsity prior) that must differ in *nothing* but the kernel/prior
block, so any difference in regret or identification is attributable to the parameterization.
The only way to guarantee "everything else is identical" is to make everything else literally
the vendored SAASBO reference's code path (`saasgp.py`, `saasbo.py`, `util.py`), imported rather
than copied wherever it is unchanged -- hence one module holding all four cells side by side.

This first section holds the kernels. All four have the reference's signature
`(X, Z, params, noise, include_noise)` and return an (n, m) covariance matrix, so prediction can
reach any cell through `KERNELS` without knowing how that cell is parameterized. Three of them
are built from a Matern-5/2 component per coordinate, centered under the uniform reference
measure U[0,1] at the same 64-node Gauss-Legendre grid `synthobj.kernel` uses (Lu et al. 2022,
eq. 8) -- that grid is imported from `synthobj.kernel`, never recomputed, so the objectives and
the surrogate cannot drift apart. The fourth cell *is* the reference's `saasgp.matern_kernel`.

`sagp/__init__.py` is this module's package and therefore runs before its body, so float64 and
the cpu platform are already in force when the arrays below are created.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

import saasgp
from synthobj.kernel import GL_NODES as _GL_NODES_NUMPY
from synthobj.kernel import GL_WEIGHTS as _GL_WEIGHTS_NUMPY

# (structure in {"additive", "product"}, sparsity prior in {"amplitude", "lengthscale"}).
CellKey = tuple[str, str]

# --- kernels ---

# 64-node Gauss-Legendre quadrature on [0,1], realizing the uniform reference measure U[0,1];
# the very grid `synthobj.kernel` centers the synthetic objectives' components against.
GL_NODES = jnp.asarray(_GL_NODES_NUMPY)
GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS_NUMPY)

_ROOT_FIVE = math.sqrt(5.0)


def matern52_1d(r: ArrayLike) -> Array:
    """Unit-variance Matern-5/2 covariance at scaled distance r >= 0: (1 + s + s^2/3) e^-s, s = sqrt(5) r.

    Elementwise, any shape. Written in r rather than r^2 on purpose: the equivalent r^2 form
    needs a sqrt, whose derivative at zero distance is infinite, and that produces NaN gradients
    at repeated inputs -- of which every k(X, X) has a whole diagonal.
    """
    s = _ROOT_FIVE * jnp.asarray(r)
    return (1.0 + s + s * s / 3.0) * jnp.exp(-s)


def v_of_ell(ell: ArrayLike) -> Array:
    """Average marginal variance of the centered unit-amplitude component under U[0,1].

    v(ell) = 1 - c(ell) with c(ell) = sum_{q,q'} w_q w_q' k(|t_q - t_q'| / ell), the double
    quadrature mean of the raw kernel. Reproduces `synthobj.kernel.v`. `ell` may be a scalar or
    a (D,) vector of per-coordinate lengthscales; the result has the same shape.
    """
    ell = jnp.asarray(ell)
    r = jnp.abs(GL_NODES[:, None, None] - GL_NODES[:, None]) / jnp.atleast_1d(ell)  # (Q, Q, D)
    weights = GL_WEIGHTS[:, None] * GL_WEIGHTS  # (Q, Q)
    c = jnp.tensordot(weights, matern52_1d(r), axes=((0, 1), (0, 1)))  # (D,)
    return (1.0 - c).reshape(ell.shape)


def _quad_mean(X: Array, ell_vec: Array) -> Array:
    """m_i(x) = sum_q w_q k(|x - t_q| / ell_i) at every entry of X; X (n, D), ell_vec (D,) -> (n, D)."""
    r = jnp.abs(X[:, None, :] - GL_NODES[:, None]) / ell_vec  # (n, Q, D)
    return jnp.sum(GL_WEIGHTS[:, None] * matern52_1d(r), axis=1)


def _kbar_all(X: Array, Z: Array, ell_vec: Array, normalize: bool) -> Array:
    """Per-coordinate centered Matern-5/2 kernels; X (n, D), Z (m, D), ell_vec (D,) -> (n, m, D).

    Entry [a, b, i] is k~_i(X[a, i], Z[b, i]) = k_i - m_i(X[a, i]) - m_i(Z[b, i]) + c_i, i.e.
    coordinate i's kernel projected onto the orthogonal complement of the constants under U[0,1]
    (Lu et al. 2022, eq. 8), divided by v(ell_i) when `normalize`. Kept as one broadcast
    elementwise expression feeding the caller's reduction, the form `saasgp.matern_kernel`
    itself uses, so that XLA is free to fuse the (n, m, D) tensor into that reduction rather
    than the caller having to materialize it -- the acquisition optimizer scores 5000 test
    points against 200 training points in D = 100, where that tensor would be 800 MB. (Whether
    XLA takes the opportunity is its own cost model's call: measured on the pinned stack at that
    size it fuses for `kernel_additive_amplitude` but not for `kernel_product_amplitude`.)
    """
    v = v_of_ell(ell_vec)  # (D,)
    k = matern52_1d(jnp.abs(X[:, None, :] - Z) / ell_vec)  # (n, m, D)
    kbar = k - _quad_mean(X, ell_vec)[:, None, :] - _quad_mean(Z, ell_vec) + (1.0 - v)
    return kbar / v if normalize else kbar


def _kbar_diag(X: Array, ell_vec: Array, normalize: bool) -> Array:
    """The diagonal of `_kbar_all(X, X, ell_vec, normalize)`, in O(n D Q); X (n, D) -> (n, D).

    At zero distance k_i = 1, so k~_i(x_i, x_i) = 1 - 2 m_i(x_i) + c_i.
    """
    v = v_of_ell(ell_vec)
    kbar = 1.0 - 2.0 * _quad_mean(X, ell_vec) + (1.0 - v)
    return kbar / v if normalize else kbar


def centered_matern52_1d(x: Array, z: Array, ell: ArrayLike, normalize: bool = False) -> Array:
    """Centered Matern-5/2 kernel of one coordinate; x (n,), z (m,) -> (n, m).

    k~(x, z) = k(|x - z| / ell) - m(x) - m(z) + c, divided by v(ell) = 1 - c when `normalize`.
    This is the D = 1 case of `_kbar_all`, and delegates to it, so the properties the tests
    assert here (zero quadrature integral, unit quadrature-mean variance after normalization)
    are properties of the code the cells below actually run.
    """
    return _kbar_all(x[:, None], z[:, None], jnp.atleast_1d(ell), normalize)[:, :, 0]


@partial(jit, static_argnums=(4,))
def kernel_additive_amplitude(
    X: Array, Z: Array, params: dict[str, Array], noise: ArrayLike, include_noise: bool
) -> Array:
    """Additive structure, amplitude sparsity: k(x, z) = sum_i a_sq_i kbar_i(x_i, z_i).

    `params`: "a_sq" (D,), "kernel_ell" (D,). Every component is normalized, so
    E_nu[kbar_i(x, x)] = 1 and a_sq_i is component i's variance under the reference measure
    whatever its lengthscale -- the property that makes an amplitude threshold well defined.
    """
    kbar = _kbar_all(X, Z, params["kernel_ell"], True)
    k = jnp.sum(params["a_sq"] * kbar, axis=-1)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


@partial(jit, static_argnums=(4,))
def kernel_additive_lengthscale(
    X: Array, Z: Array, params: dict[str, Array], noise: ArrayLike, include_noise: bool
) -> Array:
    """Additive structure, lengthscale (SAAS) sparsity: k(x, z) = var * sum_i k~_i(x_i, z_i).

    `params`: "kernel_var" scalar, "kernel_inv_length_sq" (D,) = rho_i, the SAAS prior's native
    scale, with ell_i = rho_i^-0.5. The components are deliberately *not* normalized: sparsity
    here means rho_i -> 0, i.e. ell_i -> infinity, which is exactly what drives v(ell_i) -- and
    with it the component -- to zero, so dividing v out would undo the prior's shrinkage.
    """
    ell = params["kernel_inv_length_sq"] ** -0.5
    k = params["kernel_var"] * jnp.sum(_kbar_all(X, Z, ell, False), axis=-1)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


@partial(jit, static_argnums=(4,))
def kernel_product_amplitude(
    X: Array, Z: Array, params: dict[str, Array], noise: ArrayLike, include_noise: bool
) -> Array:
    """Product structure, amplitude sparsity: k(x, z) = prod_i (1 + a_sq_i kbar_i(x_i, z_i)).

    `params`: "a_sq" (D,), "kernel_ell" (D,). Expanding the product gives 1 plus every
    interaction of the normalized components, so a_sq_i = 0 removes coordinate i from all of
    them at once.
    """
    kbar = _kbar_all(X, Z, params["kernel_ell"], True)
    k = jnp.prod(1.0 + params["a_sq"] * kbar, axis=-1)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


@partial(jit, static_argnums=(4,))
def kernel_product_lengthscale(
    X: Array, Z: Array, params: dict[str, Array], noise: ArrayLike, include_noise: bool
) -> Array:
    """Product structure, lengthscale (SAAS) sparsity: the reference's ARD Matern-5/2 itself.

    `params`: "kernel_var" scalar, "kernel_inv_length_sq" (D,). This cell is SAASBO, so it is
    `saasgp.matern_kernel` -- imported, never copied -- and its noise term is that function's.
    """
    return saasgp.matern_kernel(
        X, Z, params["kernel_var"], params["kernel_inv_length_sq"], noise, include_noise
    )


def _diag_additive_amplitude(X: Array, params: dict[str, Array]) -> Array:
    """sum_i a_sq_i kbar_i(x_i, x_i); X (n, D) -> (n,)."""
    return jnp.sum(params["a_sq"] * _kbar_diag(X, params["kernel_ell"], True), axis=-1)


def _diag_additive_lengthscale(X: Array, params: dict[str, Array]) -> Array:
    """var * sum_i k~_i(x_i, x_i); X (n, D) -> (n,)."""
    ell = params["kernel_inv_length_sq"] ** -0.5
    return params["kernel_var"] * jnp.sum(_kbar_diag(X, ell, False), axis=-1)


def _diag_product_amplitude(X: Array, params: dict[str, Array]) -> Array:
    """prod_i (1 + a_sq_i kbar_i(x_i, x_i)); X (n, D) -> (n,)."""
    return jnp.prod(1.0 + params["a_sq"] * _kbar_diag(X, params["kernel_ell"], True), axis=-1)


def _diag_product_lengthscale(X: Array, params: dict[str, Array]) -> Array:
    """var * ones(n): the ARD Matern-5/2's marginal variance is constant in x."""
    return params["kernel_var"] * jnp.ones(X.shape[-2])


# (kernel, diagonal) per cell: the one place prediction has to look to serve any of the four.
KERNELS: dict[CellKey, tuple[Callable[..., Array], Callable[..., Array]]] = {
    ("additive", "amplitude"): (kernel_additive_amplitude, _diag_additive_amplitude),
    ("additive", "lengthscale"): (kernel_additive_lengthscale, _diag_additive_lengthscale),
    ("product", "amplitude"): (kernel_product_amplitude, _diag_product_amplitude),
    ("product", "lengthscale"): (kernel_product_lengthscale, _diag_product_lengthscale),
}


def cell_kernel_diag(key: CellKey, X: Array, params: dict[str, Array]) -> Array:
    """Prior marginal variance diag k(X, X) of cell `key` at X; X (n, D) -> (n,), O(n D Q).

    Neither noise nor jitter: prediction adds `noise + 1e-6` on top, which for
    ("product", "lengthscale") reproduces the reference's `saasgp.kernel_diag(var, noise)`
    exactly. Never forms the (n, n) matrix -- the acquisition optimizer asks for this at
    thousands of test points at a time, and only that one cell has a diagonal constant in x.
    """
    return KERNELS[key][1](X, params)
