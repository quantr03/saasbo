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
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array, jit
from jax.typing import ArrayLike
from scipy.optimize import brentq

import saasgp
from synthobj.kernel import GL_NODES as _GL_NODES_NUMPY
from synthobj.kernel import GL_WEIGHTS as _GL_WEIGHTS_NUMPY
from synthobj.objective import ACTIVE_EPS

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


# --- cells ---

# Prior calibration, plan decisions D2 and D3. `ACTIVE_EPS` -- a coordinate is active when it
# carries more than 2 % of the variance -- is imported from `synthobj.objective` rather than
# restated, so the objectives' labels and the surrogate's prior are calibrated against one cutoff
# and cannot drift apart.

# The reference SAASGP's own sparsity hyperparameter, on its rho scale.
ALPHA_LENGTHSCALE: float = 0.1

# v(ELL_EPS) = ACTIVE_EPS: the lengthscale at which a unit-amplitude centered component's variance
# under U[0,1] falls to the active cutoff, so "share > eps" <=> "ell < ELL_EPS" <=> "rho > RHO_EPS".
# That equivalence is what makes the two priors' *active counts* comparable below, and it is also
# the lengthscale cells' native active rule. Solved at import rather than hardcoded so it tracks
# `v_of_ell` instead of silently disagreeing with it: v is strictly decreasing (v(0.5) = 0.282,
# v(50) = 5.6e-5), so the bracket contains exactly one root, and brentq costs a few dozen 64x64
# quadrature sums.
ELL_EPS: float = float(brentq(lambda ell: float(v_of_ell(ell)) - ACTIVE_EPS, 0.5, 50.0))
RHO_EPS: float = ELL_EPS**-2.0

# Both priors are the same half-Cauchy scale mixture, theta_i = tausq * lam_i with tausq ~
# HC(alpha) and lam_i ~ HC(1), whose prior-predictive active count #{i : theta_i > c} depends on
# (alpha, c) only through c / alpha. Matching the a^2-count at c = ACTIVE_EPS to the reference's
# rho-count at c = RHO_EPS therefore fixes alpha in closed form -- and matches the whole count
# distribution, not merely its median (plan D2, checked by `test_alpha_matches_reference_count`).
ALPHA_AMPLITUDE: float = ALPHA_LENGTHSCALE * ACTIVE_EPS / RHO_EPS

# LogNormal(mu, sigma) on each coordinate's lengthscale in the amplitude cells, where the
# amplitude carries the sparsity and ell is left a free shape parameter: median 1 (one wiggle
# across [0,1]), 95 % interval [0.053, 18.9]. That covers the study families' 0.06-3 without being
# tuned to them, and puts only 1 % below 0.03, discouraging the ell -> 0 corner where a normalized
# component degenerates into white noise and competes with `kernel_noise` (plan D3). The
# lengthscale cells have no ell prior at all: there the half-Cauchy on rho *is* the lengthscale
# prior, and adding a second one would change the model being compared.
ELL_PRIOR: tuple[float, float] = (0.0, 1.5)


def model_product_lengthscale(
    X: Array, Y: Array, *, alpha: float, fixed_noise: float | None
) -> None:
    """SAASBO itself: ARD Matern-5/2 under the SAAS prior rho_i = tausq * lam_i, tausq ~ HC(alpha).

    The vendored `SAASGP.model` body verbatim, with `self.alpha` -> `alpha`, `self.learn_noise` ->
    `fixed_noise is None`, `self.observation_variance` -> `fixed_noise`, and `self.kernel` -> this
    cell's kernel, which delegates to the very `saasgp.matern_kernel` the reference calls. Copied
    rather than imported only because the plan needs a module-level model carrying these site
    names in this order; `test_product_lengthscale_log_joint_matches_reference` pins the copy to
    the reference class's log joint so the copy cannot drift.

    Sites in order: `kernel_var` ~ LogNormal(0, 10); `kernel_noise` ~ LogNormal(0, 10), present
    only when the noise is learned (`fixed_noise is None`); `kernel_tausq` ~ HalfCauchy(alpha), the
    global shrinkage; `_kernel_inv_length_sq` ~ HalfCauchy(1) per coordinate, the local shrinkage;
    and the deterministic `kernel_inv_length_sq` = tausq * `_kernel_inv_length_sq`.
    """
    N, P = X.shape

    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = (
        numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        if fixed_noise is None
        else fixed_noise
    )
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(alpha))

    # note we use deterministic to reparameterize the geometry
    inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
    inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)

    k = kernel_product_lengthscale(
        X, X, {"kernel_var": var, "kernel_inv_length_sq": inv_length_sq}, noise, True
    )
    numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)


def model_additive_lengthscale(
    X: Array, Y: Array, *, alpha: float, fixed_noise: float | None
) -> None:
    """Additive structure under the SAAS prior: `model_product_lengthscale` with the other kernel.

    Identical to `model_product_lengthscale` site for site and prior for prior -- same names, same
    order, same distributions -- so that a difference between this cell and SAASBO is attributable
    to the kernel's structure alone. Sites: see `model_product_lengthscale`.
    """
    N, P = X.shape

    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = (
        numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        if fixed_noise is None
        else fixed_noise
    )
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(alpha))

    inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
    inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)

    k = kernel_additive_lengthscale(
        X, X, {"kernel_var": var, "kernel_inv_length_sq": inv_length_sq}, noise, True
    )
    numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)


def model_additive_amplitude(
    X: Array, Y: Array, *, alpha: float, fixed_noise: float | None, ell_prior: tuple[float, float]
) -> None:
    """Additive structure with the sparsity moved onto the amplitudes: a_sq_i = tausq * lam_i.

    The same half-Cauchy scale mixture as the lengthscale cells, applied to the components'
    variances instead of their inverse squared lengthscales -- with `alpha = ALPHA_AMPLITUDE` the
    two induce the same prior-predictive active count (plan D2). Because every component is
    normalized, a_sq_i is coordinate i's variance under the reference measure whatever its
    lengthscale, so shrinking a_sq_i to zero removes the coordinate outright rather than merely
    flattening it; the lengthscale is then a free shape parameter and gets its own prior (D3).

    Sites in order: `kernel_noise` ~ LogNormal(0, 10) when learned; `kernel_tausq` ~
    HalfCauchy(alpha); `_a_sq` ~ HalfCauchy(1) per coordinate; the deterministic `a_sq` = tausq *
    `_a_sq`; and `kernel_ell` ~ LogNormal(*ell_prior) per coordinate. There is no `kernel_var`
    site: `a_sq` already carries every component's scale, and a global factor on top of it would
    be unidentifiable against tausq.
    """
    N, P = X.shape

    noise = (
        numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        if fixed_noise is None
        else fixed_noise
    )
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(alpha))

    # As in the reference: the deterministic reparameterization is what gives NUTS a geometry it
    # can move in, since sampling a_sq directly would have tausq's scale baked into every step.
    a_sq = numpyro.sample("_a_sq", dist.HalfCauchy(jnp.ones(P)))
    a_sq = numpyro.deterministic("a_sq", tausq * a_sq)
    ell = numpyro.sample("kernel_ell", dist.LogNormal(jnp.full(P, ell_prior[0]), ell_prior[1]))

    k = kernel_additive_amplitude(X, X, {"a_sq": a_sq, "kernel_ell": ell}, noise, True)
    numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)


def model_product_amplitude(
    X: Array, Y: Array, *, alpha: float, fixed_noise: float | None, ell_prior: tuple[float, float]
) -> None:
    """Product structure with amplitude sparsity: `model_additive_amplitude` with the other kernel.

    Identical to `model_additive_amplitude` site for site and prior for prior, so that a
    difference between the two amplitude cells is attributable to the kernel's structure alone.
    Sites: see `model_additive_amplitude`.
    """
    N, P = X.shape

    noise = (
        numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        if fixed_noise is None
        else fixed_noise
    )
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(alpha))

    a_sq = numpyro.sample("_a_sq", dist.HalfCauchy(jnp.ones(P)))
    a_sq = numpyro.deterministic("a_sq", tausq * a_sq)
    ell = numpyro.sample("kernel_ell", dist.LogNormal(jnp.full(P, ell_prior[0]), ell_prior[1]))

    k = kernel_product_amplitude(X, X, {"a_sq": a_sq, "kernel_ell": ell}, noise, True)
    numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)


@dataclass(frozen=True)
class Cell:
    """One of the four surrogates: everything inference, prediction and the readouts need of it.

    Bundling the model with its kernel, its sites and its alpha is what lets `fit`, `posterior`
    and `identify` be written once against `Cell` and be literally the same code for all four --
    the design's central requirement, since any per-cell branch downstream would be a confound.
    """

    structure: str  # "additive" or "product"
    prior: str  # "amplitude" or "lengthscale"
    model: Callable[..., None]  # bind alpha/fixed_noise (and ell_prior) before handing to NUTS
    kernel: Callable[..., Array]  # (X, Z, params, noise, include_noise) -> (n, m)
    kernel_diag: Callable[..., Array]  # (X, params) -> (n,), without noise or jitter
    native_site: str  # the site the cell's own sparsity lives on: what its active rule thresholds
    sites: tuple[str, ...]  # every site to retain, sampled and deterministic, in declared order
    alpha_default: float  # calibrated so all four cells' priors have the same active count

    @property
    def key(self) -> CellKey:
        """(structure, prior): this cell's key in `CELLS` and in `KERNELS`."""
        return (self.structure, self.prior)


# The four cells of the 2x2. `sites` includes `kernel_noise` unconditionally; a fit with
# `fixed_noise` set drops it, because then it is not a site at all.
CELLS: dict[CellKey, Cell] = {
    cell.key: cell
    for cell in (
        Cell(
            structure="additive",
            prior="amplitude",
            model=model_additive_amplitude,
            kernel=kernel_additive_amplitude,
            kernel_diag=_diag_additive_amplitude,
            native_site="a_sq",
            sites=("kernel_noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"),
            alpha_default=ALPHA_AMPLITUDE,
        ),
        Cell(
            structure="additive",
            prior="lengthscale",
            model=model_additive_lengthscale,
            kernel=kernel_additive_lengthscale,
            kernel_diag=_diag_additive_lengthscale,
            native_site="kernel_inv_length_sq",
            sites=(
                "kernel_var",
                "kernel_noise",
                "kernel_tausq",
                "_kernel_inv_length_sq",
                "kernel_inv_length_sq",
            ),
            alpha_default=ALPHA_LENGTHSCALE,
        ),
        Cell(
            structure="product",
            prior="amplitude",
            model=model_product_amplitude,
            kernel=kernel_product_amplitude,
            kernel_diag=_diag_product_amplitude,
            native_site="a_sq",
            sites=("kernel_noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"),
            alpha_default=ALPHA_AMPLITUDE,
        ),
        Cell(
            structure="product",
            prior="lengthscale",
            model=model_product_lengthscale,
            kernel=kernel_product_lengthscale,
            kernel_diag=_diag_product_lengthscale,
            native_site="kernel_inv_length_sq",
            sites=(
                "kernel_var",
                "kernel_noise",
                "kernel_tausq",
                "_kernel_inv_length_sq",
                "kernel_inv_length_sq",
            ),
            alpha_default=ALPHA_LENGTHSCALE,
        ),
    )
}
