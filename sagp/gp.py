"""sagp.gp: the four surrogate cells -- kernels, models, NUTS inference and prediction.

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
measure U[0,1] at a 64-node Gauss-Legendre grid (Lu et al. 2022, eq. 8) -- the same 64-node rule
`synthobj.kernel` uses, pinned equal by test rather than imported, so the objectives and the
surrogate cannot drift apart. The fourth cell *is* the reference's `saasgp.matern_kernel`.

`sagp/__init__.py` is this module's package and therefore runs before its body, so float64 and
the cpu platform are already in force when the arrays below are created.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array, jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss
from numpyro.infer import MCMC, NUTS
from scipy.optimize import brentq

import saasgp
from sagp.diagnostics import DiagThresholds, Diagnostics, diagnose
from util import chunk_vmap

# (structure in {"additive", "product"}, sparsity prior in {"amplitude", "lengthscale"}).
CellKey = tuple[str, str]

# --- kernels ---

# 64-node Gauss-Legendre quadrature on [0,1], realizing the uniform reference measure U[0,1];
# defined here rather than imported from `synthobj.kernel`, so `sagp.gp` never imports the
# objectives -- `test_gp_constants_are_bit_identical_to_synthobjs` pins the two grids equal.
_NODES, _WEIGHTS = leggauss(64)
GL_NODES, GL_WEIGHTS = jnp.asarray(0.5 * (_NODES + 1)), jnp.asarray(0.5 * _WEIGHTS)

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


def kbar_all(X: Array, Z: Array, ell_vec: Array, normalize: bool) -> Array:
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
    """The diagonal of `kbar_all(X, X, ell_vec, normalize)`, in O(n D Q); X (n, D) -> (n, D).

    At zero distance k_i = 1, so k~_i(x_i, x_i) = 1 - 2 m_i(x_i) + c_i.
    """
    v = v_of_ell(ell_vec)
    kbar = 1.0 - 2.0 * _quad_mean(X, ell_vec) + (1.0 - v)
    return kbar / v if normalize else kbar


def centered_matern52_1d(x: Array, z: Array, ell: ArrayLike, normalize: bool = False) -> Array:
    """Centered Matern-5/2 kernel of one coordinate; x (n,), z (m,) -> (n, m).

    k~(x, z) = k(|x - z| / ell) - m(x) - m(z) + c, divided by v(ell) = 1 - c when `normalize`.
    This is the D = 1 case of `kbar_all`, and delegates to it, so the properties the tests
    assert here (zero quadrature integral, unit quadrature-mean variance after normalization)
    are properties of the code the cells below actually run.
    """
    return kbar_all(x[:, None], z[:, None], jnp.atleast_1d(ell), normalize)[:, :, 0]


@partial(jit, static_argnums=(4,))
def kernel_additive_amplitude(
    X: Array, Z: Array, params: dict[str, Array], noise: ArrayLike, include_noise: bool
) -> Array:
    """Additive structure, amplitude sparsity: k(x, z) = sum_i a_sq_i kbar_i(x_i, z_i).

    `params`: "a_sq" (D,), "kernel_ell" (D,). Every component is normalized, so
    E_nu[kbar_i(x, x)] = 1 and a_sq_i is component i's variance under the reference measure
    whatever its lengthscale -- the property that makes an amplitude threshold well defined.
    """
    kbar = kbar_all(X, Z, params["kernel_ell"], True)
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
    k = params["kernel_var"] * jnp.sum(kbar_all(X, Z, ell, False), axis=-1)
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
    kbar = kbar_all(X, Z, params["kernel_ell"], True)
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
# carries more than 2 % of the variance -- is restated here rather than imported from
# `synthobj.objective`, so `sagp.gp` never imports the objectives;
# `test_gp_constants_are_bit_identical_to_synthobjs` pins the two constants equal, so the
# objectives' labels and the surrogate's prior stay calibrated against one cutoff.
ACTIVE_EPS: float = 0.02

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
    and `experiments.identify` be written once against `Cell` and be literally the same code for
    all four -- the design's central requirement, since any per-cell branch downstream would be a
    confound.
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


# --- inference ---


@dataclass(frozen=True)
class NUTSConfig:
    """The reference's sampler settings, shared by every cell so the budget is never a confound.

    `SAASGP`'s own defaults except `max_tree_depth`, which is the 6 the reference's driver passes
    rather than the class's 7. `num_samples // thinning` = 16 draws are retained for prediction;
    diagnostics run on all `num_samples` un-thinned draws, as the reference's `summary` call does.
    """

    num_warmup: int = 512
    num_samples: int = 256
    thinning: int = 16
    max_tree_depth: int = 6
    num_chains: int = 1

    def __post_init__(self) -> None:
        """Reject a budget whose damage would only show up later, in a fit or in a readout.

        A non-positive count reaches NumPyro as an empty chain and fails somewhere inside it,
        hours into a run; a `num_samples` that `thinning` does not divide silently retains
        `ceil(num_samples / thinning)` draws instead of the `num_samples // thinning` every cost
        estimate and every "16 retained draws" claim is written against. `replace` in `fit`'s
        refit path changes `num_warmup` alone, so a config that passes here stays valid there.
        """
        for name in ("num_warmup", "num_samples", "thinning", "max_tree_depth", "num_chains"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"NUTSConfig.{name} must be a positive int; got {value!r}")
        if self.num_samples % self.thinning != 0:
            raise ValueError(
                f"NUTSConfig: num_samples {self.num_samples} is not a multiple of thinning "
                f"{self.thinning}, so the retained count would not be num_samples // thinning"
            )


# Row-chunking policy for `FittedGP.posterior` (plan rulings R19, R21), identical for every cell.
# Two of the four kernels materialize an (n_test, n, D) broadcast that XLA declines to fuse --
# Task 1 measured 420 MB (additive/lengthscale) and 775 MB (product/amplitude) peaks at
# 5000 x 200 x 100 -- and `cell_kernel_diag` builds an unfused (n_test, Q, D) tensor of its own
# (461 MB jitted at 5000 x 100) whose size does not fall with n. The threshold is therefore
# compared against `n_test * (n + Q) * D_used`, the two tensors' shapes added, so that the
# quadrature axis counts even where n is small: at n = 20, D = 100 and 5000 candidates the
# kernel's tensor is only 1e7 elements while the quadrature one is 256 MB per sample on its own.
# Above 2**24 of those elements `posterior` evaluates X_test in row blocks, kernel and diagonal
# together in the same call; the per-block peak is then `chunk_vmap`'s 8-sample batch times one
# block's tensors, about 430 MB at n = 200, D = 100. Blocking the test axis can change no
# modelling quantity -- each test point's mean and variance depend on no other test point -- so
# the most it can move is the last ulp, and it is deliberately out of reach of the bit-for-bit
# comparison against the reference (D <= 5, n <= 30), which stays on the single-call path. The
# two numbers are cell-independent on purpose: a per-cell rule would be a confound.
_CHUNK_THRESHOLD: int = 2**24
_CHUNK_ROWS: int = 256

# The two `FittedGP.cell` strings that are not a `CellKey`: Task 7's MAP references, which carry
# the lengthscale cells' parameter names on the reference ARD Matern-5/2.
_MAP_REFERENCES: frozenset[str] = frozenset({"dsp_map", "oracle_S"})


def _chunk_size(S: int) -> int:
    """`util.chunk_vmap`'s batch for S retained samples: the largest divisor of S that is <= 8.

    8 is the reference's own chunk, but `util.get_chunks` builds its ragged final chunk with an
    `np.arange` in a module that never imports numpy, so any S with `S % chunk_size != 0` and
    `S > chunk_size` raises `NameError` deep inside a prediction. Taking a *divisor* keeps the
    reference's batch wherever it already divides (16 -> 8, 8 -> 8, 1 -> 1, the study's own
    counts) and never hands `get_chunks` a remainder anywhere else (12 -> 6, 22 -> 2, 9 -> 3),
    without touching the vendored file. Chunking changes no predicted quantity -- the samples are
    independent of each other -- so a different batch is a different memory peak and nothing more.
    """
    return max(c for c in range(1, min(8, S) + 1) if S % c == 0)


class FittedGP:
    """One cell's posterior on standardized, negated targets: what `fit` returns and the loop
    (`sagp.bo`) and the readouts (`sagp.readouts`) consume.

    The training data is kept under the reference's own attribute names (`X_train`, `Y_train`) so
    that `saasbo.optimize_ei`'s incumbent lookup runs against this object unchanged. `samples`
    holds the retained constrained draws, one entry per site of `cell.sites` present in the trace
    (`kernel_noise` is absent when the noise was fixed), each of leading dimension S. `attempts`
    carries one `Diagnostics` per NUTS attempt, so a run's log can count excluded fits rather
    than averaging them in silently; its length is the row's `nuts_attempts`. One attribute is not
    set here: `fit_map` attaches a `map_result` dict to the objects it builds, since a MAP fit's
    quality lives in the optimizer's outcome and not in `attempts`; read it with `getattr`.
    """

    def __init__(
        self,
        cell: CellKey | str,
        X_train: ArrayLike,
        Y_train: ArrayLike,
        samples: dict[str, Array],
        fixed_noise: float | None,
        active: np.ndarray | None,
        status: str,
        status_reason: str,
        attempts: tuple[Diagnostics, ...],
    ) -> None:
        self.cell = cell  # a CellKey, or "dsp_map"/"oracle_S" for the MAP references
        self.X_train = jnp.asarray(X_train, dtype=jnp.float64)  # (n, D)
        self.Y_train = jnp.asarray(Y_train, dtype=jnp.float64)  # (n,)
        self.samples = {site: jnp.asarray(draws) for site, draws in samples.items()}
        self.fixed_noise = fixed_noise  # None when kernel_noise was learned
        self.active = active  # the coordinates the oracle reference was given, else None
        self.status = status  # "ok", "refit" or "excluded"
        self.status_reason = status_reason
        self.attempts = tuple(attempts)
        # Cholesky factors of the S training kernel matrices, (S, n, n). Filled on the first
        # prediction and kept, as the reference keeps its `Ls`: the acquisition optimizer
        # scores thousands of candidates against one fit and must not refactorize per call.
        self._Ls: Array | None = None

    def _is_map_reference(self) -> bool:
        """True when `cell` names one of Task 7's MAP references; raises on any other string.

        Both `_kernel` and `param_sites` fall back to the lengthscale cells' kernel and parameter
        names for those two, so without this check a mistyped cell name would be served the
        product/lengthscale cell silently -- a wrong surrogate reported under the wrong label.
        """
        if isinstance(self.cell, tuple):
            return False
        if self.cell not in _MAP_REFERENCES:
            raise ValueError(
                f"unknown cell {self.cell!r}: expected a CellKey in CELLS or one of "
                f"{sorted(_MAP_REFERENCES)}"
            )
        return True

    def _kernel(self) -> tuple[Callable[..., Array], Callable[..., Array]]:
        """This cell's (kernel, diagonal) pair out of `KERNELS`.

        Task 7's MAP references (`cell` = "dsp_map" or "oracle_S") are the reference ARD
        Matern-5/2 carrying the lengthscale cells' parameter names, so they predict through
        ("product", "lengthscale")'s entry. Prediction then has one code path for all six values
        of `cell`, and what distinguishes the oracle is `active` alone.
        """
        if self._is_map_reference():
            return KERNELS[("product", "lengthscale")]
        return KERNELS[self.cell]

    def param_sites(self) -> tuple[str, ...]:
        """The names of this cell's kernel parameters, in the order prediction passes them around.

        `util.chunk_vmap` indexes a *tuple* of arrays, so the per-sample parameters cannot travel
        through it as a dict: they go positionally in this order and are rebuilt into the dict the
        kernels take inside the vmapped function. The MAP references share the lengthscale cells'
        names for the reason given in `_kernel`.
        """
        if self._is_map_reference() or self.cell[1] == "lengthscale":
            return ("kernel_var", "kernel_inv_length_sq")
        return ("a_sq", "kernel_ell")

    def params(self, s: int) -> dict[str, Array]:
        """Retained sample `s`'s kernel parameters, in the form its kernel and diagonal take."""
        return {site: self.samples[site][s] for site in self.param_sites()}

    def noises(self) -> Array:
        """(S,): the observation variance carried by each retained sample.

        `kernel_noise` when it was learned, else `fixed_noise` repeated. The reference splits
        these two: `compute_choleskys` uses `observation_variance` while `posterior` hard-codes
        1e-6 in the predictive variance. We use the configured value in both places, which agrees
        with the reference exactly at 1e-6 -- the only fixed variance this study uses, and the one
        the bit-for-bit test pins.
        """
        if "kernel_noise" in self.samples:
            return self.samples["kernel_noise"]
        return self.fixed_noise * jnp.ones(self.samples[self.param_sites()[0]].shape[0])

    def columns(self, X: Array) -> Array:
        """`X` restricted to `active`, the coordinates Task 7's oracle reference was given.

        `X_train` itself stays full-D -- `saasbo.optimize_ei`'s incumbent lookup and the run log
        read it, and candidates arrive full-D -- so the restriction happens here, at every kernel
        and diagonal evaluation, rather than the callers having to track which width they hold.
        """
        return X if self.active is None else X[:, self.active]

    def _compute_choleskys(self, chunk_size: int | None = None) -> None:
        """`SAASGP.compute_choleskys` generalized to the cell; fills the cache `self._Ls`.

        The reference's body with `self.kernel(X, X, var, inv_length_sq, noise, True)` replaced by
        this cell's kernel over this cell's parameters. `chunk_size` defaults to `_chunk_size(S)`,
        which keeps `util.get_chunks` off its latent `np` NameError at every S -- 1 for Task 7's
        MAP references, 16 for a cell, and whatever a non-default `--nuts` retains.
        """
        kernel, _ = self._kernel()
        sites = self.param_sites()
        X = self.columns(self.X_train)

        def _cholesky(*sample: Array) -> tuple[Array]:
            # `zip` stops at `sites`, so the trailing noise argument is not taken for a parameter.
            k_XX = kernel(X, X, dict(zip(sites, sample)), sample[-1], True)
            return (cho_factor(k_XX, lower=True)[0],)

        vmap_args = tuple(self.samples[site] for site in sites) + (self.noises(),)
        if chunk_size is None:
            chunk_size = _chunk_size(vmap_args[0].shape[0])
        self._Ls = chunk_vmap(_cholesky, vmap_args, chunk_size=chunk_size)[0]

    def _predict(
        self, X_test: Array, L: Array, params: dict[str, Array], noise: ArrayLike
    ) -> tuple[Array, Array]:
        """`SAASGP.predict` generalized to the cell: mean and noisy predictive variance at X_test.

        The reference's four lines in the reference's order, for one sample, with its unused
        `rng_key` dropped. The one generalization is the prior diagonal: `saasgp.kernel_diag(var,
        noise)` is the scalar `var + noise + 1e-6`, constant in x because the ARD Matern's
        marginal variance is, whereas the three centered kernels' diagonals vary with x. It
        becomes `diag_fn(X_test, params) + noise + 1e-6`, `diag_fn` being the cell's own diagonal
        out of `_kernel`, and for ("product", "lengthscale") that diagonal is `var * ones`;
        multiplying by one is exact and the sum is associated the same way, so this reproduces
        the reference's line bit for bit.

        `X_test` arrives full-D: `active` is applied here, to it and to the training inputs alike.
        """
        kernel, diag_fn = self._kernel()
        X, X_p = self.columns(self.X_train), self.columns(X_test)

        k_pX = kernel(X_p, X, params, noise, False)
        mean = jnp.matmul(k_pX, cho_solve((L, True), self.Y_train))

        k_pp = diag_fn(X_p, params) + noise + 1.0e-6
        L_kXp = solve_triangular(L, jnp.transpose(k_pX), lower=True)
        diag_cov = k_pp - (L_kXp * L_kXp).sum(axis=0)

        return mean, diag_cov

    def posterior(self, X_test: ArrayLike) -> tuple[Array, Array]:
        """Per retained sample, the posterior mean and *noisy* predictive variance at X_test.

        `(S, n_test)` each -- `SAASGP.posterior`'s shapes and, on ("product", "lengthscale"), its
        values to the last bit. The Cholesky factors are computed on the first call and cached,
        as the reference caches its own.

        Above `_CHUNK_THRESHOLD` broadcast elements the test points are evaluated in blocks of
        `_CHUNK_ROWS` rows and concatenated along the test axis; see that constant for why, and
        for why doing so cannot change what is predicted.
        """
        X_test = jnp.asarray(X_test)
        if self._Ls is None:
            self._compute_choleskys()

        sites = self.param_sites()
        vmap_args = tuple(self.samples[site] for site in sites) + (self.noises(), self._Ls)
        chunk_size = _chunk_size(self._Ls.shape[0])

        def _block(X_block: Array) -> tuple[Array, Array]:
            def _one(*sample: Array) -> tuple[Array, Array]:
                # `zip` stops at `sites`; the trailing noise and Cholesky are not parameters.
                return self._predict(X_block, sample[-1], dict(zip(sites, sample)), sample[-2])

            return chunk_vmap(_one, vmap_args, chunk_size=chunk_size)

        n_train, D = self.X_train.shape
        d_used = D if self.active is None else len(self.active)
        if X_test.shape[0] * (n_train + GL_NODES.size) * d_used <= _CHUNK_THRESHOLD:
            return _block(X_test)

        blocks = [
            _block(X_test[start : start + _CHUNK_ROWS])
            for start in range(0, X_test.shape[0], _CHUNK_ROWS)
        ]
        return (
            jnp.concatenate([mean for mean, _ in blocks], axis=1),
            jnp.concatenate([var for _, var in blocks], axis=1),
        )

    def alphas(self) -> Array:
        """(S, n): K^-1 y per retained sample -- the weights `_predict`'s mean contracts k_pX with.

        Task 5's readouts need them on their own (a component mean, an exact Sobol index) and must
        get exactly the vector prediction uses, so they come off the same cached factors rather
        than from a second solve against a freshly built kernel matrix.
        """
        if self._Ls is None:
            self._compute_choleskys()
        return vmap(lambda L: cho_solve((L, True), self.Y_train))(self._Ls)


def _run_nuts(
    model: Callable[..., None],
    X: ArrayLike,
    Y: ArrayLike,
    key: Array,
    nuts: NUTSConfig,
) -> tuple[dict[str, Array], dict[str, Array], float]:
    """`SAASGP.run_inference` copied, with a caller-supplied key, the extra fields and no printing.

    The sampler call is the reference's line for line -- same kernel, same budget, same
    non-progress-bar path -- which is what lets a ("product", "lengthscale") fit driven by the key
    `SAASGP.fit` derives reproduce the reference's draws bit for bit. Three changes, each one the
    plan requires (D1): `extra_fields` collects the divergence and step counters the reference
    discards, `key` replaces the class's own `PRNGKey(seed)` split so the caller owns the seeding,
    and the verbose printing and the `summary` call move to `_diagnose`, which needs the summary
    on the log scale. `X` and `Y` reach `mcmc.run` exactly as the caller passed them, untouched,
    for the same bit-for-bit reason.

    Returns the un-thinned flat samples, the extra fields, and the run's wall-clock seconds.
    """
    if nuts.num_chains != 1:
        raise ValueError(
            f"num_chains must be 1; got {nuts.num_chains}. The plan pins one chain per fit, and "
            "`_diagnose` pools with group_by_chain=False, so more chains would be flattened into "
            "one and their split-R-hat would compare the halves of the concatenation."
        )
    start = time.perf_counter()
    kernel = NUTS(model, max_tree_depth=nuts.max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=nuts.num_warmup,
        num_samples=nuts.num_samples,
        num_chains=nuts.num_chains,
        progress_bar=False,
    )
    mcmc.run(key, X, Y, extra_fields=("diverging", "num_steps"))

    flat_samples = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields()
    return flat_samples, extra, time.perf_counter() - start


def fit(
    X: ArrayLike,
    y: ArrayLike,
    key: Array,
    cell: CellKey | Cell,
    *,
    alpha: float | None = None,
    fixed_noise: float | None = None,
    nuts: NUTSConfig = NUTSConfig(),
    thresholds: DiagThresholds = DiagThresholds(),
    ell_prior: tuple[float, float] = ELL_PRIOR,
) -> FittedGP:
    """NUTS fit of `cell` to (X in [0,1]^D, y already standardized and negated by the caller).

    `key` is used *unchanged* by the first attempt, so a caller handing over the key `SAASGP.fit`
    derives reproduces the reference bit for bit; the refit uses `fold_in(key, 1)`, disjoint from
    it and from the loop's exception-retry key. `alpha` defaults to the cell's calibrated value
    (0.1 on the rho scale, `ALPHA_AMPLITUDE` on the a^2 scale), so the four cells' priors have the
    same prior-predictive active count without the caller having to know which scale it is on.
    `fixed_noise=None` learns `kernel_noise` ~ LogNormal(0, 10); a positive float fixes it.
    `ell_prior` reaches the amplitude cells only -- the lengthscale cells have no ell prior at all.

    On a failed verdict the refit is a *fresh* chain with twice the warm-up and no state reused,
    as the brief's "no warm start" requires: status "ok" if attempt 0 passed, "refit" if attempt 1
    did, "excluded" if neither. An excluded fit still returns the second attempt's draws, because
    the loop has to keep querying; `status`, `status_reason` and `attempts` are what let the run's
    log count those rows instead of averaging them in silently. Diagnostics never raise; an
    exception out of JAX or NumPyro propagates, for `sagp.bo.run_bo`'s failure policy to handle.
    """
    if fixed_noise is not None and fixed_noise == 0.0:
        raise ValueError(
            "fixed_noise=0.0 is ambiguous: the vendored reference spells 'learn the noise' as "
            "observation_variance=0.0, while here it would fix the observation variance at zero. "
            "Pass fixed_noise=None to learn kernel_noise, or a positive variance to fix it."
        )
    if not isinstance(cell, Cell):
        cell = CELLS[cell]
    if alpha is None:
        alpha = cell.alpha_default

    hyperparameters: dict[str, object] = {"alpha": alpha, "fixed_noise": fixed_noise}
    if cell.prior == "amplitude":
        hyperparameters["ell_prior"] = ell_prior
    model = partial(cell.model, **hyperparameters)

    flat, extra, wall_s = _run_nuts(model, X, y, key, nuts)
    attempts = [diagnose(flat, extra, thresholds, wall_s)]

    if not attempts[-1].passed:
        flat, extra, wall_s = _run_nuts(
            model,
            X,
            y,
            jax.random.fold_in(key, 1),
            replace(nuts, num_warmup=2 * nuts.num_warmup),
        )
        attempts.append(diagnose(flat, extra, thresholds, wall_s))

    if not attempts[-1].passed:
        status = "excluded"
    else:
        status = "ok" if len(attempts) == 1 else "refit"

    return FittedGP(
        cell=cell.key,
        X_train=X,
        Y_train=y,
        # `flat` is attempt 0's when it passed and attempt 1's otherwise: the attempt whose
        # verdict decided the status is the attempt whose draws prediction gets.
        samples={site: flat[site][:: nuts.thinning] for site in cell.sites if site in flat},
        fixed_noise=fixed_noise,
        active=None,
        status=status,
        status_reason="; ".join(
            f"attempt {i}: {attempt.reason}"
            for i, attempt in enumerate(attempts)
            if not attempt.passed
        ),
        attempts=tuple(attempts),
    )


# --- standardize ---

# `standardize` stays beside `fit` because it is `fit`'s precondition. Both `sagp.bo.run_bo` and
# `experiments.identify` need it.


def standardize(y: ArrayLike) -> tuple[np.ndarray, float, float]:
    """Standardize `y` to zero mean, unit variance (ddof 0), and negate it: z = -(y - mean) / std.

    The negation turns `synthobj`'s maximization convention into the one the vendored NUTS models
    were written for -- `saasgp.py`/`saasbo.py` minimize -- applied once here rather than at every
    call site. `mean` and `std` come back as plain Python floats, not numpy scalars, since
    `identify` records them verbatim in a plain dict.
    """
    y = np.asarray(y)
    mean, std = float(y.mean()), float(y.std())
    return -(y - mean) / std, mean, std
