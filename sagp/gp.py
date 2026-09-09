"""sagp.gp: the four surrogate cells -- kernels, models, NUTS inference and prediction.

The thesis compares four Gaussian-process cells ({additive, product} kernel structure x {amplitude,
lengthscale} sparsity prior) that must differ in *nothing* but the kernel/prior block, so
everything else is literally the vendored SAASBO reference's code path (`saasgp.py`, `saasbo.py`,
`util.py`), imported rather than copied wherever unchanged. All four kernels take `(X, Z, params,
noise, include_noise)` and return an (n, m) matrix, so prediction reaches any cell through
`KERNELS`.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import botorch.settings
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from botorch.models.fully_bayesian import (
    MIN_INFERRED_NOISE_LEVEL,
    PyroModel,
    SaasFullyBayesianSingleTaskGP,
    SaasPyroModel,
    reshape_and_detach,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood
from gpytorch.means import Mean
from jax import Array, jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss
from numpyro.infer import MCMC, NUTS
from scipy.optimize import brentq

import saasgp
from sagp.diagnostics import DiagThresholds, Diagnostics, diagnose
from sagp.kernels_torch import (
    CenteredAdditiveAmplitudeKernel,
    CenteredAdditiveLengthscaleKernel,
    CenteredProductAmplitudeKernel,
)
from util import chunk_vmap

# (structure in {"additive", "product"}, sparsity prior in {"amplitude", "lengthscale"}).
CellKey = tuple[str, str]

# --- kernels ---

# 64-node Gauss-Legendre quadrature on [0,1], the reference measure U[0,1]. Defined here so
# `sagp.gp` never imports the objectives; a test pins it equal to `synthobj.kernel`'s.
_NODES, _WEIGHTS = leggauss(64)
GL_NODES, GL_WEIGHTS = jnp.asarray(0.5 * (_NODES + 1)), jnp.asarray(0.5 * _WEIGHTS)

_ROOT_FIVE = math.sqrt(5.0)


def matern52_1d(r: ArrayLike) -> Array:
    """Unit-variance Matern-5/2 covariance at scaled distance r >= 0: (1 + s + s^2/3) e^-s, s = sqrt(5) r.

    Elementwise. In r rather than r^2, whose sqrt has an infinite derivative at zero distance.
    """
    s = _ROOT_FIVE * jnp.asarray(r)
    return (1.0 + s + s * s / 3.0) * jnp.exp(-s)


def v_of_ell(ell: ArrayLike) -> Array:
    """Average marginal variance of the centered unit-amplitude component under U[0,1].

    v(ell) = 1 - sum_{q,q'} w_q w_q' k(|t_q - t_q'| / ell), reproducing `synthobj.kernel.v`.
    `ell` may be a scalar or a (D,) vector of per-coordinate lengthscales -- both live, `brentq`
    passing a scalar and `kbar_all` a vector -- and the result has the same shape.
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

    Entry [a, b, i] is coordinate i's kernel projected onto the orthogonal complement of the
    constants under U[0,1] (Lu et al. 2022, eq. 8), divided by v(ell_i) when `normalize`. One
    broadcast expression feeding the caller's reduction, so XLA may fuse the (n, m, D) tensor --
    800 MB at 5000 test points, n = 200, D = 100.
    """
    v = v_of_ell(ell_vec)  # (D,)
    k = matern52_1d(jnp.abs(X[:, None, :] - Z) / ell_vec)  # (n, m, D)
    kbar = k - _quad_mean(X, ell_vec)[:, None, :] - _quad_mean(Z, ell_vec) + (1.0 - v)
    return kbar / v if normalize else kbar


def _kbar_diag(X: Array, ell_vec: Array, normalize: bool) -> Array:
    """The diagonal of `kbar_all(X, X, ell_vec, normalize)`, in O(n D Q); X (n, D) -> (n, D)."""
    v = v_of_ell(ell_vec)
    kbar = 1.0 - 2.0 * _quad_mean(X, ell_vec) + (1.0 - v)
    return kbar / v if normalize else kbar


def centered_matern52_1d(x: Array, z: Array, ell: ArrayLike, normalize: bool = False) -> Array:
    """Centered Matern-5/2 kernel of one coordinate; x (n,), z (m,) -> (n, m).

    The D = 1 case of `kbar_all`, delegating to it, so what is tested here is what the cells run.
    """
    return kbar_all(x[:, None], z[:, None], jnp.atleast_1d(ell), normalize)[:, :, 0]


@partial(jit, static_argnums=(4,))
def kernel_additive_amplitude(
    X: Array, Z: Array, params: dict[str, Array], noise: ArrayLike, include_noise: bool
) -> Array:
    """Additive structure, amplitude sparsity: k(x, z) = sum_i a_sq_i kbar_i(x_i, z_i).

    `params`: "a_sq", "kernel_ell" (D,); normalized, so a_sq_i is coordinate i's variance under nu.
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

    `params`: "kernel_var", "kernel_inv_length_sq" (D,) = rho_i, ell_i = rho_i^-0.5. Deliberately
    *not* normalized: sparsity is rho_i -> 0, which drives v(ell_i) and the component to zero, so
    dividing v out would undo the shrinkage.
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

    `params`: "a_sq", "kernel_ell" (D,); a_sq_i = 0 removes coordinate i from every interaction.
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

    `params`: "kernel_var", "kernel_inv_length_sq" (D,). SAASBO itself: `saasgp.matern_kernel`.
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

    No noise or jitter -- prediction adds `noise + 1e-6` -- and never forms the (n, n) matrix.
    """
    return KERNELS[key][1](X, params)


# --- cells ---

# A coordinate is active when it carries more than 2 % of the variance. Restated here rather than
# imported, and pinned equal by test, so labels and prior share one cutoff.
ACTIVE_EPS: float = 0.02

# The reference SAASGP's own sparsity hyperparameter, on its rho scale.
ALPHA_LENGTHSCALE: float = 0.1

# v(ELL_EPS) = ACTIVE_EPS: the lengthscale at which a unit-amplitude centered component's variance
# under U[0,1] falls to the active cutoff, so "share > eps" <=> "ell < ELL_EPS" <=> "rho > RHO_EPS"
# -- the equivalence that makes the two priors' active counts comparable, and the lengthscale
# cells' own active rule. Solved at import so it tracks `v_of_ell`, strictly decreasing
# (v(0.5) = 0.282, v(50) = 5.6e-5), so the bracket holds one root.
ELL_EPS: float = float(brentq(lambda ell: float(v_of_ell(ell)) - ACTIVE_EPS, 0.5, 50.0))
RHO_EPS: float = ELL_EPS**-2.0

# Both priors are the same half-Cauchy scale mixture, theta_i = tausq * lam_i with tausq ~
# HC(alpha) and lam_i ~ HC(1), whose prior-predictive active count #{i : theta_i > c} depends on
# (alpha, c) only through c / alpha. Matching the a^2-count at ACTIVE_EPS to the reference's
# rho-count at RHO_EPS therefore fixes alpha in closed form, and matches the whole count
# distribution, not its median (`test_alpha_matches_reference_count`).
ALPHA_AMPLITUDE: float = ALPHA_LENGTHSCALE * ACTIVE_EPS / RHO_EPS

# LogNormal(mu, sigma) on each coordinate's lengthscale in the amplitude cells, where ell is a
# free shape parameter: median 1 (one wiggle across [0,1]), 95 % interval [0.053, 18.9], only 1 %
# below 0.03 -- discouraging the ell -> 0 corner where a normalized component degenerates into
# white noise and competes with `kernel_noise`. The lengthscale cells have none: there the
# half-Cauchy on rho *is* the lengthscale prior.
ELL_PRIOR: tuple[float, float] = (0.0, 1.5)


class ProductLengthscalePyroModel(SaasPyroModel):
    """SAASBO itself: BoTorch's `SaasPyroModel`, whose `sample` is inherited untouched.

    Two changes only. `alpha` is an attribute rather than `sample_lengthscale`'s default, because
    the study sets it per fit; and postprocessing keeps `kernel_tausq` and `_kernel_inv_length_sq`,
    which BoTorch deletes and the diagnostics and the sparsity readouts both read.
    """

    alpha: float = ALPHA_LENGTHSCALE

    def sample_inv_length_sq(self, dim: int) -> tuple[Array, Array]:
        """The SAAS block: rho_i = tausq * lam_i, tausq ~ HC(alpha), lam_i ~ HC(1) -> (rho, ell).

        BoTorch's sites, names and order, so a trace of this model is a trace of `SaasPyroModel`.
        The lengthscale cells' kernels take rho and BoTorch's GPyTorch modules take ell, so both
        are returned rather than either being recovered from the other.
        """
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(jnp.array(self.alpha)))
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(dim)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)
        lengthscale = numpyro.deterministic("lengthscale", 1.0 / jnp.sqrt(inv_length_sq))
        return inv_length_sq, lengthscale

    def sample_lengthscale(self, dim: int, alpha: float | None = None) -> Array:
        """What the inherited `sample` calls; `alpha` is `self.alpha`, the argument is ignored.

        The argument is kept because BoTorch's signature has it and a caller may pass it by name.
        """
        return self.sample_inv_length_sq(dim)[1]

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Array]) -> dict[str, torch.Tensor]:
        """BoTorch's postprocessing without its two deletions: every site survives, as torch.

        `lengthscale` is recomputed from the retained draws exactly as BoTorch computes it, so the
        kernel this loads is the kernel NUTS sampled.
        """
        inv_length_sq = (
            jnp.expand_dims(mcmc_samples["kernel_tausq"], axis=-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = 1.0 / jnp.sqrt(inv_length_sq)
        return {
            site: torch.tensor(
                np.asarray(draws), dtype=self.train_X.dtype, device=self.train_X.device
            )
            for site, draws in mcmc_samples.items()
        }


class AdditiveLengthscalePyroModel(ProductLengthscalePyroModel):
    """Additive structure under the SAAS prior: the same sites in the same order, other kernel."""

    def sample(self) -> None:
        """`MaternPyroModel.sample`'s body with `kernel_additive_lengthscale` in place of ARD."""
        outputscale = self.sample_outputscale(
            concentration=self._outputscale_prior_concentration,
            rate=self._outputscale_prior_rate,
        )
        mean = self.sample_mean()
        noise = self.sample_noise()
        inv_length_sq, _ = self.sample_inv_length_sq(dim=self.ard_num_dims)
        X = self.train_X_jax
        K_noiseless = kernel_additive_lengthscale(
            X, X, {"kernel_var": outputscale, "kernel_inv_length_sq": inv_length_sq}, 0.0, False
        )
        self.sample_observations(mean=mean, K_noiseless=K_noiseless, noise=noise)

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, torch.Tensor]
    ) -> tuple[Mean, Kernel, Likelihood, None]:
        """The retained draws as a batched GPyTorch model: (mean, covariance, likelihood, None).

        The trailing `None` is BoTorch's input-warping transform, which no cell here uses.
        """
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        batch_shape = torch.Size([len(mcmc_samples["mean"])])
        mean_module = self._build_mean_module(
            mcmc_samples=mcmc_samples, batch_shape=batch_shape, **tkwargs
        )
        covar_module = ScaleKernel(
            CenteredAdditiveLengthscaleKernel(
                ard_num_dims=self.ard_num_dims, batch_shape=batch_shape
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale, new_value=mcmc_samples["outputscale"]
        )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale, new_value=mcmc_samples["lengthscale"]
        )
        likelihood = _likelihood_from_samples(self, mcmc_samples, batch_shape, **tkwargs)
        return mean_module, covar_module, likelihood, None


class AdditiveAmplitudePyroModel(PyroModel):
    """Additive structure with the sparsity on the amplitudes: a_sq_i = tausq * lam_i.

    The lengthscale cells' half-Cauchy scale mixture applied to the components' variances, at
    `alpha = ALPHA_AMPLITUDE` for the same prior-predictive active count. Components being
    normalized, ell gets a LogNormal prior of its own, and there is no outputscale: it would be
    unidentifiable against tausq. The mean and noise priors are BoTorch's, so the four cells differ
    in nothing but their kernel/prior block.
    """

    alpha: float = ALPHA_AMPLITUDE
    ell_prior: tuple[float, float] = ELL_PRIOR
    _jax_kernel = staticmethod(kernel_additive_amplitude)
    _torch_kernel = CenteredAdditiveAmplitudeKernel

    def sample_amplitudes(self, dim: int) -> Array:
        """a_sq_i = tausq * lam_i, tausq ~ HC(alpha), lam_i ~ HC(1); the deterministic (dim,)."""
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(jnp.array(self.alpha)))
        a_sq = numpyro.sample("_a_sq", dist.HalfCauchy(jnp.ones(dim)))
        return numpyro.deterministic("a_sq", tausq * a_sq)

    def sample_ell(self, dim: int) -> Array:
        """ell_i ~ LogNormal(ell_prior), the amplitude cells' only lengthscale prior; (dim,)."""
        return numpyro.sample(
            "kernel_ell",
            dist.LogNormal(jnp.full(dim, self.ell_prior[0]), self.ell_prior[1]),
        )

    def sample(self) -> None:
        """Sites in order: mean, noise (when it is learned), the amplitudes, ell, then Y."""
        mean = self.sample_mean()
        noise = self.sample_noise()
        a_sq = self.sample_amplitudes(dim=self.ard_num_dims)
        ell = self.sample_ell(dim=self.ard_num_dims)
        X = self.train_X_jax
        K_noiseless = self._jax_kernel(X, X, {"a_sq": a_sq, "kernel_ell": ell}, 0.0, False)
        self.sample_observations(mean=mean, K_noiseless=K_noiseless, noise=noise)

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Array]) -> dict[str, torch.Tensor]:
        """Every site as torch: nothing dropped, and `a_sq` is a draw rather than recomputed."""
        return {
            site: torch.tensor(
                np.asarray(draws), dtype=self.train_X.dtype, device=self.train_X.device
            )
            for site, draws in mcmc_samples.items()
        }

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs) -> dict[str, torch.Tensor]:
        """Ones with the keys and shapes `load_mcmc_samples` reads; BoTorch loads a state dict."""
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "a_sq": torch.ones(num_mcmc_samples, self.ard_num_dims, **tkwargs),
            "kernel_ell": torch.ones(num_mcmc_samples, self.ard_num_dims, **tkwargs),
        }
        return self._common_dummy_samples(mcmc_samples, num_mcmc_samples, **tkwargs)

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, torch.Tensor]
    ) -> tuple[Mean, Kernel, Likelihood, None]:
        """The retained draws as a batched GPyTorch model: (mean, covariance, likelihood, None).

        The amplitudes are the kernel's own parameter, so there is no `ScaleKernel` to wrap it in.
        """
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        batch_shape = torch.Size([len(mcmc_samples["mean"])])
        mean_module = self._build_mean_module(
            mcmc_samples=mcmc_samples, batch_shape=batch_shape, **tkwargs
        )
        covar_module = self._torch_kernel(
            ard_num_dims=self.ard_num_dims, batch_shape=batch_shape
        ).to(**tkwargs)
        covar_module.a_sq = reshape_and_detach(
            target=covar_module.a_sq, new_value=mcmc_samples["a_sq"]
        )
        covar_module.lengthscale = reshape_and_detach(
            target=covar_module.lengthscale, new_value=mcmc_samples["kernel_ell"]
        )
        likelihood = _likelihood_from_samples(self, mcmc_samples, batch_shape, **tkwargs)
        return mean_module, covar_module, likelihood, None


class ProductAmplitudePyroModel(AdditiveAmplitudePyroModel):
    """Product structure with amplitude sparsity: `AdditiveAmplitudePyroModel`'s other kernel."""

    _jax_kernel = staticmethod(kernel_product_amplitude)
    _torch_kernel = CenteredProductAmplitudeKernel


def _likelihood_from_samples(
    pyro_model: PyroModel,
    mcmc_samples: dict[str, torch.Tensor],
    batch_shape: torch.Size,
    **tkwargs,
) -> Likelihood:
    """BoTorch's likelihood block (`fully_bayesian.py:553-569`), shared by the cells written here.

    A fixed `train_Yvar` expanded to (S, n), else a learned noise floored at
    `MIN_INFERRED_NOISE_LEVEL` -- the floor `sample_noise` already added to the sampled value.
    """
    if pyro_model.train_Yvar is not None:
        return FixedNoiseGaussianLikelihood(
            # Reshape to shape ``num_mcmc_samples x N``
            noise=pyro_model.train_Yvar.squeeze(-1).expand(
                batch_shape[0], len(pyro_model.train_Yvar)
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
    likelihood = GaussianLikelihood(
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
    ).to(**tkwargs)
    likelihood.noise_covar.noise = reshape_and_detach(
        target=likelihood.noise_covar.noise,
        new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
    )
    return likelihood


def model_product_lengthscale(
    X: Array, Y: Array, *, alpha: float, fixed_noise: float | None
) -> None:
    """SAASBO itself: ARD Matern-5/2 under the SAAS prior rho_i = tausq * lam_i, tausq ~ HC(alpha).

    The vendored `SAASGP.model` body verbatim, its `self.*` hyperparameters replaced by the
    arguments and its sites kept in order; `test_product_lengthscale_log_joint_matches_reference`
    pins the copy to the reference's log joint.
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

    Identical to `model_product_lengthscale` site for site, so only the kernel structure differs.
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

    The lengthscale cells' half-Cauchy scale mixture applied to the components' variances, at
    `alpha = ALPHA_AMPLITUDE` for the same prior-predictive active count. Components being
    normalized, shrinking a_sq_i to zero removes the coordinate rather than flattening it, ell gets
    a prior of its own, and there is no `kernel_var`: it would be unidentifiable against tausq.
    """
    N, P = X.shape

    noise = (
        numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        if fixed_noise is None
        else fixed_noise
    )
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(alpha))

    # As in the reference: the deterministic reparameterization gives NUTS a geometry it can move
    # in, sampling a_sq directly having tausq's scale baked into every step.
    a_sq = numpyro.sample("_a_sq", dist.HalfCauchy(jnp.ones(P)))
    a_sq = numpyro.deterministic("a_sq", tausq * a_sq)
    ell = numpyro.sample("kernel_ell", dist.LogNormal(jnp.full(P, ell_prior[0]), ell_prior[1]))

    k = kernel_additive_amplitude(X, X, {"a_sq": a_sq, "kernel_ell": ell}, noise, True)
    numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)


def model_product_amplitude(
    X: Array, Y: Array, *, alpha: float, fixed_noise: float | None, ell_prior: tuple[float, float]
) -> None:
    """Product structure with amplitude sparsity: `model_additive_amplitude` with the other kernel.

    Identical to `model_additive_amplitude` site for site, so only the kernel structure differs.
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

    One `Cell` per surrogate lets `fit`, `posterior` and `experiments.identify` be written once,
    branch-free.
    """

    structure: str  # "additive" or "product"
    prior: str  # "amplitude" or "lengthscale"
    model: Callable[..., None]  # bind alpha/fixed_noise (and ell_prior) before handing to NUTS
    pyro_model: type[PyroModel]  # the same cell for BoTorch; `CellGP` instantiates it per fit
    kernel: Callable[..., Array]  # (X, Z, params, noise, include_noise) -> (n, m)
    kernel_diag: Callable[..., Array]  # (X, params) -> (n,), without noise or jitter
    native_site: str  # the site the cell's own sparsity lives on: what its active rule thresholds
    sites: tuple[str, ...]  # every site to retain, sampled and deterministic, in declared order
    alpha_default: float  # calibrated so all four cells' priors have the same active count

    @property
    def key(self) -> CellKey:
        """(structure, prior): this cell's key in `CELLS` and in `KERNELS`."""
        return (self.structure, self.prior)


# The four cells of the 2x2. `sites` lists `kernel_noise` unconditionally; a fit with
# `fixed_noise` set drops it, since then it is not a site at all.
CELLS: dict[CellKey, Cell] = {
    cell.key: cell
    for cell in (
        Cell(
            structure="additive",
            prior="amplitude",
            model=model_additive_amplitude,
            pyro_model=AdditiveAmplitudePyroModel,
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
            pyro_model=AdditiveLengthscalePyroModel,
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
            pyro_model=ProductAmplitudePyroModel,
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
            pyro_model=ProductLengthscalePyroModel,
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

    `SAASGP`'s defaults but `max_tree_depth` 6, the reference driver's; 16 draws are retained.
    """

    num_warmup: int = 512
    num_samples: int = 256
    thinning: int = 16
    max_tree_depth: int = 6
    num_chains: int = 1

    def __post_init__(self) -> None:
        """Reject a budget whose damage would only show up later, in a fit or in a readout.

        A non-positive count reaches NumPyro as an empty chain and fails hours into a run; a
        `num_samples` that `thinning` does not divide retains `ceil(num_samples / thinning)` draws,
        not the 16 assumed elsewhere.
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


class CellGP(SaasFullyBayesianSingleTaskGP):
    """BoTorch's SAAS GP with a cell's `PyroModel` bound per instance.

    `SaasFullyBayesianSingleTaskGP` reads the `PyroModel` to build off a class attribute, so a
    subclass per cell would be four of them; setting that attribute on the instance, before
    `__init__` reads it, lets this one class serve all four. Input-scaling validation is off
    because the study standardizes with ddof 0 while BoTorch's check uses ddof 1, which would warn
    on every fit with n < 52.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor | None = None,
        *,
        cell: Cell,
        alpha: float | None = None,
        ell_prior: tuple[float, float] = ELL_PRIOR,
    ) -> None:
        """(n, D) inputs in [0,1]^D and (n, 1) targets already standardized and negated.

        `train_Yvar` None learns the noise. `alpha` None takes the cell's calibrated default, and
        `ell_prior` reaches the amplitude cells only.
        """
        self._pyro_model_class = cell.pyro_model
        with botorch.settings.validate_input_scaling(False):
            super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.cell = cell
        self.pyro_model.alpha = cell.alpha_default if alpha is None else alpha
        self.pyro_model.ell_prior = ell_prior


# Row-chunking policy for `FittedGP.posterior`, identical for every cell -- a per-cell rule would
# be a confound. Two kernels materialize an (n_test, n, D) broadcast XLA declines to fuse, and
# `cell_kernel_diag` an (n_test, Q, D) tensor whose size does not fall with n, so the threshold is
# compared against `n_test * (n + Q) * D_used`. Above 2**24 such elements `posterior` evaluates
# X_test in row blocks, kernel and diagonal in the same call; a test point's mean and variance
# depend on no other test point, so blocking changes no modelling quantity.
_CHUNK_THRESHOLD: int = 2**24
_CHUNK_ROWS: int = 256

# The two `FittedGP.cell` strings that are not a `CellKey`: the MAP references, on the reference
# ARD Matern-5/2 under the lengthscale cells' parameter names.
_MAP_REFERENCES: frozenset[str] = frozenset({"dsp_map", "oracle_S"})


def _chunk_size(S: int) -> int:
    """`util.chunk_vmap`'s batch for S retained samples: the largest divisor of S that is <= 8.

    `util.get_chunks` builds a ragged final chunk with an `np.arange` in a module that never
    imports numpy, so a remainder raises `NameError` inside a prediction; a divisor of S keeps the
    reference's 8 where it divides. The samples are independent, so chunking changes nothing
    predicted.
    """
    return max(c for c in range(1, min(8, S) + 1) if S % c == 0)


class FittedGP:
    """One cell's posterior on standardized, negated targets: what `fit` returns.

    The loop (`sagp.bo`) and the readouts (`sagp.readouts`) consume it, and `references.fit_map`
    builds one for each MAP reference. The training data keeps the reference's names (`X_train`,
    `Y_train`) so `saasbo.optimize_ei`'s incumbent lookup runs against this object unchanged;
    `samples` holds the retained constrained draws, one entry per site of `cell.sites` present in
    the trace, leading dimension S.
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
        # Cholesky factors of the S training kernel matrices, (S, n, n), filled on the first
        # prediction and kept as the reference keeps its `Ls`: the acquisition optimizer scores
        # thousands of candidates per fit and must not refactorize.
        self._Ls: Array | None = None

    def _is_map_reference(self) -> bool:
        """True when `cell` names one of the MAP references; raises on any other string.

        Without it an unknown cell name would be served the lengthscale cells' kernel silently.
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
        """This cell's (kernel, diagonal) pair out of `KERNELS`."""
        if self._is_map_reference():
            return KERNELS[("product", "lengthscale")]
        return KERNELS[self.cell]

    def param_sites(self) -> tuple[str, ...]:
        """The names of this cell's kernel parameters, in the order prediction passes them around.

        `util.chunk_vmap` indexes a *tuple*, so parameters travel positionally in this order.
        """
        if self._is_map_reference() or self.cell[1] == "lengthscale":
            return ("kernel_var", "kernel_inv_length_sq")
        return ("a_sq", "kernel_ell")

    def params(self, s: int) -> dict[str, Array]:
        """Retained sample `s`'s kernel parameters, in the form its kernel and diagonal take."""
        return {site: self.samples[site][s] for site in self.param_sites()}

    def noises(self) -> Array:
        """(S,): the observation variance carried by each retained sample.

        `kernel_noise` when learned, else `fixed_noise` repeated. The reference splits the two --
        `compute_choleskys` uses `observation_variance`, `posterior` hard-codes 1e-6 -- and we use
        the configured value in both, agreeing at the 1e-6 the bit-for-bit test pins.
        """
        if "kernel_noise" in self.samples:
            return self.samples["kernel_noise"]
        return self.fixed_noise * jnp.ones(self.samples[self.param_sites()[0]].shape[0])

    def columns(self, X: Array) -> Array:
        """`X` restricted to `active`, the coordinates the oracle reference was given."""
        return X if self.active is None else X[:, self.active]

    def _compute_choleskys(self, chunk_size: int | None = None) -> None:
        """`SAASGP.compute_choleskys` generalized to the cell; fills the cache `self._Ls`.

        `chunk_size` defaults to `_chunk_size(S)`, keeping `util.get_chunks` off its `np`
        NameError.
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

        The reference's four lines in its order, for one sample, minus its unused `rng_key`. The
        one generalization is the prior diagonal, `diag_fn(X_test, params) + noise + 1e-6`, which
        for ("product", "lengthscale") is the reference's scalar exactly.
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
        values to the last bit. Above `_CHUNK_THRESHOLD` broadcast elements the test points go in
        blocks of `_CHUNK_ROWS`.
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
        """(S, n): K^-1 y per retained sample, the weights `_predict`'s mean contracts."""
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

    The sampler call is the reference's line for line, so a fit driven by the key `SAASGP.fit`
    derives reproduces its draws bit for bit. Three changes: `extra_fields` collects the counters
    the reference discards, `key` replaces its `PRNGKey(seed)` split, and `summary` moves to
    `diagnose`.
    """
    if nuts.num_chains != 1:
        raise ValueError(
            f"num_chains must be 1; got {nuts.num_chains}. One chain per fit is the reference's "
            "setting, and diagnose pools with group_by_chain=False, so more chains would be "
            "flattened into one and their split-R-hat would compare the halves of the "
            "concatenation."
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

    Attempt 0 uses `key` *unchanged*, which is what reproduces the reference bit for bit; the
    refit uses `fold_in(key, 1)`, disjoint from it and from the loop's exception retry, which uses
    `fold_in(key, 2)` (`sagp.bo._fit_with_retry`). That refit is a *fresh* chain with twice the
    warm-up, giving "ok", "refit" or "excluded" -- whose draws still come back, because the loop
    has to keep querying.

    `alpha` None takes the cell's calibrated default: `ALPHA_LENGTHSCALE` on the rho scale, or
    `ALPHA_AMPLITUDE` on the a^2 scale. `fixed_noise` None learns `kernel_noise ~ LogNormal(0, 10)`
    while a positive value fixes the observation variance at it; 0.0 is rejected. `nuts` is the
    sampler budget (`NUTSConfig()`), `thresholds` the gate each attempt is judged against
    (`DiagThresholds()`), and `ell_prior` reaches the amplitude cells only.
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
        # verdict decided the status is the one whose draws prediction gets.
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

# `standardize` stays beside `fit` because it is `fit`'s precondition; `sagp.bo.run_bo` and
# `experiments.identify` both need it.


def standardize(y: ArrayLike) -> tuple[np.ndarray, float, float]:
    """Standardize `y` to zero mean, unit variance (ddof 0), and negate it: z = -(y - mean) / std.

    `mean` and `std` are plain floats; the negation is the vendored code's minimization convention.
    """
    y = np.asarray(y)
    mean, std = float(y.mean()), float(y.std())
    return -(y - mean) / std, mean, std
