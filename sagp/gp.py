"""sagp.gp: the four surrogate cells -- kernels, models, NUTS inference and prediction.

The thesis compares four Gaussian-process cells ({additive, product} kernel structure x {amplitude,
lengthscale} sparsity prior) that must differ in *nothing* but the kernel/prior block, so every
cell is a BoTorch `PyroModel` differing only there, fitted by BoTorch's own NUTS scheme and
predicted through the batched GPyTorch model BoTorch loads the draws into. The JAX kernels beside
them all take `(X, Z, params)` and return a noise-free (n, m) matrix, so a readout reaches any cell
through `KERNELS`.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import botorch.settings
import gpytorch.settings
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from botorch.models.fully_bayesian import (
    MIN_INFERRED_NOISE_LEVEL, PyroModel, SaasFullyBayesianSingleTaskGP, SaasPyroModel,
    reshape_and_detach,
)
from botorch.models.transforms.input import FilterFeatures, InputTransform
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood
from gpytorch.means import Mean
from jax import Array, jit
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss
from numpyro.infer import MCMC, NUTS
from scipy.optimize import brentq

from sagp.diagnostics import DiagThresholds, Diagnostics, diagnose
from sagp.kernels_torch import (
    CenteredAdditiveAmplitudeKernel, CenteredAdditiveLengthscaleKernel,
    CenteredProductAmplitudeKernel,
)

# (structure in {"additive", "product"}, sparsity prior in {"amplitude", "lengthscale"}).
CellKey = tuple[str, str]

# What a cell's `load_mcmc_samples` hands back; the trailing entry is BoTorch's input-warping
# transform, which no cell here uses.
_LoadedModules = tuple[Mean, Kernel, Likelihood, None]

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


@jit
def kernel_additive_amplitude(X: Array, Z: Array, params: dict[str, Array]) -> Array:
    """Additive structure, amplitude sparsity: k(x, z) = sum_i a_sq_i kbar_i(x_i, z_i).

    `params`: "a_sq", "kernel_ell" (D,); normalized, so a_sq_i is coordinate i's variance under nu.
    """
    kbar = kbar_all(X, Z, params["kernel_ell"], True)
    return jnp.sum(params["a_sq"] * kbar, axis=-1)  # N_X N_Z


@jit
def kernel_additive_lengthscale(X: Array, Z: Array, params: dict[str, Array]) -> Array:
    """Additive structure, lengthscale (SAAS) sparsity: k(x, z) = var * sum_i k~_i(x_i, z_i).

    `params`: "outputscale", "kernel_inv_length_sq" (D,) = rho_i, ell_i = rho_i^-0.5. Deliberately
    *not* normalized: sparsity is rho_i -> 0, which drives v(ell_i) and the component to zero, so
    dividing v out would undo the shrinkage.
    """
    ell = params["kernel_inv_length_sq"] ** -0.5
    return params["outputscale"] * jnp.sum(kbar_all(X, Z, ell, False), axis=-1)  # N_X N_Z


@jit
def kernel_product_amplitude(X: Array, Z: Array, params: dict[str, Array]) -> Array:
    """Product structure, amplitude sparsity: k(x, z) = prod_i (1 + a_sq_i kbar_i(x_i, z_i)).

    `params`: "a_sq", "kernel_ell" (D,); a_sq_i = 0 removes coordinate i from every interaction.
    """
    kbar = kbar_all(X, Z, params["kernel_ell"], True)
    return jnp.prod(1.0 + params["a_sq"] * kbar, axis=-1)  # N_X N_Z


@jit
def kernel_product_lengthscale(X: Array, Z: Array, params: dict[str, Array]) -> Array:
    """Product structure, lengthscale (SAAS) sparsity: SAASBO's own ARD Matern-5/2.

    `params`: "outputscale", "kernel_inv_length_sq" (D,) = rho_i, ell_i = rho_i^-0.5.
    `botorch.models.fully_bayesian.matern52_kernel`'s arithmetic in its own order, generalized to
    Z != X, so this cell is bit for bit the kernel BoTorch samples the model under. The squared
    distance is floored at 1e-30 before the square root, which is what keeps the derivative finite
    where two points coincide.
    """
    lengthscale = params["kernel_inv_length_sq"] ** -0.5
    scaled_X, scaled_Z = X / lengthscale, Z / lengthscale
    diff = jnp.expand_dims(scaled_X, -2) - jnp.expand_dims(scaled_Z, -3)
    distance = jnp.sqrt(jnp.maximum(jnp.sum(diff**2, axis=-1), 1e-30))
    sqrt5_dist = _ROOT_FIVE * distance
    matern = (1 + sqrt5_dist + 5.0 / 3.0 * distance**2) * jnp.exp(-sqrt5_dist)
    return params["outputscale"] * matern  # N_X N_Z


def _diag_additive_amplitude(X: Array, params: dict[str, Array]) -> Array:
    """sum_i a_sq_i kbar_i(x_i, x_i); X (n, D) -> (n,)."""
    return jnp.sum(params["a_sq"] * _kbar_diag(X, params["kernel_ell"], True), axis=-1)


def _diag_additive_lengthscale(X: Array, params: dict[str, Array]) -> Array:
    """var * sum_i k~_i(x_i, x_i); X (n, D) -> (n,)."""
    ell = params["kernel_inv_length_sq"] ** -0.5
    return params["outputscale"] * jnp.sum(_kbar_diag(X, ell, False), axis=-1)


def _diag_product_amplitude(X: Array, params: dict[str, Array]) -> Array:
    """prod_i (1 + a_sq_i kbar_i(x_i, x_i)); X (n, D) -> (n,)."""
    return jnp.prod(1.0 + params["a_sq"] * _kbar_diag(X, params["kernel_ell"], True), axis=-1)


def _diag_product_lengthscale(X: Array, params: dict[str, Array]) -> Array:
    """var * ones(n): the ARD Matern-5/2's marginal variance is constant in x."""
    return params["outputscale"] * jnp.ones(X.shape[-2])


# (kernel, diagonal) per cell: the one place a readout has to look to serve any of the four.
KERNELS: dict[CellKey, tuple[Callable[..., Array], Callable[..., Array]]] = {
    ("additive", "amplitude"): (kernel_additive_amplitude, _diag_additive_amplitude),
    ("additive", "lengthscale"): (kernel_additive_lengthscale, _diag_additive_lengthscale),
    ("product", "amplitude"): (kernel_product_amplitude, _diag_product_amplitude),
    ("product", "lengthscale"): (kernel_product_lengthscale, _diag_product_lengthscale),
}


def cell_kernel_diag(key: CellKey, X: Array, params: dict[str, Array]) -> Array:
    """Prior marginal variance diag k(X, X) of cell `key` at X; X (n, D) -> (n,), O(n D Q).

    Noise-free, as the kernels are, and it never forms the (n, n) matrix.
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
# white noise and competes with `noise`. The lengthscale cells have none: there the half-Cauchy on
# rho *is* the lengthscale prior.
ELL_PRIOR: tuple[float, float] = (0.0, 1.5)


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
            X, X, {"outputscale": outputscale, "kernel_inv_length_sq": inv_length_sq}
        )
        self.sample_observations(mean=mean, K_noiseless=K_noiseless, noise=noise)

    def load_mcmc_samples(self, mcmc_samples: dict[str, torch.Tensor]) -> _LoadedModules:
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
        K_noiseless = self._jax_kernel(X, X, {"a_sq": a_sq, "kernel_ell": ell})
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

    def load_mcmc_samples(self, mcmc_samples: dict[str, torch.Tensor]) -> _LoadedModules:
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


@dataclass(frozen=True)
class Cell:
    """One of the four surrogates: everything inference, prediction and the readouts need of it.

    One `Cell` per surrogate lets `fit`, `posterior` and `experiments.identify` be written once,
    branch-free.
    """

    structure: str  # "additive" or "product"
    prior: str  # "amplitude" or "lengthscale"
    pyro_model: type[PyroModel]  # the cell itself; `CellGP` instantiates it per fit
    kernel: Callable[..., Array]  # (X, Z, params) -> (n, m), noise-free
    kernel_diag: Callable[..., Array]  # (X, params) -> (n,), noise-free
    native_site: str  # the site the cell's own sparsity lives on: what its active rule thresholds
    sites: tuple[str, ...]  # every site to retain, sampled and deterministic, in declared order
    alpha_default: float  # calibrated so all four cells' priors have the same active count

    @property
    def key(self) -> CellKey:
        """(structure, prior): this cell's key in `CELLS` and in `KERNELS`."""
        return (self.structure, self.prior)


# The four cells of the 2x2. `sites` lists `noise` unconditionally; a fit with `fixed_noise` set
# drops it, since then it is not a site at all.
CELLS: dict[CellKey, Cell] = {
    cell.key: cell
    for cell in (
        Cell(
            structure="additive", prior="amplitude", pyro_model=AdditiveAmplitudePyroModel,
            kernel=kernel_additive_amplitude, kernel_diag=_diag_additive_amplitude,
            native_site="a_sq", alpha_default=ALPHA_AMPLITUDE,
            sites=("mean", "noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"),
        ),
        Cell(
            structure="additive", prior="lengthscale", pyro_model=AdditiveLengthscalePyroModel,
            kernel=kernel_additive_lengthscale, kernel_diag=_diag_additive_lengthscale,
            native_site="kernel_inv_length_sq", alpha_default=ALPHA_LENGTHSCALE,
            sites=(
                "outputscale", "mean", "noise", "kernel_tausq",
                "_kernel_inv_length_sq", "kernel_inv_length_sq", "lengthscale",
            ),
        ),
        Cell(
            structure="product", prior="amplitude", pyro_model=ProductAmplitudePyroModel,
            kernel=kernel_product_amplitude, kernel_diag=_diag_product_amplitude,
            native_site="a_sq", alpha_default=ALPHA_AMPLITUDE,
            sites=("mean", "noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"),
        ),
        Cell(
            structure="product", prior="lengthscale", pyro_model=ProductLengthscalePyroModel,
            kernel=kernel_product_lengthscale, kernel_diag=_diag_product_lengthscale,
            native_site="kernel_inv_length_sq", alpha_default=ALPHA_LENGTHSCALE,
            sites=(
                "outputscale", "mean", "noise", "kernel_tausq",
                "_kernel_inv_length_sq", "kernel_inv_length_sq", "lengthscale",
            ),
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
        input_transform: InputTransform | None = None,
    ) -> None:
        """(n, D) inputs in [0,1]^D and (n, 1) targets already standardized.

        `train_Yvar` None learns the noise. `alpha` None takes the cell's calibrated default, and
        `ell_prior` reaches the amplitude cells only. `input_transform` is BoTorch's own, and the
        MAP references use it (`FilterFeatures`) to be fitted on their `active` columns alone while
        the loop keeps handing them full-D points.
        """
        self._pyro_model_class = cell.pyro_model
        with botorch.settings.validate_input_scaling(False):
            super().__init__(
                train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar,
                input_transform=input_transform,
            )
        self.cell = cell
        self.pyro_model.alpha = cell.alpha_default if alpha is None else alpha
        self.pyro_model.ell_prior = ell_prior


# The two `FittedGP.cell` strings that are not a `CellKey`: the MAP references, on the reference
# ARD Matern-5/2 under the lengthscale cells' parameter names.
_MAP_REFERENCES: frozenset[str] = frozenset({"dsp_map", "oracle_S"})


def _check_cell(cell: CellKey | str) -> bool:
    """True when `cell` names a MAP reference, False for a `CellKey`; raises on anything else.

    Without it an unknown cell name would be served the lengthscale cells' kernel silently.
    """
    if isinstance(cell, tuple):
        return False
    if cell not in _MAP_REFERENCES:
        raise ValueError(
            f"unknown cell {cell!r}: expected a CellKey in CELLS or one of "
            f"{sorted(_MAP_REFERENCES)}"
        )
    return True


def _load_draws(model: CellGP, samples: dict[str, ArrayLike]) -> None:
    """Load retained draws into `model` and leave it ready to predict.

    Shared by `fit` and `FittedGP.from_draws`. `samples` must carry what the cell's
    `load_mcmc_samples` reads -- `mean` always, `noise` unless the noise is fixed, and the cell's
    two kernel parameters -- with leading dimension S.
    """
    # Torch draws pass straight through (`fit`'s already are); everything else goes through
    # `np.array` rather than `asarray`, a JAX array's numpy view being read-only.
    torch_draws = {
        site: draws.to(torch.float64)
        if torch.is_tensor(draws)
        else torch.as_tensor(np.array(draws, dtype=np.float64))
        for site, draws in samples.items()
    }
    model.load_mcmc_samples(torch_draws)
    model.eval()


class FittedGP:
    """One cell's posterior on standardized targets: what `fit` returns.

    The loop (`sagp.bo`) and the readouts (`sagp.readouts`) consume it, and `references.fit_map`
    builds one for each MAP reference. The training data keeps the reference's names (`X_train`,
    `Y_train`) and stays JAX, because the readouts hand it to JAX code; `samples` holds the
    retained constrained draws, one entry per site of `cell.sites` present in the trace, leading
    dimension S; `model` is those same draws loaded into a batched GPyTorch model, which is what
    prediction and the acquisition optimizer run on.
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
        model: SaasFullyBayesianSingleTaskGP,
    ) -> None:
        self.cell = cell  # a CellKey, or "dsp_map"/"oracle_S" for the MAP references
        self.X_train = jnp.asarray(X_train, dtype=jnp.float64)  # (n, D)
        self.Y_train = jnp.asarray(Y_train, dtype=jnp.float64)  # (n,)
        self.samples = {site: jnp.asarray(draws) for site, draws in samples.items()}
        self.fixed_noise = fixed_noise  # None when the noise was learned
        self.active = active  # the coordinates the oracle reference was given, else None
        self.status = status  # "ok" or "excluded"
        self.status_reason = status_reason
        self.attempts = tuple(attempts)
        self.model = model  # fitted, in eval mode: prediction goes through it

    @classmethod
    def from_draws(
        cls,
        cell: CellKey | str,
        X: ArrayLike,
        y: ArrayLike,
        samples: dict[str, ArrayLike],
        *,
        fixed_noise: float | None = None,
        active: np.ndarray | None = None,
        status: str = "ok",
        status_reason: str = "",
        attempts: tuple[Diagnostics, ...] = (),
    ) -> FittedGP:
        """A `FittedGP` for draws that did not come from `fit`: builds the model and loads them.

        `references.fit_map` builds its two MAP references this way (one draw each), and the
        readout and loop tests build hand-made draws this way rather than paying for a sampler.
        The MAP references share the lengthscale cells' kernel, and `active` becomes the model's
        own `FilterFeatures` transform, so they are fitted on those columns while every caller
        keeps handing them full-D points.
        """
        Xt = torch.as_tensor(np.array(X, dtype=np.float64))
        yt = torch.as_tensor(np.array(y, dtype=np.float64))[:, None]
        yvar = None if fixed_noise is None else torch.full_like(yt, fixed_noise)
        if _check_cell(cell):
            filter_features = (
                None
                if active is None
                else FilterFeatures(torch.as_tensor(np.array(active), dtype=torch.int64))
            )
            model = CellGP(
                Xt, yt, yvar, cell=CELLS[("product", "lengthscale")],
                input_transform=filter_features,
            )
        else:
            model = CellGP(Xt, yt, yvar, cell=CELLS[cell])
        _load_draws(model, samples)
        return cls(
            cell=cell, X_train=X, Y_train=y, samples=samples, fixed_noise=fixed_noise,
            active=active, status=status, status_reason=status_reason, attempts=attempts,
            model=model,
        )

    def _is_map_reference(self) -> bool:
        """True when `cell` names one of the MAP references; raises on any other string."""
        return _check_cell(self.cell)

    def param_sites(self) -> tuple[str, ...]:
        """This cell's kernel parameters, in the order the readouts pass them around."""
        if self._is_map_reference() or self.cell[1] == "lengthscale":
            return ("outputscale", "kernel_inv_length_sq")
        return ("a_sq", "kernel_ell")

    def params(self, s: int) -> dict[str, Array]:
        """Retained sample `s`'s kernel parameters, in the form its kernel and diagonal take."""
        return {site: self.samples[site][s] for site in self.param_sites()}

    def columns(self, X: Array) -> Array:
        """`X` restricted to `active`, the coordinates the oracle reference was given."""
        return X if self.active is None else X[:, self.active]

    def _num_draws(self) -> int:
        """S, the number of retained draws; every per-draw readout is this long."""
        return int(self.samples["mean"].shape[0])

    @property
    def device(self) -> torch.device:
        """Where `model` lives -- a GPU for a cell `fit` ran there -- so where its inputs must go."""
        return self.model.train_inputs[0].device

    def posterior(self, X_test: ArrayLike, observation_noise: bool = True) -> tuple[Array, Array]:
        """Per retained sample, the posterior mean and *noisy* predictive variance at X_test.

        `(S, n_test)` each; `observation_noise=False` gives the latent variance the acquisition
        uses. Each test point is its own batch entry, so no joint (n_test, n_test) covariance is
        ever formed and GPyTorch's square-input diagonal heuristic cannot fire when S == n_test.
        """
        X = torch.as_tensor(np.array(X_test, dtype=np.float64), device=self.device)[:, None, :]
        with torch.no_grad(), gpytorch.settings.cholesky_max_tries(9):
            post = self.model.posterior(X, observation_noise=observation_noise)
            mean = post.mean.reshape(X.shape[0], -1).T
            variance = post.variance.reshape(X.shape[0], -1).T
        return jnp.asarray(mean.cpu().numpy()), jnp.asarray(variance.cpu().numpy())

    def alphas(self) -> Array:
        """(S, n): K^_s^-1 (z - c_s), the weights the posterior mean contracts with k_*, per draw.

        The constant c_s is `means()`. GPyTorch caches exactly this vector as its prediction
        strategy's `mean_cache`, built on the first prediction, so one warm-up call is what it
        takes to read the vector the posterior mean itself uses.
        """
        if self.model.prediction_strategy is None:
            self.posterior(np.asarray(self.X_train)[:1])
        cache = self.model.prediction_strategy.mean_cache.detach()
        return jnp.asarray(cache.reshape(self._num_draws(), -1).cpu().numpy())

    def means(self) -> Array:
        """(S,): the constant mean of each retained draw, the intercept the posterior mean adds."""
        return self.samples["mean"].reshape(-1)

    def noises(self) -> Array:
        """(S,): the observation variance prediction actually uses, per retained draw.

        Read off the likelihood rather than off the draws: BoTorch clamps a learned `noise` at
        `MIN_INFERRED_NOISE_LEVEL` on the way in, and a fixed value arrives already clamped, so
        the sampled number and the predicted one need not agree.
        """
        noise = self.model.likelihood.noise.detach().reshape(self._num_draws(), -1)[:, 0]
        return jnp.asarray(noise.cpu().numpy())


def _run_nuts(model: CellGP, nuts: NUTSConfig, seed: int) -> tuple[MCMC, float]:
    """`fit_fully_bayesian_model_nuts`'s sampler lines (BoTorch `fit.py:375-389`), copied.

    Copied rather than called because the study needs the sampler itself: `diagnose` reads the
    un-thinned draws and the `diverging`/`num_steps` counters BoTorch collects nothing of and
    discards. Returns the finished `MCMC` and the wall-clock seconds it ran for.
    """
    if nuts.num_chains != 1:
        raise ValueError(
            f"num_chains must be 1; got {nuts.num_chains}. One chain per fit is BoTorch's own "
            "setting, and diagnose pools with group_by_chain=False, so more chains would be "
            "flattened into one and their split-R-hat would compare the halves of the "
            "concatenation."
        )
    start = time.perf_counter()
    model.train()
    kernel = NUTS(model.pyro_model.sample, dense_mass=True, max_tree_depth=nuts.max_tree_depth)
    mcmc = MCMC(
        kernel, num_warmup=nuts.num_warmup, num_samples=nuts.num_samples, progress_bar=False
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
    return mcmc, time.perf_counter() - start


def fit(
    X: ArrayLike,
    y: ArrayLike,
    seed: int,
    cell: CellKey | Cell,
    *,
    alpha: float | None = None,
    fixed_noise: float | None = None,
    nuts: NUTSConfig = NUTSConfig(),
    thresholds: DiagThresholds = DiagThresholds(),
    ell_prior: tuple[float, float] = ELL_PRIOR,
) -> FittedGP:
    """NUTS fit of `cell` to (X in [0,1]^D, y already standardized by the caller).

    `seed` is BoTorch's int seed: the chain is driven by `jax.random.PRNGKey(seed)`, so this fit
    reproduces `fit_fully_bayesian_model_nuts(..., seed=seed)` on the same model to the bit. One
    attempt, judged once against `thresholds`: the fit is "ok" or "excluded", and its draws come
    back either way, because the loop has to keep querying with an excluded fit.

    `alpha` None takes the cell's calibrated default: `ALPHA_LENGTHSCALE` on the rho scale, or
    `ALPHA_AMPLITUDE` on the a^2 scale. `fixed_noise` None learns `noise`, while a value fixes the
    observation variance at it -- and must be at least 1e-4, because BoTorch clamps `train_Yvar`
    there before the sampler sees it, so a smaller number would silently be raised. `nuts` is the
    sampler budget (`NUTSConfig()`), `thresholds` the gate the attempt is judged against
    (`DiagThresholds()`), and `ell_prior` reaches the amplitude cells only.
    """
    if fixed_noise is not None and not fixed_noise >= MIN_INFERRED_NOISE_LEVEL:
        raise ValueError(
            f"fixed_noise={fixed_noise!r} is below BoTorch's floor {MIN_INFERRED_NOISE_LEVEL}: "
            "train_Yvar is clamped there before the sampler sees it, so the fit would run at "
            f"{MIN_INFERRED_NOISE_LEVEL} while reporting the smaller value. Pass fixed_noise=None "
            f"to learn the noise, or a variance >= {MIN_INFERRED_NOISE_LEVEL}."
        )
    cell = CELLS[cell] if not isinstance(cell, Cell) else cell
    Xt = torch.as_tensor(np.array(X, dtype=np.float64), device=torch_device())
    yt = torch.as_tensor(np.array(y, dtype=np.float64), device=torch_device())[:, None]
    yvar = None if fixed_noise is None else torch.full_like(yt, fixed_noise)
    model = CellGP(Xt, yt, yvar, cell=cell, alpha=alpha, ell_prior=ell_prior)

    mcmc, wall_s = _run_nuts(model, nuts, seed)
    flat, extra = mcmc.get_samples(), mcmc.get_extra_fields()
    diag = diagnose(flat, extra, thresholds, wall_s)

    # `dict(flat)` because BoTorch's postprocessing writes `lengthscale` into its argument.
    samples = model.pyro_model.postprocess_mcmc_samples(mcmc_samples=dict(flat))
    samples = {site: draws[:: nuts.thinning] for site, draws in samples.items()}
    _load_draws(model, samples)

    retained = {
        site: jnp.asarray(samples[site].cpu().numpy()) for site in cell.sites if site in samples
    }
    return FittedGP(
        cell=cell.key, X_train=X, Y_train=y, samples=retained, fixed_noise=fixed_noise,
        active=None, status="ok" if diag.passed else "excluded", status_reason=diag.reason,
        attempts=(diag,), model=model,
    )


def torch_device() -> torch.device:
    """Where `fit` builds a cell's model: onto the GPU when JAX runs the chain on one, else the CPU.

    The acquisition and the readouts' predictions go through that model, so they run there too;
    a test that builds BoTorch's side of a comparison builds it here, so the comparison stays exact.
    """
    return torch.device("cuda" if jax.default_backend() == "gpu" else "cpu")


# --- standardize ---

# `standardize` stays beside `fit` because it is `fit`'s precondition; `sagp.bo.run_bo` and
# `experiments.identify` both need it.


def standardize(y: ArrayLike) -> tuple[np.ndarray, float, float]:
    """Standardize `y` to zero mean, unit variance (ddof 0): z = (y - mean) / std.

    `mean` and `std` are plain floats. There is no sign flip: the loop maximizes what the
    objective maximizes, and `sagp.bo.run_bo`'s incumbent is `max(z)`.
    """
    y = np.asarray(y)
    mean, std = float(y.mean()), float(y.std())
    return (y - mean) / std, mean, std
