"""sagp.references: the study's two non-cell surrogates, both MAP fits under a fixed prior.

The dimension-scaled-prior (DSP) reference is Hvarfner et al. 2024's "vanilla BO" model -- an ARD
Matern-5/2 fit by MAP to the full design -- and the oracle is the identical fit restricted to the
objective's true active coordinates. Both predict through `FittedGP` with a single retained
"sample", exactly like a cell at S = 1, so the BO loop cannot tell either reference from a cell.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import Array, jit
from jax.scipy.linalg import cho_factor, cho_solve
from jax.typing import ArrayLike
from scipy.optimize import minimize

from sagp.gp import FittedGP, kernel_product_lengthscale


# --- MAP references ---

# Hvarfner et al. 2024's "vanilla BO" hyperparameters, transcribed from BoTorch's
# `botorch.models.utils.gpytorch_modules` rather than imported, since nothing here may import
# torch: `get_covar_module_with_dim_scaled_prior` puts LogNormal(sqrt(2) + log(D)/2, sqrt(3)) on
# every lengthscale behind a GreaterThan(2.5e-2) floor, and
# `get_gaussian_likelihood_with_lognormal_prior` puts LogNormal(-4, 1) on the noise behind a
# GreaterThan(1e-4) floor. The dimension-scaled location is the whole point of that paper: it
# grows the prior lengthscale with D so that the prior over functions does not concentrate on
# ever more wiggly ones as dimensions are added, which is why a plain ARD GP is a fair reference
# for a sparse one at all.
_ELL_FLOOR: float = 0.025
_ELL_PRIOR_SCALE: float = math.sqrt(3.0)
_NOISE_FLOOR: float = 1.0e-4
_NOISE_PRIOR_LOC: float = -4.0
_NOISE_PRIOR_SCALE: float = 1.0


def _ell_prior_loc(D_eff: int) -> float:
    """The dimension-scaled lengthscale prior's log-location, sqrt(2) + log(D_eff)/2.

    `D_eff` is the width the GP actually sees, so the oracle reference gets the prior of a
    |S|-dimensional problem rather than of the D-dimensional one it was carved out of.
    """
    return math.sqrt(2.0) + math.log(D_eff) / 2.0


def _dsp_start(D_eff: int) -> np.ndarray:
    """The unconstrained start point: both priors' modes exp(loc - scale^2), through the floors.

    Those modes are BoTorch's own initial values (`GreaterThan(..., initial_value=prior.mode)`).
    L-BFGS-B moves in u, so the start is the u whose constrained value is the mode, i.e. the
    inverse log(value - floor) of the map `_dsp_neg_log_joint` applies.
    """
    ell0 = math.exp(_ell_prior_loc(D_eff) - _ELL_PRIOR_SCALE**2.0)
    noise0 = math.exp(_NOISE_PRIOR_LOC - _NOISE_PRIOR_SCALE**2.0)
    return np.concatenate(
        [np.full(D_eff, math.log(ell0 - _ELL_FLOOR)), [math.log(noise0 - _NOISE_FLOOR)]]
    )


def _dsp_neg_log_joint(u: Array, X: Array, y: Array, D_eff: int) -> Array:
    """Negative log joint of the DSP reference at the unconstrained parameters `u`.

    `u` is concat(the D_eff lengthscale parameters, the noise parameter), mapped onto BoTorch's
    floors by ell_i = 0.025 + exp(u_i) and noise = 1e-4 + exp(u[-1]). The floors are BoTorch's
    `GreaterThan` bounds; `exp` is this module's unconstraining map, where GPyTorch's constraint
    would use its softplus (BoTorch in fact passes `transform=None` on both of these and lets the
    optimizer's box bounds hold the floor). The two agree on the interior and differ only in the
    path L-BFGS-B walks, not in the model it walks over. `X` arrives already restricted to the
    oracle's coordinates when there are any, and the kernel is the reference ARD Matern-5/2 at
    sigma_f^2 = 1 -- Hvarfner's model has no outputscale.

    The value is -(log N(y | 0, K) + sum_i log LogNormal(ell_i; sqrt(2) + log(D_eff)/2, sqrt(3))
    + log LogNormal(noise; -4, 1)), the log-priors taken on the *constrained* values with no
    Jacobian term. That is BoTorch's MAP objective, not a variant of it: `fit_gpytorch_mll`
    maximizes the log marginal likelihood plus the priors' log-densities at the transformed
    values, so leaving the Jacobian out is what reproduces the reference rather than an omission.

    `D_eff` is redundant with the shapes of `u` and `X` and taken anyway so that it can be a
    static argument of the jitted gradient below, keeping log(D_eff) on the Python side.
    """
    ell = _ELL_FLOOR + jnp.exp(u[:D_eff])
    noise = _NOISE_FLOOR + jnp.exp(u[D_eff])

    k_XX = kernel_product_lengthscale(
        X, X, {"kernel_var": 1.0, "kernel_inv_length_sq": ell**-2.0}, noise, True
    )
    L = cho_factor(k_XX, lower=True)[0]
    log_lik = (
        -0.5 * jnp.dot(y, cho_solve((L, True), y))
        - jnp.sum(jnp.log(jnp.diag(L)))
        - 0.5 * y.shape[0] * math.log(2.0 * math.pi)
    )
    log_prior = jnp.sum(
        dist.LogNormal(_ell_prior_loc(D_eff), _ELL_PRIOR_SCALE).log_prob(ell)
    ) + dist.LogNormal(_NOISE_PRIOR_LOC, _NOISE_PRIOR_SCALE).log_prob(noise)

    return -(log_lik + log_prior)


# Compiled once per (n, D_eff) -- jit caches on the argument shapes and on the static `D_eff` --
# and therefore once per fit rather than once per L-BFGS-B iteration, which is the only reason a
# few hundred iterations of a dense Cholesky are affordable inside the acquisition loop.
_dsp_value_and_grad = jit(jax.value_and_grad(_dsp_neg_log_joint), static_argnums=(3,))


def fit_map(
    X: ArrayLike,
    y: ArrayLike,
    *,
    active: ArrayLike | None = None,
    maxiter: int = 500,
) -> FittedGP:
    """MAP fit of Hvarfner et al. 2024's dimension-scaled-prior GP: the study's "vanilla BO" reference.

    An ARD Matern-5/2 at sigma_f^2 = 1 -- BoTorch's default under that prior is an RBF, but the
    thesis holds the kernel family fixed across every method it compares, and Matern-5/2 is the
    one it holds -- with ell_i ~ LogNormal(sqrt(2) + log(D_eff)/2, sqrt(3)) and
    noise ~ LogNormal(-4, 1) behind BoTorch's floors (see `_dsp_neg_log_joint` for the objective
    and the floors' parameterization). L-BFGS-B on the jitted value-and-gradient, started at the
    two priors' modes exp(loc - scale^2), which are BoTorch's own initial values. No PRNG is
    involved anywhere: two calls on the same data return the same GP bit for bit.

    `active=None` is the DSP reference and fits all D coordinates; `active=S` is the oracle, which
    fits `X[:, S]` alone and is otherwise the identical procedure -- the two differ in the
    information they are given and in nothing else. Either way the returned `FittedGP` holds the
    *full-D* design and answers full-D test points (`FittedGP._columns` applies the restriction at
    every kernel evaluation), so the BO loop cannot tell either reference from a cell.

    The one further departure from a stock BoTorch `SingleTaskGP` is the mean function: that model
    fits a constant mean, and this one is zero-mean, like the four cells and like the vendored
    reference they all predict through. The targets are standardized before they get here, so the
    constant it would fit is ~0 anyway, and holding the mean fixed across every method keeps the
    comparison about the kernel and the prior.

    The MAP estimate comes back as a single retained "sample", so `posterior` is (1, n_test) like
    a cell's at S = 1. `map_result` records the optimizer's outcome -- final and initial objective,
    iterations, convergence flag -- which is the only trace of this fit's quality there is: a MAP
    fit has no `attempts`, and it is left empty rather than filled with a fabricated diagnostic.
    A fit L-BFGS-B *abandoned* -- an abnormal line-search termination -- comes back with
    `status="excluded"` and the optimizer's message as `status_reason` (R30, amended by R41),
    exactly as a NUTS fit that failed its diagnostics does. Hitting `maxiter` is not that: it
    leaves the last iterate, which is a descent step from the prior mode and a usable surrogate,
    so it stays "ok" and says so in `status_reason` rather than throwing the iteration away.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if active is not None:
        active = np.asarray(active, dtype=int)
    Xa = X if active is None else X[:, active]
    D_eff = Xa.shape[1]

    u0 = _dsp_start(D_eff)

    def neg_log_joint(u: np.ndarray) -> tuple[float, np.ndarray]:
        """(value, gradient) at `u` as float64 numpy, which is what `minimize(jac=True)` takes."""
        value, grad = _dsp_value_and_grad(jnp.asarray(u), Xa, y, D_eff)
        return float(value), np.asarray(grad, dtype=float)

    fun0, _ = neg_log_joint(u0)
    result = minimize(neg_log_joint, u0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})

    u = jnp.asarray(result.x)
    ell = _ELL_FLOOR + jnp.exp(u[:D_eff])
    noise = _NOISE_FLOOR + jnp.exp(u[D_eff])

    # R41: `success=False` covers two outcomes the study must not conflate. An abnormal line
    # search means L-BFGS-B could not take the step and the iterate is wherever it gave up, which
    # is the excluded case R30 introduced; the iteration limit means it was still descending, and
    # its last iterate is the MAP estimate to the accuracy `maxiter` bought. Matching on the
    # message is what scipy offers -- `status` is 2 for both -- and "ABNORMAL" is the word its
    # (Fortran) line-search messages carry.
    message = str(result.message)
    if "ABNORMAL" in message.upper():
        status, status_reason = "excluded", f"L-BFGS-B: {message}"
    elif not result.success:
        status, status_reason = "ok", "L-BFGS-B: iteration limit; last iterate used"
    else:
        status, status_reason = "ok", ""

    fitted = FittedGP(
        cell="oracle_S" if active is not None else "dsp_map",
        X_train=X,
        Y_train=y,
        samples={
            "kernel_var": jnp.array([1.0]),
            "kernel_inv_length_sq": (ell**-2.0)[None, :],
            "kernel_noise": noise[None],
        },
        fixed_noise=None,
        active=active,
        status=status,
        status_reason=status_reason,
        attempts=(),
    )
    fitted.map_result = {
        "fun": float(result.fun),
        "nit": int(result.nit),
        "success": bool(result.success),
        "fun0": fun0,
    }
    return fitted
