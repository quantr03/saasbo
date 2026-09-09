"""sagp.references: the study's two non-cell surrogates, both MAP fits under a fixed prior.

The dimension-scaled-prior (DSP) reference is Hvarfner et al. 2024's "vanilla BO" model, an ARD
Matern-5/2 fit by MAP to the full design; the oracle is the same fit restricted to the objective's
true active coordinates. Both predict through `FittedGP` at one retained "sample", like a cell at
S = 1, so the BO loop cannot tell either from a cell. The third reference, `propose_sobol`, fits
nothing and so reaches the loop as a proposer rather than as a surrogate.
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
# `get_gaussian_likelihood_with_lognormal_prior` LogNormal(-4, 1) on the noise behind a
# GreaterThan(1e-4) floor. Scaling the location with D keeps the prior over functions from
# concentrating on ever wigglier ones as D grows -- what makes a plain ARD GP a fair reference.
_ELL_FLOOR: float = 0.025
_ELL_PRIOR_SCALE: float = math.sqrt(3.0)
_NOISE_FLOOR: float = 1.0e-4
_NOISE_PRIOR_LOC: float = -4.0
_NOISE_PRIOR_SCALE: float = 1.0


def _ell_prior_loc(D_eff: int) -> float:
    """The dimension-scaled prior's log-location sqrt(2) + log(D_eff)/2, at the width the GP
    actually sees -- so the oracle gets a |S|-dimensional problem's prior, not the D-dimensional
    one's."""
    return math.sqrt(2.0) + math.log(D_eff) / 2.0


def _dsp_start(D_eff: int) -> np.ndarray:
    """The unconstrained start point: both priors' modes exp(loc - scale^2), BoTorch's own initial
    values, put through the inverse log(value - floor) of the map `_dsp_neg_log_joint` applies,
    since L-BFGS-B moves in u."""
    ell0 = math.exp(_ell_prior_loc(D_eff) - _ELL_PRIOR_SCALE**2.0)
    noise0 = math.exp(_NOISE_PRIOR_LOC - _NOISE_PRIOR_SCALE**2.0)
    return np.concatenate(
        [np.full(D_eff, math.log(ell0 - _ELL_FLOOR)), [math.log(noise0 - _NOISE_FLOOR)]]
    )


def _dsp_neg_log_joint(u: Array, X: Array, y: Array, D_eff: int) -> Array:
    """Negative log joint of the DSP reference at the unconstrained parameters `u`.

    `u` is concat(the D_eff lengthscale parameters, the noise parameter), mapped onto BoTorch's
    `GreaterThan` floors by ell_i = 0.025 + exp(u_i) and noise = 1e-4 + exp(u[-1]) -- `exp` where
    GPyTorch's constraint would use softplus, which agrees on the interior (BoTorch passes
    `transform=None` on both and lets the box bounds hold the floor) and differs only in the path
    L-BFGS-B walks. `X` arrives already restricted to the oracle's coordinates when there are any,
    and the kernel is the reference ARD Matern-5/2 at sigma_f^2 = 1 (no outputscale). The value is
    -(log N(y | 0, K) + the two log-priors above), the priors taken on the *constrained* values
    with no Jacobian term -- `fit_gpytorch_mll`'s own objective, so omitting it reproduces the
    reference rather than being a mistake. `D_eff` is redundant with the shapes of `u` and `X`,
    and taken anyway so that it can be a static argument of the jitted gradient below.
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


# Compiled once per (n, D_eff) -- jit caches on the shapes and on the static `D_eff` -- so once
# per fit, not per L-BFGS-B iteration: what makes a few hundred dense Choleskys affordable here.
_dsp_value_and_grad = jit(jax.value_and_grad(_dsp_neg_log_joint), static_argnums=(3,))


def fit_map(
    X: ArrayLike,
    y: ArrayLike,
    *,
    active: ArrayLike | None = None,
    maxiter: int = 500,
) -> FittedGP:
    """MAP fit of Hvarfner et al. 2024's dimension-scaled-prior GP: the "vanilla BO" reference.

    An ARD Matern-5/2 at sigma_f^2 = 1 -- BoTorch's default under that prior is an RBF, but the
    kernel family is held fixed across every method compared -- under the priors and floors
    `_dsp_neg_log_joint` documents, by L-BFGS-B on the jitted value-and-gradient from the priors'
    modes exp(loc - scale^2), BoTorch's own initial values. No PRNG anywhere: two calls on the
    same data return the same GP bit for bit. The mean is zero rather than the constant a stock
    `SingleTaskGP` fits, matching the cells and the vendored reference they predict through.
    `active=None` fits all D coordinates; `active=S` is the oracle, the identical procedure on
    `X[:, S]` alone. Either way the returned `FittedGP` holds the *full-D* design and answers
    full-D test points (`FittedGP.columns` restricts at every kernel evaluation).

    `map_result` carries the optimizer's outcome -- objective, iterations, convergence flag -- a
    MAP fit's only trace of quality, since it has no `attempts`. A fit L-BFGS-B *abandoned* comes
    back `status="excluded"` with the optimizer's message, as a NUTS fit that failed its
    diagnostics does; hitting `maxiter` does not, since the last iterate is a descent step from
    the prior mode and a usable surrogate.
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

    # `success=False` covers two outcomes that must not be conflated: an abnormal line search
    # leaves the iterate wherever L-BFGS-B gave up, the iteration limit leaves it still descending.
    # `status` is 2 for both, so the message is all scipy offers; "ABNORMAL" is the word its
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


# --- the Sobol search ---


def propose_sobol(
    fitted: FittedGP | None,
    y_target: float,
    rngs: object,
    t: int,
    *,
    sequence: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Row `t` of a fixed sequence: the quasi-random search reference, as a `run_bo` proposer.

    Nothing is fitted, so `fitted` is `None` and the acquisition value NaN. `sequence`'s first
    `n_init` rows are the shared initial design (`bo.initial_design`), which this continues.
    """
    return np.asarray(sequence[t], dtype=float), float("nan")
