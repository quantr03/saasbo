"""sagp.references: the study's two MAP references and its quasi-random search.

The dimension-scaled-prior (DSP) reference is Hvarfner et al. 2024's "vanilla BO" -- an ARD
Matern-5/2 at the MAP under a lengthscale prior whose location grows with the dimension -- and the
oracle is that same fit restricted to the objective's true active coordinates. What runs here is
Papenmeier et al. (2025)'s reference implementation of it rather than a transcription of the
paper: `_dsp_model` is his `build_model(method="dsp", noiseless=False)` and `_fit_mll` his
`fit_mll`. Two things in that construction are passed rather than defaulted, and both are the
point: the Matern-5/2 covariance, because `SingleTaskGP`'s own default is an RBF under a different
prior, and `outcome_transform=None`, because `run_bo` standardizes its targets itself.

Both references predict through `FittedGP` at one retained "sample", like a cell at S = 1, so the
BO loop cannot tell either from a cell; the oracle's restriction is a `FilterFeatures` input
transform rather than a narrowed design, so `optimize_acqf` goes on searching [0, 1]^D. The third
reference, `propose_sobol`, fits nothing and reaches the loop as a proposer, not a surrogate.

Three priors change from the JAX transcription this replaces, all because Papenmeier's model is a
stock `SingleTaskGP` where that one was hand-rolled: the kernel gains an outputscale under
BoTorch's default `Gamma(2, 0.15)` where it was held at 1, the noise moves from `LogNormal(-4, 1)`
to `Gamma(1.1, 0.05)`, and the lengthscales lose their `GreaterThan(2.5e-2)` floor for GPyTorch's
plain `Positive`. The location constant is `SQRT2`, the paper's sqrt(2), where Papenmeier's code
carries the float32-era literal 1.41: what this study claims to run is Hvarfner et al.'s prior,
and 1.41 reproduces his runs rather than that prior.
"""
from __future__ import annotations

import math

import botorch.settings
import gpytorch.settings
import jax.numpy as jnp
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import FilterFeatures
from botorch.models.utils.gpytorch_modules import (
    SQRT2, SQRT3, get_gaussian_likelihood_with_gamma_prior,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from jax import Array
from jax.typing import ArrayLike
from linear_operator.utils.errors import NotPSDError

from sagp.gp import FittedGP

# --- the DSP model ---

# LogNormal(sqrt(2) + log(D_eff)/2, sqrt(3)) on every lengthscale: scaling the location with the
# width the GP actually sees is what keeps the prior over functions from concentrating on ever
# wigglier ones as D grows, and so what makes a plain ARD GP a fair reference at all.
DSP_LOC_CONST: float = SQRT2
DSP_SCALE: float = SQRT3

# `_fit_mll`'s outcome, and the (status, status_reason) each earns. A fit that only survived the
# Adam fallback still moved off the prior mode, so it stays "ok" as Papenmeier's own runs do --
# though the hazard `_fit_mll` documents makes that "ok" weaker than it looks; one where Adam hit
# a `NotPSDError` too was left wherever it gave up, and is excluded exactly as a NUTS fit that
# failed its diagnostics is.
_FALLBACK_STATUS: dict[str, tuple[str, str]] = {
    "none": ("ok", ""),
    "adam": ("ok", "fit_gpytorch_mll failed; Adam fallback"),
    "failed": ("excluded", "fit_gpytorch_mll failed; Adam fallback failed: NotPSDError"),
}


def _dsp_model(X: np.ndarray, z: np.ndarray, active: np.ndarray | None) -> SingleTaskGP:
    """Papenmeier's `build_model(method="dsp", noiseless=False)`, verbatim in structure.

    `X` is the full-D design in [0,1]^D and `z` the standardized targets. `active` is the oracle's
    coordinates, which become a `FilterFeatures` transform: the model is fitted on those columns
    alone -- an |S|-wide ARD kernel under the prior at *that* width -- while every caller keeps
    handing it full-D points. The lengthscales start at the prior's mode exp(loc - scale^2), what
    `Method.initial_lengthscale` seeds `ls_init="prior_mode"` with, and the noise is learned
    (`noiseless=False`) because this study's objectives are observed with noise.
    """
    D_eff = X.shape[1] if active is None else len(active)
    loc = DSP_LOC_CONST + math.log(D_eff) / 2.0
    base_kernel = MaternKernel(
        nu=2.5, ard_num_dims=D_eff,
        lengthscale_prior=LogNormalPrior(torch.tensor(loc), torch.tensor(DSP_SCALE)),
    )
    covar_module = ScaleKernel(base_kernel=base_kernel, outputscale_prior=GammaPrior(2.0, 0.15))
    covar_module.base_kernel.lengthscale = torch.full((D_eff,), math.exp(loc - DSP_SCALE**2.0))
    likelihood = get_gaussian_likelihood_with_gamma_prior()
    input_transform = (
        None if active is None
        else FilterFeatures(torch.as_tensor(np.array(active), dtype=torch.int64))
    )
    Xt = torch.as_tensor(np.array(X, dtype=np.float64))
    zt = torch.as_tensor(np.array(z, dtype=np.float64))
    # Off because the study standardizes with ddof 0 while BoTorch's check uses ddof 1, which
    # would warn on every fit with n < 52 -- `sagp.gp.CellGP` turns it off for the same reason.
    with botorch.settings.validate_input_scaling(False):
        return SingleTaskGP(
            Xt, zt[:, None], covar_module=covar_module, likelihood=likelihood,
            outcome_transform=None, input_transform=input_transform,
        )


def _fit_mll(mll: ExactMarginalLogLikelihood, n_iter: int = 500) -> str:
    """Papenmeier's `fit_mll`, returning "none" | "adam" | "failed" where his code prints.

    BoTorch's own fitter under `cholesky_max_tries(9)`; on the `ModelFittingError` it raises once
    every restart has failed, Adam for `n_iter` steps at lr 0.01 over `model.parameters()` on
    -mll; on a `NotPSDError` inside that fallback, give up and leave the model in eval mode at
    whatever values it had reached. The return value is what `fit_map` turns into a `status`.

    The fallback carries a hazard, transcribed with it: the noise's `GreaterThan(1e-4)` is built
    with `transform=None`, so `raw_noise` *is* the noise and only the box bounds
    `fit_gpytorch_mll` hands L-BFGS-B hold the floor. Adam has none, and at this study's scale its
    500 steps walk the noise from its starting 2.0 through zero -- so a fit that reaches here
    usually leaves by the `NotPSDError` door rather than the `"adam"` one.
    """
    model = mll.model
    with gpytorch.settings.cholesky_max_tries(9):
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            try:
                train_x = model.train_inputs[0]
                train_y = model.train_targets
                model.train()
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.01)
                for _ in range(n_iter):
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y.flatten())
                    loss.backward()
                    optimizer.step()
                model.eval()
            except NotPSDError:
                model.eval()
                return "failed"
            return "adam"
    return "none"


def _one_draw(value: torch.Tensor) -> Array:
    """One fitted scalar hyperparameter as the (1,) "sample" array `FittedGP` reads."""
    return jnp.asarray(value.detach().reshape(1).numpy(), dtype=jnp.float64)


def fit_map(X: ArrayLike, y: ArrayLike, *, active: ArrayLike | None = None) -> FittedGP:
    """The DSP reference at the MAP (`active=None`), or the oracle (`active` = the true S).

    `y` is already standardized by the caller. The fit is `_fit_mll` on the model's exact marginal
    log likelihood, run inside a forked RNG at seed 0: `fit_gpytorch_mll` re-samples the free
    parameters from their priors on every restart after the first, out of torch's global RNG, so
    without the fork this would be neither a pure function of its inputs nor reproducible from a
    run's own seed. Either way the returned `FittedGP` holds the *full-D* design and answers
    full-D test points; the oracle's restriction lives in the model's input transform.
    `map_result` is a MAP fit's only trace of quality, since it has no `attempts`: the objective
    `_fit_mll` reached -- GPyTorch's MLL, the log joint over the number of data points -- and
    which of the two fitters produced it.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if active is not None:
        active = np.asarray(active, dtype=int)

    model = _dsp_model(X, y, active)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        fallback = _fit_mll(mll)
    model.eval()
    # The MLL is only defined in train mode, where the model reverts to its untransformed training
    # inputs and `forward` applies the input transform itself; `eval` again leaves it ready to
    # predict, which is the state `FittedGP` requires. The read-back runs under the same
    # `cholesky_max_tries(9)` `_fit_mll` fitted under, so it cannot fail a factorization the fit
    # itself survived.
    with torch.no_grad(), gpytorch.settings.cholesky_max_tries(9):
        model.train()
        mll_value = float(mll(model(*model.train_inputs), model.train_targets))
        model.eval()

    ell = jnp.asarray(model.covar_module.base_kernel.lengthscale.detach().reshape(-1).numpy())
    status, status_reason = _FALLBACK_STATUS[fallback]
    fitted = FittedGP(
        cell="oracle_S" if active is not None else "dsp_map",
        X_train=X, Y_train=y,
        samples={
            "outputscale": _one_draw(model.covar_module.outputscale),
            "mean": _one_draw(model.mean_module.constant),
            "noise": _one_draw(model.likelihood.noise),
            "kernel_inv_length_sq": (ell**-2.0)[None, :],
            "lengthscale": ell[None, :],
        },
        fixed_noise=None, active=active, status=status, status_reason=status_reason,
        attempts=(), model=model,
    )
    fitted.map_result = {"mll": mll_value, "fallback": fallback}
    return fitted


# --- the Sobol search ---


def propose_sobol(
    fitted: FittedGP | None, best_f: float, rngs: object, t: int, *, sequence: np.ndarray
) -> tuple[np.ndarray, float]:
    """Row `t` of a fixed sequence: the quasi-random search reference, as a `run_bo` proposer.

    Nothing is fitted, so `fitted` is `None` and the acquisition value NaN. `sequence`'s first
    `n_init` rows are the shared initial design (`bo.initial_design`), which this continues.
    """
    return np.asarray(sequence[t], dtype=float), float("nan")
