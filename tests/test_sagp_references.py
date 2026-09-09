"""Tests for sagp.references: Papenmeier's `dsp` model as this study's two MAP references.

The BO study has two references that are not one of the four NUTS-fitted cells but that the loop
must not be able to tell apart from one: Hvarfner et al. 2024's "vanilla BO" -- an ARD Matern-5/2
under the dimension-scaled lengthscale prior, fitted at the MAP and served as a single posterior
"sample" -- and that same model handed the objective's true active set. The claim `sagp.references`
makes is narrow: that what runs is Papenmeier et al. (2025)'s `build_model(method="dsp",
noiseless=False)` followed by his `fit_mll`, and not something nearby. So the construction is
pinned by building the same four GPyTorch/BoTorch modules again here, and the objective those
modules define is pinned by recomputing GPyTorch's marginal log likelihood from scratch in numpy
and scipy -- neither of which shares a line with the module under test.

The rest is the interface the loop sees: one draw, `posterior` on full-D test points whichever
reference produced the fit, an oracle whose inactive coordinates cannot reach the kernel, and a
`status` that says when `fit_gpytorch_mll` had to fall back to Adam and when even Adam failed.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp as gp
import sagp.references as references

import itertools
import math
import warnings

import numpy as np
import pytest
import torch
from botorch.exceptions import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import FilterFeatures
from botorch.models.utils.gpytorch_modules import (
    SQRT2, SQRT3, get_gaussian_likelihood_with_gamma_prior,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator.utils.errors import NotPSDError
from scipy.stats import gamma, lognorm, multivariate_normal, qmc

from synthobj.families import make_family, noise_rng

# `scipy.stats.qmc.Sobol` warns whenever the point count is not a power of two, and n = 24 is not;
# the design is the one the plan specifies for this check, so the warning is about it rather than
# about a mistake here.
_SOBOL_WARNING = "The balance properties of Sobol' points require n to be a power of 2."

D, N, N_TEST = 6, 24, 5

# The prior mode Papenmeier seeds the lengthscales at, at this width: exp(loc - sqrt(3)^2).
_PRIOR_LOC = SQRT2 + math.log(D) / 2.0
_PRIOR_MODE = math.exp(_PRIOR_LOC - 3.0)


def _data():
    """`aligned3` at D = 6: a 24-point Sobol design, observed with noise, standardized."""
    objective = make_family("aligned3", 0, D=D)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SOBOL_WARNING)
        X = qmc.Sobol(D, scramble=True, seed=0).random(N)
    z, _, _ = gp.standardize(objective.observe(X, noise_rng(0)))
    return objective, X, np.asarray(z)


def _test_points():
    """`N_TEST` full-D points to predict at; fixed, since nothing here is about the design."""
    return np.random.default_rng(0).random((N_TEST, D))


# --- the construction ---


def test_dsp_model_is_papenmeiers_construction():
    # The whole claim of this reference is that it *is* the `dsp` row of Papenmeier et al. (2025),
    # so the four module constructions are written out again here rather than read off the module
    # under test: a prior that is merely prior-shaped would pass every other test in this file.
    # `SingleTaskGP`'s own defaults -- an RBF covariance and a `Standardize` outcome transform --
    # are both overridden, and that is what the last three assertions are for.
    _, X, z = _data()
    expected_base = MaternKernel(
        nu=2.5,
        ard_num_dims=D,
        lengthscale_prior=LogNormalPrior(torch.tensor(_PRIOR_LOC), torch.tensor(SQRT3)),
    )
    expected_covar = ScaleKernel(
        base_kernel=expected_base, outputscale_prior=GammaPrior(2.0, 0.15)
    )
    expected_covar.base_kernel.lengthscale = torch.full((D,), math.exp(_PRIOR_LOC - SQRT3**2.0))
    expected_likelihood = get_gaussian_likelihood_with_gamma_prior()

    model = references._dsp_model(X, z, None)

    base = model.covar_module.base_kernel
    assert type(model) is SingleTaskGP
    assert isinstance(model.mean_module, ConstantMean)
    assert base.nu == expected_base.nu == 2.5
    assert base.ard_num_dims == expected_base.ard_num_dims == D == 6
    assert float(base.lengthscale_prior.loc) == float(expected_base.lengthscale_prior.loc)
    assert float(base.lengthscale_prior.loc) == SQRT2 + math.log(6.0) / 2.0
    assert float(base.lengthscale_prior.scale) == SQRT3
    assert torch.equal(base.lengthscale, expected_covar.base_kernel.lengthscale)
    # `sqrt(3) ** 2` is 2.9999999999999996, not the 3.0 Papenmeier hardcodes, and the constraint's
    # softplus round-trip moves the last bit again: the two agree to a relative 1e-12, not exactly.
    assert base.lengthscale.reshape(-1).tolist() == pytest.approx([_PRIOR_MODE] * D, rel=1e-12)
    # No lengthscale floor: `GreaterThan(2.5e-2)` was the old JAX transcription's, not Papenmeier's.
    assert type(base.raw_lengthscale_constraint).__name__ == "Positive"
    assert float(model.covar_module.outputscale_prior.concentration) == 2.0
    assert float(model.covar_module.outputscale_prior.rate) == 0.15
    noise_prior = model.likelihood.noise_covar.noise_prior
    assert float(noise_prior.concentration) == 1.1
    assert float(noise_prior.rate) == 0.05
    assert float(model.likelihood.noise_covar.raw_noise_constraint.lower_bound) == 1.0e-4
    noise = model.likelihood.noise.detach()
    assert float(noise) == float(expected_likelihood.noise.detach())
    assert float(noise) == pytest.approx((1.1 - 1.0) / 0.05, rel=1e-15)
    assert not hasattr(model, "outcome_transform")
    assert not hasattr(model, "input_transform")
    assert model.train_targets.dtype is torch.float64


def test_oracle_sees_only_the_active_columns():
    # The oracle is the same model behind a `FilterFeatures` transform, which has to do two things
    # at once: give the GP an |S|-dimensional problem -- an |S|-wide ARD kernel under the prior at
    # that width, not the D-dimensional one's -- while every caller, `optimize_acqf` included,
    # keeps handing it full-D points. So the inactive coordinates must not reach the kernel at all,
    # which is what the bit-identical prediction under a perturbed inactive column pins.
    _, X, z = _data()
    active = np.array([1, 4])

    model = references._dsp_model(X, z, active)

    assert isinstance(model.input_transform, FilterFeatures)
    assert model.input_transform.feature_indices.tolist() == [1, 4]
    assert model.covar_module.base_kernel.ard_num_dims == 2
    assert float(model.covar_module.base_kernel.lengthscale_prior.loc) == (
        SQRT2 + math.log(2.0) / 2.0
    )

    fitted = references.fit_map(X, z, active=active)
    X_test = _test_points()
    perturbed = X_test.copy()
    perturbed[:, 0] = 1.0 - perturbed[:, 0]

    assert fitted.cell == "oracle_S"
    assert fitted.samples["kernel_inv_length_sq"].shape == (1, 2)
    assert np.array_equal(np.asarray(fitted.active), active)
    mean, var = fitted.posterior(X_test)
    assert mean.shape == (1, N_TEST) and var.shape == (1, N_TEST)
    mean_p, var_p = fitted.posterior(perturbed)
    assert np.array_equal(np.asarray(mean), np.asarray(mean_p))
    assert np.array_equal(np.asarray(var), np.asarray(var_p))


def test_map_objective_matches_a_from_scratch_recomputation():
    # The check nothing else here can make: the objective `fit_mll` maximized, written again from
    # the model's definition in numpy and scipy and sharing no line with `sagp.references` --
    # explicit ARD Matern-5/2 under a ScaleKernel, `multivariate_normal` for the marginal
    # likelihood and `lognorm`/`gamma` for the three priors. GPyTorch's
    # `ExactMarginalLogLikelihood` takes the priors' log-probabilities on the *constrained* values
    # with no Jacobian term and then divides the whole sum by the number of data points, so the
    # recomputation is compared against `map_result["mll"] * n`.
    _, X, z = _data()

    fitted = references.fit_map(X, z)

    outputscale = float(fitted.samples["outputscale"][0])
    ell = np.asarray(fitted.samples["lengthscale"][0])
    noise = float(fitted.samples["noise"][0])
    constant = float(fitted.samples["mean"][0])

    dsq = np.sum((X[:, None, :] - X[None, :, :]) ** 2.0 / ell**2.0, axis=-1)
    r = math.sqrt(5.0) * np.sqrt(dsq)
    K = outputscale * (1.0 + r + 5.0 / 3.0 * dsq) * np.exp(-r) + noise * np.eye(N)
    expected = (
        multivariate_normal.logpdf(z, mean=np.full(N, constant), cov=K)
        + lognorm(s=SQRT3, scale=math.exp(_PRIOR_LOC)).logpdf(ell).sum()
        + gamma(a=2.0, scale=1.0 / 0.15).logpdf(outputscale)
        + gamma(a=1.1, scale=1.0 / 0.05).logpdf(noise)
    )

    assert fitted.map_result["mll"] * N == pytest.approx(expected, rel=1e-8)


# --- the fit ---


def test_map_lengthscales_are_shorter_on_the_active_set():
    # What makes this a *reference* rather than an arbitrary GP: the dimension-scaled prior is
    # meant to let the marginal likelihood find the few coordinates that matter, and on `aligned3`
    # -- three active coordinates out of six -- it has to.
    objective, X, z = _data()

    fitted = references.fit_map(X, z)

    ell = np.asarray(fitted.samples["lengthscale"][0])
    S = list(objective.labels.S)
    off = [i for i in range(D) if i not in S]
    assert np.mean(ell[S]) < np.mean(ell[off])


def test_fit_map_is_deterministic():
    # `fit_gpytorch_mll` re-samples the free parameters from their priors on every retry after the
    # first, out of torch's global RNG, so without the seeded fork inside `fit_map` a run that
    # reports the DSP reference's regret would not be reproducible from its own seed alone.
    _, X, z = _data()

    first, second = references.fit_map(X, z), references.fit_map(X, z)

    assert first.samples.keys() == second.samples.keys()
    for site, draws in first.samples.items():
        assert np.array_equal(np.asarray(draws), np.asarray(second.samples[site]))
    assert first.map_result == second.map_result


def test_fit_map_reports_the_fallback(monkeypatch):
    # Papenmeier's `fit_mll` prints where this has to record: a fit that only survived the Adam
    # fallback is still a usable surrogate and stays "ok", while one where Adam hit a
    # `NotPSDError` too was left at its initial values and is excluded, exactly as a NUTS fit that
    # failed its diagnostics is. Both reach the run log through `map_result["fallback"]`.
    _, X, z = _data()
    real_step, taken, ADAM_STEPS = torch.optim.Adam.step, itertools.count(), 20

    def failing_fit(mll, **kwargs):
        raise ModelFittingError("x")

    def truncated_step(self, *args, **kwargs):
        """Papenmeier's 500 Adam steps, of which only the first `ADAM_STEPS` really step.

        `get_gaussian_likelihood_with_gamma_prior` builds `GreaterThan(1e-4, transform=None)`, so
        `raw_noise` *is* the noise and the floor is enforced by the box bounds `fit_gpytorch_mll`
        hands L-BFGS-B -- which Adam does not have. At lr 0.01 the full 500 steps walk the noise
        from its starting 2.0 down through zero, the loss goes NaN, and the model this reference
        would return is not positive definite: with this data the fallback ends in the very
        `NotPSDError` the second half of this test injects. That is a property of the optimizer
        being transcribed, not of the bookkeeping under test, so the loop is cut short here
        instead -- far enough in for the lengthscales to have moved, short of the cliff.
        """
        return real_step(self, *args, **kwargs) if next(taken) < ADAM_STEPS else None

    monkeypatch.setattr(references, "fit_gpytorch_mll", failing_fit)
    monkeypatch.setattr(torch.optim.Adam, "step", truncated_step)
    fitted = references.fit_map(X, z)

    assert fitted.status == "ok"
    assert fitted.status_reason == "fit_gpytorch_mll failed; Adam fallback"
    assert fitted.map_result["fallback"] == "adam"
    assert next(taken) == 500  # `n_iter`, Papenmeier's own count
    # Adam ran: the lengthscales are no longer the prior mode they were all seeded at.
    ell = np.asarray(fitted.samples["lengthscale"][0])
    assert np.all(np.abs(ell - _PRIOR_MODE) > 1.0e-3)

    def failing_step(self, *args, **kwargs):
        raise NotPSDError("y")

    monkeypatch.setattr(torch.optim.Adam, "step", failing_step)
    failed = references.fit_map(X, z)

    assert failed.status == "excluded"
    assert failed.status_reason == "fit_gpytorch_mll failed; Adam fallback failed: NotPSDError"
    assert failed.map_result["fallback"] == "failed"
    # Nothing stepped, so the excluded fit is the prior mode it started at.
    assert np.asarray(failed.samples["lengthscale"][0]).tolist() == pytest.approx(
        [_PRIOR_MODE] * D, rel=1e-12
    )


def test_dsp_posterior_has_one_draw():
    # The interface the BO loop and the readouts see: S = 1, so every per-draw quantity is one
    # entry long and `posterior` is (1, n_test) exactly as a cell's is at S = 1.
    _, X, z = _data()

    fitted = references.fit_map(X, z)
    mean, var = fitted.posterior(_test_points())

    assert fitted.cell == "dsp_map"
    assert fitted.active is None
    assert fitted.status == "ok" and fitted.status_reason == ""
    assert fitted.param_sites() == ("outputscale", "kernel_inv_length_sq")
    assert mean.shape == (1, N_TEST) and var.shape == (1, N_TEST)
    assert np.all(np.isfinite(mean)) and np.all(np.asarray(var) > 0.0)
    assert fitted.means().shape == (1,)
    assert fitted.noises().shape == (1,)
    assert fitted.alphas().shape == (1, N)


# --- the Sobol search ---


def test_propose_sobol_walks_the_sequence():
    # The quasi-random reference fits nothing, so it reaches `run_bo` as a proposer: row `t` of a
    # sequence fixed before the run, and no acquisition value to report.
    sequence = np.arange(30.0).reshape(10, 3)

    x, acq_value = references.propose_sobol(None, 0.0, None, 4, sequence=sequence)

    assert np.array_equal(x, sequence[4])
    assert math.isnan(acq_value)
