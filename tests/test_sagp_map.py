"""Tests for sagp.gp's MAP references: DSP-by-MAP and the oracle restricted to S.

The BO study has two references that are not one of the four NUTS-fitted cells but that the loop
must not be able to tell apart from one: Hvarfner et al. 2024's "vanilla BO" -- an ARD Matern-5/2
under the dimension-scaled lengthscale prior, fitted by MAP and served as a single posterior
"sample" -- and that same model handed the objective's true active set. Both halves are pinned
here: that the MAP fit is a fit (its objective falls from the prior mode, and the lengthscales it
learns are shorter on the active coordinates than off them), and that what comes back is a
`FittedGP` the loop can query -- `posterior` on full-D test points, shapes (1, n_test), whichever
reference produced it.

Neither of those catches a *mistranscribed* prior, though: the fit would still be a fit and the
lengthscales would still come out short on S under the wrong constants. Since the whole claim of
this reference is that it is Hvarfner et al.'s model and not something nearby, the constants are
pinned directly -- against their arithmetic, and against a from-scratch recomputation of the whole
objective in numpy/scipy that shares no line of code with `sagp.gp`.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp as gp

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.stats import lognorm, multivariate_normal, qmc

from synthobj.families import make_family, noise_rng

# `scipy.stats.qmc.Sobol` warns whenever the point count is not a power of two, and n = 60 is not;
# the design is the one the plan specifies for this check, so the warning is about it rather than
# about a mistake here.
_SOBOL_WARNING = "The balance properties of Sobol' points require n to be a power of 2."

D, N, N_TEST = 10, 60, 7


def _data():
    """`aligned3` at D = 10: a 60-point Sobol design, observed with noise, standardized."""
    objective = make_family("aligned3", 0, D=D)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SOBOL_WARNING)
        X = qmc.Sobol(D, scramble=True, seed=0).random(N)
    z, _, _ = gp.standardize(objective.observe(X, noise_rng(0)))
    return objective, X, z


def _test_points():
    """`N_TEST` full-D points to predict at; fixed, since nothing here is about the design."""
    return np.random.default_rng(0).random((N_TEST, D))


def test_dsp_map_posterior_has_one_sample():
    # The interface the BO loop sees: one "sample", so `posterior` is (1, n_test) exactly as a
    # cell's is at S = 1, and a usable predictive variance at points that are not in the design.
    _, X, z = _data()

    fitted = gp.fit_map(X, z)
    mean, var = fitted.posterior(_test_points())

    assert fitted.cell == "dsp_map"
    assert fitted.active is None
    assert fitted.status == "ok" and fitted.status_reason == ""
    assert mean.shape == (1, N_TEST) and var.shape == (1, N_TEST)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(var))
    assert np.all(np.asarray(var) > 0.0)


def test_map_lengthscales_are_shorter_on_the_active_set():
    # What makes this a *reference* rather than an arbitrary GP: the dimension-scaled prior is
    # meant to let the marginal likelihood find the few coordinates that matter, and on `aligned3`
    # -- three active coordinates out of ten -- it has to.
    objective, X, z = _data()

    fitted = gp.fit_map(X, z)

    ell = np.asarray(fitted.samples["kernel_inv_length_sq"][0]) ** -0.5
    S = list(objective.labels.S)
    off = [i for i in range(D) if i not in S]
    assert np.median(ell[S]) < np.median(ell[off])


def test_oracle_fits_the_active_coordinates_only():
    # The oracle is the same fit on |S| coordinates, and the restriction has to be invisible from
    # outside: it stores the full-D design (the incumbent lookup and the run log read it) and
    # answers full-D test points, while its parameters are only as wide as S.
    objective, X, z = _data()

    oracle = gp.fit_map(X, z, active=np.array(objective.labels.S))
    mean, var = oracle.posterior(_test_points())

    assert oracle.cell == "oracle_S"
    assert isinstance(oracle.active, np.ndarray) and oracle.active.dtype.kind == "i"
    assert objective.labels.S == (4, 8, 9)
    assert oracle.samples["kernel_inv_length_sq"].shape == (1, 3)
    assert oracle.X_train.shape == (N, D)
    assert mean.shape == (1, N_TEST) and var.shape == (1, N_TEST)


def test_map_objective_falls_from_the_prior_mode():
    # L-BFGS-B is started at the priors' modes, so a fit that never moved would still return a
    # sane-looking GP; only the objective at the optimum against the objective at the start
    # distinguishes "fitted" from "returned the initial values".
    _, X, z = _data()

    fitted = gp.fit_map(X, z)

    assert fitted.map_result["nit"] > 0
    assert fitted.map_result["fun"] < fitted.map_result["fun0"]


def test_fit_map_is_deterministic():
    # No PRNG anywhere in the fit: a run that reports the DSP reference's regret must be
    # reproducible from the seed alone, and re-fitting the same data has to give the same GP.
    _, X, z = _data()

    first, second = gp.fit_map(X, z), gp.fit_map(X, z)

    for site, draws in first.samples.items():
        assert np.array_equal(np.asarray(draws), np.asarray(second.samples[site]))


# --- the prior itself ---


def test_prior_constants_are_hvarfners():
    # The transcription from BoTorch (`get_covar_module_with_dim_scaled_prior` and
    # `get_gaussian_likelihood_with_lognormal_prior`), which nothing else here would catch: every
    # test above passes just as well under a prior that is merely prior-shaped.
    assert gp._ELL_FLOOR == 0.025
    assert gp._ELL_PRIOR_SCALE == pytest.approx(math.sqrt(3.0))
    assert gp._ell_prior_loc(10) == pytest.approx(math.sqrt(2.0) + math.log(10.0) / 2.0)
    assert gp._ell_prior_loc(3) == pytest.approx(math.sqrt(2.0) + math.log(3.0) / 2.0)
    assert gp._NOISE_FLOOR == 1.0e-4
    assert gp._NOISE_PRIOR_LOC == -4.0
    assert gp._NOISE_PRIOR_SCALE == 1.0


@pytest.mark.parametrize("D_eff", [D, 3])
def test_start_point_is_the_two_prior_modes(D_eff):
    # BoTorch's initial values are the priors' modes exp(loc - scale^2); L-BFGS-B walks in u, so
    # what has to be the mode is the *constrained* start, not `u0` itself.
    u0 = gp._dsp_start(D_eff)

    ell0 = gp._ELL_FLOOR + np.exp(u0[:D_eff])
    noise0 = gp._NOISE_FLOOR + np.exp(u0[D_eff])
    assert u0.shape == (D_eff + 1,)
    assert ell0 == pytest.approx(math.exp(math.sqrt(2.0) + math.log(D_eff) / 2.0 - 3.0), rel=1e-12)
    assert noise0 == pytest.approx(math.exp(-5.0), rel=1e-12)


@pytest.mark.parametrize("D_eff", [D, 3])
def test_objective_matches_a_from_scratch_recomputation(D_eff):
    # The check the rest of the file cannot make: the objective, written again from the paper's
    # description in numpy and scipy, sharing no line with `sagp.gp` -- explicit ARD Matern-5/2 at
    # unit variance, `scipy.stats.multivariate_normal` for the marginal likelihood and
    # `scipy.stats.lognorm` for the two priors, on the constrained values with no Jacobian. Run at
    # both widths the study uses: the full design and the oracle's |S| = 3.
    n = 20
    rng = np.random.default_rng(3)
    X, y = rng.random((n, D_eff)), rng.standard_normal(n)
    u = 0.3 * rng.standard_normal(D_eff + 1)

    value = float(gp._dsp_neg_log_joint(u, X, y, D_eff))

    ell = 0.025 + np.exp(u[:D_eff])
    noise = 1.0e-4 + np.exp(u[D_eff])
    # `saasgp.matern_kernel` clips the squared distance at 1e-12 before the sqrt, so that the
    # gradient at zero distance stays finite; that moves the diagonal by 2.5e-12, far inside the
    # tolerance below, and is not reproduced here.
    dsq = np.sum((X[:, None, :] - X[None, :, :]) ** 2 / ell**2, axis=-1)
    r = math.sqrt(5.0) * np.sqrt(dsq)
    K = (1.0 + r + 5.0 / 3.0 * dsq) * np.exp(-r) + (noise + 1.0e-6) * np.eye(n)
    ell_prior = lognorm(
        s=math.sqrt(3.0), scale=math.exp(math.sqrt(2.0) + math.log(D_eff) / 2.0)
    )
    expected = -(
        multivariate_normal.logpdf(y, mean=np.zeros(n), cov=K)
        + ell_prior.logpdf(ell).sum()
        + lognorm(s=1.0, scale=math.exp(-4.0)).logpdf(noise)
    )
    assert value == pytest.approx(expected, rel=1e-9)


def test_a_failed_optimization_is_an_excluded_row(monkeypatch):
    # R30: the loop reads `status` the same way whatever produced the fit, so a MAP fit L-BFGS-B
    # could not converge has to be countable as excluded rather than quietly averaged in.
    _, X, z = _data()

    def stub(fun, u0, **kwargs):
        """`minimize` that evaluates the start and reports failure, so no optimizer runs here."""
        value, _ = fun(u0)
        return OptimizeResult(
            x=u0, fun=value, nit=0, success=False, message="ABNORMAL_TERMINATION_IN_LNSRCH"
        )

    monkeypatch.setattr(gp, "minimize", stub)
    fitted = gp.fit_map(X, z)

    assert fitted.status == "excluded"
    assert fitted.status_reason == "L-BFGS-B: ABNORMAL_TERMINATION_IN_LNSRCH"
    assert fitted.map_result["success"] is False
