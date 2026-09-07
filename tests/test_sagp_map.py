"""Tests for sagp.gp's MAP references: DSP-by-MAP and the oracle restricted to S.

The BO study has two references that are not one of the four NUTS-fitted cells but that the loop
must not be able to tell apart from one: Hvarfner et al. 2024's "vanilla BO" -- an ARD Matern-5/2
under the dimension-scaled lengthscale prior, fitted by MAP and served as a single posterior
"sample" -- and that same model handed the objective's true active set. Both halves are pinned
here: that the MAP fit is a fit (its objective falls from the prior mode, and the lengthscales it
learns are shorter on the active coordinates than off them), and that what comes back is a
`FittedGP` the loop can query -- `posterior` on full-D test points, shapes (1, n_test), whichever
reference produced it.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp as gp

import warnings

import numpy as np
from scipy.stats import qmc

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
    assert oracle.samples["kernel_inv_length_sq"].shape == (1, len(objective.labels.S))
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
