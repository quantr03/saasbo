"""Tests for `sagp.bo`: the acquisition, the seeded reference optimizer, and the initial design.

`log_h` and `log1mexp` are the pieces of LogEI that exist only for numerical reasons, so they are
checked against the naive formulas where those are accurate and against finiteness where they are
not, and `log_ei` is checked to be the log of the reference's own EI wherever the reference has not
underflowed. Then `optimize_ei`, whose five-line departure from `saasbo.optimize_ei` must leave it
a maximizer under both acquisitions. Last the initial design, whose first `n_init` points must be
the run's own Sobol sequence -- what makes the Sobol reference a continuation of that design
rather than a new one.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erfcx
from scipy.stats import norm, qmc

import saasbo
from sagp.bo import (
    ACQUISITIONS,
    initial_design,
    log1mexp,
    log_ei,
    log_ei_sum,
    log_h,
    optimize_ei,
)
from sagp.gp import FittedGP


def _hand_made_gp() -> FittedGP:
    """A product/lengthscale `FittedGP` built by hand (S = 3, D = 4, n = 12), no inference involved.

    The acquisition only ever sees `posterior`, so a fitted GP assembled from arbitrary but valid
    draws exercises it exactly as a real fit would, deterministically and in milliseconds.
    """
    rng = np.random.default_rng(0)
    X = rng.random((12, 4))
    y = rng.standard_normal(12)
    samples = {
        "kernel_var": jnp.asarray([1.0, 0.7, 1.3]),
        "kernel_inv_length_sq": jnp.asarray(rng.gamma(2.0, 1.0, size=(3, 4))),
        "kernel_noise": jnp.asarray([1.0e-3, 5.0e-3, 2.0e-3]),
    }
    return FittedGP(
        cell=("product", "lengthscale"),
        X_train=X,
        Y_train=y,
        samples=samples,
        fixed_noise=None,
        active=None,
        status="ok",
        status_reason="",
        attempts=(),
    )



# --- the acquisition ---


def _log_h_reference(z: float) -> float:
    """log h(z) for z < 0 in float64 numpy/scipy: the tail reference `log_h` is checked against.

    h(z) = phi(z) [1 - (-z) Phi(z)/phi(z)] with the Mills ratio written as
    Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt 2), so the bracket is a `log1p` of a quantity that
    never needs the two cancelling O(z^2) terms. `scipy.special.erfcx` and `numpy.log1p` are the
    only arithmetic here, which is what makes this independent of the JAX code under test.
    """
    ratio = -z * np.sqrt(np.pi / 2.0) * erfcx(-z / np.sqrt(2.0))
    return -0.5 * z * z - 0.5 * np.log(2.0 * np.pi) + float(np.log1p(-ratio))


def test_log_h_matches_naive_and_stays_finite():
    z = np.linspace(-6.0, 6.0, 200)
    h = norm.pdf(z) + z * norm.cdf(z)
    got = np.asarray(log_h(jnp.asarray(z)))
    assert np.all(np.abs(np.exp(got) - h) <= 1e-12 * np.maximum(1.0, h))

    # Where h itself underflows, only the log form survives: it must stay finite, keep decreasing,
    # and keep a finite gradient, since that gradient is what the L-BFGS-B restarts follow.
    tails = [float(log_h(jnp.asarray(z_far))) for z_far in (-10.0, -20.0, -40.0)]
    grads = [float(jax.grad(log_h)(z_far)) for z_far in (-10.0, -20.0, -40.0)]
    assert all(np.isfinite(tails)) and all(np.isfinite(grads))
    assert tails[0] > tails[1] > tails[2]
    assert norm.pdf(-40.0) + -40.0 * norm.cdf(-40.0) == 0.0  # the naive form has no value here

    # Five decades of tail against an independent scipy reference (ruling R45). "Finite and
    # decreasing" is not enough on its own: the form this replaced satisfied all of the above and
    # was still 0.04 out at z = -5e3, NaN by -3e4 and 3x wrong in the gradient at -1e5, because it
    # subtracted two O(z^2) terms. `erfcx` is the scaled complementary error function both forms
    # are written in, so this reference shares no line of code with `log_h` but is exact.
    for z_far in (-10.0, -100.0, -1000.0, -10000.0, -100000.0):
        value, grad = float(log_h(jnp.asarray(z_far))), float(jax.grad(log_h)(z_far))
        assert np.isfinite(value) and np.isfinite(grad)
        assert value == pytest.approx(_log_h_reference(z_far), rel=1e-8)
        # Central differences at a relative step: the reference has no closed-form derivative
        # here that is any better conditioned than the value it differentiates.
        step = 1e-6 * abs(z_far)
        finite_difference = (
            _log_h_reference(z_far + step) - _log_h_reference(z_far - step)
        ) / (2.0 * step)
        assert grad == pytest.approx(finite_difference, rel=1e-5)

    x = np.linspace(-30.0, -1.0e-3, 500)
    naive = np.log(1.0 - np.exp(x))
    # `1 - exp(x)` cancels as x -> 0 (three digits at x = -1e-3), so the naive reference is only
    # accurate to its own conditioning; the tolerance is the brief's 1e-14 plus exactly that.
    conditioning = 4.0 * np.finfo(float).eps * np.exp(x) / (-np.expm1(x))
    assert np.all(np.abs(np.asarray(log1mexp(jnp.asarray(x))) - naive) <= 1e-14 + conditioning)


def test_log_ei_equals_log_of_reference_ei():
    gp = _hand_made_gp()
    rng = np.random.default_rng(7)
    X_test = rng.random((20, 4))
    y_target = float(np.asarray(gp.Y_train).min())

    reference = np.asarray(saasbo.ei(X_test, y_target, gp))
    ours = np.asarray(log_ei(X_test, y_target, gp))
    usable = reference > 1e-100
    assert usable.sum() == 20
    assert np.max(np.abs(ours[usable] - np.log(reference[usable]))) < 1e-9

    # 40 standard deviations below the posterior: every per-sample EI underflows to exactly zero,
    # so the reference stops distinguishing candidates and stops having a gradient. LogEI does not.
    mu, var = gp.posterior(X_test)
    far = float(np.asarray(mu).min() - 40.0 * float(np.sqrt(np.asarray(var)).max()))
    assert np.all(np.asarray(saasbo.ei(X_test, far, gp)) == 0.0)
    values = np.asarray(log_ei(X_test, far, gp))
    assert np.all(np.isfinite(values))
    grad = np.asarray(jax.grad(lambda x: log_ei_sum(x, far, gp))(jnp.asarray(X_test)))
    assert np.all(np.isfinite(grad)) and np.abs(grad).max() > 0.0


@pytest.mark.parametrize("acq_name", ["ei", "logei"])
def test_optimize_ei_reference_acq_is_within_bounds_and_improves(acq_name):
    gp = _hand_made_gp()
    acq = ACQUISITIONS[acq_name]
    y_target = float(np.asarray(gp.Y_train).min())

    x_best, value = optimize_ei(
        gp, y_target, sobol_seed=1, jitter_rng=np.random.default_rng(1), acq=acq
    )
    assert x_best.shape == (4,)
    assert np.all(x_best >= 0.0) and np.all(x_best <= 1.0)

    # The candidate set the run's seeds define, rebuilt here: L-BFGS-B starts from its best points
    # and cannot return a worse one, so the reported value is a floor on the whole set's best.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        candidates = qmc.Sobol(4, scramble=True, seed=1).random(5000)
    incumbent = np.asarray(gp.X_train)[int(np.asarray(gp.Y_train).argmin())]
    candidates[0, :] = np.clip(
        incumbent + 0.001 * np.random.default_rng(1).standard_normal((1, 4)), a_min=0.0, a_max=1.0
    )
    best_candidate = float(np.max(np.asarray(acq(jnp.asarray(candidates), y_target, gp))))
    assert value >= best_candidate - 1e-12


# --- the initial design ---


def test_initial_design_is_the_first_rows_of_the_runs_sobol_sequence():
    # What makes the Sobol reference a *continuation* of the shared design rather than a new one.
    assert np.array_equal(initial_design(5, 5, 1), initial_design(5, 15, 1)[:5])
