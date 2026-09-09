"""Tests for `sagp.bo`: the acquisition, the seeding, the initial design, the loop.

Three things are pinned. First, the acquisition the loop maximizes *is* BoTorch's
`LogExpectedImprovement` -- checked against the analytic sample-average formula written out in
numpy/scipy, and against the vendored code's own convention on negated targets, which the sign
flips make the very same number. Second, every random draw of iteration `t` is a function of
`(seed, t)` alone: the acquisition optimizer takes two seeds, one for its raw Sobol candidates and
one for torch's global RNG, and a run that repeats them repeats the point. Third, the two
fallbacks -- a fit that raises twice, and a proposer that raises -- both query the same seeded
Sobol point and mark the row "excluded". `run_bo` is driven through its adapters alone, with no
run directory and no inference at all, which is what the split is for.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples
from scipy.special import erfcx, logsumexp
from scipy.stats import norm

from sagp import bo
from sagp.bo import BOState, initial_design, iteration_rngs, propose_ei, run_bo
from sagp.gp import FittedGP, standardize
from sagp.references import propose_sobol
from synthobj.families import make_family


def _hand_made_gp() -> FittedGP:
    """A product/lengthscale `FittedGP` built by hand (S = 3, D = 4, n = 12), no inference involved.

    The acquisition only ever sees the fitted model, so a posterior assembled from arbitrary but
    valid draws exercises it exactly as a real fit would, deterministically and in milliseconds.
    """
    rng = np.random.default_rng(0)
    X = rng.random((12, 4))
    y = rng.standard_normal(12)
    rho = rng.gamma(2.0, 1.0, size=(3, 4))
    samples = {
        "outputscale": np.array([1.0, 0.7, 1.3]),
        "mean": np.array([0.0, 0.2, -0.1]),
        "noise": np.array([1.0e-3, 5.0e-3, 2.0e-3]),
        "kernel_inv_length_sq": rho,
        "lengthscale": rho**-0.5,
    }
    return FittedGP.from_draws(("product", "lengthscale"), X, y, samples)


# --- the acquisition ---


def _log_h_reference(z: np.ndarray) -> np.ndarray:
    """log h(z) for z < 0 in float64 numpy/scipy: the tail reference BoTorch is checked against.

    h(z) = phi(z) [1 - (-z) Phi(z)/phi(z)] with the Mills ratio written as
    Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt 2), so the bracket is a `log1p` of a quantity that
    never needs the two cancelling O(z^2) terms. `scipy.special.erfcx` and `numpy.log1p` are the
    only arithmetic here, which is what makes this independent of the code under test.
    """
    z = np.asarray(z, dtype=float)
    ratio = -z * np.sqrt(np.pi / 2.0) * erfcx(-z / np.sqrt(2.0))
    return -0.5 * z * z - 0.5 * np.log(2.0 * np.pi) + np.log1p(-ratio)


def _log_h(z: np.ndarray) -> np.ndarray:
    """log h(z), h(z) = phi(z) + z Phi(z), over all z: naive above -1, the erfcx tail below it.

    Each branch runs on inputs clamped into its own domain, because `np.where` evaluates both
    everywhere and the naive branch's log(0) far into the tail would warn.
    """
    z = np.asarray(z, dtype=float)
    upper = np.maximum(z, -1.0)
    naive = np.log(norm.pdf(upper) + upper * norm.cdf(upper))
    return np.where(z > -1.0, naive, _log_h_reference(np.minimum(z, -1.0)))


def test_log_ei_is_botorchs_ensemble_log_expected_improvement():
    """What the loop maximizes, written out: logmeanexp over draws of log(sigma) + log h(u).

    BoTorch's `LogExpectedImprovement` clamps the *latent* variance at 1e-12 and its
    `average_over_ensemble_models` decorator reduces the draw dimension by `logmeanexp`, so the
    whole acquisition is this one line of numpy over `FittedGP.posterior(..., False)`.
    """
    gp = _hand_made_gp()
    X_test = np.random.default_rng(7).random((9, 4))
    best_f = float(np.asarray(gp.Y_train).max())

    acq = LogExpectedImprovement(gp.model, best_f=best_f)
    ours = acq(torch.as_tensor(X_test)[:, None, :]).detach().numpy()
    assert ours.shape == (9,)

    mu, var = (np.asarray(moment) for moment in gp.posterior(X_test, observation_noise=False))
    sigma = np.maximum(np.sqrt(var), 1e-6)  # BoTorch's min_var = 1e-12 on the variance
    per_draw = np.log(sigma) + _log_h((mu - best_f) / sigma)
    expected = logsumexp(per_draw, axis=0) - np.log(sigma.shape[0])
    np.testing.assert_allclose(ours, expected, atol=1e-10, rtol=0)

    # The vendored convention on the negated targets is the same number: EI of (y_target - (-mu))
    # against y_target = -best_f is EI of (mu - best_f), both sign flips cancelling. This is what
    # licenses `standardize` to stop negating without the acquisition changing meaning.
    y_target = -best_f
    old = logsumexp(np.log(sigma) + _log_h((y_target - (-mu)) / sigma), 0) - np.log(sigma.shape[0])
    np.testing.assert_array_equal(old, expected)

    # 40 standard deviations above the posterior: every per-draw EI underflows to exactly zero, so
    # the reference's own EI would stop distinguishing candidates. The log form does not.
    far = float(mu.max() + 40.0 * sigma.max())
    far_acq = LogExpectedImprovement(gp.model, best_f=far)
    ours_far = far_acq(torch.as_tensor(X_test)[:, None, :]).detach().numpy()
    per_draw_far = np.log(sigma) + _log_h((mu - far) / sigma)
    expected_far = logsumexp(per_draw_far, axis=0) - np.log(sigma.shape[0])
    assert np.all(np.isfinite(ours_far))
    np.testing.assert_allclose(ours_far, expected_far, rtol=1e-8, atol=0)


def test_propose_ei_returns_a_cube_point_with_the_acquisition_value():
    gp = _hand_made_gp()
    best_f = float(np.asarray(gp.Y_train).max())

    x, value = propose_ei(gp, best_f, iteration_rngs(0, 5), 5, raw_samples=64, num_restarts=2)

    assert x.shape == (4,)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.isfinite(value)
    # The reported value is the acquisition at the point returned, not at some other restart's.
    acq = LogExpectedImprovement(gp.model, best_f=best_f)
    at_x = float(acq(torch.as_tensor(x)[None, None, :]).detach())
    assert value == pytest.approx(at_x, rel=1e-8)


def test_propose_ei_is_a_pure_function_of_the_iteration_rngs(monkeypatch):
    """Both seeds matter: `options["seed"]` reaches only the raw Sobol draw, torch's the rest."""
    gp = _hand_made_gp()
    best_f = float(np.asarray(gp.Y_train).max())
    propose = partial(propose_ei, raw_samples=64, num_restarts=2)
    rngs = iteration_rngs(0, 5)

    x, _ = propose(gp, best_f, rngs, 5)
    again, _ = propose(gp, best_f, rngs, 5)
    assert np.array_equal(x, again)

    other_torch, _ = propose(gp, best_f, rngs._replace(torch_seed=rngs.torch_seed + 1), 5)
    assert not np.array_equal(x, other_torch)

    # `sobol_seed` is checked at the candidate set rather than at the point returned. On a problem
    # this small both candidate sets lead L-BFGS-B into the same basin, so the two points agree to
    # about 1e-7 and at some `t` to the bit -- asserting they differ would pin the optimizer's
    # last bits, which any BoTorch or scipy release may move. What must hold is that the seed is
    # handed to `optimize_acqf` as `options["seed"]`, and that it changes the raw draw.
    seen: list = []
    real_optimize_acqf = bo.optimize_acqf

    def recording(acq, **kwargs):
        seen.append(kwargs["options"])
        return real_optimize_acqf(acq, **kwargs)

    monkeypatch.setattr(bo, "optimize_acqf", recording)
    propose(gp, best_f, rngs, 5)
    assert seen[0]["seed"] == rngs.sobol_seed
    bounds = torch.stack([torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)])
    drawn = [draw_sobol_samples(bounds=bounds, n=64, q=1, seed=s)
             for s in (rngs.sobol_seed, rngs.sobol_seed + 1)]
    assert not torch.equal(*drawn)


# --- the seeding ---


def test_iteration_rngs_streams_are_distinct_and_deterministic():
    def seeds(rngs):
        return (
            rngs.nuts_seed, rngs.retry_seed, rngs.sobol_seed, rngs.torch_seed, rngs.fallback_seed
        )

    here = seeds(iteration_rngs(0, 3))
    assert len(set(here)) == len(here)  # one counter would couple the streams; five salts do not
    assert here == seeds(iteration_rngs(0, 3))
    assert not set(here) & set(seeds(iteration_rngs(0, 4)))
    assert not set(here) & set(seeds(iteration_rngs(1, 3)))


# --- the initial design ---


def test_initial_design_is_the_first_rows_of_the_runs_sobol_sequence():
    # What makes the Sobol reference a *continuation* of the shared design rather than a new one.
    assert np.array_equal(initial_design(5, 5, 1), initial_design(5, 15, 1)[:5])


# --- the loop ---


def _objective():
    """The fixed D = 5 problem the loop runs on: three active coordinates, an exact `f_star`."""
    return make_family("aligned3", 0, D=5)


def _stub_surrogate(X, z, seed):
    """A valid product/lengthscale posterior over two draws, built without any inference."""
    samples = {
        "outputscale": np.ones(2),
        "mean": np.zeros(2),
        "noise": np.full(2, 0.01),
        "kernel_inv_length_sq": np.ones((2, 5)),
        "lengthscale": np.ones((2, 5)),
    }
    return FittedGP.from_draws(("product", "lengthscale"), X, z, samples)


_PROPOSE = partial(propose_ei, raw_samples=32, num_restarts=1)


def test_run_bo_core_is_a_pure_function_of_seed_state_and_adapters():
    """The loop's whole contract, with the surrogate stubbed and no run directory anywhere.

    A fit is the expensive part of an iteration and none of it is the loop's business, so the
    surrogate here returns a hand-made posterior: what is pinned is that the trajectory is a
    function of `(seed, state, adapters)` alone -- the same seed gives the same run, a `BOState`
    resumes it exactly where a checkpoint would, `on_iteration` sees the arrays the loop returns,
    and `surrogate=None` is the model-free path the Sobol reference takes.
    """
    obj = _objective()
    seen: list = []

    a = run_bo(obj, _stub_surrogate, 0, T=8, n_init=5, propose=_PROPOSE, on_iteration=seen.append)
    assert [it.t for it in seen] == [5, 6, 7] and a.X.shape == (8, 5) and seen[-1].state.X is a.X
    assert all(it.fit_calls == 1 and it.status == "ok" for it in seen)

    b = run_bo(obj, _stub_surrogate, 0, T=8, n_init=5, propose=_PROPOSE)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)

    c = run_bo(
        obj, _stub_surrogate, 0, T=8, n_init=5, propose=_PROPOSE,
        state=BOState(a.X[:6], a.y[:6], a.f[:6]),
    )
    assert np.array_equal(c.X, a.X)

    s = run_bo(
        obj, None, 3, T=8, n_init=5,
        propose=partial(propose_sobol, sequence=initial_design(5, 8, 3)),
        on_iteration=seen.append,
    )
    assert np.array_equal(s.X, initial_design(5, 8, 3))
    assert seen[-1].fit_calls == 0 and np.isnan(seen[-1].fit_wall_s)


def test_run_bo_hands_the_proposer_the_standardized_incumbent():
    """`best_f` is `max(standardize(y))`: the loop maximizes, and the fit sees the same targets."""
    obj = _objective()
    seen: list = []
    incumbents: list = []

    def propose(fitted, best_f, rngs, t):
        incumbents.append(best_f)
        return _PROPOSE(fitted, best_f, rngs, t)

    run_bo(obj, _stub_surrogate, 0, T=8, n_init=5, propose=propose, on_iteration=seen.append)
    expected = [float(np.max(standardize(it.state.y[: it.t])[0])) for it in seen]
    assert incumbents == expected


def test_a_proposer_that_raises_takes_the_seeded_sobol_fallback():
    obj = _objective()
    seen: list = []
    messages: list = []

    def propose(fitted, best_f, rngs, t):
        raise RuntimeError("boom")

    run_bo(
        obj, _stub_surrogate, 0, T=8, n_init=5, propose=propose,
        on_iteration=seen.append, log=messages.append,
    )

    assert [it.t for it in seen] == [5, 6, 7]
    for it in seen:
        assert it.status == "excluded"
        assert it.reason.startswith("exception: RuntimeError")
        assert np.isnan(it.acq_value)
        assert it.fit_calls == 1  # the fit itself was fine, so it is not retried
        fallback = initial_design(5, 1, iteration_rngs(0, it.t).fallback_seed)[0]
        assert np.array_equal(it.x, fallback)
    assert any("the proposer raised" in message for message in messages)


def test_a_fit_that_raises_twice_takes_the_seeded_sobol_fallback():
    obj = _objective()
    seen: list = []

    def always_raises(X, z, seed):
        raise RuntimeError("no fit")

    run_bo(
        obj, always_raises, 0, T=6, n_init=5, propose=_PROPOSE,
        on_iteration=seen.append, log=lambda message: None,
    )
    assert len(seen) == 1
    assert seen[0].fit_calls == 2 and seen[0].status == "excluded"
    assert seen[0].fitted is None and np.isnan(seen[0].acq_value)
    assert np.array_equal(seen[0].x, initial_design(5, 1, iteration_rngs(0, 5).fallback_seed)[0])

    # One failure is retried on the retry seed, not on the seed the first attempt already used.
    calls: list = []

    def raises_once(X, z, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("first attempt")
        return _stub_surrogate(X, z, seed)

    retried: list = []
    run_bo(
        obj, raises_once, 0, T=6, n_init=5, propose=_PROPOSE,
        on_iteration=retried.append, log=lambda message: None,
    )
    assert retried[0].fit_calls == 2 and retried[0].status == "ok"
    rngs = iteration_rngs(0, 5)
    assert calls == [rngs.nuts_seed, rngs.retry_seed]
