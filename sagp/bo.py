"""sagp.bo: the Bayesian-optimization loop -- one loop, the methods differing only in adapters.

`run_bo` is the loop every method shares, differing only in its `surrogate`, `propose` and
`on_iteration` adapters. The acquisition is BoTorch's analytic `LogExpectedImprovement`, maximized
by `optimize_acqf` under Papenmeier et al. (2025)'s protocol: no acquisition code is copied into
this repo any more, and the optimizer is BoTorch's own rather than a copy of the reference's. The
loop maximizes, as the thesis's objectives do, so `gp.standardize` hands the fit
`z = (y - mean) / std` with no sign flip and the incumbent the acquisition improves on is
`best_f = max(z)`.
"""
from __future__ import annotations

import time
import traceback
import warnings
from collections.abc import Callable
from typing import NamedTuple

import gpytorch
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from scipy.stats import qmc

from sagp.gp import FittedGP, standardize
from synthobj.families import noise_rng


# --- seeding ---


class IterRNG(NamedTuple):
    """Every random draw iteration `t` needs, derived from `(seed, t)` and nothing else.

    Six streams, not one: they are consumed by four libraries, and a single counter would couple
    draws that have to be able to move independently -- the fit's retry must not rerun the chain
    the first attempt already ran, and the acquisition maximizer needs two seeds that reach two
    different generators.
    """

    nuts_seed: int  # BoTorch's int seed for the fit: PRNGKey(nuts_seed)
    retry_seed: int  # the fit's one retry after an exception
    sobol_seed: int  # optimize_acqf's options["seed"]: the raw Sobol candidates
    torch_seed: int  # torch.manual_seed in propose_ei: the RAASP draws and the Boltzmann starts
    noise_rng: np.random.Generator  # `synthobj`'s observation noise for the point chosen at t
    fallback_seed: int  # scramble seed of the Sobol point queried when the fit or maximizer fails


def _salted_seed(seed: int, t: int, salt: int) -> int:
    """A 32-bit seed from `(seed, t)` under `salt`: one stream's entry, disjoint from the rest."""
    return int(np.random.SeedSequence([seed, t, salt]).generate_state(1)[0])


def iteration_rngs(seed: int, t: int) -> IterRNG:
    """All randomness of iteration `t`, as a pure function of `(seed, t)`."""
    return IterRNG(
        nuts_seed=_salted_seed(seed, t, 0x4E),
        retry_seed=_salted_seed(seed, t, 0x2E),
        sobol_seed=_salted_seed(seed, t, 0xCA),
        torch_seed=_salted_seed(seed, t, 0x70),
        noise_rng=noise_rng(seed, run=t),
        fallback_seed=_salted_seed(seed, t, 0xFA),
    )


def initial_design(D: int, n_init: int, seed: int) -> np.ndarray:
    """The reference's initial design: `n_init` points of a scrambled Sobol sequence on [0,1]^D.

    Identical per seed, so comparisons are paired; `qmc.Sobol`'s power-of-two warning is muted.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return qmc.Sobol(D, scramble=True, seed=seed).random(n_init)


# --- the acquisition step ---


def propose_ei(
    fitted: FittedGP,
    best_f: float,
    rngs: IterRNG,
    t: int,
    *,
    raw_samples: int = 512,
    num_restarts: int = 5,
    sample_around_best_sigma: float = 1e-3,
    batch_limit: int = 1,
    maxiter: int = 200,
) -> tuple[np.ndarray, float]:
    """Papenmeier et al. (2025)'s acquisition step: BoTorch's LogEI, maximized by `optimize_acqf`.

    His settings, and why the ones that are not BoTorch defaults are here. The `num_restarts=5`
    L-BFGS-B starts are Boltzmann-sampled from `raw_samples=512` raw Sobol candidates, and
    `batch_limit=1` optimizes them one at a time: batching changes the L-BFGS-B iterates, not
    only the speed, because scipy then descends on their sum. `sample_around_best` appends
    `raw_samples` further candidates perturbed from the top 5 % of the design at sigma = 1e-3
    (RAASP; from D = 20 up BoTorch also perturbs a random coordinate subset, each coordinate with
    probability min(20/D, 1)). `maxiter=200` is BoTorch's own default, named because it is part of
    the protocol, and `cholesky_max_tries(9)` is his loop's: a fully Bayesian posterior on a
    growing design fails at GPyTorch's default of 3 often enough to cost iterations their model.

    Two seeds, because `options["seed"]` reaches only the raw Sobol draw: the RAASP perturbations
    and the Boltzmann selection come from torch's *global* RNG, so `rngs.torch_seed` is set inside
    `fork_rng` -- seeded, without any generator outside this call moving. `t` is unused; it
    belongs to the proposer protocol, for the proposers that walk a sequence.
    """
    D = fitted.X_train.shape[1]
    bounds = torch.stack([torch.zeros(D, dtype=torch.float64), torch.ones(D, dtype=torch.float64)])
    acq = LogExpectedImprovement(model=fitted.model, best_f=best_f)
    options = {
        "batch_limit": batch_limit, "maxiter": maxiter, "sample_around_best": True,
        "sample_around_best_sigma": sample_around_best_sigma, "seed": rngs.sobol_seed,
    }
    with torch.random.fork_rng(), gpytorch.settings.cholesky_max_tries(9):
        torch.manual_seed(rngs.torch_seed)
        x, value = optimize_acqf(
            acq, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
            options=options,
        )
    return x.detach().numpy()[0], float(value)


# --- the loop ---


class BOState(NamedTuple):
    """Every point evaluated so far: the design, its observations, and the noise-free values."""

    X: np.ndarray  # (t, D), the design in the unit cube
    y: np.ndarray  # (t,), the observations the loop optimizes against -- noisy unless `noiseless`
    f: np.ndarray  # (t,), the objective's own noise-free values, which the regret is taken on


class Iteration(NamedTuple):
    """What one iteration produced, handed whole to `on_iteration` after the append."""

    t: int
    x: np.ndarray  # the point chosen at t
    y: float  # its observation, and
    f: float  # the objective's noise-free value there
    state: BOState
    fitted: FittedGP | None  # None for a model-free method, and for a fit that raised twice
    acq_value: float  # in the acquisition's own units; NaN when nothing was optimized
    fit_wall_s: float
    acq_wall_s: float
    fit_calls: int
    status: str  # the fit's "ok"/"excluded", or "excluded" for either fallback
    reason: str
    y_mean: float  # the standardization this iteration fitted under (`gp.standardize`)
    y_std: float


def _fit_with_retry(
    t: int,
    surrogate: Callable[..., FittedGP] | None,
    X: np.ndarray,
    z: np.ndarray,
    rngs: IterRNG,
    log: Callable[[str], None],
) -> tuple[FittedGP | None, int, float, str | None]:
    """Fit, retrying once on an exception; returns (fitted, fit_calls, wall_s, failure reason).

    A fit whose *diagnostics* failed is not a failure here: `fit` returns it with
    `status="excluded"` and the loop queries with it anyway, since stopping at every bad chain
    would bias the regret. An *exception* leaves no surrogate: logged, then retried on
    `rngs.retry_seed`, a stream of its own, so the retry cannot rerun the chain `rngs.nuts_seed`
    already ran.
    """
    if surrogate is None:
        return None, 0, float("nan"), None
    start = time.perf_counter()
    try:
        fitted = surrogate(X, z, rngs.nuts_seed)
        return fitted, 1, time.perf_counter() - start, None
    except Exception:
        log(f"t={t}: the fit raised; retrying once\n{traceback.format_exc()}")
    try:
        fitted = surrogate(X, z, rngs.retry_seed)
        return fitted, 2, time.perf_counter() - start, None
    except Exception as error:
        log(f"t={t}: the retry raised; querying at random\n{traceback.format_exc()}")
        return None, 2, time.perf_counter() - start, f"exception: {type(error).__name__}: {error}"


def _propose_with_fallback(
    t: int, propose: Callable[..., tuple[np.ndarray, float]], fitted: FittedGP | None,
    best_f: float, rngs: IterRNG, D: int, log: Callable[[str], None],
) -> tuple[np.ndarray, float, float, str | None]:
    """`propose(fitted, best_f, rngs, t)`, under Papenmeier's policy on an exception.

    His loop catches whatever the maximizer raises -- a Cholesky failure on the posterior above
    all -- logs it and queries at random rather than ending the run. That random point is the same
    seeded Sobol point a twice-failed fit takes, and the reason returned marks the row.
    """
    start = time.perf_counter()
    try:
        x, acq_value = propose(fitted, best_f, rngs, t)
        return x, float(acq_value), time.perf_counter() - start, None
    except Exception as error:
        log(f"t={t}: the proposer raised; querying at random\n{traceback.format_exc()}")
        return (
            initial_design(D, 1, rngs.fallback_seed)[0], float("nan"),
            time.perf_counter() - start, f"exception: {type(error).__name__}: {error}",
        )


def _observe(objective: object, x: np.ndarray, noiseless: bool, rngs: IterRNG) -> float:
    """One observation of the objective at `x`: noisy through `rngs.noise_rng`, or `f(x)` itself."""
    if noiseless:
        return float(objective(x))
    return float(objective.observe(x[None, :], rngs.noise_rng)[0])


def run_bo(
    objective: object,
    surrogate: Callable[..., FittedGP] | None,
    seed: int,
    *,
    T: int = 200,
    n_init: int = 20,
    propose: Callable[..., tuple[np.ndarray, float]] = propose_ei,
    noiseless: bool = False,
    state: BOState | None = None,
    on_iteration: Callable[[Iteration], None] | None = None,
    log: Callable[[str], None] = print,
) -> BOState:
    """Run `T` evaluations of `objective` under `surrogate` and `propose`; returns the final state.

    The first `n_init` points are the shared Sobol design of `seed` and are not iterations; from `t
    = n_init` on, each iteration standardizes, fits, maximizes the acquisition, evaluates, appends
    and hands itself to `on_iteration`. The adapters: `surrogate(X, z, seed) -> FittedGP` is the
    fit, `None` being the model-free method, which costs no fit call and reports a NaN
    `fit_wall_s`, and `propose(fitted, best_f, rngs, t) -> (x, acq_value)` chooses the next point.
    `state` is a resume: iterating starts at `len(state.X)` and `n_init` is ignored.

    `z` is `standardize(state.y)`'s first return -- standardized, not negated, since the loop
    maximizes what `objective` maximizes -- and `best_f = max(z)` is the incumbent the acquisition
    improves on. `rngs` is `iteration_rngs(seed, t)`, an `IterRNG`. Two failures fall back on the
    same seeded Sobol point and mark the row "excluded": a fit that raised twice, leaving nothing
    to query with, and a proposer that raised, which is Papenmeier's Cholesky policy. `log(str)`
    takes those exception messages and defaults to `print`; `on_iteration(Iteration)` runs after
    each append, `Iteration.state` being the very arrays this function goes on to return.
    """
    D = objective.D
    if state is None:
        X = initial_design(D, n_init, seed)
        f = np.asarray(objective(X), dtype=float)
        y = np.array(
            [_observe(objective, X[t], noiseless, iteration_rngs(seed, t)) for t in range(n_init)]
        )
        state = BOState(X, y, f)

    for t in range(len(state.X), T):
        z, y_mean, y_std = standardize(state.y)
        best_f = float(np.max(z))
        rngs = iteration_rngs(seed, t)

        fitted, fit_calls, fit_wall_s, failure = _fit_with_retry(
            t, surrogate, state.X, z, rngs, log
        )
        if failure is not None:
            # Both attempts raised: query at random rather than end the run, and mark the row so
            # the analysis can count these rather than read them as ordinary iterations.
            x_next = initial_design(D, 1, rngs.fallback_seed)[0]
            acq_value, acq_wall_s = float("nan"), float("nan")
            status, reason = "excluded", failure
        else:
            x_next, acq_value, acq_wall_s, failure = _propose_with_fallback(
                t, propose, fitted, best_f, rngs, D, log
            )
            status, reason = ("ok", "") if fitted is None else (fitted.status, fitted.status_reason)
            if failure is not None:
                status, reason = "excluded", failure

        # A proposer is an argument, so what it returns is normalized here: the state below is
        # numpy's, whatever the proposer computed in.
        x_next = np.asarray(x_next, dtype=float)
        f_next = float(objective(x_next))
        y_next = _observe(objective, x_next, noiseless, rngs)

        state = BOState(
            np.vstack([state.X, x_next]),
            np.append(state.y, y_next),
            np.append(state.f, f_next),
        )
        if on_iteration is not None:
            on_iteration(Iteration(
                t=t, x=x_next, y=y_next, f=f_next, state=state, fitted=fitted,
                acq_value=acq_value, fit_wall_s=fit_wall_s, acq_wall_s=acq_wall_s,
                fit_calls=fit_calls, status=status, reason=reason, y_mean=y_mean, y_std=y_std,
            ))

    return state


if __name__ == "__main__":
    raise SystemExit("sagp.bo has no command line; run python -m experiments.run_bo instead")
