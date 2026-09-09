"""sagp.bo: the Bayesian-optimization loop -- the vendored SAASBO driver, generalized.

`run_bo` generalizes `saasbo.run_saasbo` into one loop every method shares, differing only in the
`surrogate`, `propose` and `on_iteration` adapters. The acquisition is LogEI (Ament et al. 2023),
the exact log of the reference's sample-averaged EI, and `acq="ei"` is the vendored `saasbo.ei`
itself. `synthobj` maximizes where the vendored code minimizes, so `gp.standardize` negates on the
way in.
"""
from __future__ import annotations

import math
import time
import traceback
import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import Array, value_and_grad
from jax.scipy.special import erfcx, logsumexp
from jax.scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import qmc

import saasbo
from sagp.gp import FittedGP, standardize
from synthobj.families import noise_rng


# --- seeding ---


class IterRNG(NamedTuple):
    """Every random draw iteration `t` needs, derived from `(seed, t)` and nothing else.

    Five streams, not one: four libraries consume them and one counter would couple their draws.
    """

    key: Array  # the fit's PRNG key; `fit` uses it unchanged, so a run can match the reference
    sobol_seed: int  # scramble seed of the acquisition optimizer's candidate set
    jitter_rng: np.random.Generator  # the incumbent's jitter inside `optimize_ei`
    noise_rng: np.random.Generator  # `synthobj`'s observation noise for the point chosen at t
    fallback_seed: int  # scramble seed of the random query used when the fit raises twice


def _salted_seed(seed: int, t: int, salt: int) -> int:
    """A 32-bit seed from `(seed, t)` under `salt`: one stream's entry, disjoint from the rest."""
    return int(np.random.SeedSequence([seed, t, salt]).generate_state(1)[0])


def iteration_rngs(seed: int, t: int) -> IterRNG:
    """All randomness of iteration `t`, as a pure function of `(seed, t)`."""
    return IterRNG(
        key=jax.random.fold_in(jax.random.PRNGKey(seed), t),
        sobol_seed=_salted_seed(seed, t, 0xCA),
        jitter_rng=np.random.default_rng(np.random.SeedSequence([seed, t, 0x1A])),
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


# --- acquisition ---

_LOG2 = math.log(2.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def log1mexp(x: Array) -> Array:
    """log(1 - exp(x)) for x < 0, accurate at both ends (Machler 2012).

    The switch at -log 2 picks the better-conditioned branch, and each branch runs on inputs
    *clamped into its own domain* (and likewise in `log_h`): under `jnp.where` both branches are
    computed everywhere, and a NaN in the one not taken would poison the other's gradient.
    """
    upper = jnp.maximum(x, -_LOG2)  # the naive branch never sees the tail's inputs
    lower = jnp.minimum(x, -_LOG2)  # and the tail branch never sees log1p(-1) = -inf
    # `maximum`/`minimum` split the derivative at a tie, halving d/dx exactly at the branch point.
    return jnp.where(x > -_LOG2, jnp.log(-jnp.expm1(upper)), jnp.log1p(-jnp.exp(lower)))


def log_h(z: Array) -> Array:
    """log h(z), h(z) = phi(z) + z Phi(z): the standardized improvement, stably (Ament et al. 2023).

    h is the whole content of EI, EI(x) = std * h((y_target - mu)/std), and underflows to zero in
    float64 at about z = -37, where the reference's EI loses its gradient. The log form goes
    through `log1mexp` of the paper's erfcx Mills ratio, which keeps the mantissa lost by log
    Phi(z) + z^2/2. As in `log1mexp`, each branch is evaluated on inputs clamped into its own
    domain (`upper`, `lower`): under `jnp.where` both branches run everywhere, and a NaN in the one
    not taken -- the naive branch's 0/0 far into the tail, the tail branch's `log(-z)` at z > 0 --
    would poison the gradient of the branch that is taken.
    """
    upper = jnp.maximum(z, -1.0)
    lower = jnp.minimum(z, -1.0)
    naive = jnp.log(norm.pdf(upper) + upper * norm.cdf(upper))
    log_mills = jnp.log(-lower) + 0.5 * jnp.log(jnp.pi / 2) + jnp.log(erfcx(-lower / jnp.sqrt(2)))
    tail = -0.5 * lower**2 - _HALF_LOG_2PI + log1mexp(log_mills)
    return jnp.where(z > -1.0, naive, tail)


def log_ei(x: Array, y_target: float, gp: FittedGP, xi: float = 0.0) -> Array:
    """log EI at `x`, with `saasbo.ei`'s signature and conventions (minimization, std floor 1e-6).

    `logsumexp(log EI_s) - log S` is the log of the reference's sample-average, so the argmax
    matches `saasbo.ei`'s; a degenerate sample's NaN becomes `finfo.min`, neutral in `logsumexp`.
    """
    mu, var = gp.posterior(x)
    std = jnp.maximum(jnp.sqrt(var), 1e-6)
    scaled = (y_target - xi - mu) / std
    values = jnp.nan_to_num(jnp.log(std) + log_h(scaled), nan=-jnp.inf)
    return logsumexp(values, axis=0) - jnp.log(values.shape[0])


def log_ei_sum(x: Array, y_target: float, gp: FittedGP, xi: float = 0.0) -> Array:
    """`log_ei(...).sum()`: `saasbo.ei_grad`'s analogue, the scalar L-BFGS-B differentiates."""
    return log_ei(x, y_target, gp, xi).sum()


# "ei" is the vendored reference itself, so the reproduction test can run this loop against
# `run_saasbo` with nothing but the seeding changed.
ACQUISITIONS: dict[str, Callable[..., Array]] = {"logei": log_ei, "ei": saasbo.ei}


def optimize_ei(
    gp: FittedGP,
    y_target: float,
    sobol_seed: int,
    jitter_rng: np.random.Generator,
    xi: float = 0.0,
    num_restarts_ei: int = 5,
    num_init: int = 5000,
    acq: Callable[..., Array] = log_ei,
) -> tuple[np.ndarray, float]:
    """`saasbo.optimize_ei` with five changed lines; `acq=saasbo.ei` reproduces it exactly.

    The changes: the candidate Sobol set and the jitter come from `iteration_rngs`, not global
    state; the acquisition is a callable where the reference names `ei`; the value is returned too.
    The rest is the reference's -- 5000 candidates, jitter 1e-3, `top_k`, `maxfun=100`, bounds.
    """

    # Helper function for optimizing EI
    def negative_ei_and_grad(x, y_target, gp, xi):
        # Compute EI and its gradient and then flip the signs since L-BFGS-B minimizes
        x = jnp.array(x.copy())[None, :]
        ei_val, ei_val_grad = value_and_grad(lambda x, yt, g, xi: acq(x, yt, g, xi).sum())(
            x, y_target, gp, xi
        )
        return -1 * ei_val.item(), -1 * np.array(ei_val_grad)

    dim = gp.X_train.shape[-1]
    with warnings.catch_warnings(record=True):  # Suppress qmc.Sobol UserWarning
        X_rand = qmc.Sobol(dim, scramble=True, seed=sobol_seed).random(num_init)

    # Make sure x_best is in the set of candidate EI maximizers
    x_best = gp.X_train[gp.Y_train.argmin(), :]
    X_rand[0, :] = np.clip(
        x_best + 0.001 * jitter_rng.standard_normal((1, dim)), a_min=0.0, a_max=1.0
    )
    X_rand = jnp.array(X_rand)

    ei_rand = acq(X_rand, y_target, gp)
    _, top_inds = lax.top_k(ei_rand, num_restarts_ei)
    X_init = X_rand[top_inds, :]

    x_best, y_best = None, -float("inf")
    for x0 in X_init:
        x, fx, _ = fmin_l_bfgs_b(
            func=negative_ei_and_grad,
            x0=x0,
            fprime=None,
            bounds=[(0.0, 1.0) for _ in range(dim)],
            args=(y_target, gp, 0.0),
            maxfun=100,  # this limits computational cost
        )
        fx = -1 * fx  # Back to maximization

        if fx > y_best:
            x_best, y_best = x.copy(), fx

    return x_best, y_best


# --- the loop ---


def propose_ei(
    fitted: FittedGP,
    y_target: float,
    rngs: IterRNG,
    t: int,
    *,
    acq: Callable[..., Array] = log_ei,
    num_init: int = 5000,
    num_restarts_ei: int = 5,
) -> tuple[np.ndarray, float]:
    """`optimize_ei` driven by iteration `t`'s streams: the loop's default proposer.

    `t` is unused here; it belongs to the proposer protocol, for those that walk a sequence.
    """
    x, value = optimize_ei(
        fitted,
        y_target,
        rngs.sobol_seed,
        rngs.jitter_rng,
        num_restarts_ei=num_restarts_ei,
        num_init=num_init,
        acq=acq,
    )
    return np.asarray(x, dtype=float), float(value)


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
    status: str  # the fit's "ok"/"refit"/"excluded", or "excluded" for a fit that raised twice
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
    `fold_in(rngs.key, 2)`, disjoint from both keys `fit` uses itself (`key` for attempt 0 and
    `fold_in(key, 1)` for its refit) -- changing either constant would make this retry rerun
    `fit`'s refit chain.
    """
    if surrogate is None:
        return None, 0, float("nan"), None
    start = time.perf_counter()
    try:
        fitted = surrogate(X, z, rngs.key)
        return fitted, 1, time.perf_counter() - start, None
    except Exception:
        log(f"t={t}: the fit raised; retrying once\n{traceback.format_exc()}")
    try:
        fitted = surrogate(X, z, jax.random.fold_in(rngs.key, 2))
        return fitted, 2, time.perf_counter() - start, None
    except Exception as error:
        log(f"t={t}: the retry raised; querying at random\n{traceback.format_exc()}")
        return None, 2, time.perf_counter() - start, f"exception: {type(error).__name__}: {error}"


def _observe(
    objective: object, x: np.ndarray, noiseless: bool, rngs: IterRNG
) -> float:
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
    and hands itself to `on_iteration`. The adapters: `surrogate(X, z, key) -> FittedGP` is the
    fit, `None` being the model-free method, which costs no fit call and reports a NaN
    `fit_wall_s`, and `propose(fitted, y_target, rngs, t) -> (x, acq_value)` chooses the next
    point. `state` is a resume: iterating starts at `len(state.X)` and `n_init` is ignored.

    `z` is `standardize(state.y)`'s first return: the standardized *and negated* targets (the loop
    maximizes `objective`; the vendored code path minimizes `z`), and `y_target = min(z)`. `rngs`
    is `iteration_rngs(seed, t)`, an `IterRNG`. `log(str)` takes the two exception-policy messages
    (the fit raised; the retry raised) and defaults to `print`; `on_iteration(Iteration)` runs after
    each append, `Iteration.state` being the very arrays this function goes on to return.
    """
    D = objective.D
    if state is None:
        X = initial_design(D, n_init, seed)
        f = np.asarray(objective(X), dtype=float)
        y = np.array(
            [
                _observe(objective, X[t], noiseless, iteration_rngs(seed, t))
                for t in range(n_init)
            ]
        )
        state = BOState(X, y, f)

    for t in range(len(state.X), T):
        z, y_mean, y_std = standardize(state.y)
        y_target = float(np.min(z))
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
            start = time.perf_counter()
            x_next, acq_value = propose(fitted, y_target, rngs, t)
            acq_wall_s = time.perf_counter() - start
            status, reason = ("ok", "") if fitted is None else (fitted.status, fitted.status_reason)

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
            on_iteration(
                Iteration(
                    t=t,
                    x=x_next,
                    y=y_next,
                    f=f_next,
                    state=state,
                    fitted=fitted,
                    acq_value=acq_value,
                    fit_wall_s=fit_wall_s,
                    acq_wall_s=acq_wall_s,
                    fit_calls=fit_calls,
                    status=status,
                    reason=reason,
                    y_mean=y_mean,
                    y_std=y_std,
                )
            )

    return state


if __name__ == "__main__":
    raise SystemExit("sagp.bo has no command line; run python -m experiments.run_bo instead")
