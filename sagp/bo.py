"""sagp.bo: the Bayesian-optimization loop -- the vendored SAASBO driver, generalized.

`saasbo.run_saasbo` is one loop hard-wired to one surrogate. The thesis compares seven methods
(the four cells, a Sobol search, and the two MAP references) whose regret differences must be
attributable to the surrogate alone, so the loop around them has to be *literally* the same code:
this module is that loop, and every difference between the methods reaches it as an argument --
`surrogate` and `propose`, with `on_iteration` for whoever wants to watch. It reaches `sagp.gp`
only through `standardize` and `FittedGP.posterior` and never asks how a cell is parameterized --
a per-method branch here would be exactly the confound the design is built to avoid -- and `gp.py`
in turn never sees a budget or an acquisition. Which method a run takes, where it writes and how
it is resumed are `experiments.run_bo`'s and `experiments.runlog`'s; nothing here knows about
either.

What is new relative to the reference driver is bookkeeping, not method: every random draw of
iteration `t` is a pure function of `(seed, t)` (`iteration_rngs`), so a killed run resumes
bit-identically from its `state` and paired runs across methods share the same design and the
same observation noise; every iteration is handed whole to `on_iteration`, which is where a run
directory's row, readouts and retained samples get written; and a fit that raises is retried once
and then replaced by a random query rather than being allowed to end the run.

The acquisition is LogEI (Ament et al. 2023) combined over the retained samples by log-mean-exp,
which is the exact log of the reference's sample-averaged EI: the argmax is unchanged in exact
arithmetic, and what differs is only that the log form still has a gradient where the reference's
EI has underflowed to zero -- the regime a sparse GP in 100 dimensions spends most of its time in.
`acq="ei"` runs the vendored `saasbo.ei` itself, which is what the reference-reproduction test
needs.

`synthobj` maximizes and the vendored code minimizes: `gp.standardize` negates once, on the way
into the GP, so every line copied from the reference runs unchanged while `y`, `best_f` and the
regret in the logs stay in the objective's own units.

`python -m experiments.run_bo --help` is the study's entry point: one (family, seed, cell) per
invocation, resumable, with its whole provenance written into the run directory's `manifest.json`.
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

    Five independent streams rather than one, because they are consumed by four different
    libraries (JAX, `scipy.stats.qmc`, NumPy's Generator, `synthobj`'s own noise stream) and a
    single counter shared across them would make each one's draws depend on how many the others
    took -- so a change to, say, the number of EI restarts would silently move the observation
    noise. Salting each stream keeps them disjoint and keeps resume a function of `(seed, t)`.
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
    """All randomness of iteration `t` (plan section 4), as a pure function of `(seed, t)`.

    `t` is the index of the point being chosen -- equivalently the number of points already
    evaluated -- so the initial design's point `t` draws its observation noise from the same
    stream the loop would have used for it. Resume therefore needs nothing but `(seed, t)`: no
    generator state is ever carried across iterations, and none is stored in the checkpoint.
    """
    return IterRNG(
        key=jax.random.fold_in(jax.random.PRNGKey(seed), t),
        sobol_seed=_salted_seed(seed, t, 0xCA),
        jitter_rng=np.random.default_rng(np.random.SeedSequence([seed, t, 0x1A])),
        noise_rng=noise_rng(seed, run=t),
        fallback_seed=_salted_seed(seed, t, 0xFA),
    )


def initial_design(D: int, n_init: int, seed: int) -> np.ndarray:
    """The reference's initial design: `n_init` points of a scrambled Sobol sequence on [0,1]^D.

    Identical for every method at a given seed, which is what makes the comparison paired: two
    methods differ from their first fitted iteration onward and not before. `qmc.Sobol` warns when
    the count is not a power of two (the balance property); the reference suppresses that warning
    and so do we, since `n_init` is a budget, not a choice about balance. The Sobol *search*
    reference draws `T` rows of this same sequence, whose first `n_init` rows are exactly this
    design.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return qmc.Sobol(D, scramble=True, seed=seed).random(n_init)


# --- acquisition ---

_LOG2 = math.log(2.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def log1mexp(x: Array) -> Array:
    """log(1 - exp(x)) for x < 0, accurate at both ends (Machler 2012).

    `log(-expm1(x))` loses nothing as x -> 0 but cancels for very negative x; `log1p(-exp(x))` is
    the other way round. The switch is at -log 2, where the two are equally conditioned. Each
    branch is evaluated on inputs *clamped into its own domain* rather than on the raw `x`: under
    `jnp.where` both branches are computed everywhere, and a NaN in the branch not taken poisons
    the gradient of the branch that is (the standard JAX where-NaN trap), which would leave the
    acquisition optimizer with a NaN step deep in the tail -- the regime this function exists for.
    """
    upper = jnp.maximum(x, -_LOG2)  # the naive branch never sees the tail's inputs
    lower = jnp.minimum(x, -_LOG2)  # and the tail branch never sees log1p(-1) = -inf
    # `maximum`/`minimum` split the derivative evenly at a tie, so the clamps halve d/dx exactly
    # at the branch point (and likewise in `log_h`): one point, measure zero, harmless here.
    return jnp.where(x > -_LOG2, jnp.log(-jnp.expm1(upper)), jnp.log1p(-jnp.exp(lower)))


def log_h(z: Array) -> Array:
    """log h(z), h(z) = phi(z) + z Phi(z): the standardized improvement, stably (Ament et al. 2023).

    h is the whole content of EI -- EI(x) = std * h((y_target - mu)/std) -- and it underflows to
    exactly zero in float64 at about z = -37, which is where the reference's EI stops having a
    gradient and its optimizer stops moving. Writing h(z) = phi(z) [1 - (-z) Phi(z)/phi(z)] and
    taking the log turns the cancellation into `log1mexp` of a log Mills ratio, which stays finite
    and differentiable arbitrarily far into the tail. The Mills ratio itself is the paper's own
    erfcx form, Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt 2) for z < 0, evaluated as one scaled
    special function rather than as log Phi(z) + z^2/2: those two terms are both O(z^2) and cancel
    to an O(1) result, which costs the whole mantissa in the tail this branch exists for (0.04
    absolute at z = -5e3, NaN by z = -3e4, and a gradient 3x wrong at z = -1e5 -- the gradient
    being what the L-BFGS-B restarts follow). Above z = -1 the naive form is both accurate and
    cheaper. As in `log1mexp`, each branch is clamped into its own domain, since the naive
    branch's gradient at z = -40 is 0/0 and the tail branch's `log(-z)` is NaN at z > 0.
    """
    upper = jnp.maximum(z, -1.0)
    lower = jnp.minimum(z, -1.0)
    naive = jnp.log(norm.pdf(upper) + upper * norm.cdf(upper))
    log_mills = jnp.log(-lower) + 0.5 * jnp.log(jnp.pi / 2) + jnp.log(erfcx(-lower / jnp.sqrt(2)))
    tail = -0.5 * lower**2 - _HALF_LOG_2PI + log1mexp(log_mills)
    return jnp.where(z > -1.0, naive, tail)


def log_ei(x: Array, y_target: float, gp: FittedGP, xi: float = 0.0) -> Array:
    """log EI at `x`, with `saasbo.ei`'s signature and conventions (minimization, std floor 1e-6).

    The reference averages EI over the retained posterior samples; the log of that average is
    `logsumexp(log EI_s) - log S`, which is what this returns -- so under exact arithmetic
    `log_ei` and `log(saasbo.ei)` have the same argmax and the same ranking of candidates, and the
    only difference is that this one still separates candidates after every per-sample EI has
    underflowed to zero. A degenerate sample's NaN reaches `nan_to_num`'s own neginf pass and so
    comes out as `finfo.min` rather than as a literal -inf; inside `logsumexp` that is the neutral
    element of the combination all the same, since it exponentiates to exactly zero against any
    other sample -- where the reference maps such a sample to 0, the neutral element of its own.
    """
    mu, var = gp.posterior(x)
    std = jnp.maximum(jnp.sqrt(var), 1e-6)
    scaled = (y_target - xi - mu) / std
    values = jnp.nan_to_num(jnp.log(std) + log_h(scaled), nan=-jnp.inf)
    return logsumexp(values, axis=0) - jnp.log(values.shape[0])


def log_ei_sum(x: Array, y_target: float, gp: FittedGP, xi: float = 0.0) -> Array:
    """`log_ei(...).sum()`: `saasbo.ei_grad`'s analogue, the scalar L-BFGS-B differentiates."""
    return log_ei(x, y_target, gp, xi).sum()


# The acquisitions a run may take. "ei" is the vendored reference itself, kept so that the
# reproduction test can run this loop against `run_saasbo` with nothing but the seeding changed.
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

    The changes are the seeding the study's reproducibility requires (the candidate Sobol set and
    the incumbent's jitter both come from `iteration_rngs`, where the reference draws them from
    global state), the acquisition as a callable in the two places the reference names `ei`, and
    returning the value as well as the point so the row can log it. Everything else -- the 5000
    candidates, the jitter's 1e-3, `top_k`, `maxfun=100`, the bounds, the sign flips -- is the
    reference's, deliberately unexamined: it is held fixed across every method compared.
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

    `t` is unused here -- the acquisition's randomness is already in `rngs` -- and is part of the
    proposer protocol for the proposers that walk a sequence instead of optimizing, above all
    `references.propose_sobol`.
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
    """Every point evaluated so far: the design, its observations, and the noise-free values.

    The whole of what an iteration depends on besides `(seed, t)`, which is what makes it both the
    loop's running state and the thing a checkpoint has to hold: `run_bo(..., state=...)` resumes
    from one and cannot tell it from the state its own earlier iterations would have built.
    """

    X: np.ndarray  # (t, D), the design in the unit cube
    y: np.ndarray  # (t,), the observations the loop optimizes against -- noisy unless `noiseless`
    f: np.ndarray  # (t,), the objective's own noise-free values, which the regret is taken on


class Iteration(NamedTuple):
    """What one iteration produced, handed whole to `on_iteration` after the append.

    Everything a run's row needs and nothing derived from it: the loop reports, and what a row
    should say about an iteration is the observer's business (`experiments.runlog.RunLogger`).
    `state` is the arrays *after* this point was appended -- the very objects `run_bo` returns --
    so an observer's `state.y.max()` is the incumbent including this iteration.
    """

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

    A fit whose *diagnostics* failed is not a failure here -- `fit` returns it with
    `status="excluded"` and the loop queries with it anyway, because a study that stopped at every
    bad chain would report a survivorship-biased regret. An *exception* out of JAX or NumPyro is
    different: it leaves no surrogate at all, so it is logged with its traceback, retried once on
    a key disjoint from both of `fit`'s own (`fold_in(key, 2)`; `fit`'s refit uses 1), and if that
    raises too the caller falls back to a random query. The run continues either way. A `None`
    surrogate is the model-free method: no call, no clock, and nothing that could fail.
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

    The first `n_init` points are the shared Sobol design of `seed` and are not iterations: they
    have no fit and nothing to report but their observation. From `t = n_init` on, each iteration
    standardizes what has been observed, fits, maximizes the acquisition, evaluates, appends, and
    hands the whole of itself to `on_iteration` (plan section 4, steps 2-6).

    The three adapters are what makes this the *same* loop for all seven methods.
    `surrogate(X, z, key) -> FittedGP` is the fit, and `None` is the model-free method -- the Sobol
    search -- which costs no fit call and reports a NaN `fit_wall_s`; `propose(fitted, y_target,
    rngs, t) -> (x, acq_value)` chooses the next point, `propose_ei` by default and
    `references.propose_sobol` for that same search; `on_iteration(Iteration)` is the observer,
    called once per iteration after the append, and `log(str)` receives the two messages of the
    exception policy. Nothing in here writes a file or reads a configuration.

    `state` is a resume: iterating starts at `len(state.X)` and the run continues bit-identically,
    since iteration `t` is a function of `(seed, t)` and of the points before it alone. Without one
    the design is drawn and observed here, each of its points from the stream iteration `t` would
    have used for it, so a resumed run and an uninterrupted one draw the same noise.
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
            # Both attempts raised: query at random rather than end the run, and mark the
            # iteration so the analysis can count these instead of reading them as ordinary ones.
            x_next = initial_design(D, 1, rngs.fallback_seed)[0]
            acq_value, acq_wall_s = float("nan"), float("nan")
            status, reason = "excluded", failure
        else:
            start = time.perf_counter()
            x_next, acq_value = propose(fitted, y_target, rngs, t)
            acq_wall_s = time.perf_counter() - start
            status, reason = ("ok", "") if fitted is None else (fitted.status, fitted.status_reason)

        # A proposer is an argument, so what it returns is normalized here rather than trusted:
        # the objective and the state below are numpy's, whatever the proposer computed in.
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
