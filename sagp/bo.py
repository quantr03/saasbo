"""sagp.bo: the Bayesian-optimization loop -- the vendored SAASBO driver, generalized.

`saasbo.run_saasbo` is one loop hard-wired to one surrogate. The thesis compares seven methods
(the four cells, a Sobol search, and the two MAP references) whose regret differences must be
attributable to the surrogate alone, so the loop around them has to be *literally* the same code:
this module is that loop, and it reaches `sagp.gp` only through `fit`, `fit_map`,
`FittedGP.posterior`, `readouts`, `standardize` and the two config dataclasses. It never asks how
a cell is parameterized -- a per-method branch here would be exactly the confound the design is
built to avoid -- and `gp.py` in turn never sees a budget or an acquisition.

What is new relative to the reference driver is bookkeeping, not method: every random draw of
iteration `t` is a pure function of `(seed, t)` (`iteration_rngs`), so a killed run resumes
bit-identically from its checkpoint and paired runs across methods share the same design and the
same observation noise; every iteration appends one row of the plan's schema, the per-coordinate
readouts and the retained samples; and a fit that raises is retried once and then replaced by a
random query rather than being allowed to end the run.

The acquisition is LogEI (Ament et al. 2023) combined over the retained samples by log-mean-exp,
which is the exact log of the reference's sample-averaged EI: the argmax is unchanged in exact
arithmetic, and what differs is only that the log form still has a gradient where the reference's
EI has underflowed to zero -- the regime a sparse GP in 100 dimensions spends most of its time in.
`acq="ei"` runs the vendored `saasbo.ei` itself, which is what the reference-reproduction test
needs.

`synthobj` maximizes and the vendored code minimizes: `gp.standardize` negates once, on the way
into the GP, so every line copied from the reference runs unchanged while `y`, `best_f` and the
regret in the logs stay in the objective's own units.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import Array, value_and_grad
from jax.scipy.special import log_ndtr, logsumexp
from jax.scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import qmc

import saasbo
from sagp.gp import (
    CELLS,
    ELL_PRIOR,
    DiagThresholds,
    FittedGP,
    NUTSConfig,
    fit,
    fit_map,
    readouts,
    standardize,
)
from synthobj.families import noise_rng

# The seven methods a run can take, as they are spelled on the command line and in a run's path.
METHODS: list[str] = [f"{structure}/{prior}" for (structure, prior) in CELLS] + [
    "sobol",
    "dsp_map",
    "oracle_S",
]


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
    return jnp.where(x > -_LOG2, jnp.log(-jnp.expm1(upper)), jnp.log1p(-jnp.exp(lower)))


def log_h(z: Array) -> Array:
    """log h(z), h(z) = phi(z) + z Phi(z): the standardized improvement, stably (Ament et al. 2023).

    h is the whole content of EI -- EI(x) = std * h((y_target - mu)/std) -- and it underflows to
    exactly zero in float64 at about z = -37, which is where the reference's EI stops having a
    gradient and its optimizer stops moving. Writing h(z) = phi(z) [1 - (-z) Phi(z)/phi(z)] and
    taking the log turns the cancellation into `log1mexp` of a log Mills ratio, which stays
    finite and differentiable arbitrarily far into the tail: this is the paper's erfcx form
    rewritten through Phi/phi, so it needs no erfcx (JAX has none). Above z = -1 the naive form is
    both accurate and cheaper. As in `log1mexp`, each branch is clamped into its own domain, since
    the naive branch's gradient at z = -40 is 0/0 and the tail branch's `log(-z)` is NaN at z > 0.
    """
    upper = jnp.maximum(z, -1.0)
    lower = jnp.minimum(z, -1.0)
    naive = jnp.log(norm.pdf(upper) + upper * norm.cdf(upper))
    log_mills = jnp.log(-lower) + log_ndtr(lower) + 0.5 * lower**2 + _HALF_LOG_2PI
    tail = -0.5 * lower**2 - _HALF_LOG_2PI + log1mexp(log_mills)
    return jnp.where(z > -1.0, naive, tail)


def log_ei(x: Array, y_target: float, gp: FittedGP, xi: float = 0.0) -> Array:
    """log EI at `x`, with `saasbo.ei`'s signature and conventions (minimization, std floor 1e-6).

    The reference averages EI over the retained posterior samples; the log of that average is
    `logsumexp(log EI_s) - log S`, which is what this returns -- so under exact arithmetic
    `log_ei` and `log(saasbo.ei)` have the same argmax and the same ranking of candidates, and the
    only difference is that this one still separates candidates after every per-sample EI has
    underflowed to zero. NaN (a degenerate sample) maps to -inf, the neutral element of the
    combination, where the reference maps it to 0, the neutral element of its own.
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


# --- configuration ---


@dataclass(frozen=True)
class RunConfig:
    """Every setting of one run, resolved once and written verbatim into `manifest.json`.

    Frozen and hashed (`config_hash`) because it is the identity of a run directory: resume
    refuses to continue a checkpoint under a different configuration, which is the only automatic
    protection there is against silently splicing two half-runs of different methods together.
    Two fields are excluded from that hash, because they say where the run lives and how far it
    goes rather than what it is: `out_dir`, so a moved or copied run directory still resumes, and
    `T`, so a run killed by a wall-clock limit can be resumed with a longer budget -- the very
    situation resume exists for. Extending `T` changes nothing already computed: iteration `t`
    depends on `(seed, t)` and the points before it, and `T` reaches the loop only through the
    end-of-run Sobol readout (`t == T - 1`), so the shortened run's last row carries a
    `sobol_hat` the longer run would have computed later.
    """

    family: str
    seed: int
    D: int
    method: str
    T: int
    n_init: int
    acq: str = "logei"
    alpha: float | None = None
    fixed_noise: float | None = None
    noiseless: bool = False
    nuts: NUTSConfig = NUTSConfig()
    thresholds: DiagThresholds = DiagThresholds()
    ell_prior: tuple[float, float] = ELL_PRIOR
    num_init_candidates: int = 5000
    num_restarts_ei: int = 5
    sobol_every: int = 25
    sobol_n: int = 2048
    out_dir: str = "runs"


def config_hash(cfg: RunConfig) -> str:
    """sha256 over the settings, less `out_dir` and `T`: the identity resume checks against."""
    fields = dataclasses.asdict(cfg)
    for excluded in ("out_dir", "T"):
        fields.pop(excluded)
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def run_dir_for(out_dir: str | os.PathLike[str], cfg: RunConfig) -> Path:
    """`<out_dir>/<family>/<method>/seed{seed:02d}`, with the method's "/" flattened to "-"."""
    return Path(out_dir) / cfg.family / cfg.method.replace("/", "-") / f"seed{cfg.seed:02d}"


# --- the run directory ---

# The plan's row schema, less the trailing x_0 ... x_{D-1} which depend on D. `reason` is the
# fit's own status_reason, or the exception that replaced the fit.
_ROW_FIELDS: tuple[str, ...] = (
    "t", "method", "family", "seed", "y", "f", "best_obs", "best_f", "regret", "acq_value",
    "fit_wall_s", "acq_wall_s", "fit_calls", "nuts_attempts", "status", "reason",
    "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
    "num_steps_mean", "y_mean", "y_std", "sobol_computed",
)
# The six numbers of the retained attempt's `Diagnostics` that the row carries (its `wall_s`,
# `passed` and `reason` are already in `fit_wall_s`, `status` and `reason`).
_DIAG_FIELDS: tuple[str, ...] = (
    "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
    "num_steps_mean",
)
_COORD_FIELDS: tuple[str, ...] = ("t", "i", "native_median", "p_active", "sobol_hat")


class RunLogger:
    """One run directory: the manifest, the checkpoint, the two CSVs, the samples and `log.txt`.

    Every write goes through here so that the *order* of writes is stated in one place. The
    checkpoint is written first, by tmp-file and `os.replace`, and the appends follow: a kill
    between the two then leaves the logs one iteration behind the checkpoint, which `truncate_to`
    rolls back, rather than leaving a duplicated or half-written row that no reader could detect.
    """

    def __init__(self, run_dir: Path, cfg: RunConfig) -> None:
        self.dir = run_dir
        self.cfg = cfg
        self.samples_dir = run_dir / "samples"
        self.iterations = run_dir / "iterations.csv"
        self.coords = run_dir / "coords.csv"
        self.checkpoint_path = run_dir / "checkpoint.npz"
        self.manifest = run_dir / "manifest.json"
        self.log_path = run_dir / "log.txt"
        self.row_fields = _ROW_FIELDS + tuple(f"x_{i}" for i in range(cfg.D))
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Append one line to `log.txt` (timings, statuses, tracebacks): the run's narrative."""
        with self.log_path.open("a") as handle:
            handle.write(message.rstrip("\n") + "\n")

    def write_manifest(self) -> None:
        """The minimal manifest of plan ruling R6: the `RunConfig` fields plus `config_hash`.

        Task 9 extends it with the environment, the git commit, the objective's labels and the
        reference constants; what has to be here already is the hash, because resume compares it.
        """
        fields = dataclasses.asdict(self.cfg)
        fields["config_hash"] = config_hash(self.cfg)
        self.manifest.write_text(json.dumps(fields, indent=2, sort_keys=True, default=str) + "\n")

    def manifest_hash(self) -> str | None:
        """The `config_hash` recorded in `manifest.json`, or None if there is no manifest."""
        if not self.manifest.exists():
            return None
        return json.loads(self.manifest.read_text()).get("config_hash")

    def reset(self) -> None:
        """Empty the run directory of everything a previous run wrote (a fresh, non-resumed run)."""
        for path in (self.iterations, self.coords, self.checkpoint_path, self.log_path):
            path.unlink(missing_ok=True)
        for path in self.samples_dir.glob("t*.npz"):
            path.unlink()

    def checkpoint(
        self, X: np.ndarray, y: np.ndarray, f: np.ndarray, t_done: int, statuses: list[str]
    ) -> None:
        """The state a resume needs, written atomically: everything else is derivable from it.

        `statuses` holds one entry per *iteration* (t from `n_init` to `t_done`), not per point:
        the initial design has no fit and therefore no status. The config hash rides along so a
        checkpoint can be told apart from another run's by looking at the file alone.
        """
        tmp = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                X=X,
                y=y,
                f=f,
                t_done=t_done,
                statuses=np.array(statuses, dtype="<U16"),
                config_hash=config_hash(self.cfg),
            )
        os.replace(tmp, self.checkpoint_path)

    def load_checkpoint(self) -> dict[str, np.ndarray] | None:
        """`checkpoint.npz` as a dict, or None if this run has not completed an iteration yet."""
        if not self.checkpoint_path.exists():
            return None
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def append_row(self, row: dict[str, object]) -> None:
        """Append one iteration to `iterations.csv`, writing the header if the file is new."""
        new = not self.iterations.exists() or self.iterations.stat().st_size == 0
        with self.iterations.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.row_fields)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def append_coords(self, t: int, readout: dict[str, object]) -> None:
        """Append this iteration's D per-coordinate readouts to `coords.csv` (long format).

        Long rather than wide because D differs between studies and `sobol_hat` is only computed
        on some iterations; a long table needs no schema change when either moves.
        """
        new = not self.coords.exists() or self.coords.stat().st_size == 0
        native_median = np.asarray(readout["native_median"], dtype=float)
        p_active = np.asarray(readout["p_active"], dtype=float)
        sobol_hat = np.asarray(readout["sobol_hat"], dtype=float)
        with self.coords.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(_COORD_FIELDS)
            for i in range(native_median.shape[0]):
                writer.writerow(
                    [t, i, float(native_median[i]), float(p_active[i]), float(sobol_hat[i])]
                )

    def save_samples(self, t: int, fitted: FittedGP) -> None:
        """The retained draws of iteration `t`, with the status and attempts that produced them.

        This is the only place `bo.py` touches `fitted.samples`, and it does not look inside: what
        a cell's sites mean is `gp.py`'s business, and every later analysis re-reads them from here.
        """
        arrays = {site: np.asarray(draws) for site, draws in fitted.samples.items()}
        with (self.samples_dir / f"t{t:03d}.npz").open("wb") as handle:
            np.savez(
                handle, status=fitted.status, nuts_attempts=len(fitted.attempts), **arrays
            )

    def last_logged_t(self) -> int | None:
        """The largest `t` with a *complete* row in `iterations.csv`, or None if there is none.

        A row shorter than the header is a row a kill interrupted mid-write; counting it as
        logged would leave the file with a torn line no reader could parse.
        """
        if not self.iterations.exists():
            return None
        with self.iterations.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return None
        width = len(rows[0])
        complete = [int(row[0]) for row in rows[1:] if len(row) == width]
        return max(complete) if complete else None

    def truncate_to(self, t_done: int) -> None:
        """Drop every logged iteration after `t_done`, so the logs agree with the checkpoint."""
        for path in (self.iterations, self.coords):
            _truncate_csv(path, t_done)
        for path in self.samples_dir.glob("t*.npz"):
            if int(path.stem[1:]) > t_done:
                path.unlink()


def _truncate_csv(path: Path, t_done: int) -> None:
    """Rewrite `path` keeping its header and the rows whose first column (`t`) is <= `t_done`."""
    if not path.exists():
        return
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    width = len(rows[0])
    kept = [row for row in rows[1:] if len(row) == width and int(row[0]) <= t_done]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0])
        writer.writerows(kept)


# --- one iteration ---


def _fit_for(
    method: str, cfg: RunConfig, X: np.ndarray, z: np.ndarray, key: Array, labels: object
) -> FittedGP | None:
    """The surrogate this method fits to the standardized design, or None for the Sobol search.

    The four cells differ from each other only in the `Cell` handed to one `fit` call, and the two
    MAP references only in whether `fit_map` is told which coordinates matter -- which is the
    point: nothing downstream of here can tell them apart.
    """
    if method == "sobol":
        return None
    if method == "dsp_map":
        return fit_map(X, z)
    if method == "oracle_S":
        return fit_map(X, z, active=np.asarray(labels.S))
    return fit(
        X,
        z,
        key,
        cell=CELLS[tuple(method.split("/"))],
        alpha=cfg.alpha,
        fixed_noise=cfg.fixed_noise,
        nuts=cfg.nuts,
        thresholds=cfg.thresholds,
        ell_prior=cfg.ell_prior,
    )


def _fit_with_retry(
    t: int,
    cfg: RunConfig,
    X: np.ndarray,
    z: np.ndarray,
    rngs: IterRNG,
    labels: object,
    logger: RunLogger,
) -> tuple[FittedGP | None, int, float, str | None]:
    """Fit, retrying once on an exception; returns (fitted, fit_calls, wall_s, failure reason).

    A fit whose *diagnostics* failed is not a failure here -- `fit` returns it with
    `status="excluded"` and the loop queries with it anyway, because a study that stopped at every
    bad chain would report a survivorship-biased regret. An *exception* out of JAX or NumPyro is
    different: it leaves no surrogate at all, so it is logged with its traceback, retried once on
    a key disjoint from both of `fit`'s own (`fold_in(key, 2)`; `fit`'s refit uses 1), and if that
    raises too the caller falls back to a random query. The run continues either way.
    """
    if cfg.method == "sobol":
        return None, 0, float("nan"), None
    start = time.perf_counter()
    try:
        fitted = _fit_for(cfg.method, cfg, X, z, rngs.key, labels)
        return fitted, 1, time.perf_counter() - start, None
    except Exception:
        logger.log(f"t={t}: the fit raised; retrying once\n{traceback.format_exc()}")
    try:
        fitted = _fit_for(cfg.method, cfg, X, z, jax.random.fold_in(rngs.key, 2), labels)
        return fitted, 2, time.perf_counter() - start, None
    except Exception as error:
        logger.log(f"t={t}: the retry raised; querying at random\n{traceback.format_exc()}")
        return None, 2, time.perf_counter() - start, f"exception: {type(error).__name__}: {error}"


def _propose(
    cfg: RunConfig,
    fitted: FittedGP | None,
    y_target: float,
    rngs: IterRNG,
    sobol_seq: np.ndarray | None,
    t: int,
) -> tuple[np.ndarray, float, float]:
    """The next query: the acquisition's maximizer, or row `t` of the Sobol search's sequence."""
    start = time.perf_counter()
    if fitted is None:
        return np.asarray(sobol_seq[t], dtype=float), float("nan"), time.perf_counter() - start
    x_next, acq_value = optimize_ei(
        fitted,
        y_target,
        rngs.sobol_seed,
        rngs.jitter_rng,
        num_restarts_ei=cfg.num_restarts_ei,
        num_init=cfg.num_init_candidates,
        acq=ACQUISITIONS[cfg.acq],
    )
    return np.asarray(x_next, dtype=float), float(acq_value), time.perf_counter() - start


def _observe(objective: object, x: np.ndarray, cfg: RunConfig, rngs: IterRNG) -> float:
    """One observation of the objective at `x`: noisy through `rngs.noise_rng`, or `f(x)` itself."""
    if cfg.noiseless:
        return float(objective(x))
    return float(objective.observe(x[None, :], rngs.noise_rng)[0])


def _iteration(
    t: int,
    objective: object,
    cfg: RunConfig,
    X: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    logger: RunLogger,
    sobol_seq: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], FittedGP | None, dict | None]:
    """Plan section 4, steps 2-5: standardize, fit, propose, observe, and build the row.

    Returns the extended (X, y, f) together with the row, the fit and its readouts; the *writes*
    are the caller's, because their order (checkpoint first) is a property of the run and not of
    one iteration.
    """
    z, y_mean, y_std = standardize(y)
    y_target = float(np.min(z))
    rngs = iteration_rngs(cfg.seed, t)

    fitted, fit_calls, fit_wall_s, failure = _fit_with_retry(
        t, cfg, X, z, rngs, objective.labels, logger
    )
    if failure is not None:
        # Both attempts raised: query at random rather than end the run, and mark the row so the
        # analysis can count these iterations instead of reading them as ordinary ones.
        x_next = initial_design(cfg.D, 1, rngs.fallback_seed)[0]
        acq_value, acq_wall_s = float("nan"), float("nan")
        status, reason = "excluded", failure
    else:
        x_next, acq_value, acq_wall_s = _propose(cfg, fitted, y_target, rngs, sobol_seq, t)
        status, reason = ("ok", "") if fitted is None else (fitted.status, fitted.status_reason)

    f_next = float(objective(x_next))
    y_next = _observe(objective, x_next, cfg, rngs)

    X = np.vstack([X, x_next])
    y = np.append(y, y_next)
    f = np.append(f, f_next)

    # Sobol indices are the expensive readout, so they are computed on a schedule and at the last
    # iteration (the row an end-of-run analysis reads); the native summaries are free and always on.
    compute_sobol = fitted is not None and (t % cfg.sobol_every == 0 or t == cfg.T - 1)
    readout = (
        None
        if fitted is None
        else readouts(fitted, compute_sobol=compute_sobol, sobol_n=cfg.sobol_n)
    )
    diagnostics = fitted.attempts[-1] if fitted is not None and fitted.attempts else None

    row: dict[str, object] = {
        "t": t,
        "method": cfg.method,
        "family": cfg.family,
        "seed": cfg.seed,
        "y": y_next,
        "f": f_next,
        # What the loop sees (best_obs) and the truth it is scored on (best_f): under noise they
        # differ, and the design brief's regret is the noise-free one, which cannot go negative.
        "best_obs": float(y.max()),
        "best_f": float(f.max()),
        "regret": float(objective.f_star - f.max()),
        "acq_value": acq_value,
        "fit_wall_s": fit_wall_s,
        "acq_wall_s": acq_wall_s,
        "fit_calls": fit_calls,
        "nuts_attempts": 0 if fitted is None else len(fitted.attempts),
        "status": status,
        "reason": reason,
        **{
            field: (
                float("nan") if diagnostics is None else getattr(diagnostics, field)
            )
            for field in _DIAG_FIELDS
        },
        "y_mean": y_mean,
        "y_std": y_std,
        "sobol_computed": int(compute_sobol),
        **{f"x_{i}": float(x_next[i]) for i in range(cfg.D)},
    }
    return X, y, f, row, fitted, readout


def _iteration_line(row: dict[str, object], fitted: FittedGP | None) -> str:
    """One `log.txt` line: what happened at this iteration, in the units the row logs."""
    parts = [
        f"t={row['t']}",
        f"method={row['method']}",
        f"status={row['status']}",
        f"fit_calls={row['fit_calls']}",
        f"nuts_attempts={row['nuts_attempts']}",
        f"fit_wall_s={row['fit_wall_s']:.3f}",
        f"acq_wall_s={row['acq_wall_s']:.3f}",
        f"acq_value={row['acq_value']:.6g}",
        f"y={row['y']:.6g}",
        f"best_f={row['best_f']:.6g}",
        f"regret={row['regret']:.6g}",
        f"sobol_computed={row['sobol_computed']}",
    ]
    if row["reason"]:
        parts.append(f"reason={row['reason']!r}")
    # The MAP references have no `attempts`; their fit's quality is the optimizer's outcome, so
    # that dict is the only diagnostic they can report and it goes here rather than in the row.
    map_result = getattr(fitted, "map_result", None)
    if map_result is not None:
        parts.append(f"map_result={map_result}")
    return " ".join(parts)


# --- the loop ---


def run_bo(
    objective: object,
    method: str,
    seed: int,
    *,
    T: int = 200,
    n_init: int = 20,
    out_dir: str | os.PathLike[str],
    resume: bool = True,
    **overrides: object,
) -> Path:
    """Run `method` on `objective` for `T` evaluations; returns the run directory (plan section 4).

    The first `n_init` points are the shared Sobol design and are not rows: they have no fit and
    nothing to log but their observation, which the checkpoint carries. From `t = n_init` on, each
    iteration standardizes what has been observed, fits, maximizes the acquisition, evaluates, and
    writes -- checkpoint first, then the row, the coordinates and the samples.

    `resume=True` (the default) continues an existing run directory: the checkpoint's config hash
    must match this configuration or the run stops with a `ValueError` rather than splicing two
    different runs together, and the logs are rolled back to the last iteration that is complete
    in *both* the checkpoint and `iterations.csv`, so an interrupted write is redone rather than
    left as a hole. Redone iterations are bit-identical, since iteration `t` is a function of
    `(seed, t)` and of the points before it alone. `**overrides` sets any other `RunConfig` field
    (an unknown one is a `TypeError` from the dataclass, which is the check).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}: expected one of {METHODS}")
    cfg = RunConfig(
        family=objective.labels.family,
        seed=int(seed),
        D=int(objective.D),
        method=method,
        T=int(T),
        n_init=int(n_init),
        out_dir=str(out_dir),
        **overrides,
    )
    if cfg.acq not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {cfg.acq!r}: expected one of {sorted(ACQUISITIONS)}")

    run_dir = run_dir_for(out_dir, cfg)
    logger = RunLogger(run_dir, cfg)
    state = logger.load_checkpoint() if resume else None

    if state is None:
        logger.reset()
        logger.write_manifest()
        X = initial_design(cfg.D, cfg.n_init, cfg.seed)
        f = np.asarray(objective(X), dtype=float)
        y = np.array(
            [
                _observe(objective, X[t], cfg, iteration_rngs(cfg.seed, t))
                for t in range(cfg.n_init)
            ]
        )
        statuses: list[str] = []
        start = cfg.n_init
        logger.log(f"start method={cfg.method} family={cfg.family} seed={cfg.seed} T={cfg.T}")
    else:
        recorded = logger.manifest_hash()
        if recorded != config_hash(cfg):
            raise ValueError(
                f"{run_dir} was written under a different configuration "
                f"(manifest config_hash {recorded}, this run's {config_hash(cfg)}): "
                "resume only continues an identical run"
            )
        logged = logger.last_logged_t()
        t_done = cfg.n_init - 1 if logged is None else min(int(state["t_done"]), logged)
        X = state["X"][: t_done + 1]
        y = state["y"][: t_done + 1]
        f = state["f"][: t_done + 1]
        statuses = [str(s) for s in state["statuses"][: max(0, t_done - cfg.n_init + 1)]]
        logger.truncate_to(t_done)
        start = t_done + 1
        logger.log(
            f"resume at t={start} with T={cfg.T} "
            f"(checkpoint t_done={int(state['t_done'])}, last logged row {logged})"
        )

    # The Sobol search continues the design's own sequence, so its first n_init rows are the
    # design: only the reference itself needs the remaining T - n_init rows.
    sobol_seq = initial_design(cfg.D, cfg.T, cfg.seed) if cfg.method == "sobol" else None

    for t in range(start, cfg.T):
        X, y, f, row, fitted, readout = _iteration(t, objective, cfg, X, y, f, logger, sobol_seq)
        statuses.append(str(row["status"]))
        # Checkpoint first (plan section 4, step 6): the logs may then lag it by one iteration,
        # which resume rolls back, but they can never gain a duplicate or a torn row.
        logger.checkpoint(X, y, f, t, statuses)
        logger.append_row(row)
        if readout is not None:
            logger.append_coords(t, readout)
        if fitted is not None:
            logger.save_samples(t, fitted)
        logger.log(_iteration_line(row, fitted))

    return run_dir
