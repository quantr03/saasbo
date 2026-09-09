"""experiments.run_bo: the study's BO entry point -- one (family, seed, method) per invocation.

The seven methods differ only in the fit they dispatch -- the four cells through `sagp.gp.fit`,
the two MAP references through `sagp.references.fit_map`, and the Sobol search through no fit at
all -- and are otherwise the same run: `sagp.bo`'s seeding and acquisition, with the run directory
of `experiments.runlog` around them. `python -m experiments.run_bo --help` is the command line:
resumable, with the whole provenance of a run written into its `manifest.json`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path

import jax
import numpy as np
from jax import Array

from experiments.runlog import (
    _ATTEMPT0_FIELDS,
    _DIAG_FIELDS,
    RunConfig,
    RunLogger,
    config_hash,
    run_dir_for,
)
from sagp.bo import ACQUISITIONS, IterRNG, initial_design, iteration_rngs, optimize_ei
from sagp.gp import CELLS, FittedGP, NUTSConfig, fit, standardize
from sagp.readouts import readouts
from sagp.references import fit_map
from synthobj.families import make_family
from synthobj.objective import SyntheticObjective


# The seven methods a run can take, as they are spelled on the command line and in a run's path.
METHODS: list[str] = [f"{structure}/{prior}" for (structure, prior) in CELLS] + [
    "sobol",
    "dsp_map",
    "oracle_S",
]


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
    attempt0 = fitted.attempts[0] if fitted is not None and fitted.attempts else None

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
        **{
            f"a0_{field}": (float("nan") if attempt0 is None else getattr(attempt0, field))
            for field in _ATTEMPT0_FIELDS
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
    writes -- checkpoint first, then the coordinates and the samples, and the row last.

    `resume=True` (the default) continues an existing run directory: the checkpoint's config hash
    must match this configuration or the run stops with a `ValueError` rather than splicing two
    different runs together, and the logs are rolled back to the last iteration that is complete
    in *both* the checkpoint and `iterations.csv`, so an interrupted write is redone rather than
    left as a hole. Redone iterations are bit-identical, since iteration `t` is a function of
    `(seed, t)` and of the points before it alone. A commit or a package version that has moved
    since the run started is warned about in `log.txt` and recorded in the manifest's `resumed`
    list rather than asserted (`RunLogger.note_resume`). `**overrides` sets any other `RunConfig`
    field (an unknown one is a `TypeError` from the dataclass, which is the check).
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
        logger.write_manifest(objective)
        logger.write_environment_lock()
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
        logger.note_resume()

    # The Sobol search continues the design's own sequence, so its first n_init rows are the
    # design: only the reference itself needs the remaining T - n_init rows.
    sobol_seq = initial_design(cfg.D, cfg.T, cfg.seed) if cfg.method == "sobol" else None

    for t in range(start, cfg.T):
        X, y, f, row, fitted, readout = _iteration(t, objective, cfg, X, y, f, logger, sobol_seq)
        statuses.append(str(row["status"]))
        # Checkpoint first (plan section 4, step 6): the logs may then lag it by one iteration,
        # which resume rolls back, but they can never gain a duplicate or a torn row. The
        # `iterations.csv` row goes *last* of all (ruling R34), so that a complete row is a
        # promise that this iteration's coordinates and samples are on disk too -- a kill between
        # the row and the samples would otherwise leave a hole resume has no way to see.
        logger.checkpoint(X, y, f, t, statuses)
        if readout is not None:
            logger.append_coords(t, readout)
        if fitted is not None:
            logger.save_samples(t, fitted)
        logger.append_row(row)
        logger.log(_iteration_line(row, fitted))

    return run_dir


# --- the command line ---


def _nuts_config(text: str) -> NUTSConfig:
    """`--nuts W,S,K` -> `NUTSConfig(W, S, K)`: warmup draws, samples, thinning.

    The other two fields are deliberately not exposed: `max_tree_depth` is the reference driver's
    6 and `num_chains` is 1, and both are held identical across every cell so that the sampler's
    budget can never be the reason two cells differ. The plan's fallback is passed here as
    `--nuts 512,256,16`, which is also the default.
    """
    try:
        warmup, samples, thinning = (int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated ints 'warmup,samples,thinning'; got {text!r}"
        ) from None
    return NUTSConfig(warmup, samples, thinning)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_bo",
        description=(
            "Run one (family, seed, cell) of the study into "
            "<out>/<family>/<cell with '/' as '-'>/seed{seed:02d}/, resuming it if it exists."
        ),
    )
    parser.add_argument("--family", required=True, help="synthobj family, e.g. aligned10")
    parser.add_argument(
        "--seed", required=True, type=int,
        help="run seed: fixes the initial design, the observation noise and every later draw",
    )
    parser.add_argument(
        "--cell", required=True, choices=METHODS,
        help="the method to run: a structure/prior cell, or one of the three references",
    )
    parser.add_argument("--out", required=True, type=Path, help="root of the run directories")
    parser.add_argument(
        "--T", type=int, default=200,
        help="total evaluations, initial design included (default 200)",
    )
    parser.add_argument(
        "--n-init", type=int, default=20,
        help="size of the shared Sobol initial design (default 20)",
    )
    parser.add_argument("--D", type=int, default=100, help="ambient dimension (default 100)")
    parser.add_argument(
        "--acq", choices=list(ACQUISITIONS), default="logei",
        help="LogEI, or the vendored reference's own EI (default logei)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="global-shrinkage scale; the cell's own prior default when unset",
    )
    parser.add_argument(
        "--fixed-noise", type=float, default=None,
        help="fix the observation noise at this value instead of inferring it",
    )
    parser.add_argument(
        "--noiseless", action="store_true",
        help="observe f(x) itself rather than the family's noisy observation",
    )
    parser.add_argument(
        "--sobol-every", type=int, default=25,
        help="compute the Sobol readout every N iterations and at t = T-1 (default 25)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="start over, discarding whatever the run directory already holds",
    )
    parser.add_argument(
        "--nuts", type=_nuts_config, default=NUTSConfig(), metavar="W,S,K",
        help="NUTS warmup, samples and thinning (default 512,256,16); tree depth stays 6",
    )
    parser.add_argument(
        "--num-init-candidates", type=int, default=5000,
        help="candidates the acquisition optimizer scores each iteration (default 5000)",
    )
    parser.add_argument(
        "--num-restarts-ei", type=int, default=5,
        help="L-BFGS-B restarts from the best of those candidates (default 5)",
    )
    parser.add_argument(
        "--objective-dir", type=Path, default=None,
        help="load <dir>/<family>/seed{seed:02d}_D{D}.npz instead of rebuilding the objective",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved config and the objective, then exit without writing anything",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """The `RunConfig` a parsed command line asks for: the one place a flag becomes a setting.

    Separate from `main` so that `--dry-run` prints the very object the run would be given, and
    so that two command lines can be compared without either being run -- which is how the study
    checks that its seven methods differ in `method` and in nothing else.
    """
    return RunConfig(
        family=args.family,
        seed=args.seed,
        D=args.D,
        method=args.cell,
        T=args.T,
        n_init=args.n_init,
        acq=args.acq,
        alpha=args.alpha,
        fixed_noise=args.fixed_noise,
        noiseless=args.noiseless,
        nuts=args.nuts,
        num_init_candidates=args.num_init_candidates,
        num_restarts_ei=args.num_restarts_ei,
        sobol_every=args.sobol_every,
        out_dir=str(args.out),
    )


def _objective_stem(args: argparse.Namespace) -> Path:
    """`<dir>/<family>/seed{seed:02d}_D{D}`: this run's stem under `--objective-dir`."""
    return args.objective_dir / args.family / f"seed{args.seed:02d}_D{args.D}"


def _load_objective(args: argparse.Namespace) -> SyntheticObjective:
    """The problem to run: the study grid's stored objective, or one rebuilt from its family.

    `--objective-dir` points at what `python -m synthobj.generate` wrote, whose stems are
    `<dir>/<family>/seed{seed:02d}_D{D}`. Loading rather than rebuilding is what lets a cluster
    array of 160 tasks share one objective per (family, seed) -- `f_star` included, which is
    searched for rather than derived and is the constant every regret in the study is taken
    against.
    """
    if args.objective_dir is None:
        return make_family(args.family, args.seed, args.D)
    return SyntheticObjective.load(_objective_stem(args))


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (or `sys.argv`) and run one cell; 0 on success, 2 on a bad argument.

    The two ways a command line can be wrong reach exit 2 by different routes. Argparse handles
    what it can express -- a missing `--out`, a `--cell` outside `METHODS`, a `--nuts` that is not
    three ints -- and raises `SystemExit(2)`, which is caught here so that `main` returns an int
    to its caller rather than raising through it. An unknown `--family`, or an `--objective-dir`
    with no such stem under it, cannot be expressed as `choices` (which families exist is
    `synthobj.families`' business, and gamma variants are parameterized names), so it is checked
    here instead. Anything the *run* raises is left to propagate -- a checkpoint under a different
    configuration above all: that is not an argument error, and a cluster log should carry its
    traceback rather than a bare exit code.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)

    cfg = resolve_config(args)
    try:
        objective = _load_objective(args)
    except (KeyError, FileNotFoundError) as error:
        # Two different mistakes reach the same exit code, and naming `--family` for both would
        # send the operator of a cluster job looking in the wrong place: under `--objective-dir`
        # the family is usually fine and it is the *file* that the grid never wrote, so the
        # message names the stem that was looked for.
        if args.objective_dir is None:
            print(f"error: no objective for --family {args.family!r}: {error}", file=sys.stderr)
        else:
            print(
                f"error: no objective file at {_objective_stem(args)}.npz for "
                f"--objective-dir {args.objective_dir}: {error}",
                file=sys.stderr,
            )
        return 2

    if args.dry_run:
        print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
        print(
            f"objective family={objective.labels.family} D={objective.D} "
            f"f_star={objective.f_star!r} S={list(objective.labels.S)}"
        )
        return 0

    # Every `RunConfig` field except the seven `run_bo` takes as arguments of its own. Derived
    # rather than listed, so that a field added to the config cannot silently stop reaching the
    # run while `--dry-run` goes on printing it.
    named = {"family", "seed", "D", "method", "T", "n_init", "out_dir"}
    overrides = {
        field.name: getattr(cfg, field.name)
        for field in dataclasses.fields(cfg)
        if field.name not in named
    }
    run_dir = run_bo(
        objective,
        cfg.method,
        cfg.seed,
        T=cfg.T,
        n_init=cfg.n_init,
        out_dir=cfg.out_dir,
        resume=not args.no_resume,
        **overrides,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
