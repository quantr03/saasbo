"""experiments.run_bo: the study's BO entry point -- one (family, seed, method) per invocation.

The seven methods differ only in the two adapters they hand `sagp.bo.run_bo` -- the four cells
fit through `sagp.gp.fit`, the two MAP references through `sagp.references.fit_map`, and the Sobol
search fits nothing and proposes from its own sequence -- and are otherwise the same run: one
loop, with the run directory of `experiments.runlog` observing it. `surrogate_for` and
`propose_for` are that whole dispatch, and `run` is what resolves a configuration, a run directory
and a resume around them. `python -m experiments.run_bo --help` is the command line: resumable,
with the whole provenance of a run written into its `manifest.json`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

import sagp.bo
from experiments.runlog import RunConfig, RunLogger, config_hash, run_dir_for
from sagp.bo import BOState, initial_design, propose_ei
from sagp.gp import CELLS, NUTSConfig, fit
from sagp.references import fit_map, propose_sobol
from synthobj.families import make_family
from synthobj.objective import SyntheticObjective


# The seven methods a run can take, as they are spelled on the command line and in a run's path.
METHODS: list[str] = [f"{structure}/{prior}" for (structure, prior) in CELLS] + [
    "sobol",
    "dsp_map",
    "oracle_S",
]


# --- the method's adapters ---


def surrogate_for(cfg: RunConfig, labels: object) -> Callable[..., object] | None:
    """The surrogate `run_bo` fits each iteration under this method, or None for the Sobol search.

    The whole of the study's method dispatch, and the only place it exists: the four cells differ
    from each other only in the `Cell` handed to one `fit` call, and the two MAP references only
    in whether `fit_map` is told which coordinates matter. Nothing downstream of here -- the loop
    above all -- can tell them apart, which is what makes the comparison a comparison.

    Each branch closes over the module-level `fit`/`fit_map` rather than binding them now, so a
    test that patches `experiments.run_bo.fit` still reaches the fit this run makes.
    """
    if cfg.method == "sobol":
        return None
    if cfg.method == "dsp_map":
        return lambda X, z, seed: fit_map(X, z)
    if cfg.method == "oracle_S":
        active = np.asarray(labels.S)
        return lambda X, z, seed: fit_map(X, z, active=active)
    cell = CELLS[tuple(cfg.method.split("/"))]
    return lambda X, z, seed: fit(
        X,
        z,
        seed,
        cell=cell,
        alpha=cfg.alpha,
        fixed_noise=cfg.fixed_noise,
        nuts=cfg.nuts,
        thresholds=cfg.thresholds,
        ell_prior=cfg.ell_prior,
    )


def propose_for(cfg: RunConfig) -> Callable[..., tuple[np.ndarray, float]]:
    """How this method chooses its next query: the acquisition's maximizer, or the next Sobol row.

    The six model-based methods share one proposer at one operating point -- BoTorch's LogEI under
    `optimize_acqf`, with the five maximizer sizes the config carries -- because a budget that
    varied by method would be an alternative explanation for every regret difference the study
    reports. The Sobol search continues the design's own sequence, so its first `n_init` rows
    *are* the shared initial design: only the reference itself needs the remaining T - n_init
    rows, and it is handed all T at once because row `t` must not depend on which iteration drew
    it.
    """
    if cfg.method == "sobol":
        return partial(propose_sobol, sequence=initial_design(cfg.D, cfg.T, cfg.seed))
    return partial(
        propose_ei,
        raw_samples=cfg.raw_samples,
        num_restarts=cfg.num_restarts,
        sample_around_best_sigma=cfg.sample_around_best_sigma,
        batch_limit=cfg.batch_limit,
        maxiter=cfg.maxiter,
    )


# --- one run ---


def run(
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

    Everything one run needs that the loop itself does not: which surrogate and which proposer the
    method names (`surrogate_for`, `propose_for`), where the run lives, and what its resume is.
    The first `n_init` points are the shared Sobol design and are not rows: they have no fit and
    nothing to log but their observation, which the checkpoint carries. From `t = n_init` on,
    `sagp.bo.run_bo` runs the iterations and `RunLogger.record` writes each of them -- checkpoint
    first, then the coordinates and the samples, and the row last.

    `resume=True` (the default) continues an existing run directory: the checkpoint's config hash
    must match this configuration or the run stops with a `ValueError` rather than splicing two
    different runs together, and the logs are rolled back to the last iteration that is complete
    in *both* the checkpoint and `iterations.csv`, so an interrupted write is redone rather than
    left as a hole. What the loop is then handed is that rolled-back checkpoint as a `BOState`,
    which it cannot tell from the state its own earlier iterations built: redone iterations are
    bit-identical, since iteration `t` is a function of `(seed, t)` and of the points before it
    alone. A commit or a package version that has moved since the run started is warned about in
    `log.txt` and recorded in the manifest's `resumed` list rather than asserted
    (`RunLogger.note_resume`). `**overrides` sets any other `RunConfig` field (an unknown one is a
    `TypeError` from the dataclass, which is the check).
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

    run_dir = run_dir_for(out_dir, cfg)
    logger = RunLogger(run_dir, cfg)
    logger.f_star = float(objective.f_star)
    checkpoint = logger.load_checkpoint() if resume else None

    if checkpoint is None:
        logger.reset()
        logger.write_manifest(objective)
        logger.write_environment_lock()
        state = None
        logger.statuses = []
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
        t_done = cfg.n_init - 1 if logged is None else min(int(checkpoint["t_done"]), logged)
        state = BOState(
            checkpoint["X"][: t_done + 1],
            checkpoint["y"][: t_done + 1],
            checkpoint["f"][: t_done + 1],
        )
        logger.statuses = [
            str(s) for s in checkpoint["statuses"][: max(0, t_done - cfg.n_init + 1)]
        ]
        logger.truncate_to(t_done)
        logger.log(
            f"resume at t={t_done + 1} with T={cfg.T} "
            f"(checkpoint t_done={int(checkpoint['t_done'])}, last logged row {logged})"
        )
        logger.note_resume()

    sagp.bo.run_bo(
        objective,
        surrogate_for(cfg, objective.labels),
        cfg.seed,
        T=cfg.T,
        n_init=cfg.n_init,
        propose=propose_for(cfg),
        noiseless=cfg.noiseless,
        state=state,
        on_iteration=logger.record,
        log=logger.log,
    )
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
        "--raw-samples", type=int, default=512,
        help="raw Sobol candidates the maximizer draws (default 512); held fixed across methods",
    )
    parser.add_argument(
        "--num-restarts", type=int, default=5,
        help="L-BFGS-B restarts off those candidates (default 5); held fixed across methods",
    )
    parser.add_argument(
        "--sample-around-best-sigma", type=float, default=1e-3,
        help="sigma of the RAASP perturbations (default 0.001); held fixed across methods",
    )
    parser.add_argument(
        "--batch-limit", type=int, default=1,
        help="restarts optimized at once (default 1); held fixed across methods",
    )
    parser.add_argument(
        "--maxiter", type=int, default=200,
        help="L-BFGS-B iterations per restart (default 200); held fixed across methods",
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
        alpha=args.alpha,
        fixed_noise=args.fixed_noise,
        noiseless=args.noiseless,
        nuts=args.nuts,
        raw_samples=args.raw_samples,
        num_restarts=args.num_restarts,
        sample_around_best_sigma=args.sample_around_best_sigma,
        batch_limit=args.batch_limit,
        maxiter=args.maxiter,
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

    # Every `RunConfig` field except the seven `run` takes as arguments of its own. Derived
    # rather than listed, so that a field added to the config cannot silently stop reaching the
    # run while `--dry-run` goes on printing it.
    named = {"family", "seed", "D", "method", "T", "n_init", "out_dir"}
    overrides = {
        field.name: getattr(cfg, field.name)
        for field in dataclasses.fields(cfg)
        if field.name not in named
    }
    run_dir = run(
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
