"""The `sagp` pilot: what one study run of each cell actually costs, and how often it refits.

Plan section 5's D8 estimated the four cells' cost from flop counts and assumed the three
centered cells were about 1.5x the reference. A gradient measurement on 2026-09-07 found them
15-55x instead -- they evaluate one exponential per (pair, coordinate), and two of them
materialize an (n, n, D) tensor XLA declines to fuse -- which moves the study's compute plan from
"a few hundred core-hours" into "several thousand". That number is now load-bearing, so it has to
be reproducible rather than quoted from a session log: this script is the measurement, split into
stages short enough to run on a laptop (ledger ruling R26 -- no fit here takes longer than about
fifteen minutes).

    PYTHONPATH=. /opt/anaconda3/envs/saasbo/bin/python scripts/pilot_sagp.py --stage grad

Four stages, each of which prints its rows as it measures them and rewrites its own section of
`--out` after every row, so a run killed partway through still leaves everything it finished:

  grad   per-gradient time per cell at n = 50/100/200, D = 100, and the fit estimate it implies
  fit    two production-budget fits per (cell, n), their compile time, steps/iteration and one
         timed `optimize_ei`
  refit  ten fits per cell, for the D4 diagnostic gate's false-failure rate
  alpha  the D2 prior-predictive active-count table that calibrates `ALPHA_AMPLITUDE`
  all    `alpha` first (it is free), then `grad`, `fit`, `refit`, plus a run-cost section
         combining them

`grad` and `alpha` together take under fifteen seconds. `fit` and `refit` are the whole cost of
`--stage all`: at the sizes below, and with the per-gradient times `--stage grad` measured on
2026-09-08, `fit` is ≈ 20-30 minutes and `refit` ≈ 40-70, so budget one to two hours for `all`.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import warnings
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parent.parent

# A script under `scripts/` puts its own directory on `sys.path`, not the repo root, so an
# invocation without `PYTHONPATH=.` fails on `import sagp` -- which is exactly how the first run
# of the 2026-09-07 measurement died. Adding the root here makes the invocation form irrelevant.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from sagp.bo import optimize_ei  # noqa: E402
from sagp.gp import (  # noqa: E402
    ACTIVE_EPS,
    ALPHA_AMPLITUDE,
    ALPHA_LENGTHSCALE,
    CELLS,
    ELL_EPS,
    ELL_PRIOR,
    RHO_EPS,
    CellKey,
    NUTSConfig,
    fit,
    standardize,
)
from synthobj.families import make_family, noise_rng  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-09-07-sagp-pilot.md"

D = 100  # the study's ambient dimension; every stage below is at this width

# `RunConfig`'s acquisition settings, passed to `optimize_ei` explicitly rather than left to its
# defaults so that the numbers in the fit section's caption and the numbers actually timed are the
# same two constants and cannot drift apart.
ACQ_CANDIDATES = 5000
ACQ_RESTARTS = 5

# The cells in the order the 2026-09-07 cost measurement reports them -- the reference cell
# (SAASBO itself) first -- so a reader can compare row for row with the table quoted in the grad
# section rather than re-sorting two tables by eye.
CELL_ORDER: tuple[CellKey, ...] = (
    ("product", "lengthscale"),
    ("additive", "amplitude"),
    ("additive", "lengthscale"),
    ("product", "amplitude"),
)

# The controller's 2026-09-07 measurement (M3 laptop, mean of 5 jitted gradients of the log-joint,
# float64, learned noise, D = 100), in ms at n = 50/100/200. Quoted here, and printed beside the
# reproduced table, because "15-55x the reference" is the claim the study's compute plan rests on
# and a reader of the pilot file should be able to see both numbers at once.
CONTROLLER_GRAD_MS: dict[CellKey, tuple[float, float, float]] = {
    ("product", "lengthscale"): (0.4, 0.5, 3.1),
    ("additive", "amplitude"): (7.6, 20.0, 45.0),
    ("additive", "lengthscale"): (6.6, 56.0, 173.0),
    ("product", "amplitude"): (7.3, 29.0, 135.0),
}

GRAD_NS = (50, 100, 200)
GRAD_TIMED_CALLS = 5

# NUTS at `max_tree_depth = 6` takes between 40 and 63 leapfrog steps per iteration; `fit` records
# the mean it actually took as `num_steps_mean`, which is what the fit stage measures and the cost
# stage uses in place of this range.
STEPS_RANGE = (40, 63)
NUTS_ITERATIONS = NUTSConfig().num_warmup + NUTSConfig().num_samples  # 768 gradients-worth of iterations

# Plan D8's run-cost formula: sum_{n=20}^{199} cost(n) = 62.0 * cost(200) over a T = 200 run's 180
# fits, each diagnostic refit adds 1.67x a fit (D4's doubled warm-up), and every fitted iteration
# pays one acquisition optimization.
FITS_PER_RUN = 62.0
REFIT_FACTOR = 1.67
ACQ_PER_RUN = 180

# The stage whose fits are the small smoke variant, and the sizes each stage uses.
FIT_SIZES: tuple[tuple[CellKey, int], ...] = (
    (("product", "lengthscale"), 200),
    (("additive", "amplitude"), 100),
)
FIT_SIZES_SMALL: tuple[tuple[CellKey, int], ...] = (
    (("product", "lengthscale"), 50),
    (("additive", "amplitude"), 50),
)
REFIT_SIZES: tuple[tuple[CellKey, int], ...] = (
    (("product", "lengthscale"), 100),
    (("additive", "amplitude"), 50),
)
REFIT_KEYS = 10

ALPHA_DRAWS = 10_000

HEADER = """# sagp pilot: per-gradient cost, fit diagnostics and the alpha table

Generated by `scripts/pilot_sagp.py` (plan task 11). Each section is written by its own stage and
carries the date, host and core count of the run that produced it, so sections may come from
different runs; `--stage all` refreshes all of them. Regenerate with

```
PYTHONPATH=. /opt/anaconda3/envs/saasbo/bin/python scripts/pilot_sagp.py --stage all
```

(on a laptop, under `caffeinate -di`: a sleeping machine does not advance `perf_counter`, so a
suspended run reports times that are wrong rather than merely late).

"""

SECTION_PREFIX = "## Stage: "
STAGE_SECTIONS = ("grad", "fit", "refit", "alpha", "cost")


# --- output file ---


def _read_sections(out: Path) -> dict[str, str]:
    """The stage sections `out` already holds, as `{stage: body}`; `{}` if it does not exist."""
    if not out.exists():
        return {}
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in out.read_text().splitlines():
        if line.startswith(SECTION_PREFIX):
            current = line[len(SECTION_PREFIX) :].strip()
            bodies[current] = []
        elif current is not None:
            bodies[current].append(line)
    return {stage: "\n".join(body).strip() for stage, body in bodies.items()}


def _write_section(out: Path, stage: str, body: str) -> None:
    """Replace `stage`'s section of `out` (appending it in canonical order), atomically.

    Every stage calls this after each row it measures rather than once at the end, because the
    expected way to run the expensive stages on a laptop is to kill them when they have taken long
    enough -- and a killed run should leave the rows it finished, in a file that is never
    half-written. Rewriting the whole document each time keeps the section order fixed no matter
    which order the stages were run in.
    """
    sections = _read_sections(out)
    sections[stage] = body.strip()
    text = HEADER + "\n".join(
        f"{SECTION_PREFIX}{name}\n\n{sections[name]}\n" for name in STAGE_SECTIONS if name in sections
    )
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, out)


def _run_label(note: str = "") -> str:
    """The provenance line every section opens with: when, where, and on how many cores."""
    parts = [
        f"Measured {date.today().isoformat()} on `{platform.node()}`",
        f"`os.cpu_count()` = {os.cpu_count()}",
    ]
    if note:
        parts.append(note)
    return ", ".join(parts) + "."


def _cell_name(cell: CellKey) -> str:
    return "/".join(cell)


# --- stage 1: per-gradient time ---


def _grad_params(cell: CellKey, rng: np.random.Generator) -> dict[str, object]:
    """A plausible interior point of each cell's parameter space, as the 2026-09-07 script used.

    The gradient's cost does not depend on *where* it is evaluated, only on the shapes, so the
    point is chosen to be ordinary (lengthscales spanning the study families' range, a small
    noise) rather than to be a posterior draw -- which would require the fit this stage exists to
    price. Only the sampled sites appear: the deterministic ones are recomputed by the model.
    """
    if CELLS[cell].prior == "amplitude":
        return {
            "kernel_noise": 0.05,
            "kernel_tausq": 0.05,
            "_a_sq": rng.uniform(0.1, 2.0, D),
            "kernel_ell": rng.uniform(0.3, 1.5, D),
        }
    return {
        "kernel_var": 1.0,
        "kernel_noise": 0.05,
        "kernel_tausq": 0.1,
        "_kernel_inv_length_sq": rng.uniform(0.1, 5.0, D),
    }


def _time_gradient(cell: CellKey, n: int, rng: np.random.Generator) -> float:
    """Seconds per jitted gradient of `cell`'s log-joint at (n, D): mean of `GRAD_TIMED_CALLS`.

    One untimed call first, so the reported time is the steady-state cost NUTS pays 768 times per
    fit and not the compilation it pays once. The data are random and standardized because the
    cost depends on the shapes alone.
    """
    X = rng.random((n, D))
    y = rng.standard_normal(n)
    y = (y - y.mean()) / y.std()

    hyperparameters: dict[str, object] = {"alpha": CELLS[cell].alpha_default, "fixed_noise": None}
    if CELLS[cell].prior == "amplitude":
        hyperparameters["ell_prior"] = ELL_PRIOR
    model = partial(CELLS[cell].model, **hyperparameters)
    params = _grad_params(cell, rng)

    grad_fn = jax.jit(jax.grad(lambda p: log_density(model, (X, y), {}, p)[0]))
    jax.block_until_ready(grad_fn(params))

    times = []
    for _ in range(GRAD_TIMED_CALLS):
        start = time.perf_counter()
        jax.block_until_ready(grad_fn(params))
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


def _grad_cell(t_grad: float) -> str:
    """One table cell: per-gradient ms and the fit it implies over `STEPS_RANGE` leapfrog steps."""
    minutes = [NUTS_ITERATIONS * steps * t_grad / 60.0 for steps in STEPS_RANGE]
    return f"{1e3 * t_grad:.3g} ms -> {minutes[0]:.3g}-{minutes[1]:.3g} min/fit"


def _grad_body(measured: dict[CellKey, dict[int, float]]) -> str:
    head = "| cell | " + " | ".join(f"n = {n}" for n in GRAD_NS) + " |\n"
    head += "|---" * (len(GRAD_NS) + 1) + "|\n"
    rows = "".join(
        f"| {_cell_name(cell)} | "
        + " | ".join(_grad_cell(measured[cell][n]) if n in measured[cell] else "" for n in GRAD_NS)
        + " |\n"
        for cell in CELL_ORDER
        if measured.get(cell)  # a cell reached but not yet measured would be an all-empty row
    )
    quoted = "".join(
        f"| {_cell_name(cell)} | " + " | ".join(f"{ms} ms" for ms in CONTROLLER_GRAD_MS[cell]) + " |\n"
        for cell in CELL_ORDER
    )
    return (
        _run_label()
        + "\n\nOne jitted gradient of each cell's log-joint at D = 100 (`jax.grad` of\n"
        "`numpyro.infer.util.log_density` on random standardized data with a learned noise), the\n"
        f"mean of {GRAD_TIMED_CALLS} timed calls after one untimed compile call. The fit estimate\n"
        f"is `{NUTS_ITERATIONS} x steps x t_grad` over the {STEPS_RANGE[0]}-{STEPS_RANGE[1]}\n"
        "leapfrog steps per iteration that `max_tree_depth = 6` allows.\n\n" + head + rows + "\n"
        "The 2026-09-07 measurement this stage reproduces (same laptop, same procedure), for\n"
        "comparison:\n\n"
        "| cell | " + " | ".join(f"n = {n}" for n in GRAD_NS) + " |\n"
        + "|---" * (len(GRAD_NS) + 1) + "|\n" + quoted
    )


def stage_grad(out: Path) -> dict[CellKey, dict[int, float]]:
    """Per-gradient time for every cell at every `GRAD_NS`, printed and written as it is measured."""
    rng = np.random.default_rng(0)
    measured: dict[CellKey, dict[int, float]] = {}
    for cell in CELL_ORDER:
        measured[cell] = {}
        for n in GRAD_NS:
            t_grad = _time_gradient(cell, n, rng)
            measured[cell][n] = t_grad
            print(f"grad {_cell_name(cell):22s} n={n:3d}  {_grad_cell(t_grad)}", flush=True)
            _write_section(out, "grad", _grad_body(measured))
    return measured


# --- stage 2/3: fits ---


def _design(objective: object, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`identify()`'s design: `n` scrambled Sobol points, observed with noise, standardized.

    The same construction the identification runs use, so a fit timed here is a fit of the study's
    own data rather than of a differently conditioned problem. `qmc.Sobol` warns when `n` is not a
    power of two; the reference suppresses that warning and so does this.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        X = qmc.Sobol(objective.D, scramble=True, seed=seed).random(n)
    y = objective.observe(X, noise_rng(seed, run=0))
    z, _, _ = standardize(y)
    return X, z


def _fit_row(cell: CellKey, n: int, X: np.ndarray, z: np.ndarray, label: str, key) -> dict[str, object]:
    """One timed `fit` at the production budget, flattened into a row of the stage's table.

    Both the deciding attempt's diagnostics (unprefixed) and attempt 0's (`a0_`) are recorded. On
    a fit that passed first time they are the same numbers; on a `refit` or `excluded` fit the
    `a0_` set is the only record of *how far outside* the gate the first attempt fell, which is
    what makes the refit rate interpretable rather than just a count.
    """
    start = time.perf_counter()
    fitted = fit(X, z, key, cell)
    wall_s = time.perf_counter() - start
    diag, first = fitted.attempts[-1], fitted.attempts[0]
    return {
        "cell": cell,
        "n": n,
        "run": label,
        "wall_s": wall_s,
        "attempts": len(fitted.attempts),
        "status": fitted.status,
        "r_hat_max": diag.r_hat_max,
        "n_eff_min": diag.n_eff_min,
        "divergences": diag.divergences,
        "num_steps_mean": diag.num_steps_mean,
        "a0_r_hat_max": first.r_hat_max,
        "a0_n_eff_min": first.n_eff_min,
        "a0_divergences": first.divergences,
        "fitted": fitted,
    }


def _print_fit_row(row: dict[str, object]) -> None:
    print(
        f"fit {_cell_name(row['cell']):22s} n={row['n']:3d} run={row['run']}"
        f"  wall={row['wall_s']:.1f}s attempts={row['attempts']} status={row['status']}"
        f" r_hat_max={row['r_hat_max']:.3f} n_eff_min={row['n_eff_min']:.1f}"
        f" div={row['divergences']} steps={row['num_steps_mean']:.1f}"
        + (f" acq={row['acq_s']:.1f}s" if "acq_s" in row else ""),
        flush=True,
    )


_FIT_COLUMNS = ("cell", "n", "run", "wall_s", "attempts", "status", "r_hat_max", "n_eff_min", "divergences", "num_steps_mean", "acq_s")


def _fit_body(rows: list[dict[str, object]], compile_s: dict[tuple[CellKey, int], float], small: bool) -> str:
    note = "`--n-fit-small` (a smoke run of the stage, not the production sizes)" if small else ""
    table = "| " + " | ".join(_FIT_COLUMNS) + " |\n" + "|---" * len(_FIT_COLUMNS) + "|\n"
    for row in rows:
        values = []
        for column in _FIT_COLUMNS:
            value = row.get(column)
            if column == "cell":
                values.append(_cell_name(row["cell"]))
            elif value is None:
                values.append("")
            elif isinstance(value, float):
                # 4 significant figures, not 3: it keeps `r_hat_max` readable against the 1.1 gate
                # and keeps a wall of over 1000 s out of scientific notation.
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        table += "| " + " | ".join(values) + " |\n"
    compiles = "".join(
        f"- {_cell_name(cell)}, n = {n}: {seconds:.1f} s\n" for (cell, n), seconds in compile_s.items()
    )
    planned = ", ".join(f"{_cell_name(cell)} at n = {n}" for cell, n in (FIT_SIZES_SMALL if small else FIT_SIZES))
    return (
        _run_label(note)
        + "\n\nProduction `NUTSConfig()` (512 warm-up / 256 samples / thinning 16, tree depth 6) on\n"
        "`make_family(\"aligned10\", 0)` at D = 100, fitted to a scrambled Sobol design observed\n"
        "through `noise_rng(0)` and standardized. Each (cell, n) is fitted twice under the **same**\n"
        "key `PRNGKey(0)`: `fit` is deterministic given its key, so run 2 repeats run 1's chain\n"
        "exactly and the two walls can differ only by the compilation run 1 paid. `acq_s` times one\n"
        f"`optimize_ei` against that fit at the production {ACQ_CANDIDATES} candidates /\n"
        f"{ACQ_RESTARTS} restarts with `sobol_seed=0` and `default_rng(0)`; run 1 pays JAX\n"
        "compilation for both the fit and the acquisition, run 2 pays neither.\n\n"
        f"This stage fits {planned}; a pair missing from the table below was not reached before the\n"
        "run ended.\n\n"
        + table
        + "\nCompile time = run 1's wall minus run 2's at the same (cell, n), the two runs being the\n"
        "same fit under the same key, so this is compilation and nothing else:\n\n"
        + compiles
    )


def stage_fit(out: Path, small: bool) -> dict[str, dict[CellKey, float]]:
    """Two production-budget fits per (cell, n) plus one timed acquisition on each."""
    objective = make_family("aligned10", 0, D=D)
    sizes = FIT_SIZES_SMALL if small else FIT_SIZES
    rows: list[dict[str, object]] = []
    compile_s: dict[tuple[CellKey, int], float] = {}
    steps: dict[CellKey, float] = {}
    acq: dict[CellKey, float] = {}

    for cell, n in sizes:
        X, z = _design(objective, n)
        # The loop's own target: the incumbent of the standardized, negated observations.
        y_target = float(np.min(z))
        pair: list[dict[str, object]] = []
        # The SAME key for both runs (ruling R38). `fit` is deterministic given its key, so run 2
        # repeats run 1's trajectory exactly: the two walls differ by the compilation run 1 paid
        # and by nothing else. Two different keys would have mixed compile time with fit-to-fit
        # variation in the number of leapfrog steps, and could make the difference negative.
        for label in ("1", "2"):
            row = _fit_row(cell, n, X, z, label, jax.random.PRNGKey(0))
            start = time.perf_counter()
            optimize_ei(
                row["fitted"],
                y_target,
                sobol_seed=0,
                jitter_rng=np.random.default_rng(0),
                num_restarts_ei=ACQ_RESTARTS,
                num_init=ACQ_CANDIDATES,
            )
            row["acq_s"] = time.perf_counter() - start
            _print_fit_row(row)
            pair.append(row)
            rows.append(row)
            _write_section(out, "fit", _fit_body(rows, compile_s, small))
        compile_s[(cell, n)] = pair[0]["wall_s"] - pair[1]["wall_s"]
        # Run 2's numbers are the steady-state ones: its fit and its acquisition both ran against
        # code JAX had already compiled, which is the state all but the first of a run's 180
        # iterations are in.
        steps[cell] = pair[1]["num_steps_mean"]
        acq[cell] = pair[1]["acq_s"]
        print(f"fit {_cell_name(cell):22s} n={n:3d} compile={compile_s[(cell, n)]:.1f}s", flush=True)
        _write_section(out, "fit", _fit_body(rows, compile_s, small))

    return {"steps": steps, "acq": acq}


_REFIT_COLUMNS = (
    "cell", "n", "k", "status", "attempts",
    "r_hat_max", "n_eff_min", "divergences",
    "a0_r_hat_max", "a0_n_eff_min", "a0_divergences",
    "wall_s",
)


def _refit_body(rows: list[dict[str, object]]) -> str:
    table = "| " + " | ".join(_REFIT_COLUMNS) + " |\n" + "|---" * len(_REFIT_COLUMNS) + "|\n"
    for row in rows:
        values = [
            _cell_name(row["cell"]) if column == "cell" else
            (f"{row[column]:.4g}" if isinstance(row[column], float) else str(row[column]))
            for column in _REFIT_COLUMNS
        ]
        table += "| " + " | ".join(values) + " |\n"

    summary = ""
    for cell, n in REFIT_SIZES:
        group = [row for row in rows if row["cell"] == cell and row["n"] == n]
        if not group:
            continue
        counts = {status: sum(row["status"] == status for row in group) for status in ("ok", "refit", "excluded")}
        r = (counts["refit"] + counts["excluded"]) / len(group)
        r_hats = [row["r_hat_max"] for row in group]
        n_effs = [row["n_eff_min"] for row in group]
        summary += (
            f"- {_cell_name(cell)}, n = {n}, {len(group)} fits: ok {counts['ok']}, refit"
            f" {counts['refit']}, excluded {counts['excluded']} -> refit rate r ="
            f" {r:.2f}, exclusion rate {counts['excluded'] / len(group):.2f}; `r_hat_max`"
            f" min/median/max {min(r_hats):.3f}/{float(np.median(r_hats)):.3f}/{max(r_hats):.3f},"
            f" `n_eff_min` min/median/max {min(n_effs):.1f}/{float(np.median(n_effs)):.1f}/{max(n_effs):.1f}\n"
        )

    planned = ", ".join(f"{_cell_name(cell)} at n = {n}" for cell, n in REFIT_SIZES)
    return (
        _run_label()
        + "\n\nThe D4 gate's false-failure rate: the production budget on the same `aligned10` design\n"
        f"as the fit stage, fitted {REFIT_KEYS} times with `fold_in(PRNGKey(0), k)`. `r` below is the\n"
        "fraction of fits that ran a second attempt -- both `refit` and `excluded` paid for one --\n"
        "which is the `r` the run-cost formula's `(1 + 1.67 r)` factor takes.\n\n"
        "The unprefixed diagnostics are the *deciding* attempt's (the one whose draws are returned);\n"
        "the `a0_` ones are attempt 0's. On an `ok` row the two are the same numbers; on a `refit`\n"
        "or `excluded` row the `a0_` set says how far attempt 0 fell outside the gate\n"
        "(`r_hat_max <= 1.1`, `n_eff_min >= 16`, `divergences <= 5`), which is what separates a gate\n"
        "that fires on genuinely bad chains from one whose thresholds are simply too tight.\n\n"
        f"This stage fits {planned}; a pair missing below was not reached before the run ended.\n\n"
        + table
        + "\n"
        + summary
    )


def stage_refit(out: Path) -> dict[CellKey, float]:
    """Ten fits per cell at the D4 sizes; returns each cell's refit rate for the cost stage."""
    objective = make_family("aligned10", 0, D=D)
    rows: list[dict[str, object]] = []
    rates: dict[CellKey, float] = {}
    for cell, n in REFIT_SIZES:
        X, z = _design(objective, n)
        for k in range(REFIT_KEYS):
            row = _fit_row(cell, n, X, z, str(k), jax.random.fold_in(jax.random.PRNGKey(0), k))
            row["k"] = k
            rows.append(row)
            print(
                f"refit {_cell_name(cell):22s} n={n:3d} k={k}  wall={row['wall_s']:.1f}s"
                f" status={row['status']} r_hat_max={row['r_hat_max']:.3f}"
                f" n_eff_min={row['n_eff_min']:.1f} div={row['divergences']}",
                flush=True,
            )
            _write_section(out, "refit", _refit_body(rows))
        group = [row for row in rows if row["cell"] == cell and row["n"] == n]
        rates[cell] = sum(row["status"] != "ok" for row in group) / len(group)
    return rates


# --- stage 4: the alpha table ---


def _active_counts(rng: np.random.Generator, alpha: float, cutoff: float) -> np.ndarray:
    """#{i : tausq * lam_i > cutoff} per draw, tausq ~ HalfCauchy(alpha), lam_i ~ HalfCauchy(1).

    Drawn in numpy as |standard Cauchy| times the scale -- exactly `dist.HalfCauchy(scale)` -- so
    the calibration is checked independently of the JAX code it calibrates.
    """
    tausq = alpha * np.abs(rng.standard_cauchy(ALPHA_DRAWS))[:, None]
    lam = np.abs(rng.standard_cauchy((ALPHA_DRAWS, D)))
    return np.count_nonzero(tausq * lam > cutoff, axis=1)


def stage_alpha(out: Path) -> None:
    """Plan D2's prior-predictive active-count table: the reference's count and the a^2 cell's."""
    rng = np.random.default_rng(0)
    table = "| prior scale | alpha | cutoff | median | q25 | q75 |\n" + "|---" * 6 + "|\n"
    for label, alpha, cutoff in (
        ("reference (rho)", ALPHA_LENGTHSCALE, RHO_EPS),
        ("amplitude (a^2)", ALPHA_AMPLITUDE, ACTIVE_EPS),
    ):
        counts = _active_counts(rng, alpha, cutoff)
        quantiles = [float(np.quantile(counts, q)) for q in (0.5, 0.25, 0.75)]
        table += (
            f"| {label} | {alpha:.4g} | {cutoff:.4g} | "
            + " | ".join(f"{q:.0f}" for q in quantiles)
            + " |\n"
        )
        print(
            f"alpha {label:16s} alpha={alpha:.4g} cutoff={cutoff:.4g}"
            f" median={quantiles[0]:.0f} q25={quantiles[1]:.0f} q75={quantiles[2]:.0f}",
            flush=True,
        )

    body = (
        _run_label()
        + f"\n\n{ALPHA_DRAWS} draws of the prior-predictive active count #{{i : theta_i > cutoff}} at\n"
        "D = 100, for the reference's rho scale and the amplitude cells' a^2 scale. Both priors are the\n"
        "same half-Cauchy scale mixture, so the count depends on (alpha, cutoff) only through\n"
        "cutoff / alpha and matching them fixes `ALPHA_AMPLITUDE` in closed form (plan D2); this table\n"
        "is the numerical check that the whole count distribution matches, not only its median.\n\n"
        + table
        + f"\nConstants: `ELL_EPS` = {ELL_EPS:.4f} (v(ELL_EPS) = ACTIVE_EPS = {ACTIVE_EPS}), `RHO_EPS` ="
        f" {RHO_EPS:.4f}, `ALPHA_AMPLITUDE` = ALPHA_LENGTHSCALE * ACTIVE_EPS / RHO_EPS ="
        f" {ALPHA_AMPLITUDE:.5f}.\n"
    )
    _write_section(out, "alpha", body)


# --- the run-cost section ---


def stage_cost(
    out: Path,
    grad: dict[CellKey, dict[int, float]],
    fits: dict[str, dict[CellKey, float]],
    refit_rates: dict[CellKey, float],
) -> None:
    """Plan D8's `62 * t_fit * (1 + 1.67 r) + 180 * t_acq` per cell, in core-hours per run."""
    reference: CellKey = ("product", "lengthscale")
    n_grad = max(n for n in GRAD_NS if n in grad[reference])

    columns = ("cell", f"t_grad(n = {n_grad}) ms", "steps/iter", "t_fit min", "r", "t_acq s", "core-hours per run")
    table = "| " + " | ".join(columns) + " |\n" + "|---" * len(columns) + "|\n"
    borrowed = []
    for cell in CELL_ORDER:
        t_grad = grad[cell][n_grad]
        steps = fits["steps"].get(cell, fits["steps"][reference])
        t_acq = fits["acq"].get(cell, fits["acq"][reference])
        r = refit_rates.get(cell, refit_rates[reference])
        if cell not in fits["steps"] or cell not in refit_rates:
            borrowed.append(_cell_name(cell))
        t_fit = NUTS_ITERATIONS * steps * t_grad
        core_hours = (FITS_PER_RUN * t_fit * (1.0 + REFIT_FACTOR * r) + ACQ_PER_RUN * t_acq) / 3600.0
        table += (
            f"| {_cell_name(cell)} | {1e3 * t_grad:.3g} | {steps:.1f} | {t_fit / 60.0:.3g} |"
            f" {r:.2f} | {t_acq:.3g} | {core_hours:.3g} |\n"
        )
        print(f"cost {_cell_name(cell):22s} {core_hours:.3g} core-hours per T = 200, D = 100 run", flush=True)

    note = (
        f"\n{', '.join(borrowed)} fall back to the reference cell's steps/iteration, acquisition\n"
        "wall and refit rate wherever the fit and refit stages did not fit that cell themselves:\n"
        "only two of the four are fitted, and steps/iteration is a property of the sampler's\n"
        "adaptation rather than of the kernel's cost.\n\n"
        "`t_acq` is **not** such a property, and borrowing it is the one place this table is\n"
        "optimistic. The acquisition scores its 5000 candidates through the very kernel the\n"
        "gradient uses, and additive/lengthscale and product/amplitude are the two whose\n"
        "`(n_test, n, D)` broadcast XLA declines to fuse, so a borrowed reference `t_acq`\n"
        f"understates their `{ACQ_PER_RUN} x t_acq` term -- their core-hours above are a lower bound\n"
        "in it. Only `--stage fit` at those cells measures it.\n"
        if borrowed
        else ""
    )
    body = (
        _run_label()
        + "\n\nPlan D8's run cost for one T = 200, D = 100 run, per cell:\n\n"
        "```\n"
        f"t_fit  = {NUTS_ITERATIONS} x steps x t_grad(n = {n_grad})\n"
        f"cost   = {FITS_PER_RUN:.0f} x t_fit x (1 + {REFIT_FACTOR} r) + {ACQ_PER_RUN} x t_acq\n"
        "```\n\n"
        f"`t_grad` is the grad stage's largest measured n; `steps` and `t_acq` are the fit stage's\n"
        "measured `num_steps_mean` and acquisition wall; `r` is the refit stage's refit rate. The 62\n"
        "is sum_{n=20}^{199} cost(n) / cost(200) over a run's 180 fits and the 1.67 is a doubled\n"
        "warm-up refit's cost, both from the plan.\n\n" + table + note
    )
    _write_section(out, "cost", body)


# --- CLI ---


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/pilot_sagp.py",
        description=(
            "Measure what one sagp study run costs and how often the D4 diagnostic gate refits, "
            "writing one markdown section per stage into --out."
        ),
        epilog=(
            "Stages print each row as they measure it and rewrite their section after every row, so "
            "a run killed partway through keeps what it finished. `grad` and `alpha` take seconds; "
            "`fit` is roughly 20-30 minutes and `refit` roughly 40-70."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("grad", "fit", "refit", "alpha", "all"),
        help="which measurement to run; 'all' runs the four and adds the run-cost section",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"markdown file to write (default {DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--n-fit-small",
        action="store_true",
        help="run --stage fit at n = 50 for both cells (a smoke run of the stage, minutes not tens of minutes)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested stage(s) and write their sections; 0 on success, 2 on a bad argument."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # `alpha` runs first under `--stage all` because it is free (a second of numpy) and needs
    # nothing from the others: an `all` run killed in its first minute still leaves the D2 table.
    if args.stage in ("alpha", "all"):
        stage_alpha(out)
    grad = stage_grad(out) if args.stage in ("grad", "all") else None
    fits = stage_fit(out, args.n_fit_small) if args.stage in ("fit", "all") else None
    refit_rates = stage_refit(out) if args.stage in ("refit", "all") else None
    if args.stage == "all":
        stage_cost(out, grad, fits, refit_rates)

    print(f"stage {args.stage} finished in {time.perf_counter() - start:.1f} s; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
