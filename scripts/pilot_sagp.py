"""The `sagp` pilot: what one study run of each cell actually costs, and how often the gate fires.

Plan section 5's D8 estimated the four cells' cost from flop counts and assumed the three
centered cells were about 1.5x the reference. A gradient measurement on 2026-09-07 found them
15-55x instead -- they evaluate one exponential per (pair, coordinate), and two of them
materialize an (n, n, D) tensor XLA declines to fuse -- which moves the study's compute plan from
"a few hundred core-hours" into "several thousand". That number is now load-bearing, so it has to
be reproducible rather than quoted from a session log: this script is the measurement.

    PYTHONPATH=. /opt/anaconda3/envs/saasbo/bin/python scripts/pilot_sagp.py --stage grad

Everything here measures BoTorch's scheme: a cell is a `PyroModel` whose `sample` is the log
density NUTS differentiates, a fit is one NUTS run judged once against the diagnostic gate, and
the acquisition is `optimize_acqf` over BoTorch's `LogExpectedImprovement`.

Five stages, each of which prints its rows as it measures them and rewrites its own section of
`--out` after every row, so a run killed partway through still leaves everything it finished:

  grad   per-gradient time per cell at n = 50/100/200, D = 100, and the fit estimate it implies
  fit    two fits per (cell, n), their compile time, steps/iteration and one timed `propose_ei`
  gate   ten fits per (cell, n), for the rate at which the gate marks one `excluded`
  alpha  the D2 prior-predictive active-count table that calibrates `ALPHA_AMPLITUDE`
  cost   `grad` and `fit`, then the run-cost section combining them
  all    `alpha` first (it is free), then `grad`, `fit`, `gate` and the run-cost section

`grad` and `alpha` together take a couple of minutes. `fit` prices all four cells at n = 100 and
200: the reference cell is minutes and the three centered cells hours, so the stage is meant to
be killed once the rows you need are in, and `--once` halves it by dropping the second run of
each pair. `gate` is 20 fits, roughly 40-70 minutes. `--stage fit --once --small` is the smoke
run -- one fit of each cell at n = 50 -- and `--nuts` shortens the chain under any of them.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import warnings
from datetime import date
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
import torch  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from sagp.bo import iteration_rngs, propose_ei  # noqa: E402
from sagp.gp import (  # noqa: E402
    ACTIVE_EPS,
    ALPHA_AMPLITUDE,
    ALPHA_LENGTHSCALE,
    CELLS,
    ELL_EPS,
    RHO_EPS,
    CellGP,
    CellKey,
    NUTSConfig,
    fit,
    standardize,
)
from synthobj.families import make_family, noise_rng  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-09-09-botorch-saasbo-pilot.md"

D = 100  # the study's ambient dimension; every stage below is at this width

# `RunConfig`'s two acquisition sizes, passed to `propose_ei` explicitly rather than left to its
# defaults so that the numbers in the fit section's caption and the numbers actually timed are the
# same two constants and cannot drift apart.
RAW_SAMPLES = 512
NUM_RESTARTS = 5

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

# The budget the fit and gate stages sample at; `--nuts` rebinds it. The grad and cost stages are
# about the production budget by construction -- they fit nothing -- so `NUTS_ITERATIONS` above
# stays `NUTSConfig()`'s 768 whatever `--nuts` says.
NUTS: NUTSConfig = NUTSConfig()

# Plan D8's run-cost formula: sum_{n=20}^{199} cost(n) = 62.0 * cost(200) over a T = 200 run's 180
# fits, every one of which pays one acquisition optimization. There is no refit term: the
# diagnostic gate labels a fit `excluded` and never reruns it.
FITS_PER_RUN = 62.0
ACQ_PER_RUN = 180

# The sizes each fitting stage uses. The fit stage prices every cell, because the cost table's one
# borrowed number used to be the acquisition wall and that is the term the maximizer changed.
FIT_SIZES: tuple[tuple[CellKey, int], ...] = tuple(
    (cell, n) for cell in CELL_ORDER for n in (100, 200)
)
FIT_SIZES_SMALL: tuple[tuple[CellKey, int], ...] = tuple((cell, 50) for cell in CELL_ORDER)
GATE_SIZES: tuple[tuple[CellKey, int], ...] = (
    (("product", "lengthscale"), 100),
    (("additive", "amplitude"), 50),
)
GATE_KEYS = 10

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
STAGE_SECTIONS = ("grad", "fit", "gate", "alpha", "cost")


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
    """A plausible interior point of each cell's parameter space, under BoTorch's site names.

    The gradient's cost does not depend on *where* it is evaluated, only on the shapes, so the
    point is chosen to be ordinary (lengthscales spanning the study families' range, a small
    noise) rather than to be a posterior draw -- which would require the fit this stage exists to
    price. Only the sampled sites appear: the deterministic ones are recomputed by the model, and
    the amplitude cells have no `outputscale` at all.
    """
    if CELLS[cell].prior == "amplitude":
        return {
            "mean": 0.0,
            "noise": 0.05,
            "kernel_tausq": 0.05,
            "_a_sq": rng.uniform(0.1, 2.0, D),
            "kernel_ell": rng.uniform(0.3, 1.5, D),
        }
    return {
        "outputscale": 1.0,
        "mean": 0.0,
        "noise": 0.05,
        "kernel_tausq": 0.1,
        "_kernel_inv_length_sq": rng.uniform(0.1, 5.0, D),
    }


def _time_gradient(cell: CellKey, n: int, rng: np.random.Generator) -> float:
    """Seconds per jitted gradient of `cell`'s log-joint at (n, D): mean of `GRAD_TIMED_CALLS`.

    The log density is `CellGP`'s own `pyro_model.sample`, bound to its training data and taking
    no arguments -- the very callable `fit` hands NUTS -- so what is timed here is what the
    sampler pays, not a re-derivation of it. One untimed call first, so the reported time is the
    steady-state cost NUTS pays 768 times per fit and not the compilation it pays once. The data
    are random and standardized because the cost depends on the shapes alone.
    """
    X = rng.random((n, D))
    y = rng.standard_normal(n)
    y = (y - y.mean()) / y.std()

    gp = CellGP(torch.as_tensor(X), torch.as_tensor(y)[:, None], None, cell=CELLS[cell])
    params = _grad_params(cell, rng)

    grad_fn = jax.jit(jax.grad(lambda p: log_density(gp.pyro_model.sample, (), {}, p)[0]))
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


def _fit_row(cell: CellKey, n: int, X: np.ndarray, z: np.ndarray, label: str, seed: int) -> dict[str, object]:
    """One timed `fit` at the `--nuts` budget, flattened into a row of the stage's table.

    A fit is one NUTS run judged once, so there is one set of diagnostics rather than a per-attempt
    history: `status` is `ok` or `excluded` and the draws come back either way.
    """
    start = time.perf_counter()
    fitted = fit(X, z, seed, cell, nuts=NUTS)
    wall_s = time.perf_counter() - start
    diag = fitted.attempts[-1]
    return {
        "cell": cell,
        "n": n,
        "run": label,
        "wall_s": wall_s,
        "status": fitted.status,
        "r_hat_max": diag.r_hat_max,
        "n_eff_min": diag.n_eff_min,
        "divergences": diag.divergences,
        "num_steps_mean": diag.num_steps_mean,
        "fitted": fitted,
    }


def _print_fit_row(row: dict[str, object]) -> None:
    print(
        f"fit {_cell_name(row['cell']):22s} n={row['n']:3d} run={row['run']}"
        f"  wall={row['wall_s']:.1f}s status={row['status']}"
        f" r_hat_max={row['r_hat_max']:.3f} n_eff_min={row['n_eff_min']:.1f}"
        f" div={row['divergences']} steps={row['num_steps_mean']:.1f}"
        + (f" acq={row['acq_s']:.1f}s" if "acq_s" in row else ""),
        flush=True,
    )


_FIT_COLUMNS = ("cell", "n", "run", "wall_s", "status", "r_hat_max", "n_eff_min", "divergences", "num_steps_mean", "acq_s")


def _fit_body(
    rows: list[dict[str, object]],
    compile_s: dict[tuple[CellKey, int], float],
    small: bool,
    once: bool,
) -> str:
    notes = [
        note
        for note, on in (
            ("`--small` (a smoke run of the stage, not the production sizes)", small),
            ("`--once` (one run per pair, so no compile time is separated out)", once),
            (f"`--nuts {NUTS.num_warmup},{NUTS.num_samples},{NUTS.thinning}`", NUTS != NUTSConfig()),
        )
        if on
    ]
    note = ", ".join(notes)
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
    runs = (
        "Each (cell, n) is fitted once.\n"
        if once
        else "Each (cell, n) is fitted twice under the **same** seed 0: `fit` is deterministic\n"
        "given its seed, so run 2 repeats run 1's chain exactly and the two walls can differ only\n"
        "by the compilation run 1 paid.\n"
    )
    return (
        _run_label(note)
        + f"\n\n`NUTSConfig({NUTS.num_warmup}, {NUTS.num_samples}, {NUTS.thinning})` at tree depth\n"
        f"{NUTS.max_tree_depth}, dense mass matrix, on `make_family(\"aligned10\", 0)` at D = 100,\n"
        "fitted to a scrambled Sobol design observed through `noise_rng(0)` and standardized.\n"
        + runs
        + "`acq_s` times one `propose_ei` against that fit -- BoTorch's `LogExpectedImprovement`\n"
        f"maximized by `optimize_acqf` at {RAW_SAMPLES} raw Sobol candidates plus as many RAASP\n"
        f"perturbations, {NUM_RESTARTS} L-BFGS-B restarts one at a time, seeded from\n"
        "`iteration_rngs(0, 0)`; the first run of a pair pays JAX compilation for the fit and torch\n"
        "compilation for the acquisition, the second pays neither.\n\n"
        f"This stage fits {planned}; a pair missing from the table below was not reached before the\n"
        "run ended.\n\n"
        + table
        + (
            ""
            if once
            else "\nCompile time = run 1's wall minus run 2's at the same (cell, n), the two runs being"
            "\nthe same fit under the same seed, so this is compilation and nothing else:\n\n" + compiles
        )
    )


def stage_fit(out: Path, small: bool, once: bool) -> dict[str, dict[CellKey, float]]:
    """Two fits per (cell, n) -- one under `once` -- plus one timed acquisition on each."""
    objective = make_family("aligned10", 0, D=D)
    sizes = FIT_SIZES_SMALL if small else FIT_SIZES
    labels = ("1",) if once else ("1", "2")
    rows: list[dict[str, object]] = []
    compile_s: dict[tuple[CellKey, int], float] = {}
    steps: dict[CellKey, float] = {}
    acq: dict[CellKey, float] = {}

    for cell, n in sizes:
        X, z = _design(objective, n)
        # The loop's own target: the incumbent of the standardized observations, which `run_bo`
        # maximizes.
        best_f = float(np.max(z))
        pair: list[dict[str, object]] = []
        # The SAME seed for both runs (ruling R38). `fit` is deterministic given its seed, so run 2
        # repeats run 1's trajectory exactly: the two walls differ by the compilation run 1 paid
        # and by nothing else. Two different seeds would have mixed compile time with fit-to-fit
        # variation in the number of leapfrog steps, and could make the difference negative.
        for label in labels:
            row = _fit_row(cell, n, X, z, label, 0)
            start = time.perf_counter()
            propose_ei(
                row["fitted"],
                best_f,
                iteration_rngs(0, 0),
                0,
                raw_samples=RAW_SAMPLES,
                num_restarts=NUM_RESTARTS,
            )
            row["acq_s"] = time.perf_counter() - start
            _print_fit_row(row)
            pair.append(row)
            rows.append(row)
            _write_section(out, "fit", _fit_body(rows, compile_s, small, once))
        if not once:
            compile_s[(cell, n)] = pair[0]["wall_s"] - pair[1]["wall_s"]
            print(f"fit {_cell_name(cell):22s} n={n:3d} compile={compile_s[(cell, n)]:.1f}s", flush=True)
        # The pair's last run has the steady-state numbers: its fit and its acquisition both ran
        # against code JAX and torch had already compiled, which is the state all but the first of
        # a run's 180 iterations are in. Under `--once` there is no such run, and the cost stage's
        # numbers carry that compilation.
        steps[cell] = pair[-1]["num_steps_mean"]
        acq[cell] = pair[-1]["acq_s"]
        _write_section(out, "fit", _fit_body(rows, compile_s, small, once))

    return {"steps": steps, "acq": acq}


_GATE_COLUMNS = (
    "cell", "n", "k", "status",
    "r_hat_max", "n_eff_min", "divergences", "num_steps_mean", "wall_s",
)


def _gate_body(rows: list[dict[str, object]]) -> str:
    table = "| " + " | ".join(_GATE_COLUMNS) + " |\n" + "|---" * len(_GATE_COLUMNS) + "|\n"
    for row in rows:
        values = [
            _cell_name(row["cell"]) if column == "cell" else
            (f"{row[column]:.4g}" if isinstance(row[column], float) else str(row[column]))
            for column in _GATE_COLUMNS
        ]
        table += "| " + " | ".join(values) + " |\n"

    summary = ""
    for cell, n in GATE_SIZES:
        group = [row for row in rows if row["cell"] == cell and row["n"] == n]
        if not group:
            continue
        counts = {status: sum(row["status"] == status for row in group) for status in ("ok", "excluded")}
        e = counts["excluded"] / len(group)
        r_hats = [row["r_hat_max"] for row in group]
        n_effs = [row["n_eff_min"] for row in group]
        summary += (
            f"- {_cell_name(cell)}, n = {n}, {len(group)} fits: ok {counts['ok']}, excluded"
            f" {counts['excluded']} -> exclusion rate e = {e:.2f}; `r_hat_max`"
            f" min/median/max {min(r_hats):.3f}/{float(np.median(r_hats)):.3f}/{max(r_hats):.3f},"
            f" `n_eff_min` min/median/max {min(n_effs):.1f}/{float(np.median(n_effs)):.1f}/{max(n_effs):.1f}\n"
        )

    planned = ", ".join(f"{_cell_name(cell)} at n = {n}" for cell, n in GATE_SIZES)
    return (
        _run_label()
        + "\n\nHow often the diagnostic gate fires: the same `aligned10` design as the fit stage,\n"
        f"fitted {GATE_KEYS} times on seeds k = 0..{GATE_KEYS - 1}. The gate is a **label**, not a\n"
        "trigger: a fit whose chain fails `r_hat_max <= 1.1`, `n_eff_min >= 16` or\n"
        "`divergences <= 5` is marked `excluded` and its draws are used anyway, so `e` below costs\n"
        "the study nothing in compute and buys it a per-iteration record of which posteriors the\n"
        "readouts should be read against with suspicion. It is the rate at which the thresholds\n"
        "reject a chain the sampler was content with, so a large `e` on a cell that mixes fine is\n"
        "evidence about the thresholds rather than about the cell.\n\n"
        f"This stage fits {planned}; a pair missing below was not reached before the run ended.\n\n"
        + table
        + "\n"
        + summary
    )


def stage_gate(out: Path) -> None:
    """`GATE_KEYS` fits per (cell, n) on seeds 0..9, for the rate at which the gate excludes one."""
    objective = make_family("aligned10", 0, D=D)
    rows: list[dict[str, object]] = []
    for cell, n in GATE_SIZES:
        X, z = _design(objective, n)
        for k in range(GATE_KEYS):
            row = _fit_row(cell, n, X, z, str(k), k)
            row["k"] = k
            rows.append(row)
            print(
                f"gate {_cell_name(cell):22s} n={n:3d} k={k}  wall={row['wall_s']:.1f}s"
                f" status={row['status']} r_hat_max={row['r_hat_max']:.3f}"
                f" n_eff_min={row['n_eff_min']:.1f} div={row['divergences']}",
                flush=True,
            )
            _write_section(out, "gate", _gate_body(rows))


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
) -> None:
    """Plan D8's `62 * t_fit + 180 * t_acq` per cell, in core-hours per run."""
    reference: CellKey = ("product", "lengthscale")
    n_grad = max(n for n in GRAD_NS if n in grad[reference])

    columns = ("cell", f"t_grad(n = {n_grad}) ms", "steps/iter", "t_fit min", "t_acq s", "core-hours per run")
    table = "| " + " | ".join(columns) + " |\n" + "|---" * len(columns) + "|\n"
    borrowed = []
    for cell in CELL_ORDER:
        t_grad = grad[cell][n_grad]
        steps = fits["steps"].get(cell, fits["steps"][reference])
        t_acq = fits["acq"].get(cell, fits["acq"][reference])
        if cell not in fits["steps"]:
            borrowed.append(_cell_name(cell))
        t_fit = NUTS_ITERATIONS * steps * t_grad
        core_hours = (FITS_PER_RUN * t_fit + ACQ_PER_RUN * t_acq) / 3600.0
        table += (
            f"| {_cell_name(cell)} | {1e3 * t_grad:.3g} | {steps:.1f} | {t_fit / 60.0:.3g} |"
            f" {t_acq:.3g} | {core_hours:.3g} |\n"
        )
        print(f"cost {_cell_name(cell):22s} {core_hours:.3g} core-hours per T = 200, D = 100 run", flush=True)

    note = (
        f"\n{', '.join(borrowed)} fall back to the reference cell's steps/iteration and acquisition\n"
        "wall, the fit stage not having reached them before the run ended. Steps/iteration is a\n"
        "property of the sampler's adaptation rather than of the kernel's cost, so borrowing it is\n"
        "cheap; `t_acq` is **not** such a property, and borrowing it is the one place this table is\n"
        f"optimistic. The maximizer scores {2 * RAW_SAMPLES} candidates and then differentiates the\n"
        "very kernel the gradient column prices, through every L-BFGS-B step, so a borrowed\n"
        f"reference `t_acq` understates the `{ACQ_PER_RUN} x t_acq` term of the three centered cells\n"
        "-- their core-hours above are a lower bound in it. Only `--stage fit` at those cells\n"
        "measures it.\n"
        if borrowed
        else ""
    )
    body = (
        _run_label()
        + "\n\nPlan D8's run cost for one T = 200, D = 100 run, per cell:\n\n"
        "```\n"
        f"t_fit  = {NUTS_ITERATIONS} x steps x t_grad(n = {n_grad})\n"
        f"cost   = {FITS_PER_RUN:.0f} x t_fit + {ACQ_PER_RUN} x t_acq\n"
        "```\n\n"
        f"`t_grad` is the grad stage's largest measured n; `steps` and `t_acq` are the fit stage's\n"
        "measured `num_steps_mean` and acquisition wall. The 62 is sum_{n=20}^{199} cost(n) /\n"
        "cost(200) over a run's 180 fits. There is no refit term: the diagnostic gate labels a fit\n"
        "`excluded` and never reruns it, so each of those 180 fits is paid for exactly once.\n\n"
        + table + note
    )
    _write_section(out, "cost", body)


# --- CLI ---


def _nuts_config(text: str) -> NUTSConfig:
    """`--nuts W,S,K` -> `NUTSConfig(W, S, K)`: warmup draws, samples, thinning.

    `experiments.run_bo`'s own parser, restated here rather than imported: it is private to that
    module, and this script reaches only public `sagp` names.
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
        prog="python scripts/pilot_sagp.py",
        description=(
            "Measure what one sagp study run costs under BoTorch's fit and maximizer, and how "
            "often the diagnostic gate excludes a fit, writing one markdown section per stage "
            "into --out."
        ),
        epilog=(
            "Stages print each row as they measure it and rewrite their section after every row, so "
            "a run killed partway through keeps what it finished. `grad` and `alpha` take a couple "
            "of minutes; `fit` runs the reference cell in minutes and the centered cells in hours, "
            "and `gate` is roughly 40-70 minutes. `--stage fit --once --small` is the smoke run."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("grad", "fit", "gate", "alpha", "cost", "all"),
        help=(
            "which measurement to run; 'cost' runs 'grad' and 'fit' and combines them, and 'all' "
            "runs every stage"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"markdown file to write (default {DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="--stage fit: one run per (cell, n) instead of two, so no compile time is separated out",
    )
    parser.add_argument(
        "--nuts", type=_nuts_config, default=NUTSConfig(), metavar="W,S,K",
        help="NUTS warmup, samples and thinning for the fit and gate stages (default 512,256,16)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="run --stage fit at n = 50 for every cell (a smoke run of the stage, minutes not hours)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested stage(s) and write their sections; 0 on success, 2 on a bad argument."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)

    global NUTS
    NUTS = args.nuts

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # `alpha` runs first under `--stage all` because it is free (a second of numpy) and needs
    # nothing from the others: an `all` run killed in its first minute still leaves the D2 table.
    if args.stage in ("alpha", "all"):
        stage_alpha(out)
    grad = stage_grad(out) if args.stage in ("grad", "cost", "all") else None
    fits = stage_fit(out, args.small, args.once) if args.stage in ("fit", "cost", "all") else None
    if args.stage in ("gate", "all"):
        stage_gate(out)
    if args.stage in ("cost", "all"):
        stage_cost(out, grad, fits)

    print(f"stage {args.stage} finished in {time.perf_counter() - start:.1f} s; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
