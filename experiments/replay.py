"""experiments.replay: refit a cell on the data a stored run fitted at iteration t (plan D9).

A stored BO run keeps everything its fit at iteration t saw: the checkpoint holds the design and
the observations, so `X[:t]` and `standardize(y[:t])` are that fit's inputs, and row t of
`iterations.csv` holds its standardization and its diagnostics. `run_replay` rebuilds those inputs
-- refusing a run that cannot reproduce them (`load_source`, `dataset_at`) -- and refits a cell on
them with the loop's own seed for t, `iteration_rngs(seed, t).nuts_seed`, under the run's own
settings, writing what the loop would have written about that fit. An R2-D2 cell replayed on its
half-Cauchy twin's run (`TWINS`) is diagnosed on exactly the datasets its twin was. A half-Cauchy
cell replayed on its own run is the control: the loop's own computation again, which on a CPU
reproduces the stored fit bit for bit and on a GPU measures the device's run-to-run variation. Both
go through the one code path. `compare` joins every replayed fit to the stored fit of the same
(family, seed, t) and writes the tables and the verdict of gate G2 (plan, Task 11), which decides
whether the R2-D2 cells run in the loop.

    python -m experiments.replay fit --runs runs/ --out runs_replay/ --family aligned10 --seed 3 \
        --cell additive/amplitude_r2d2 [--source additive/amplitude] [--t 50,100,199] \
        [--nuts 512,256,16] [--max-tree-depth 6] [--dry-run]
    python -m experiments.replay compare --runs runs/ --replay runs_replay/ --out <report dir>
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from experiments.runlog import RunConfig, RunLogger, config_hash, run_dir_for
from sagp.bo import iteration_rngs
from sagp.diagnostics import DiagThresholds, Diagnostics
from sagp.gp import CELLS, R2D2_K, Cell, NUTSConfig, fit, standardize
from sagp.r2d2 import log_gamma_icdf
from sagp.readouts import first_order_r2, r2d2_r2, readouts


# Each R2-D2 cell and the half-Cauchy twin whose runs it is replayed on (D3's spellings): the two
# share a kernel and a native site and differ in the prior block alone.
TWINS: dict[str, str] = {
    "additive/amplitude_r2d2": "additive/amplitude",
    "product/amplitude_r2d2": "product/amplitude",
    "additive/lengthscale_r2d2": "additive/lengthscale",
    "product/lengthscale_r2d2": "product/lengthscale",
}
# The iterations replayed: each carries the stored Sobol readout (t = 0 mod 25, and t = T - 1).
REPLAY_T: tuple[int, ...] = (50, 100, 199)

# What a replay can refit or read from: the eight cells, spelled as methods. The MAP references and
# the Sobol search run no chain, so there is nothing of theirs to diagnose.
_CELL_METHODS: tuple[str, ...] = tuple("/".join(key) for key in CELLS)
# The fit's twelve diagnostic numbers, as `iterations.csv` carries them: every `Diagnostics` field
# but the three a row states in other columns (`fit_wall_s`, `status` and `reason`).
_DIAG_FIELDS: tuple[str, ...] = tuple(
    field.name
    for field in dataclasses.fields(Diagnostics)
    if field.name not in ("wall_s", "passed", "reason")
)
# One `replay.csv` row: the in-loop row's vocabulary for the fit alone (D9). `budget` is
# "warmup/samples/thinning/max_tree_depth" -- no comma, so `cut -d,` still finds every column --
# and a file holds one budget.
_REPLAY_FIELDS: tuple[str, ...] = (
    "t", "n", "method", "source_method", "family", "seed", "nuts_seed", "budget",
    "status", "reason", *_DIAG_FIELDS,
    "fit_wall_s", "readout_wall_s", "y_mean", "y_std", "r2d2_r2", "first_order_r2", "device",
)


# --- the source run ---


@dataclass(frozen=True)
class Source:
    """A stored run as the replay reads it, checked once when it is read (`load_source`)."""

    dir: Path
    cfg: RunConfig  # rebuilt from manifest.json, and reproducing the config_hash it records
    manifest: dict[str, object]
    checkpoint: dict[str, np.ndarray]
    rows: dict[int, dict[str, str]]  # iterations.csv's complete rows, by t


def load_source(run_dir: str | os.PathLike[str]) -> Source:
    """Read a stored run directory, refusing one whose settings or checkpoint cannot be trusted.

    The `RunConfig` is rebuilt from `manifest.json` (`nuts` and `thresholds` from their dicts,
    `ell_prior` as a tuple) and must hash to the `config_hash` the manifest records, so the
    settings a replay takes from it are the ones the run ran under. The checkpoint must carry that
    same hash: a checkpoint under another one belongs to a run resumed under another
    configuration, whose later points another configuration chose.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fields = {field.name: manifest[field.name] for field in dataclasses.fields(RunConfig)}
    fields["nuts"] = NUTSConfig(**fields["nuts"])
    fields["thresholds"] = DiagThresholds(**fields["thresholds"])
    fields["ell_prior"] = tuple(fields["ell_prior"])
    cfg = RunConfig(**fields)
    recorded = manifest["config_hash"]
    if config_hash(cfg) != recorded:
        raise ValueError(
            f"{run_dir}/manifest.json does not reproduce the config_hash it records "
            f"({recorded}): its settings cannot be trusted to be the run's"
        )
    with np.load(run_dir / "checkpoint.npz", allow_pickle=False) as data:
        checkpoint = {name: data[name] for name in data.files}
    if str(checkpoint["config_hash"]) != recorded:
        raise ValueError(
            f"{run_dir}: checkpoint.npz was written under config_hash "
            f"{checkpoint['config_hash']}, the manifest records {recorded}: the run was resumed "
            "under another configuration, so its points are not all its own"
        )
    _, rows = _read_csv(run_dir / "iterations.csv")
    return Source(run_dir, cfg, manifest, checkpoint, {int(row["t"]): row for row in rows})


def dataset_at(source: Source, t: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(X[:t], z, y_mean, y_std): what the loop's fit at iteration t saw, or a `ValueError`.

    Iteration t fitted the t points evaluated before it, standardized together. Refused unless
    the checkpoint holds iteration t (`t <= t_done`), `iterations.csv` has a complete row t, and
    `y[:t]` standardizes to that row's `y_mean` and `y_std` exactly -- the check that the stored
    observations are the ones the fit was handed, and that no shorter or other dataset is fitted.
    """
    t_done = int(source.checkpoint["t_done"])
    if t > t_done:
        raise ValueError(
            f"{source.dir}: t={t} is beyond the run's last completed iteration t_done={t_done}"
        )
    row = source.rows.get(t)
    if row is None:
        raise ValueError(f"{source.dir}: iterations.csv has no complete row t={t}")
    X = source.checkpoint["X"][:t]
    z, y_mean, y_std = standardize(source.checkpoint["y"][:t])
    if float(row["y_mean"]) != y_mean or float(row["y_std"]) != y_std:
        raise ValueError(
            f"{source.dir}: y[:{t}] standardizes to y_mean={y_mean!r}, y_std={y_std!r}, but row "
            f"t={t} recorded y_mean={row['y_mean']}, y_std={row['y_std']}: the checkpoint's "
            "observations are not the ones this iteration fitted"
        )
    return X, z, y_mean, y_std


# --- the replay ---


def run_replay(
    runs: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    family: str,
    seed: int,
    method: str,
    source_method: str | None = None,
    ts: Iterable[int] = REPLAY_T,
    nuts: NUTSConfig | None = None,
    max_tree_depth: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Refit `method` on the stored run's data at every t of `ts`; returns the replay directory.

    The source is `<runs>/<family>/<source_method>/seed##`, `source_method` defaulting to the
    cell's twin for an R2-D2 cell and to the cell itself for a half-Cauchy one (the control). Every
    t is checked (`dataset_at`) before anything is fitted or written, so a refused t costs the
    others nothing. Each fit takes `alpha`, `fixed_noise`, `thresholds` and `ell_prior` from the
    source's configuration, and its budget too unless `nuts` (every field) or `max_tree_depth`
    replaces it; the readouts take `sobol_n`, and compute the Sobol index where the loop did.
    A twin replay -- a cell other than the source's -- is refused when the source set `alpha`,
    the half-Cauchy global scale there but k on an R2-D2 cell (ruling R27); a control takes the
    source's `alpha` through. Before its first timed fit the process builds the R2-D2 map's
    table at `R2D2_K`, whatever the cell (ruling R26), so `fit_wall_s` holds a fit and that n's
    compilation but not the table's one-time build.

    The replay directory, `<out>/<family>/<method>/seed##` (`run_dir_for`): `replay.csv`, one row
    per t (`_REPLAY_FIELDS`); `coords.csv` and `samples/t###.npz`, written by `RunLogger` exactly
    as the loop writes them; `manifest.json`, the source's `RunConfig` with `method` replaced, its
    hash, the source's objective, and a `source` block with the source's path, hash, commit and
    devices; and `log.txt`. A t with a complete row is skipped, and the row is written last, as in
    the loop, so what a kill interrupted is rolled back and redone. A directory holds one source
    and one budget: a call under another source or budget is refused, as is one whose source
    configuration differs from the manifest's, and `out` may not lie in `runs`. A fit that raises
    propagates and leaves its t undone: the loop's retry on `retry_seed` is not replayed. `dry_run`
    prints each t's n and seed and the budget after every check, and writes nothing.
    """
    cell = _cell(method)
    source_method = TWINS.get(method, method) if source_method is None else source_method
    _cell(source_method)
    # The stored runs are only read: under the same root a control's replay directory would be
    # its source's own, whose coordinates and draws the rollback below would then delete.
    runs_root, out_root = Path(runs).resolve(), Path(out).resolve()
    if out_root == runs_root or runs_root in out_root.parents:
        raise ValueError(f"--out {out} lies in --runs {runs}: the stored runs are only read")
    source = load_source(_run_dir(runs, family, source_method, seed))
    if (source.cfg.family, source.cfg.seed, source.cfg.method) != (family, seed, source_method):
        raise ValueError(
            f"{source.dir}/manifest.json names family={source.cfg.family}, "
            f"seed={source.cfg.seed}, method={source.cfg.method}, not the run its path does"
        )
    if method != source_method and source.cfg.alpha is not None:
        raise ValueError(
            f"{source.dir} ran at alpha={source.cfg.alpha!r}: alpha is the half-Cauchy global "
            "scale in the source's family but k on an R2-D2 cell, so a twin replay cannot carry "
            f"it over to {method} (ruling R27); only a control replay, a cell on its own run, "
            "takes the source's alpha"
        )
    budget = source.cfg.nuts if nuts is None else nuts
    if max_tree_depth is not None:
        budget = dataclasses.replace(budget, max_tree_depth=max_tree_depth)
    budget_text = _budget_text(budget)
    datasets = {t: dataset_at(source, t) for t in sorted(set(ts))}

    cfg = dataclasses.replace(source.cfg, method=method, out_dir=str(out))
    replay_dir = run_dir_for(out, cfg)
    done = _done(replay_dir, cfg, source_method, budget_text)

    if dry_run:
        print(f"source {source.dir} (method {source_method}, t_done "
              f"{int(source.checkpoint['t_done'])})")
        print(f"replay {replay_dir} (method {method}, budget {budget_text} = "
              "warmup/samples/thinning/max_tree_depth)")
        for t, (X, _, y_mean, y_std) in datasets.items():
            state = "done, skipped" if t in done else "to fit"
            print(f"t={t} n={len(X)} nuts_seed={iteration_rngs(seed, t).nuts_seed} "
                  f"y_mean={y_mean!r} y_std={y_std!r}: {state}")
        print("dry run: nothing written")
        return replay_dir

    todo = [t for t in datasets if t not in done]
    if not todo:
        return replay_dir
    # A process's one-time start-up, done before its first timed fit so that no `fit_wall_s`
    # carries it (ruling R26): the R2-D2 map's table at R2D2_K, built on first use and cached per
    # shape. Every cell's replay builds it, so every cell's first fit starts from the same state.
    start = time.perf_counter()
    jax.block_until_ready(log_gamma_icdf(np.zeros(1), R2D2_K))
    startup_s = time.perf_counter() - start
    logger = RunLogger(replay_dir, cfg)
    if logger.manifest.exists():
        logger.note_resume()
    else:
        _write_manifest(logger, source)
    logger.log(
        f"replay method={method} on {source.dir} t={todo} budget={budget_text}; "
        f"R2-D2 table at k={R2D2_K!r} ready in {startup_s:.3f} s before the first fit"
    )
    _drop_unfinished(logger, done)
    for t in todo:
        _replay_one(logger, source, cell, t, datasets[t], budget, budget_text)
    return replay_dir


def _cell(method: str) -> Cell:
    """The cell a method string names; a `ValueError` for the references and unknown names."""
    if method not in _CELL_METHODS:
        raise ValueError(f"{method!r} is not a cell: expected one of {list(_CELL_METHODS)}")
    return CELLS[tuple(method.split("/"))]


def _run_dir(root: str | os.PathLike[str], family: str, method: str, seed: int) -> Path:
    """`run_dir_for`'s directory for a run known by name: its other settings do not place it."""
    return run_dir_for(root, RunConfig(family=family, seed=seed, D=0, method=method, T=0, n_init=0))


def _budget_text(nuts: NUTSConfig) -> str:
    """A budget as `replay.csv` records it: warmup/samples/thinning/max_tree_depth."""
    return f"{nuts.num_warmup}/{nuts.num_samples}/{nuts.thinning}/{nuts.max_tree_depth}"


def _done(replay_dir: Path, cfg: RunConfig, source_method: str, budget_text: str) -> set[int]:
    """The t this replay directory has finished, after refusing one written under other terms.

    The manifest's `config_hash` is the replay's configuration, as a run's is on resume, and every
    complete row must be of this call's source and budget: the directory's path names neither, so
    without these checks a call under another would skip the t it finds done and mix the two.
    """
    manifest = replay_dir / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text()).get("config_hash")
        if recorded != config_hash(cfg):
            raise ValueError(
                f"{replay_dir} was written under a different configuration (manifest "
                f"config_hash {recorded}, this replay's {config_hash(cfg)})"
            )
    _, rows = _read_csv(replay_dir / "replay.csv")
    for row in rows:
        if row["budget"] != budget_text:
            raise ValueError(
                f"{replay_dir}/replay.csv holds rows under budget {row['budget']}, not this "
                f"call's {budget_text}: a replay directory holds one budget, so replay another "
                "budget into another --out"
            )
        if row["source_method"] != source_method:
            raise ValueError(
                f"{replay_dir}/replay.csv holds rows replayed on {row['source_method']}'s run, "
                f"not {source_method}'s"
            )
    return {int(row["t"]) for row in rows}


def _write_manifest(logger: RunLogger, source: Source) -> None:
    """The replay's manifest: `RunLogger.write_manifest`'s own, plus the run it was replayed on.

    `write_manifest` reads the objective's labels and nothing else of it, and the source's
    manifest records exactly those, so they are handed over from there rather than by rebuilding
    the objective. `devices` lists the device of each of the source's sessions, creation first.
    """
    labels = source.manifest["objective"]
    logger.write_manifest(SimpleNamespace(
        D=labels["D"],
        f_star=labels["f_star"],
        labels=SimpleNamespace(**{key: labels[key] for key in (
            "family", "seed", "S", "gamma", "noise_sd"
        )}),
    ))
    manifest = json.loads(logger.manifest.read_text())
    manifest["source"] = {
        "path": str(source.dir),
        "config_hash": source.manifest["config_hash"],
        "commit": source.manifest["git"]["commit"],
        "devices": _session_devices(source.manifest),
    }
    tmp = logger.manifest.with_name(logger.manifest.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, logger.manifest)


def _session_devices(manifest: dict[str, object]) -> list[str]:
    """The device of each of a run's sessions: its creation's, then each resume's."""
    return [manifest["env"]["jax_device"]] + [
        resume["env"]["jax_device"] for resume in manifest.get("resumed", [])
    ]


def _drop_unfinished(logger: RunLogger, done: set[int]) -> None:
    """Remove what a killed call wrote for a t it did not finish: the loop's `truncate_to`.

    The row is written last, so a t without a complete row may still have left coordinates, an
    npz or a torn row; each goes, so that redoing the t neither duplicates nor tears anything.
    """
    for path in (logger.dir / "replay.csv", logger.coords):
        _keep_rows(path, lambda row: int(row["t"]) in done)
    for path in logger.samples_dir.glob("t*.npz"):
        if int(path.stem[1:]) not in done:
            path.unlink()


def _replay_one(
    logger: RunLogger,
    source: Source,
    cell: Cell,
    t: int,
    dataset: tuple[np.ndarray, np.ndarray, float, float],
    budget: NUTSConfig,
    budget_text: str,
) -> None:
    """Fit, read out and record one t: the coordinates, the draws, then the row, as the loop does.

    `fit_wall_s` is taken around `fit` as the loop takes it (`sagp.bo`), and so includes the
    compilation every new n costs, but not the R2-D2 table `run_replay` built before the first
    fit; the draws are on the host before `fit` returns, so the clock does not stop before the
    chain has run. The two R2 medians are an amplitude cell's
    (`r2d2_r2`, and `first_order_r2` on the noise prediction uses); NaN for a lengthscale cell.
    """
    X, z, y_mean, y_std = dataset
    cfg = source.cfg
    nuts_seed = iteration_rngs(cfg.seed, t).nuts_seed
    start = time.perf_counter()
    fitted = fit(
        X, z, nuts_seed, cell=cell, alpha=cfg.alpha, fixed_noise=cfg.fixed_noise, nuts=budget,
        thresholds=cfg.thresholds, ell_prior=cfg.ell_prior,
    )
    fit_wall_s = time.perf_counter() - start

    start = time.perf_counter()
    readout = readouts(
        fitted, compute_sobol=t % cfg.sobol_every == 0 or t == cfg.T - 1, sobol_n=cfg.sobol_n
    )
    readout_wall_s = time.perf_counter() - start

    r2d2, first_order = float("nan"), float("nan")
    if cell.native_site == "a_sq":
        a_sq = fitted.samples["a_sq"]
        r2d2 = float(np.median(np.asarray(r2d2_r2(a_sq))))
        first_order = float(np.median(np.asarray(first_order_r2(a_sq, fitted.noises()))))

    diagnostics = fitted.attempts[-1]
    row = {
        "t": t,
        "n": len(X),
        "method": logger.cfg.method,
        "source_method": cfg.method,
        "family": cfg.family,
        "seed": cfg.seed,
        "nuts_seed": nuts_seed,
        "budget": budget_text,
        "status": fitted.status,
        "reason": fitted.status_reason,
        **{field: getattr(diagnostics, field) for field in _DIAG_FIELDS},
        "fit_wall_s": fit_wall_s,
        "readout_wall_s": readout_wall_s,
        "y_mean": y_mean,
        "y_std": y_std,
        "r2d2_r2": r2d2,
        "first_order_r2": first_order,
        "device": jax.devices()[0].device_kind,
    }
    logger.append_coords(t, readout)
    logger.save_samples(t, fitted)
    _append_row(logger.dir / "replay.csv", row)
    logger.log(
        f"t={t} n={row['n']} status={row['status']} fit_wall_s={fit_wall_s:.3f} "
        f"readout_wall_s={readout_wall_s:.3f} device={row['device']}"
        + (f" reason={row['reason']!r}" if row["reason"] else "")
    )


# --- CSV files ---


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """`path`'s header and its complete rows; a missing file, or a torn header, has neither.

    A complete row is a line the writer finished -- a kill can leave the last one without its
    terminator, and cut at a field boundary it would still parse -- at the header's width.
    """
    if not path.exists():
        return [], []
    with path.open(newline="") as handle:  # newline="" so the terminators survive the read
        lines = handle.read().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    if not lines:
        return [], []
    header, *rows = csv.reader(lines)
    return header, [dict(zip(header, row)) for row in rows if len(row) == len(header)]


def _keep_rows(path: Path, keep: Callable[[dict[str, str]], bool]) -> None:
    """Rewrite `path` with its header and the complete rows `keep` accepts; remove it if headless.

    A file with no complete header is removed rather than kept, because an append writes a header
    only into a file that is missing or empty.
    """
    header, rows = _read_csv(path)
    if not header:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(row for row in rows if keep(row))
    os.replace(tmp, path)


def _append_row(path: Path, row: dict[str, object]) -> None:
    """Append one row to `replay.csv`, writing the header if the file is new."""
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_REPLAY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# --- compare ---

# Gate G2 (plan, Task 11) in numbers. FAIL: an R2-D2 cell's exclusion rate at or above 0.90, or its
# median n_eff_min at or below 10. PASS: every R2-D2 cell at or below 0.60 and at or above 16.
# Harness: each control cell's exclusion rate within 0.15 of its stored fits', and its median
# |delta r_hat_max| and |delta AP_native| against them at most 0.05. Cost: an R2-D2 cell's median
# fit_wall_s at most 1.5 times its twin's control refits' on the same device.
_G2_FAIL_EXCLUSION, _G2_FAIL_N_EFF = 0.90, 10.0
_G2_PASS_EXCLUSION, _G2_PASS_N_EFF = 0.60, 16.0
_G2_HARNESS_EXCLUSION, _G2_HARNESS_R_HAT, _G2_HARNESS_AP = 0.15, 0.05, 0.05
_G2_COST_RATIO = 1.5
# The full replay: 4 R2-D2 cells x 8 families x 10 seeds x 3 t, and 4 controls x 2 x 10 x 3.
_G2_R2D2_FITS, _G2_CONTROL_FITS = 960, 240
# The three groups the per-group diagnostics attribute a gate failure to.
_GROUPS: tuple[str, ...] = ("native", "ell", "global")


def compare(
    runs: str | os.PathLike[str], replay: str | os.PathLike[str], out: str | os.PathLike[str]
) -> Path:
    """Join every replayed fit to its stored fit and write G2's tables and report; returns `out`.

    One joined row per replayed (cell, family, seed, t): the replay's fit, the source's stored fit
    of the same (family, seed, t) (`stored_`), and for an R2-D2 cell its twin's control refit
    there where one exists (`control_`). `tables/`: `replay_vs_stored.csv` (the join),
    `gate_by_cell_t.csv` (per cell and t, and over all t: exclusion rate, the r_hat_max and
    n_eff_min medians, the per-group r_hat medians and the fraction of fits whose every sampling
    iteration reached the tree cap, for the replay and for its stored fits),
    `identification.csv` (AP of native_median and of sobol_hat, and precision, recall and F1 of
    p_active > 0.5, against the manifest's S, for each replayed and each stored fit) and
    `cost.csv` (fit_wall_s by cell and device); `REPORT.md` states the G2 verdict with its numbers.
    NumPy and csv only, so it runs in the study environment.
    """
    runs, replay, out = Path(runs), Path(replay), Path(out)
    replayed = _replayed_fits(replay)
    wanted: dict[tuple[str, str, int], set[int]] = {}  # (family, source, seed) -> its t
    for (_, family, seed, t), (row, _) in replayed.items():
        wanted.setdefault((family, row["source_method"], seed), set()).add(t)
    stored_runs = {
        (family, source_method, seed): _stored_run(_run_dir(runs, family, source_method, seed), ts)
        for (family, source_method, seed), ts in wanted.items()
    }

    joined = []
    for (method, family, seed, t), (row, coord_rows) in sorted(replayed.items()):
        source_method = row["source_method"]
        stored = stored_runs[(family, source_method, seed)]
        S = stored["S"]
        replay_fit = _fit_record(row, coord_rows, S, _tree_cap(row["budget"]), row["device"])
        stored_fit = _fit_record(
            stored["rows"][t], stored["coords"].get(t, []), S, stored["tree_cap"],
            stored["devices"].get(t, "unknown"),
        )
        # The twin's control: the source cell refitted on this same run, if it was.
        control = replayed.get((source_method, family, seed, t))
        has_control = control is not None and control[0]["source_method"] == source_method
        control_fit = (
            _fit_record(control[0], control[1], S, _tree_cap(control[0]["budget"]),
                        control[0]["device"])
            if has_control and method != source_method
            else _blank(replay_fit)
        )
        role = (
            "control" if method == source_method
            else "twin" if TWINS.get(method) == source_method
            else "other"
        )
        joined.append({
            "cell": method, "source_method": source_method, "role": role, "family": family,
            "seed": seed, "t": t, "n": int(row["n"]), "budget": row["budget"],
            "nuts_seed": row["nuts_seed"],
            **replay_fit,
            "readout_wall_s": float(row["readout_wall_s"]),
            "r2d2_r2": float(row["r2d2_r2"]),
            "first_order_r2": float(row["first_order_r2"]),
            **{f"stored_{name}": value for name, value in stored_fit.items()},
            **{f"control_{name}": value for name, value in control_fit.items()},
        })

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_table(tables / "replay_vs_stored.csv", joined)
    _write_table(tables / "gate_by_cell_t.csv", _gate_table(joined))
    _write_table(tables / "identification.csv", _identification_table(joined))
    _write_table(tables / "cost.csv", _cost_table(joined))
    (out / "REPORT.md").write_text(_report(joined, runs, replay))
    return out


def _replayed_fits(
    replay: Path,
) -> dict[tuple[str, str, int, int], tuple[dict[str, str], list[dict[str, str]]]]:
    """Every replayed fit under `replay`, by (method, family, seed, t): its row, its coordinates."""
    fits = {}
    for path in sorted(replay.glob("*/*/seed*/replay.csv")):
        _, rows = _read_csv(path)
        coords = _coords_by_t(path.parent / "coords.csv", {int(row["t"]) for row in rows})
        for row in rows:
            t = int(row["t"])
            fits[(row["method"], row["family"], int(row["seed"]), t)] = (row, coords.get(t, []))
    return fits


def _stored_run(run_dir: Path, wanted: set[int]) -> dict[str, object]:
    """What `compare` reads of a stored run: rows and coordinates at `wanted`, S, cap, devices.

    The device of an iteration is its session's: the log's "resume at" lines start each session
    after the first, whose device the manifest's `env` records, and each resume's its `resumed`
    entry (as the study's analysis reads it). A redone iteration's latest line is the one kept.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    _, rows = _read_csv(run_dir / "iterations.csv")
    missing = sorted(wanted - {int(row["t"]) for row in rows})
    if missing:
        raise ValueError(f"{run_dir}/iterations.csv has no complete row at t={missing}")
    sessions = _session_devices(manifest)
    session, devices = 0, {}
    log = run_dir / "log.txt"
    for line in log.read_text().splitlines() if log.exists() else []:
        if line.startswith("resume at"):
            session += 1
        elif match := re.match(r"t=(\d+) method=", line):
            devices[int(match.group(1))] = sessions[min(session, len(sessions) - 1)]
    return {
        "rows": {int(row["t"]): row for row in rows if int(row["t"]) in wanted},
        "coords": _coords_by_t(run_dir / "coords.csv", wanted),
        "S": [int(i) for i in manifest["objective"]["S"]],
        "tree_cap": 2 ** int(manifest["nuts"]["max_tree_depth"]) - 1,
        "devices": devices,
    }


def _coords_by_t(path: Path, wanted: set[int]) -> dict[int, list[dict[str, str]]]:
    """`coords.csv`'s complete rows at the t in `wanted`, grouped by t."""
    _, rows = _read_csv(path)
    by_t: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        t = int(row["t"])
        if t in wanted:
            by_t.setdefault(t, []).append(row)
    return by_t


def _tree_cap(budget_text: str) -> int:
    """The leapfrog steps of a full tree at the budget's depth: 2^depth - 1."""
    return 2 ** int(budget_text.split("/")[3]) - 1


def _fit_record(
    row: dict[str, str], coord_rows: list[dict[str, str]], S: list[int], tree_cap: int,
    device: str,
) -> dict[str, object]:
    """One fit as `compare` reads it -- verdict, diagnostics, cost, identification -- replayed or
    stored alike, since both rows carry the loop's names."""
    return {
        "status": row["status"],
        "reason": row["reason"],
        **{field: float(row[field]) for field in _DIAG_FIELDS},
        "at_tree_cap": float(float(row["num_steps_mean"]) >= tree_cap),
        "fit_wall_s": float(row["fit_wall_s"]),
        "device": device,
        **_identification(coord_rows, S),
    }


def _blank(record: dict[str, object]) -> dict[str, object]:
    """`record`'s columns with nothing in them: "" for a text column, NaN for a number."""
    return {name: "" if isinstance(value, str) else float("nan") for name, value in record.items()}


def _identification(coord_rows: list[dict[str, str]], S: list[int]) -> dict[str, object]:
    """AP of native_median and of sobol_hat, and precision, recall and F1 of p_active > 0.5."""
    rows = sorted(coord_rows, key=lambda row: int(row["i"]))
    truth = np.zeros(len(rows), dtype=bool)
    truth[[i for i in S if i < len(rows)]] = True
    native = np.array([float(row["native_median"]) for row in rows])
    sobol = np.array([float(row["sobol_hat"]) for row in rows])
    predicted = np.array([float(row["p_active"]) > 0.5 for row in rows], dtype=bool)
    precision, recall, f1 = _precision_recall_f1(predicted, truth)
    return {
        "ap_native": _average_precision(native, truth),
        "ap_sobol": _average_precision(sobol, truth),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred_active": int(predicted.sum()),
    }


def _average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    """AP = sum_n (R_n - R_(n-1)) P_n down the ranking by score, as the study's analysis takes it.

    `sagp_analysis/latest/analyze.py`'s `average_precision`, so a replayed AP and a stored one
    are one quantity: ties keep coordinate order, NaN scores rank last. NaN when there is nothing
    to rank -- no positives, or no score at all (a Sobol readout not computed at this t).
    """
    if not truth.any() or np.isnan(scores).all():
        return float("nan")
    order = np.argsort(-np.where(np.isnan(scores), -np.inf, scores), kind="stable")
    hits = truth[order].astype(float)
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / hits.sum())


def _precision_recall_f1(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 of a predicted active set, NaN where undefined (as the analysis)."""
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if tp + fp and tp + fn and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if tp + fp else float("nan")
    return precision, recall, f1


def _numbers(values: Iterable[float]) -> np.ndarray:
    """The values that are numbers: NaN marks a missing one (a group the cell has no site in)."""
    array = np.asarray(list(values), dtype=float)
    return array[~np.isnan(array)]


def _median(values: Iterable[float]) -> float:
    """The median of the values that are numbers, or NaN when none is."""
    array = _numbers(values)
    return float(np.median(array)) if array.size else float("nan")


def _max(values: Iterable[float]) -> float:
    array = _numbers(values)
    return float(array.max()) if array.size else float("nan")


def _mean(values: Iterable[float]) -> float:
    """The mean of the values that are numbers (a share, for booleans), or NaN when none is."""
    array = _numbers(values)
    return float(array.mean()) if array.size else float("nan")


def _share(values: Iterable[float], fails: Callable[[np.ndarray], np.ndarray]) -> float:
    """The share of the values that are numbers for which `fails` holds; NaN when none is."""
    array = _numbers(values)
    return float(fails(array).mean()) if array.size else float("nan")


def _gate_stats(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    """One group of fits against the gate: `prefix` "" reads the replayed fits, "stored_" theirs."""
    return {
        f"{prefix}exclusion_rate": _mean(row[f"{prefix}status"] == "excluded" for row in rows),
        f"{prefix}r_hat_max_median": _median(row[f"{prefix}r_hat_max"] for row in rows),
        f"{prefix}n_eff_min_median": _median(row[f"{prefix}n_eff_min"] for row in rows),
        **{
            f"{prefix}r_hat_max_{group}_median": _median(
                row[f"{prefix}r_hat_max_{group}"] for row in rows
            )
            for group in _GROUPS
        },
        f"{prefix}frac_at_tree_cap": _mean(row[f"{prefix}at_tree_cap"] for row in rows),
        f"{prefix}num_steps_mean_median": _median(row[f"{prefix}num_steps_mean"] for row in rows),
    }


def _groups(joined: list[dict[str, object]]) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    """The joined rows by (cell, source_method, role), cells in `CELLS` order."""
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    order = {method: i for i, method in enumerate(_CELL_METHODS)}
    for row in sorted(joined, key=lambda row: (order[row["cell"]], row["source_method"])):
        groups.setdefault((row["cell"], row["source_method"], row["role"]), []).append(row)
    return groups


def _gate_table(joined: list[dict[str, object]]) -> list[dict[str, object]]:
    """Per (cell, source, role) and t, then over all t: the replayed fits and their stored ones."""
    table = []
    for (cell, source_method, role), rows in _groups(joined).items():
        ts = sorted({row["t"] for row in rows})
        for t in [*ts, "all"]:
            at_t = rows if t == "all" else [row for row in rows if row["t"] == t]
            table.append({
                "cell": cell, "source_method": source_method, "role": role, "t": t,
                "fits": len(at_t), **_gate_stats(at_t, ""), **_gate_stats(at_t, "stored_"),
            })
    return table


_IDENTIFICATION_COLUMNS: tuple[str, ...] = (
    "status", "ap_native", "ap_sobol", "precision", "recall", "f1", "n_pred_active",
)


def _identification_table(joined: list[dict[str, object]]) -> list[dict[str, object]]:
    """One row per replayed fit and per stored fit (each stored fit once, however often joined)."""
    replayed, stored = [], {}
    for row in joined:
        keys = {"family": row["family"], "seed": row["seed"], "t": row["t"]}
        replayed.append({"kind": "replay", "method": row["cell"], **keys,
                         **{name: row[name] for name in _IDENTIFICATION_COLUMNS}})
        stored[(row["source_method"], row["family"], row["seed"], row["t"])] = {
            "kind": "stored", "method": row["source_method"], **keys,
            **{name: row[f"stored_{name}"] for name in _IDENTIFICATION_COLUMNS},
        }
    return replayed + [stored[key] for key in sorted(stored)]


def _cost_table(joined: list[dict[str, object]]) -> list[dict[str, object]]:
    """fit_wall_s by (kind, method, device) per t and over all t, each stored fit counted once.

    A replay's `readout_wall_s` rides along; the stored rows have none (the loop does not time its
    readouts apart from the rest of the iteration).
    """
    fits: dict[tuple[str, str, str, object], list[tuple[float, float]]] = {}
    seen = set()
    for row in joined:
        stored_key = (row["source_method"], row["family"], row["seed"], row["t"])
        entries = [(("replay", row["cell"], row["device"]),
                    (row["fit_wall_s"], row["readout_wall_s"]))]
        if stored_key not in seen:
            seen.add(stored_key)
            entries.append((("stored", row["source_method"], row["stored_device"]),
                            (row["stored_fit_wall_s"], float("nan"))))
        for group, value in entries:
            for t in (row["t"], "all"):
                fits.setdefault((*group, t), []).append(value)

    def order(key: tuple[str, str, str, object]) -> tuple[object, ...]:
        kind, method, device, t = key
        return kind, method, device, float("inf") if t == "all" else t

    return [
        {
            "kind": kind, "method": method, "device": device, "t": t, "fits": len(fits[key]),
            "fit_wall_s_median": _median(fit_s for fit_s, _ in fits[key]),
            "fit_wall_s_max": _max(fit_s for fit_s, _ in fits[key]),
            "readout_wall_s_median": _median(readout_s for _, readout_s in fits[key]),
        }
        for key in sorted(fits, key=order)
        for kind, method, device, t in [key]
    ]


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    """`rows` as a CSV whose columns are the first row's keys (every row has the same keys)."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    """A number for the report, to three significant figures; "n/a" for NaN."""
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.3g}"
    return str(value)


def _md_table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _report(joined: list[dict[str, object]], runs: Path, replay: Path) -> str:
    """REPORT.md: G2's verdict first, then each of its criteria with the numbers it rests on."""
    groups = _groups(joined)
    r2d2 = {
        cell: rows for (cell, _, role), rows in groups.items() if role == "twin" and cell in TWINS
    }
    controls = {cell: rows for (cell, _, role), rows in groups.items() if role == "control"}
    n_r2d2 = sum(len(rows) for rows in r2d2.values())
    n_control = sum(len(rows) for rows in controls.values())

    cells = []  # [cell, twin, fits, exclusion, median n_eff_min, stored exclusion, median, reading]
    for cell, rows in r2d2.items():
        stats = _gate_stats(rows, "") | _gate_stats(rows, "stored_")
        exclusion, n_eff = stats["exclusion_rate"], stats["n_eff_min_median"]
        fails = exclusion >= _G2_FAIL_EXCLUSION or n_eff <= _G2_FAIL_N_EFF
        passes = exclusion <= _G2_PASS_EXCLUSION and n_eff >= _G2_PASS_N_EFF
        reading = "FAIL" if fails else "PASS" if passes else "neither"
        cells.append([cell, TWINS[cell], len(rows), exclusion, n_eff,
                      stats["stored_exclusion_rate"], stats["stored_n_eff_min_median"], reading])

    harness = []  # [cell, fits, exclusion, stored exclusion, |d|, med |d r_hat|, med |d AP|, holds]
    for cell, rows in controls.items():
        exclusion = _gate_stats(rows, "")["exclusion_rate"]
        stored = _gate_stats(rows, "stored_")["stored_exclusion_rate"]
        d_r_hat = _median(abs(row["r_hat_max"] - row["stored_r_hat_max"]) for row in rows)
        d_ap = _median(abs(row["ap_native"] - row["stored_ap_native"]) for row in rows)
        holds = (
            abs(exclusion - stored) <= _G2_HARNESS_EXCLUSION
            and d_r_hat <= _G2_HARNESS_R_HAT
            and d_ap <= _G2_HARNESS_AP
        )
        harness.append([cell, len(rows), exclusion, stored, abs(exclusion - stored), d_r_hat, d_ap,
                        "yes" if holds else "no"])

    cost = []  # [cell, device, pairs, median fit_wall_s, control's, ratio, holds]
    for cell, rows in r2d2.items():
        paired = [row for row in rows if row["control_device"] == row["device"]]
        for device in sorted({row["device"] for row in paired}):
            on_device = [row for row in paired if row["device"] == device]
            mine = _median(row["fit_wall_s"] for row in on_device)
            theirs = _median(row["control_fit_wall_s"] for row in on_device)
            ratio = mine / theirs
            cost.append([cell, device, len(on_device), mine, theirs, ratio,
                         "yes" if ratio <= _G2_COST_RATIO else "no"])

    # The two stops come first: without a harness that reproduces the stored fits no reading of
    # the R2-D2 cells can be trusted, and a cost stop is asked about before anything else runs.
    stops = []
    if any(row[-1] == "no" for row in harness):
        stops.append("harness -- the control refits do not reproduce their stored fits: stop, "
                     "fix, rerun")
    if any(row[-1] == "no" for row in cost):
        stops.append(f"cost -- an R2-D2 cell costs more than {_G2_COST_RATIO} times its control "
                     "on the same device: ask before the bulk array")
    if not cells:
        verdict = "NO R2-D2 FITS: nothing to decide"
    elif stops:
        verdict = "STOP (" + "; ".join(stops) + ")"
    elif not harness or not cost:
        verdict = ("INCONCLUSIVE: no control refit " + ("at all" if not harness else "shares a "
                   "device with an R2-D2 fit") + ", so the harness or the cost is unchecked -- "
                   "stop and ask")
    elif any(row[-1] == "FAIL" for row in cells):
        verdict = ("FAIL: the half-Cauchy amplitude signature -- nothing in-loop is launched; "
                   "Quan decides the budget for all eight cells (D10)")
    elif len(cells) == len(TWINS) and all(row[-1] == "PASS" for row in cells):
        verdict = "PASS: proceed to stage 3"
    else:
        verdict = "INCONCLUSIVE: stop and ask"
    complete = n_r2d2 == _G2_R2D2_FITS and n_control == _G2_CONTROL_FITS

    # Per group, the share of fits whose statistic fails the gate, over the fits that have one.
    thresholds = DiagThresholds()
    attribution = [
        [method, kind, len(rows)]
        + [_share((row[f"{prefix}r_hat_max_{group}"] for row in rows),
                  lambda r_hat: r_hat > thresholds.r_hat_max) for group in _GROUPS]
        + [_share((row[f"{prefix}n_eff_min_{group}"] for row in rows),
                  lambda n_eff: n_eff < thresholds.n_eff_min) for group in _GROUPS]
        for cell, rows in r2d2.items()
        for prefix, kind, method in (("", "replay", cell), ("stored_", "stored", TWINS[cell]))
    ]

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return "".join([
        "# R2-D2 replay: gate G2\n\n",
        f"Stored runs `{runs}`, replays `{replay}`; written {stamp} by "
        "`python -m experiments.replay compare`.\n\n",
        "## Verdict\n\n",
        f"**G2: {verdict}.**\n\n",
        f"Fits: {n_r2d2} of {_G2_R2D2_FITS} R2-D2 fits and {n_control} of {_G2_CONTROL_FITS} "
        "control refits"
        + ("." if complete else " -- incomplete, so this verdict is provisional.")
        + "\n\n",
        "## R2-D2 cells\n\n",
        f"FAIL at an exclusion rate of {_G2_FAIL_EXCLUSION} or more or a median n_eff_min of "
        f"{_G2_FAIL_N_EFF:g} or less; PASS at {_G2_PASS_EXCLUSION} or less and {_G2_PASS_N_EFF:g} "
        "or more (every R2-D2 cell). The twin's columns are its stored fits of the same "
        "(family, seed, t).\n\n",
        _md_table(["cell", "twin", "fits", "exclusion rate", "median n_eff_min",
                   "twin's exclusion rate", "twin's median n_eff_min", "reading"], cells),
        "\n## Harness: control refits against their stored fits\n\n",
        f"Holds at |delta exclusion rate| <= {_G2_HARNESS_EXCLUSION}, median |delta r_hat_max| "
        f"<= {_G2_HARNESS_R_HAT} and median |delta AP_native| <= {_G2_HARNESS_AP}, over the "
        "stored fits of the same (family, seed, t).\n\n",
        _md_table(["cell", "fits", "exclusion rate", "stored exclusion rate", "|delta|",
                   "median |delta r_hat_max|", "median |delta AP_native|", "holds"], harness),
        "\n## Cost: R2-D2 against its twin's control refits on the same device\n\n",
        f"Medians over the (family, seed, t) where both were fitted on one device; holds at a "
        f"ratio of {_G2_COST_RATIO} or less.\n\n",
        _md_table(["cell", "device", "pairs", "median fit_wall_s", "control's median",
                   "ratio", "holds"], cost),
        "\n## Per-group attribution\n\n",
        f"The share of fits whose group's r_hat_max exceeds {thresholds.r_hat_max} or whose "
        f"n_eff_min is below {thresholds.n_eff_min:g}; n/a where the cell has no site in the "
        "group. "
        "If the ell group fails under both prior families, the prior cannot be the lever.\n\n",
        _md_table(["method", "kind", "fits", "r_hat native", "r_hat ell", "r_hat global",
                   "n_eff native", "n_eff ell", "n_eff global"], attribution),
        "\n## Tables\n\n",
        "- `tables/replay_vs_stored.csv`: one row per replayed (cell, family, seed, t) -- the "
        "replayed fit, its source's stored fit (`stored_`), and for an R2-D2 cell its twin's "
        "control refit where one exists (`control_`).\n",
        "- `tables/gate_by_cell_t.csv`: the gate statistics per cell and t, and over all t "
        "(`at_tree_cap`: every sampling iteration at 2^depth - 1 leapfrog steps).\n",
        "- `tables/identification.csv`: AP of native_median and of sobol_hat, and precision, "
        "recall and F1 of p_active > 0.5, against S.\n",
        "- `tables/cost.csv`: fit_wall_s by cell, device and t.\n",
    ])


# --- the command line ---


def _ints(text: str) -> tuple[int, ...]:
    """`--t 50,100,199` -> (50, 100, 199)."""
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints; got {text!r}") from None


def _nuts_arg(text: str) -> NUTSConfig:
    """`--nuts W,S,K` -> `NUTSConfig(W, S, K)`: warm-up, samples, thinning (depth: its own flag)."""
    values = _ints(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"expected 'warmup,samples,thinning'; got {text!r}")
    try:
        return NUTSConfig(*values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.replay",
        description="Refit a cell on a stored run's iteration-t data (fit), or join the refits "
        "to the stored fits and write gate G2's tables and report (compare).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit_command = commands.add_parser(
        "fit",
        help="replay one (family, seed, cell) at each t",
        description="Refit --cell on <runs>/<family>/<source>/seed##'s data at each t, into "
        "<out>/<family>/<cell with '/' as '-'>/seed##/, skipping the t already done.",
    )
    fit_command.add_argument("--runs", required=True, type=Path,
                             help="root of the stored runs (only read)")
    fit_command.add_argument("--out", required=True, type=Path,
                             help="root of the replay directories")
    fit_command.add_argument("--family", required=True, help="objective family, e.g. aligned10")
    fit_command.add_argument("--seed", required=True, type=int, help="the stored run's seed")
    fit_command.add_argument("--cell", required=True, choices=_CELL_METHODS,
                             help="the cell to refit")
    fit_command.add_argument(
        "--source", default=None, choices=_CELL_METHODS,
        help="the stored run's method (default: an R2-D2 cell's twin, or the cell itself)",
    )
    fit_command.add_argument(
        "--t", type=_ints, default=REPLAY_T, metavar="T,...",
        help="iterations to replay (default 50,100,199)",
    )
    fit_command.add_argument(
        "--nuts", type=_nuts_arg, default=None, metavar="W,S,K",
        help="NUTS warm-up, samples and thinning (default: the source run's)",
    )
    fit_command.add_argument(
        "--max-tree-depth", type=int, default=None,
        help="NUTS tree depth (default: the source run's)",
    )
    fit_command.add_argument(
        "--dry-run", action="store_true",
        help="check the source and print each t's n and seed and the budget; write nothing",
    )

    compare_command = commands.add_parser(
        "compare",
        help="join every replayed fit to its stored fit; write the tables and REPORT.md",
    )
    compare_command.add_argument("--runs", required=True, type=Path,
                                 help="root of the stored runs")
    compare_command.add_argument("--replay", required=True, type=Path,
                                 help="root of the replay directories")
    compare_command.add_argument(
        "--out", required=True, type=Path,
        help="the report directory, e.g. sagp_analysis/r2d2-replay/<YYYY-MM-DD-HHMM>/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (or `sys.argv`) and run `fit` or `compare`; 0 on success, 2 on a bad argument.

    A refusal -- a source that cannot reproduce its data, a replay directory under another budget
    -- is not an argument error and propagates, so a cluster log carries its traceback.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)

    if args.command == "compare":
        print(compare(args.runs, args.replay, args.out))
        return 0
    replay_dir = run_replay(
        args.runs, args.out, family=args.family, seed=args.seed, method=args.cell,
        source_method=args.source, ts=args.t, nuts=args.nuts,
        max_tree_depth=args.max_tree_depth, dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(replay_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
