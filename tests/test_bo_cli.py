"""Tests for `python -m sagp.bo`: the study's entry point, exercised the way SLURM will run it.

Three of these go through `subprocess` rather than through `main(argv)`, because what they check
is exactly what an in-process call cannot: that `python -m sagp.bo` resolves to a module with a
`__main__` guard, that its exit code reaches the shell (a SLURM array's only signal that a task
failed), and that a real 15-iteration run leaves the seven artifacts of the plan's run directory
on disk. They are the expensive ones -- each pays JAX's import -- so everything that does not
need a process (the argument errors, `--objective-dir`'s stem) calls `main` directly instead.

`--dry-run` gets the closest reading: it is the flag a person uses to check a command line before
committing 16 hours of cluster time to it, so the test parses its JSON rather than grepping it,
and asserts the output directory does not exist afterwards -- "wrote nothing" being the whole
promise, and one an assertion about stdout alone would not catch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sagp.bo import main
from synthobj.families import make_family

_RUN = [
    "--family", "aligned3", "--D", "5", "--seed", "0", "--cell", "dsp_map", "--n-init", "5",
    # The study's acquisition budget is 5000 candidates x 5 restarts; this is the loop tests' own
    # reduced one, so that a real run fits in the suite's per-test time budget.
    "--num-init-candidates", "256", "--num-restarts-ei", "1", "--sobol-every", "5",
]


def _run(repo_root: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sagp.bo", *argv],
        cwd=repo_root, capture_output=True, text=True, timeout=240,
    )


def test_dry_run_prints_the_resolved_config_and_writes_nothing(tmp_path, repo_root):
    out = tmp_path / "runs"
    done = _run(repo_root, *_RUN, "--T", "8", "--out", str(out), "--dry-run")
    assert done.returncode == 0, done.stderr

    cfg, end = json.JSONDecoder().raw_decode(done.stdout)
    assert cfg["method"] == "dsp_map"
    fields = (cfg["family"], cfg["seed"], cfg["D"], cfg["T"], cfg["n_init"])
    assert fields == ("aligned3", 0, 5, 8, 5)
    assert cfg["out_dir"] == str(out)
    # The objective is resolved too, so a dry run also answers "which problem is this, and what is
    # the f_star my regret column will be taken against".
    tail = done.stdout[end:]
    assert "family=aligned3" in tail and "D=5" in tail and "f_star=" in tail
    assert "S=[0, 2, 3]" in tail
    assert not out.exists()


def test_an_unknown_cell_exits_2(tmp_path, repo_root):
    done = _run(repo_root, *_RUN, "--T", "8", "--out", str(tmp_path / "runs"), "--cell", "nope")
    assert done.returncode == 2
    assert "invalid choice" in done.stderr
    assert not (tmp_path / "runs").exists()


def test_a_run_writes_every_artifact_of_the_run_directory(tmp_path, repo_root):
    out = tmp_path / "runs"
    start = time.perf_counter()
    done = _run(repo_root, *_RUN, "--T", "15", "--out", str(out))
    elapsed = time.perf_counter() - start
    assert done.returncode == 0, done.stderr
    # A 15-iteration MAP run is seconds of work, and the bound is here so that a change which
    # makes it minutes is caught. It is a wall clock, though, so it also fails when the machine is
    # busy with something else entirely -- another test job, a pilot run -- which is not this
    # test's finding. The artifacts below are checked either way; only the timing is skipped.
    if os.getloadavg()[0] < os.cpu_count():
        assert elapsed < 60.0

    run_dir = out / "aligned3" / "dsp_map" / "seed00"
    assert done.stdout.strip().endswith(str(run_dir))
    for name in (
        "manifest.json", "environment.lock.txt", "iterations.csv", "coords.csv",
        "checkpoint.npz", "log.txt",
    ):
        assert (run_dir / name).stat().st_size > 0
    assert sorted(p.name for p in (run_dir / "samples").glob("t*.npz")) == [
        f"t{t:03d}.npz" for t in range(5, 15)
    ]


# --- what does not need a process ---


def test_the_other_argument_errors_also_exit_2(tmp_path):
    base = ["--family", "aligned3", "--seed", "0", "--cell", "sobol"]
    out = ["--out", str(tmp_path / "runs")]
    assert main([*base, *out, "--nuts", "8,16"]) == 2
    assert main([*base, *out, "--acq", "pi"]) == 2
    assert main(base) == 2  # no --out
    # Which families exist is `synthobj.families`' business, so this one cannot be a `choices=`
    # rejection and is checked after parsing instead -- but it is still a bad argument.
    assert main(["--family", "nope", *base[2:], *out]) == 2
    assert not (tmp_path / "runs").exists()


def test_no_resume_starts_the_run_over_and_the_default_continues_it(tmp_path):
    # The SLURM array re-submits the same tasks and must *continue* them; `--no-resume` is the
    # opt-out, so a flag wired the wrong way round would silently discard finished work on every
    # re-submission. `sobol` because this is about the run directory, not about a surrogate.
    argv = [
        "--family", "aligned3", "--D", "5", "--seed", "0", "--cell", "sobol",
        "--T", "8", "--n-init", "5", "--out", str(tmp_path / "runs"),
    ]
    assert main(argv) == 0
    log = tmp_path / "runs" / "aligned3" / "sobol" / "seed00" / "log.txt"
    assert log.read_text().count("start method=") == 1

    assert main(argv) == 0
    assert "resume at t=8" in log.read_text()
    assert log.read_text().count("start method=") == 1

    assert main([*argv, "--no-resume"]) == 0
    assert log.read_text().count("start method=") == 1  # a new log.txt, not an appended one
    assert "resume at" not in log.read_text()


def test_objective_dir_loads_the_study_grids_own_stem(tmp_path, capsys):
    # The literal layout `python -m synthobj.generate` writes, spelled out here rather than built
    # with the same f-string the CLI uses: a stem that drifted would find nothing at all.
    stem = tmp_path / "grid" / "aligned3" / "seed00_D5"
    stem.parent.mkdir(parents=True)
    stored = make_family("aligned3", 0, D=5)
    stored.save(stem)

    assert main([
        "--family", "aligned3", "--D", "5", "--seed", "0", "--cell", "sobol",
        "--out", str(tmp_path / "runs"), "--objective-dir", str(tmp_path / "grid"), "--dry-run",
    ]) == 0
    assert f"f_star={stored.f_star!r}" in capsys.readouterr().out

    # A stem that is not there is an argument error, not a traceback out of a cluster job.
    assert main([
        "--family", "aligned3", "--D", "99", "--seed", "0", "--cell", "sobol",
        "--out", str(tmp_path / "runs"), "--objective-dir", str(tmp_path / "grid"), "--dry-run",
    ]) == 2
