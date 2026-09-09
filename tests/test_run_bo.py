"""Tests for `experiments.run_bo`: the contracts of one run directory, end to end.

The loop's contracts are behavioural: two runs at the same seed are the same run bit for bit, a
resumed run equals the uninterrupted one bit for bit (including after a truncated write and after
a torn one), a changed configuration refuses to resume, and the row schema, the regret and the
Sobol schedule are what the plan says. Then the contracts a study depends on but a normal run
never exercises: an exception out of the fit costs a retry and then a seeded random query rather
than the run, and the two references that are not cells -- the Sobol search and the oracle -- run
end to end and leave the artifacts they should (the Sobol search leaves no `coords.csv` at all,
which is legal for a method that fits nothing). Every test runs a real objective through a real
fit -- `dsp_map`, whose MAP fit is milliseconds, with one NUTS-based run for the sampler's own
determinism.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

import sagp.gp
from experiments import run_bo as run_bo_module
from experiments.run_bo import run_bo
from sagp.bo import initial_design, iteration_rngs
from sagp.diagnostics import DiagThresholds, Diagnostics
from sagp.gp import NUTSConfig
from synthobj.families import make_family


# The acquisition optimizer's cost is 5 restarts x 100 L-BFGS-B evaluations per iteration, each a
# separate jitted call; one restart off 256 candidates keeps the loop tests inside the per-test
# time budget. It is still the same optimizer on the same seeded candidate set.
_LOOP_KW = dict(num_init_candidates=256, num_restarts_ei=1, sobol_every=5)
# T = 10 rather than the study's 200: every iteration triggers a fresh JAX compilation, because
# the training set it fits grows by a point, and five of them is what fits the suite's per-test
# time budget. Reproducibility is a property of every iteration, not of the last one, and five
# still spans a Sobol-schedule iteration (t = 5), the final one (t = T - 1) and three without.
_T = 10
_N_INIT = 5


def _objective():
    """The fixed D = 5 problem every loop test runs: three active coordinates, exact `f_star`."""
    return make_family("aligned3", 0, D=5)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- the loop ---


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory) -> Path:
    """One uninterrupted `dsp_map` run: the trajectory determinism and resume are compared to."""
    return run_bo(
        _objective(),
        "dsp_map",
        seed=1,
        T=_T,
        n_init=_N_INIT,
        out_dir=tmp_path_factory.mktemp("reference"),
        **_LOOP_KW,
    )


def _checkpoint(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "checkpoint.npz") as data:
        return {name: data[name] for name in data.files}


def test_two_runs_at_the_same_seed_are_bit_identical(reference_run, tmp_path):
    again = run_bo(
        _objective(), "dsp_map", seed=1, T=_T, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    first, second = _checkpoint(reference_run), _checkpoint(again)
    assert np.array_equal(first["X"], second["X"])
    assert np.array_equal(first["y"], second["y"])
    assert np.array_equal(first["f"], second["f"])
    assert first["X"].shape == (_T, 5)


def test_two_nuts_runs_at_the_same_seed_are_bit_identical(tmp_path):
    # A real chain, at a budget small enough for a non-slow test and thresholds that accept every
    # attempt: what is pinned is that the sampler's key comes from (seed, t) and nothing else.
    kwargs = dict(
        nuts=NUTSConfig(32, 32, 4),
        thresholds=DiagThresholds(float("inf"), 0.0, 10**9),
        **_LOOP_KW,
    )
    first = run_bo(
        _objective(), "product/lengthscale", seed=1, T=7, n_init=5,
        out_dir=tmp_path / "a", **kwargs,
    )
    second = run_bo(
        _objective(), "product/lengthscale", seed=1, T=7, n_init=5,
        out_dir=tmp_path / "b", **kwargs,
    )
    assert np.array_equal(_checkpoint(first)["X"], _checkpoint(second)["X"])
    assert [row["status"] for row in _read_rows(first / "iterations.csv")] == ["ok"] * 2


def test_resume_reproduces_the_uninterrupted_run(reference_run, tmp_path):
    run_bo(_objective(), "dsp_map", seed=1, T=8, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW)
    resumed = run_bo(
        _objective(), "dsp_map", seed=1, T=_T, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    assert np.array_equal(_checkpoint(reference_run)["X"], _checkpoint(resumed)["X"])
    assert np.array_equal(_checkpoint(reference_run)["y"], _checkpoint(resumed)["y"])

    rows = _read_rows(resumed / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, _T))


def test_resume_refuses_a_changed_configuration(tmp_path):
    run_bo(_objective(), "dsp_map", seed=1, T=6, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW)
    with pytest.raises(ValueError, match="different configuration"):
        run_bo(
            _objective(), "dsp_map", seed=1, T=6, n_init=_N_INIT + 1, out_dir=tmp_path,
            resume=True, **_LOOP_KW,
        )


def test_unknown_override_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        run_bo(_objective(), "dsp_map", seed=1, T=6, n_init=_N_INIT, out_dir=tmp_path, nonsense=1)


def test_rows_carry_the_schema_the_regret_and_the_sobol_schedule(reference_run):
    objective = _objective()
    X = _checkpoint(reference_run)["X"]
    rows = _read_rows(reference_run / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, _T))

    # The plan's row schema, spelled out rather than imported: this is the contract every later
    # analysis reads, so a rename in `bo.py` has to fail here.
    assert list(rows[0]) == [
        "t", "method", "family", "seed", "y", "f", "best_obs", "best_f", "regret", "acq_value",
        "fit_wall_s", "acq_wall_s", "fit_calls", "nuts_attempts", "status", "reason",
        "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
        "num_steps_mean",
        "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell", "n_eff_min_ell",
        "r_hat_max_global", "n_eff_min_global",
        "a0_r_hat_max", "a0_n_eff_min", "a0_divergences",
        "y_mean", "y_std", "sobol_computed",
        "x_0", "x_1", "x_2", "x_3", "x_4",
    ]

    regrets = [float(row["regret"]) for row in rows]
    assert all(r >= 0.0 for r in regrets)
    assert all(later <= earlier for earlier, later in zip(regrets, regrets[1:]))

    for row in rows:
        t = int(row["t"])
        assert float(row["best_f"]) == float(np.max(objective(X[: t + 1])))
        assert float(row["regret"]) == pytest.approx(objective.f_star - float(row["best_f"]))
        assert float(row["best_obs"]) == float(np.max(_checkpoint(reference_run)["y"][: t + 1]))
        assert row["method"] == "dsp_map" and row["family"] == "aligned3"
        assert int(row["fit_calls"]) == 1 and int(row["nuts_attempts"]) == 0
        assert np.isnan(float(row["r_hat_max"]))  # a MAP fit has no chain to diagnose
        assert np.array_equal([float(row[f"x_{i}"]) for i in range(5)], X[t])
        assert int(row["sobol_computed"]) == int(t % 5 == 0 or t == _T - 1)

    coords = _read_rows(reference_run / "coords.csv")
    assert len(coords) == 5 * (_T - _N_INIT)
    for row in coords:
        t = int(row["t"])
        assert np.isfinite(float(row["native_median"])) and np.isfinite(float(row["p_active"]))
        assert np.isfinite(float(row["sobol_hat"])) == (t % 5 == 0 or t == _T - 1)


def _diagnostics(r_hat_max: float, n_eff_min: float, divergences: int, passed: bool):
    """A `Diagnostics` whose pooled numbers are the caller's, for stubbing the gate's verdict."""
    return Diagnostics(
        r_hat_max=r_hat_max,
        r_hat_median=1.0,
        frac_r_hat_below_1_05=1.0,
        n_eff_min=n_eff_min,
        divergences=divergences,
        num_steps_mean=1.0,
        r_hat_max_native=r_hat_max,
        n_eff_min_native=n_eff_min,
        r_hat_max_ell=float("nan"),
        n_eff_min_ell=float("nan"),
        r_hat_max_global=1.0,
        n_eff_min_global=n_eff_min,
        wall_s=0.5,
        passed=passed,
        reason="" if passed else "stubbed failure",
    )


def test_a_refit_row_carries_both_attempts_diagnostics(tmp_path, monkeypatch):
    """Ruling R43. The unprefixed columns are the *retained* attempt's, so on a refit row they are
    the attempt that passed; without `a0_*` the trigger would be nowhere on disk at all.

    The verdict decides the path, so it is stubbed rather than waited for -- no toy chain reliably
    fails and then passes -- and the two attempts are given distinguishable numbers so that a row
    reading attempt 1 into `a0_*`, or attempt 0 into the unprefixed columns, cannot pass.
    """
    attempts = [
        _diagnostics(1.5, 4.0, 9, passed=False),
        _diagnostics(1.01, 40.0, 0, passed=True),
    ]
    remaining = iter(attempts)
    monkeypatch.setattr(
        sagp.gp, "diagnose", lambda flat, extra, thresholds, wall_s: next(remaining)
    )

    run_dir = run_bo(
        _objective(), "product/lengthscale", seed=1, T=6, n_init=_N_INIT, out_dir=tmp_path,
        nuts=NUTSConfig(32, 32, 4), **_LOOP_KW,
    )

    (row,) = _read_rows(run_dir / "iterations.csv")
    assert row["status"] == "refit" and int(row["nuts_attempts"]) == 2
    assert float(row["a0_r_hat_max"]) == 1.5
    assert float(row["a0_n_eff_min"]) == 4.0
    assert int(row["a0_divergences"]) == 9
    assert float(row["r_hat_max"]) == 1.01
    assert float(row["n_eff_min"]) == 40.0
    assert int(row["divergences"]) == 0
    # The per-group columns come off the retained attempt too, `ell` NaN because this cell has no
    # `kernel_ell` site at all.
    assert float(row["r_hat_max_native"]) == 1.01 and float(row["n_eff_min_global"]) == 40.0
    assert np.isnan(float(row["r_hat_max_ell"])) and np.isnan(float(row["n_eff_min_ell"]))

    # The npz keeps attempt 0 whole, `reason` included: the row's three numbers say the gate
    # fired, and this says on which criterion.
    with np.load(run_dir / "samples" / "t005.npz") as data:
        assert str(data["status"]) == "refit" and int(data["nuts_attempts"]) == 2
        assert float(data["a0_r_hat_max"]) == 1.5
        assert bool(data["a0_passed"]) is False
        assert str(data["a0_reason"]) == "stubbed failure"
        assert float(data["a0_wall_s"]) == 0.5


def test_resume_after_a_partial_write_rebuilds_the_missing_rows(tmp_path):
    run_dir = run_bo(
        _objective(), "dsp_map", seed=1, T=9, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    iterations = run_dir / "iterations.csv"
    before = iterations.read_text().splitlines()
    # A kill between the checkpoint and the logs: the checkpoint is at t = 8, the rows stop at 6.
    iterations.write_text("\n".join(before[:-2]) + "\n")

    run_bo(
        _objective(), "dsp_map", seed=1, T=9, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    rows = _read_rows(iterations)
    assert [int(row["t"]) for row in rows] == [5, 6, 7, 8]

    # The redone iterations are the same iterations: every column but the two wall-clock ones.
    original = _read_rows_from_text(before)
    for redone, first in zip(rows, original):
        for column in redone:
            if column not in ("fit_wall_s", "acq_wall_s"):
                assert redone[column] == first[column]

    coords = _read_rows(run_dir / "coords.csv")
    assert len({(row["t"], row["i"]) for row in coords}) == len(coords) == 4 * 5
    assert sorted(int(p.stem[1:]) for p in (run_dir / "samples").glob("t*.npz")) == [5, 6, 7, 8]


def _read_rows_from_text(lines: list[str]) -> list[dict[str, str]]:
    return list(csv.DictReader(lines))


def test_resume_redoes_an_iteration_whose_row_was_torn_by_the_kill(tmp_path):
    # Ruling R35. Under ruling R34 the row is the last thing an iteration writes, so a kill lands
    # inside it; and when it lands just after a field separator, what is left on disk parses at
    # the header's full width with an empty final field -- a row the width check cannot tell from
    # a complete one. Only the missing line terminator can, and resume has to redo that iteration.
    run_dir = run_bo(
        _objective(), "dsp_map", seed=1, T=9, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    iterations = run_dir / "iterations.csv"
    before = _read_rows(iterations)
    with iterations.open(newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    last = lines[-1].rstrip("\r\n")
    with iterations.open("w", newline="") as handle:
        handle.write("".join(lines[:-1]) + last[: last.rindex(",") + 1])

    with iterations.open(newline="") as handle:
        parsed = list(csv.reader(handle))
    assert len(parsed[-1]) == len(parsed[0]) and parsed[-1][-1] == ""  # full width, last field lost

    run_bo(
        _objective(), "dsp_map", seed=1, T=9, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    rows = _read_rows(iterations)
    assert [int(row["t"]) for row in rows] == [5, 6, 7, 8]
    for redone, first in zip(rows, before):
        for column in redone:
            if column not in ("fit_wall_s", "acq_wall_s"):
                assert redone[column] == first[column]


def test_resume_rewrites_a_header_the_kill_tore(tmp_path):
    # Ruling R37. The kill can land inside the *header* too, before any row is on disk. What it
    # leaves holds no complete row at all, so there is nothing to keep -- and leaving the fragment
    # alone would be worse than removing it: `append_row` writes a header only into a file that is
    # missing or empty, so every row of the resumed run would land underneath a torn first line.
    run_dir = run_bo(
        _objective(), "dsp_map", seed=1, T=8, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    iterations = run_dir / "iterations.csv"
    header = iterations.read_text().splitlines()[0]
    with iterations.open("w", newline="") as handle:
        handle.write(header[:10])  # cut mid-field, with no line terminator

    run_bo(
        _objective(), "dsp_map", seed=1, T=8, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    rows = _read_rows(iterations)
    assert list(rows[0]) == header.split(",")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, 8))


# --- the failure policy ---


def test_a_fit_that_raises_twice_is_replaced_by_a_seeded_random_query(tmp_path, monkeypatch):
    """Both attempts raise: the row is excluded, `fit_calls` is 2, the query is the fallback seed's.

    The run must not end. What replaces the fit is one scrambled Sobol point drawn from the
    iteration's own `fallback_seed`, so even an iteration nothing could be fitted to stays a pure
    function of `(seed, t)` -- rebuilt here from `iteration_rngs` rather than read back.
    """

    def always_raises(*args, **kwargs):
        raise RuntimeError("no surrogate today")

    monkeypatch.setattr(run_bo_module, "fit", always_raises)
    run_dir = run_bo(
        _objective(), "product/lengthscale", seed=4, T=8, n_init=_N_INIT, out_dir=tmp_path,
        nuts=NUTSConfig(32, 32, 4), **_LOOP_KW,
    )

    rows = _read_rows(run_dir / "iterations.csv")
    assert [int(row["t"]) for row in rows] == [5, 6, 7]
    for row in rows:
        t = int(row["t"])
        assert row["status"] == "excluded"
        assert int(row["fit_calls"]) == 2 and int(row["nuts_attempts"]) == 0
        assert row["reason"] == "exception: RuntimeError: no surrogate today"
        assert np.isnan(float(row["acq_value"]))
        seed = int(iteration_rngs(4, t).fallback_seed)
        fallback = qmc.Sobol(5, scramble=True, seed=seed).random(1)[0]
        assert np.array_equal([float(row[f"x_{i}"]) for i in range(5)], fallback)

    # Nothing was fitted, so there are no readouts and no draws to retain: `coords.csv` is legally
    # absent here, exactly as it is for a Sobol run, and a reader has to allow for that.
    assert not (run_dir / "coords.csv").exists()
    assert list((run_dir / "samples").glob("t*.npz")) == []
    assert "the retry raised" in (run_dir / "log.txt").read_text()


def test_a_fit_that_raises_once_is_retried_and_the_retry_is_what_the_row_reports(
    tmp_path, monkeypatch
):
    """One exception costs a second `fit` call and nothing else: the row is the retry's own."""
    real_fit = run_bo_module.fit
    calls = []

    def raises_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(run_bo_module, "fit", raises_first)
    run_dir = run_bo(
        _objective(), "product/lengthscale", seed=4, T=6, n_init=_N_INIT, out_dir=tmp_path,
        nuts=NUTSConfig(32, 32, 4), thresholds=DiagThresholds(float("inf"), 0.0, 10**9),
        **_LOOP_KW,
    )

    (row,) = _read_rows(run_dir / "iterations.csv")
    assert len(calls) == 2
    assert int(row["fit_calls"]) == 2
    # The status is the surviving fit's, not the exception's: this iteration is an ordinary one.
    assert row["status"] == "ok" and row["reason"] == ""
    assert int(row["nuts_attempts"]) == 1 and np.isfinite(float(row["acq_value"]))
    assert len(_read_rows(run_dir / "coords.csv")) == 5
    assert (run_dir / "samples" / "t005.npz").exists()


# --- the two references that are not cells ---


def test_the_sobol_reference_walks_the_seeds_own_sequence(tmp_path):
    run_dir = run_bo(
        _objective(), "sobol", seed=3, T=12, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        expected = qmc.Sobol(5, scramble=True, seed=3).random(12)
    assert np.array_equal(_checkpoint(run_dir)["X"], expected)

    rows = _read_rows(run_dir / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, 12))
    for row in rows:
        assert row["status"] == "ok" and int(row["fit_calls"]) == 0
        assert np.isnan(float(row["acq_value"])) and np.isnan(float(row["fit_wall_s"]))
        assert int(row["sobol_computed"]) == 0
    # This reference fits nothing, so it has no readouts and no draws: `coords.csv` and the
    # samples are absent for every Sobol run, which is legal rather than a missing artifact.
    assert not (run_dir / "coords.csv").exists()
    assert list((run_dir / "samples").glob("t*.npz")) == []

    # Which every reader of a run directory has to allow for, `truncate_to` included: it reads
    # `coords.csv` on each resume, and here there has never been one to read.
    run_bo(
        _objective(), "sobol", seed=3, T=14, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    assert np.array_equal(_checkpoint(run_dir)["X"], initial_design(5, 14, 3))
    rows = _read_rows(run_dir / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, 14))
    assert not (run_dir / "coords.csv").exists()


def test_the_oracle_reference_fits_the_objectives_own_S_end_to_end(tmp_path):
    objective = _objective()
    run_dir = run_bo(
        objective, "oracle_S", seed=3, T=12, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )

    rows = _read_rows(run_dir / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, 12))
    assert all(int(row["fit_calls"]) == 1 and int(row["nuts_attempts"]) == 0 for row in rows)
    assert sorted(int(p.stem[1:]) for p in (run_dir / "samples").glob("t*.npz")) == list(
        range(_N_INIT, 12)
    )

    # The one place a run's artifacts show the oracle was handed S: `fit_map(active=S)` fits
    # X[:, S] alone, so its retained draw carries |S| lengthscales where `dsp_map`'s carries D.
    with np.load(run_dir / "samples" / "t005.npz") as data:
        assert data["kernel_inv_length_sq"].shape == (1, len(objective.labels.S))

    # `readouts` widens back to D with zeros outside `active`, so the oracle reports -- correctly,
    # by construction -- that no coordinate outside S is active.
    off_S = [i for i in range(5) if i not in objective.labels.S]
    assert off_S  # aligned3 at D = 5 leaves two coordinates inert
    coords = _read_rows(run_dir / "coords.csv")
    assert len(coords) == 5 * (12 - _N_INIT)
    assert all(float(r["p_active"]) == 0.0 for r in coords if int(r["i"]) in off_S)
    assert all(float(r["native_median"]) == 0.0 for r in coords if int(r["i"]) in off_S)


@pytest.mark.slow
def test_oracle_beats_sobol_on_aligned3(tmp_path):
    """The oracle reference beats a Sobol search: the loop optimizes, it does not merely run.

    Every other loop test here pins a *mechanism* -- the same seed gives the same trajectory, a
    resume reproduces the uninterrupted run, the row schema is the plan's -- and all of them would
    still pass if the acquisition were maximized with the wrong sign or evaluated at the wrong
    point. This is the end-to-end check that it is not. `oracle_S` is handed the objective's own
    S and so fits a 3-dimensional GP inside a 20-dimensional box, which is the most favourable
    surrogate the study has; over 40 fitted iterations it has to end below a quasi-random search
    of the same budget. Three seeds and a median rather than one run, because a single Sobol
    design can be lucky, and the reduced acquisition budget of the other loop tests because what
    is compared is the two methods against each other, at settings they share.
    """
    objective = make_family("aligned3", 0, D=20)
    kwargs = dict(T=50, n_init=10, num_init_candidates=1000, num_restarts_ei=2)
    final = {
        method: [
            float(
                _read_rows(
                    run_bo(objective, method, seed=seed, out_dir=tmp_path, **kwargs)
                    / "iterations.csv"
                )[-1]["regret"]
            )
            for seed in range(3)
        ]
        for method in ("oracle_S", "sobol")
    }
    assert np.median(final["oracle_S"]) < np.median(final["sobol"])

