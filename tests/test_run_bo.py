"""Tests for `experiments.run_bo`: the contracts of one run directory, end to end.

The loop's contracts are behavioural: two runs at the same seed are the same run bit for bit, a
resumed run equals the uninterrupted one bit for bit (including after a truncated write and after
a torn one), a changed configuration refuses to resume, and the row schema, the regret and the
Sobol schedule are what the plan says. Then the contracts a study depends on but a normal run
never exercises: an exception out of the fit costs a retry and then a seeded random query rather
than the run, and the two references that are not cells -- the Sobol search and the oracle -- run
end to end and leave the artifacts they should (the Sobol search leaves no `coords.csv` at all,
which is legal for a method that fits nothing). Every test runs a real objective through a real
fit -- `dsp_map`, whose MAP fit is milliseconds, with two NUTS-based runs for the sampler's own
determinism and for the draws a cell writes into `samples/`.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

from experiments import run_bo as run_bo_module
from experiments.run_bo import run
from sagp.bo import initial_design, iteration_rngs
from sagp.diagnostics import DiagThresholds
from sagp.gp import CELLS, NUTSConfig
from synthobj.families import make_family


# The study's maximizer is 5 L-BFGS-B restarts off 512 raw Sobol candidates (plus as many RAASP
# perturbations again), each restart run to `maxiter` on a fully Bayesian posterior; one restart
# off 64 candidates keeps the loop tests inside the per-test time budget. It is still BoTorch's
# `optimize_acqf` on the same seeded candidate set.
_LOOP_KW = dict(raw_samples=64, num_restarts=1, sobol_every=5)
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
    return run(
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
    again = run(
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
    first = run(
        _objective(), "product/lengthscale", seed=1, T=7, n_init=5,
        out_dir=tmp_path / "a", **kwargs,
    )
    second = run(
        _objective(), "product/lengthscale", seed=1, T=7, n_init=5,
        out_dir=tmp_path / "b", **kwargs,
    )
    assert np.array_equal(_checkpoint(first)["X"], _checkpoint(second)["X"])
    # One NUTS run per fit now, whatever the gate makes of it: `attempts` has length 1 and the
    # status is the verdict on that one chain.
    rows = _read_rows(first / "iterations.csv")
    assert len(rows) == 2  # T = 7 off n_init = 5, so the loop below is not vacuous
    for row in rows:
        assert int(row["nuts_attempts"]) == 1
        assert row["status"] in ("ok", "excluded")


def test_resume_reproduces_the_uninterrupted_run(reference_run, tmp_path):
    run(_objective(), "dsp_map", seed=1, T=8, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW)
    resumed = run(
        _objective(), "dsp_map", seed=1, T=_T, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    assert np.array_equal(_checkpoint(reference_run)["X"], _checkpoint(resumed)["X"])
    assert np.array_equal(_checkpoint(reference_run)["y"], _checkpoint(resumed)["y"])

    rows = _read_rows(resumed / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, _T))


def test_resume_refuses_a_changed_configuration(tmp_path):
    run(_objective(), "dsp_map", seed=1, T=6, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW)
    with pytest.raises(ValueError, match="different configuration"):
        run(
            _objective(), "dsp_map", seed=1, T=6, n_init=_N_INIT + 1, out_dir=tmp_path,
            resume=True, **_LOOP_KW,
        )


def test_unknown_override_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        run(_objective(), "dsp_map", seed=1, T=6, n_init=_N_INIT, out_dir=tmp_path, nonsense=1)


def test_rows_carry_the_schema_the_regret_and_the_sobol_schedule(reference_run):
    objective = _objective()
    X = _checkpoint(reference_run)["X"]
    rows = _read_rows(reference_run / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, _T))

    # The plan's row schema, spelled out rather than imported: this is the contract every later
    # analysis reads, so a rename in `experiments/runlog.py` has to fail here.
    assert list(rows[0]) == [
        "t", "method", "family", "seed", "y", "f", "best_obs", "best_f", "regret", "acq_value",
        "fit_wall_s", "acq_wall_s", "fit_calls", "nuts_attempts", "status", "reason",
        "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
        "num_steps_mean",
        "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell", "n_eff_min_ell",
        "r_hat_max_global", "n_eff_min_global",
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


def test_samples_npz_carries_the_schema_version_and_botorch_site_names(tmp_path):
    """What `samples/t*.npz` promises a later analysis: BoTorch's site names, and version 2.

    The draws are keyed by the cell's `sites`, which are `SaasFullyBayesianSingleTaskGP`'s own
    names for them, beside the fit's `status` and `nuts_attempts`. `schema_version` is what tells
    a reader which of the two formats it has: version 1's files named the sites differently and
    carried a second attempt's whole `Diagnostics` as `a0_*` scalars, and nothing written now may
    be mistaken for one of those.
    """
    cell = CELLS[("product", "lengthscale")]
    run_dir = run(
        _objective(), "product/lengthscale", seed=1, T=6, n_init=_N_INIT, out_dir=tmp_path,
        # A gate that can never fail: this test is about the npz schema, not the gate, and a
        # 16/16/4 chain is short enough that a real gate would make `status == "ok"` a coin toss.
        nuts=NUTSConfig(16, 16, 4), thresholds=DiagThresholds(float("inf"), 0.0, 10**9),
        **_LOOP_KW,
    )

    with np.load(run_dir / "samples" / "t005.npz") as data:
        keys = set(data.files)
        # Every site of the cell: `noise` among them, since this run learns the noise rather than
        # fixing it, which is the one site a `--fixed-noise` run would not have.
        assert set(cell.sites) <= keys
        assert {"status", "nuts_attempts", "schema_version"} <= keys
        assert int(data["schema_version"]) == 2
        assert int(data["nuts_attempts"]) == 1 and str(data["status"]) == "ok"
        assert [key for key in sorted(keys) if key.startswith("a0_")] == []
        # 16 samples thinned by 4: one leading dimension, shared by every site.
        assert {data[site].shape[0] for site in cell.sites} == {4}


def test_resume_after_a_partial_write_rebuilds_the_missing_rows(tmp_path):
    run_dir = run(
        _objective(), "dsp_map", seed=1, T=9, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    iterations = run_dir / "iterations.csv"
    before = iterations.read_text().splitlines()
    # A kill between the checkpoint and the logs: the checkpoint is at t = 8, the rows stop at 6.
    iterations.write_text("\n".join(before[:-2]) + "\n")

    run(
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
    run_dir = run(
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

    run(
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
    run_dir = run(
        _objective(), "dsp_map", seed=1, T=8, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    iterations = run_dir / "iterations.csv"
    header = iterations.read_text().splitlines()[0]
    with iterations.open("w", newline="") as handle:
        handle.write(header[:10])  # cut mid-field, with no line terminator

    run(
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
    run_dir = run(
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
    run_dir = run(
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
    run_dir = run(
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
    run(
        _objective(), "sobol", seed=3, T=14, n_init=_N_INIT, out_dir=tmp_path, resume=True,
        **_LOOP_KW,
    )
    assert np.array_equal(_checkpoint(run_dir)["X"], initial_design(5, 14, 3))
    rows = _read_rows(run_dir / "iterations.csv")
    assert [int(row["t"]) for row in rows] == list(range(_N_INIT, 14))
    assert not (run_dir / "coords.csv").exists()


def test_the_oracle_reference_fits_the_objectives_own_S_end_to_end(tmp_path):
    objective = _objective()
    run_dir = run(
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
# The only two warnings this run raises that the short loop tests above do not, both of them a
# retry the code already handles: the escalating jitter ladder `psd_safe_cholesky` climbs inside a
# `fit_gpytorch_mll` restart, which `_fit_mll` runs under `cholesky_max_tries(9)` and abandons for
# Adam only once every restart has failed; and the failed `gen_candidates_scipy` restart that
# `optimize_acqf`'s seeded `retry_on_optimization_warning` runs again from fresh initial
# conditions. Neither changes a result -- both are the attempt that was discarded -- and this is
# the whole filter: nothing else is silenced.
@pytest.mark.filterwarnings(
    "ignore:A not p.d., added jitter:gpytorch.utils.warnings.NumericalWarning"
)
@pytest.mark.filterwarnings("ignore:Optimization failed in `gen_candidates_scipy`:RuntimeWarning")
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
    kwargs = dict(T=50, n_init=10, raw_samples=256, num_restarts=2)
    final = {
        method: [
            float(
                _read_rows(
                    run(objective, method, seed=seed, out_dir=tmp_path, **kwargs)
                    / "iterations.csv"
                )[-1]["regret"]
            )
            for seed in range(3)
        ]
        for method in ("oracle_S", "sobol")
    }
    assert np.median(final["oracle_S"]) < np.median(final["sobol"])

