"""Tests for `sagp.bo`: the acquisition, the seeded reference optimizer, and the loop's contracts.

Three things are pinned here, in the order the file runs them. First the acquisition: `log_h` and
`log1mexp` are the pieces of LogEI that exist only for numerical reasons, so they are checked
against the naive formulas where those are accurate and against finiteness where they are not, and
`log_ei` is checked to be the log of the reference's own EI wherever the reference has not
underflowed. Second `optimize_ei`, whose five-line departure from `saasbo.optimize_ei` must leave
it a maximizer under both acquisitions. Third the loop itself, whose contracts are behavioural: two
runs at the same seed are the same run bit for bit, a resumed run equals the uninterrupted one bit
for bit (including after a truncated write), a changed configuration refuses to resume, and the
row schema, the regret and the Sobol schedule are what the plan says.

Every loop test runs a real objective through a real fit -- `dsp_map`, whose MAP fit is
milliseconds, with one NUTS-based run for the sampler's own determinism. Two knobs are turned down
from the study's settings (`num_init_candidates`, `num_restarts_ei`, both `RunConfig` fields) to
keep each test inside the suite's per-test budget: they change how hard the acquisition is
maximized, not whether the loop is reproducible, which is what is being tested.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm, qmc

import saasbo
from sagp.bo import (
    ACQUISITIONS,
    initial_design,
    log1mexp,
    log_ei,
    log_ei_sum,
    log_h,
    optimize_ei,
    run_bo,
)
from sagp.gp import DiagThresholds, FittedGP, NUTSConfig
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


def _hand_made_gp() -> FittedGP:
    """A product/lengthscale `FittedGP` built by hand (S = 3, D = 4, n = 12), no inference involved.

    The acquisition only ever sees `posterior`, so a fitted GP assembled from arbitrary but valid
    draws exercises it exactly as a real fit would, deterministically and in milliseconds.
    """
    rng = np.random.default_rng(0)
    X = rng.random((12, 4))
    y = rng.standard_normal(12)
    samples = {
        "kernel_var": jnp.asarray([1.0, 0.7, 1.3]),
        "kernel_inv_length_sq": jnp.asarray(rng.gamma(2.0, 1.0, size=(3, 4))),
        "kernel_noise": jnp.asarray([1.0e-3, 5.0e-3, 2.0e-3]),
    }
    return FittedGP(
        cell=("product", "lengthscale"),
        X_train=X,
        Y_train=y,
        samples=samples,
        fixed_noise=None,
        active=None,
        status="ok",
        status_reason="",
        attempts=(),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- the acquisition ---


def test_log_h_matches_naive_and_stays_finite():
    z = np.linspace(-6.0, 6.0, 200)
    h = norm.pdf(z) + z * norm.cdf(z)
    got = np.asarray(log_h(jnp.asarray(z)))
    assert np.all(np.abs(np.exp(got) - h) <= 1e-12 * np.maximum(1.0, h))

    # Where h itself underflows, only the log form survives: it must stay finite, keep decreasing,
    # and keep a finite gradient, since that gradient is what the L-BFGS-B restarts follow.
    tails = [float(log_h(jnp.asarray(z_far))) for z_far in (-10.0, -20.0, -40.0)]
    grads = [float(jax.grad(log_h)(z_far)) for z_far in (-10.0, -20.0, -40.0)]
    assert all(np.isfinite(tails)) and all(np.isfinite(grads))
    assert tails[0] > tails[1] > tails[2]
    assert norm.pdf(-40.0) + -40.0 * norm.cdf(-40.0) == 0.0  # the naive form has no value here

    x = np.linspace(-30.0, -1.0e-3, 500)
    naive = np.log(1.0 - np.exp(x))
    # `1 - exp(x)` cancels as x -> 0 (three digits at x = -1e-3), so the naive reference is only
    # accurate to its own conditioning; the tolerance is the brief's 1e-14 plus exactly that.
    conditioning = 4.0 * np.finfo(float).eps * np.exp(x) / (-np.expm1(x))
    assert np.all(np.abs(np.asarray(log1mexp(jnp.asarray(x))) - naive) <= 1e-14 + conditioning)


def test_log_ei_equals_log_of_reference_ei():
    gp = _hand_made_gp()
    rng = np.random.default_rng(7)
    X_test = rng.random((20, 4))
    y_target = float(np.asarray(gp.Y_train).min())

    reference = np.asarray(saasbo.ei(X_test, y_target, gp))
    ours = np.asarray(log_ei(X_test, y_target, gp))
    usable = reference > 1e-100
    assert usable.sum() == 20
    assert np.max(np.abs(ours[usable] - np.log(reference[usable]))) < 1e-9

    # 40 standard deviations below the posterior: every per-sample EI underflows to exactly zero,
    # so the reference stops distinguishing candidates and stops having a gradient. LogEI does not.
    mu, var = gp.posterior(X_test)
    far = float(np.asarray(mu).min() - 40.0 * float(np.sqrt(np.asarray(var)).max()))
    assert np.all(np.asarray(saasbo.ei(X_test, far, gp)) == 0.0)
    values = np.asarray(log_ei(X_test, far, gp))
    assert np.all(np.isfinite(values))
    grad = np.asarray(jax.grad(lambda x: log_ei_sum(x, far, gp))(jnp.asarray(X_test)))
    assert np.all(np.isfinite(grad)) and np.abs(grad).max() > 0.0


@pytest.mark.parametrize("acq_name", ["ei", "logei"])
def test_optimize_ei_reference_acq_is_within_bounds_and_improves(acq_name):
    gp = _hand_made_gp()
    acq = ACQUISITIONS[acq_name]
    y_target = float(np.asarray(gp.Y_train).min())

    x_best, value = optimize_ei(
        gp, y_target, sobol_seed=1, jitter_rng=np.random.default_rng(1), acq=acq
    )
    assert x_best.shape == (4,)
    assert np.all(x_best >= 0.0) and np.all(x_best <= 1.0)

    # The candidate set the run's seeds define, rebuilt here: L-BFGS-B starts from its best points
    # and cannot return a worse one, so the reported value is a floor on the whole set's best.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        candidates = qmc.Sobol(4, scramble=True, seed=1).random(5000)
    incumbent = np.asarray(gp.X_train)[int(np.asarray(gp.Y_train).argmin())]
    candidates[0, :] = np.clip(
        incumbent + 0.001 * np.random.default_rng(1).standard_normal((1, 4)), a_min=0.0, a_max=1.0
    )
    best_candidate = float(np.max(np.asarray(acq(jnp.asarray(candidates), y_target, gp))))
    assert value >= best_candidate - 1e-12


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
        "num_steps_mean", "y_mean", "y_std", "sobol_computed",
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


def test_initial_design_is_the_first_rows_of_the_runs_sobol_sequence():
    # What makes the Sobol reference a *continuation* of the shared design rather than a new one.
    assert np.array_equal(initial_design(5, 5, 1), initial_design(5, 15, 1)[:5])
