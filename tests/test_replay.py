"""Tests for `experiments.replay`: refitting a cell on the data a stored run fitted at iteration t.

What gate G2 needs the replay to be, one contract per test. On a CPU, replaying a cell on its own
run is the loop's own fit again, bit for bit. A stored run that could hand the replay other data
than its loop fitted is refused before anything is fitted or written: a checkpoint or row it
lacks, a manifest that does not reproduce its hash or names another run, observations or points
the rows do not carry. An R2-D2 cell replayed on its twin's run writes the loop's own vocabulary:
the row, the coordinates and the draws. A second call skips what the first finished and redoes,
without duplicating, what a kill interrupted; a replay directory holds one configuration, one
source run and one budget, and records every session's commit and checkout. And `compare` joins
every replayed fit to the stored fit of the same (family, seed, t) -- refusing a replay not made
on that run -- and, for an R2-D2 fit, to its twin's control refit; it reads G2 only on a complete
tree at one clean commit, and the GPU probe for its cost alone. Runs are the loop tests' D = 5
problem at a 16/16/4 chain, so a fit costs seconds; G2's trees are written directly.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

import sagp.r2d2
from experiments import replay as replay_module
from experiments.replay import REPLAY_T, TWINS, compare, load_source, main, run_replay
from experiments.run_bo import run
from experiments.runlog import config_hash
from sagp.bo import iteration_rngs
from sagp.diagnostics import DiagThresholds
from sagp.gp import CELLS, NUTSConfig, R2D2_K, standardize
from synthobj.families import make_family


# `_LOOP_KW`, `_objective` and `_read_rows` are copies of tests/test_run_bo.py:36-57 (ruling R5):
# importing them would make pytest run that module's tests a second time under another name.

# The study's maximizer is 5 L-BFGS-B restarts off 512 raw Sobol candidates (plus as many RAASP
# perturbations again), each restart run to `maxiter` on a fully Bayesian posterior; one restart
# off 64 candidates keeps the loop tests inside the per-test time budget. It is still BoTorch's
# `optimize_acqf` on the same seeded candidate set.
_LOOP_KW = dict(raw_samples=64, num_restarts=1, sobol_every=5)


def _objective():
    """The fixed D = 5 problem every loop test runs: three active coordinates, exact `f_star`."""
    return make_family("aligned3", 0, D=5)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- the replay ---


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """A stored runs root holding one product/lengthscale run, rows t = 5-7, never written to.

    The refusal, resume and dry-run tests read it; the ones that tamper with it copy it first.
    """
    root = tmp_path_factory.mktemp("runs")
    run(_objective(), "product/lengthscale", seed=1, T=8, n_init=5, out_dir=root,
        nuts=NUTSConfig(16, 16, 4), **_LOOP_KW)
    return root


def _source_dir(root: Path) -> Path:
    return root / "aligned3" / "product-lengthscale" / "seed01"


def _rewrite_checkpoint(run_dir: Path, **changes: object) -> None:
    """Rewrite `checkpoint.npz` with some arrays replaced and every other one kept."""
    path = run_dir / "checkpoint.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays.update(changes)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _checkpoint_under_another_hash(run_dir: Path) -> None:
    # A run resumed under another configuration: its checkpoint names a hash its manifest does not.
    _rewrite_checkpoint(run_dir, config_hash="0" * 64)


def _drop_row_6(run_dir: Path) -> None:
    path = run_dir / "iterations.csv"
    lines = path.read_text().splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith("6,")]
    assert len(kept) == len(lines) - 1
    path.write_text("".join(kept))


def _nudge_y(run_dir: Path) -> None:
    # One observation moved by 1e-6: y[:t] no longer standardizes to what row t recorded.
    with np.load(run_dir / "checkpoint.npz") as data:
        y = data["y"].copy()
    y[0] += 1e-6
    _rewrite_checkpoint(run_dir, y=y)


def _edit_the_manifest(run_dir: Path) -> None:
    # A setting edited after the run: the manifest no longer hashes to the config_hash it records.
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sobol_every"] += 1
    path.write_text(json.dumps(manifest))


def _swap_y_5_and_6(run_dir: Path) -> None:
    # Two observations of the rows t' = 5 and 6 exchanged: y[:5] and y[:7] keep their mean and std
    # to the bit (checked here), so only the rows' own y can tell the checkpoint is not theirs.
    with np.load(run_dir / "checkpoint.npz") as data:
        y = data["y"].copy()
    swapped = y.copy()
    swapped[[5, 6]] = y[[6, 5]]
    for t in (5, 7):
        assert standardize(swapped[:t])[1:] == standardize(y[:t])[1:], "the swap moved a moment"
    _rewrite_checkpoint(run_dir, y=swapped)


def _swap_x_5_and_6(run_dir: Path) -> None:
    # Two points of the rows t' = 5 and 6 exchanged: y is untouched, so only the rows' x_* can tell.
    with np.load(run_dir / "checkpoint.npz") as data:
        X = data["X"].copy()
    X[[5, 6]] = X[[6, 5]]
    _rewrite_checkpoint(run_dir, X=X)


@pytest.mark.parametrize(
    "tamper, t, match",
    [
        (None, 8, "t_done"),
        (_checkpoint_under_another_hash, 6, "config_hash"),
        (_drop_row_6, 6, "no complete row"),
        (_nudge_y, 6, "y_mean"),
        (_edit_the_manifest, 6, "does not reproduce the config_hash"),
        (_swap_y_5_and_6, 7, r"row t=5 records"),
        (_swap_x_5_and_6, 7, r"row t=5 records"),
    ],
    ids=["t_beyond_t_done", "checkpoint_under_another_hash", "missing_row", "tampered_y",
         "manifest_edited", "y_swapped_keeping_its_moments", "x_swapped"],
)
def test_a_replay_refuses_data_its_source_loop_did_not_fit(runs, tmp_path, tamper, t, match):
    """Review Focus 1: refused -- before any t is fitted -- never fitted on other data.

    t = 5 is valid in every case and is asked for first, so a refusal that came only once the
    bad t was reached would leave t = 5's fit behind; `out` must not even exist. The last three
    cases: a manifest that does not reproduce its own hash, and a checkpoint whose rows t' in
    [n_init, t) no longer hold the y[t'] and X[t'] that `iterations.csv` recorded there, one of
    them with y[:t]'s mean and std unchanged, which the y_mean/y_std check alone would pass.
    """
    copy = tmp_path / "runs"
    shutil.copytree(runs, copy)
    if tamper is not None:
        tamper(_source_dir(copy))
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match=match):
        run_replay(copy, out, family="aligned3", seed=1, method="product/lengthscale_r2d2",
                   ts=(5, t))
    assert not out.exists()


def test_a_replay_refuses_a_source_whose_manifest_names_another_run(runs, tmp_path):
    """The source directory is found by its path, so the manifest must name the run the path
    does: a run copied under another seed's path would otherwise hand its data to that seed."""
    copy = tmp_path / "runs"
    shutil.copytree(runs, copy)
    shutil.copytree(_source_dir(copy), _source_dir(copy).with_name("seed02"))
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="not the run its path does"):
        run_replay(copy, out, family="aligned3", seed=2, method="product/lengthscale_r2d2",
                   ts=(6,))
    assert not out.exists()


@pytest.mark.parametrize("inside", ["", "aligned3"], ids=["the_runs_root", "a_directory_in_it"])
def test_a_replay_never_writes_into_the_stored_runs(runs, inside):
    """Under the runs root, a control's replay directory would be its own source run, and the
    replay's rollback of unfinished t would delete that run's coordinates and draws."""
    before = _snapshot(runs)
    with pytest.raises(ValueError, match="only read"):
        run_replay(runs, runs / inside, family="aligned3", seed=1, method="product/lengthscale",
                   ts=(6,))
    assert _snapshot(runs) == before


# At about the marker's 10 s line: an additive/amplitude run's two fits and acquisitions, the
# replay's R2-D2 fit, and the process's first build of the R2-D2 map's table.
@pytest.mark.slow
def test_an_r2d2_replay_on_its_twins_run_writes_the_loops_vocabulary(tmp_path):
    """The R2-D2 cell refitted on its twin's iteration-6 data, and everything the loop would write.

    The row names the cell and the run it was replayed on, carries the fit's diagnostics and the
    source row's standardization, and states the budget and the device; `coords.csv` holds the
    t = 6 readouts (with the Sobol index, t = 6 being T - 1); the npz holds the R2-D2 cell's own
    sites and no half-Cauchy one; the manifest is the source's configuration with `method`
    replaced, and names the source.
    """
    source = run(
        _objective(), "additive/amplitude", seed=1, T=7, n_init=5, out_dir=tmp_path / "runs",
        nuts=NUTSConfig(16, 16, 4), thresholds=DiagThresholds(float("inf"), 0.0, 10**9),
        **_LOOP_KW,
    )
    method = "additive/amplitude_r2d2"
    cell = CELLS[("additive", "amplitude_r2d2")]
    replay_dir = run_replay(tmp_path / "runs", tmp_path / "replay", family="aligned3", seed=1,
                            method=method, ts=(6,))
    assert replay_dir == tmp_path / "replay" / "aligned3" / "additive-amplitude_r2d2" / "seed01"

    (row,) = _read_rows(replay_dir / "replay.csv")
    # D9's row, spelled out: every later reader of a replay directory reads these names.
    assert list(row) == [
        "t", "n", "method", "source_method", "family", "seed", "nuts_seed", "budget",
        "status", "reason",
        "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
        "num_steps_mean",
        "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell", "n_eff_min_ell",
        "r_hat_max_global", "n_eff_min_global",
        "fit_wall_s", "readout_wall_s", "y_mean", "y_std", "r2d2_r2", "first_order_r2", "device",
    ]
    stored = next(r for r in _read_rows(source / "iterations.csv") if r["t"] == "6")
    assert row["t"] == "6" and row["n"] == "6"
    assert row["method"] == method and row["source_method"] == "additive/amplitude"
    assert row["family"] == "aligned3" and row["seed"] == "1"
    assert int(row["nuts_seed"]) == iteration_rngs(1, 6).nuts_seed
    assert row["budget"] == "16/16/4/6"
    assert row["status"] == "ok" and row["reason"] == ""  # the source's gate, which cannot fail
    assert (row["y_mean"], row["y_std"]) == (stored["y_mean"], stored["y_std"])
    assert np.isfinite(float(row["r_hat_max"])) and float(row["fit_wall_s"]) > 0.0
    assert 0.0 < float(row["r2d2_r2"]) < 1.0 and 0.0 < float(row["first_order_r2"]) < 1.0
    assert row["device"] == jax.devices()[0].device_kind

    coords = _read_rows(replay_dir / "coords.csv")
    assert [(int(r["t"]), int(r["i"])) for r in coords] == [(6, i) for i in range(5)]
    for r in coords:
        assert all(np.isfinite(float(r[k])) for k in ("native_median", "p_active", "sobol_hat"))

    with np.load(replay_dir / "samples" / "t006.npz") as data:
        assert set(data.files) == set(cell.sites) | {"status", "nuts_attempts", "schema_version"}
        assert int(data["schema_version"]) == 2 and str(data["status"]) == "ok"
        assert {data[site].shape[0] for site in cell.sites} == {4}

    manifest = json.loads((replay_dir / "manifest.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["method"] == method
    for field in ("family", "seed", "D", "T", "n_init", "nuts", "thresholds", "sobol_n"):
        assert manifest[field] == source_manifest[field]
    source_cfg = load_source(source).cfg
    assert manifest["config_hash"] == config_hash(dataclasses.replace(source_cfg, method=method))
    assert manifest["objective"] == source_manifest["objective"]
    assert manifest["source"] == {
        "path": str(source),
        "config_hash": source_manifest["config_hash"],
        "commit": source_manifest["git"]["commit"],
        "devices": [source_manifest["env"]["jax_device"]],
    }


@pytest.fixture(scope="module")
def replayed(runs, tmp_path_factory) -> tuple[Path, Path]:
    """The control replay of t = 6 on `runs`: (the replay root, the replay directory).

    Read only by the tests that use it -- a second call must write nothing, and a refused call
    nothing either; the test that damages it works on a copy.
    """
    out = tmp_path_factory.mktemp("replay")
    replay_dir = run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale",
                            ts=(6,))
    return out, replay_dir


def test_replaying_a_run_reproduces_its_stored_fit_bit_for_bit(runs, replayed):
    # The harness check that a GPU cannot give: on CPU, refitting iteration t's data with
    # iteration t's seed and the run's budget is the same computation as the loop's own fit.
    # The run and its t = 6 replay are the module's `runs` and `replayed` fixtures, which other
    # tests build anyway, so this exact check costs PART_A nothing (ruling R29, revised).
    source = _source_dir(runs)
    replay_dir = replayed[1]
    with np.load(source / "samples" / "t006.npz") as a, np.load(replay_dir / "samples" / "t006.npz") as b:
        for site in CELLS[("product", "lengthscale")].sites:
            assert np.array_equal(a[site], b[site])
    stored = next(r for r in _read_rows(source / "iterations.csv") if r["t"] == "6")
    replayed = _read_rows(replay_dir / "replay.csv")[0]
    for field in ("status", "r_hat_max", "n_eff_min", "divergences", "y_mean", "y_std"):
        assert replayed[field] == stored[field]


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _must_not_fit(*args, **kwargs):
    raise AssertionError("a t with a complete row was fitted again")


def test_a_second_call_skips_every_t_it_has_done(runs, replayed, monkeypatch):
    out, replay_dir = replayed
    before = _snapshot(replay_dir)
    monkeypatch.setattr(replay_module, "fit", _must_not_fit)
    again = run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale",
                       ts=(6,))
    assert again == replay_dir
    assert _snapshot(replay_dir) == before


def test_a_budget_other_than_the_files_is_refused(runs, replayed, monkeypatch):
    out, replay_dir = replayed
    before = _snapshot(replay_dir)
    monkeypatch.setattr(replay_module, "fit", _must_not_fit)
    # t = 7 is not done, so only the refusal stands between this call and a fit at depth 5.
    with pytest.raises(ValueError, match="budget"):
        run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale",
                   ts=(6, 7), max_tree_depth=5)
    with pytest.raises(ValueError, match="budget"):
        run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale",
                   ts=(7,), nuts=NUTSConfig(32, 32, 4))
    assert _snapshot(replay_dir) == before


def _replace_the_replays_config_hash(replay_dir: Path) -> None:
    _edit_json(replay_dir / "manifest.json", lambda manifest: manifest.update(config_hash="0" * 64))


def _replay_rows_on_another_source(replay_dir: Path) -> None:
    path = replay_dir / "replay.csv"
    rows = _read_rows(path)
    for row in rows:
        row["source_method"] = "additive/lengthscale"
    _write_csv(path, list(rows[0]), rows)


def _no_rows(replay_dir: Path) -> None:
    # What a directory holds when its first t was never finished: a manifest and a log.
    for name in ("replay.csv", "coords.csv"):
        (replay_dir / name).unlink()
    shutil.rmtree(replay_dir / "samples")


def _no_source_block(replay_dir: Path) -> None:
    # A kill between the manifest's two writes: RunLogger's, then the one adding `source`.
    _no_rows(replay_dir)
    _edit_json(replay_dir / "manifest.json", lambda manifest: manifest.pop("source"))


def _another_source(replay_dir: Path) -> None:
    # A directory begun on another run (another source method, or the same one under other
    # settings): its manifest names that run's config_hash.
    _no_rows(replay_dir)
    _edit_json(replay_dir / "manifest.json",
               lambda manifest: manifest["source"].update(config_hash="0" * 64))


def _edit_json(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "damage, match",
    [
        (_replace_the_replays_config_hash, "different configuration"),
        (_replay_rows_on_another_source, "replayed on additive/lengthscale's run"),
        (_no_source_block, "names no source run"),
        (_another_source, "was replayed on the run"),
    ],
    ids=["replay_config_hash", "rows_of_another_source", "no_source_block", "another_source"],
)
def test_a_replay_directory_is_continued_only_under_its_own_terms(
    runs, replayed, tmp_path, monkeypatch, damage, match
):
    """A second call continues a replay directory only on the configuration, the source run and
    the rows it was begun with; otherwise it is refused before anything is fitted or written. The
    source run is named by its config_hash, which names the run wherever its root is mounted."""
    out = tmp_path / "replay"
    shutil.copytree(replayed[0], out)
    replay_dir = out / replayed[1].relative_to(replayed[0])
    damage(replay_dir)
    before = _snapshot(replay_dir)
    monkeypatch.setattr(replay_module, "fit", _must_not_fit)
    with pytest.raises(ValueError, match=match):
        run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale",
                   ts=(6, 7))
    assert _snapshot(replay_dir) == before


def test_a_t_the_kill_interrupted_is_redone_without_duplicates(runs, replayed, tmp_path):
    """What a kill during t = 7 leaves -- part of its coordinates, a torn row, a torn npz -- is
    rolled back and t = 7 redone, while t = 6 is kept as it was (the loop's `truncate_to`)."""
    out = tmp_path / "replay"
    shutil.copytree(replayed[0], out)
    replay_dir = out / replayed[1].relative_to(replayed[0])
    t6_npz = (replay_dir / "samples" / "t006.npz").read_bytes()
    with (replay_dir / "coords.csv").open("a", newline="") as handle:
        handle.write("7,0,0.5,0.5,nan\r\n7,1,0.5,0.5,nan\r\n7,2,0.")
    with (replay_dir / "replay.csv").open("a", newline="") as handle:
        handle.write("7,7,product/lengthscale,product/lengthscale,aligned3,1,")
    (replay_dir / "samples" / "t007.npz").write_bytes(b"torn")

    run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale", ts=(6, 7))

    assert [r["t"] for r in _read_rows(replay_dir / "replay.csv")] == ["6", "7"]
    coords = [(int(r["t"]), int(r["i"])) for r in _read_rows(replay_dir / "coords.csv")]
    assert coords == [(6, i) for i in range(5)] + [(7, i) for i in range(5)]
    assert (replay_dir / "samples" / "t006.npz").read_bytes() == t6_npz
    with np.load(replay_dir / "samples" / "t007.npz") as data:
        assert int(data["schema_version"]) == 2


def test_a_resumed_replay_records_whether_its_checkout_was_dirty(
    runs, replayed, tmp_path, monkeypatch
):
    """G2 is read only off replays whose every session ran at one commit on a clean checkout
    (`compare`), so a replay's resume entry records `dirty` beside the commit `note_resume`
    records, as the manifest's `git` block does for the directory's creation."""
    out = tmp_path / "replay"
    shutil.copytree(replayed[0], out)
    replay_dir = out / replayed[1].relative_to(replayed[0])
    monkeypatch.setattr(replay_module, "fit", _fit_reached)
    with pytest.raises(_FitReached):
        run_replay(runs, out, family="aligned3", seed=1, method="product/lengthscale", ts=(6, 7))
    (entry,) = json.loads((replay_dir / "manifest.json").read_text())["resumed"]
    assert set(entry) == {"time", "commit", "versions_changed", "T", "env", "dirty"}
    assert isinstance(entry["dirty"], bool)


def test_a_dry_run_prints_the_datasets_seeds_and_budget_and_writes_nothing(runs, tmp_path, capsys):
    out = tmp_path / "dry"
    argv = [
        "fit", "--runs", str(runs), "--out", str(out), "--family", "aligned3", "--seed", "1",
        "--cell", "product/lengthscale_r2d2", "--t", "5,6,7", "--max-tree-depth", "5",
        "--dry-run",
    ]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert not out.exists()
    assert "product/lengthscale" in printed and "16/16/4/5" in printed
    for t in (5, 6, 7):
        assert f"t={t} n={t} nuts_seed={iteration_rngs(1, t).nuts_seed}" in printed


class _FitReached(Exception):
    """Raised by a stub `fit`: the replay got as far as its first timed fit."""


def _fit_reached(*args, **kwargs):
    raise _FitReached


@pytest.mark.parametrize("method", ["product/lengthscale", "product/lengthscale_r2d2"])
def test_every_replay_builds_the_r2d2_table_before_its_first_timed_fit(
    runs, tmp_path, monkeypatch, method
):
    """Ruling R26: a process's one-time build of the R2-D2 map's table (cached per shape) is
    done before the first timed fit, for a half-Cauchy cell as for an R2-D2 one, so no
    `fit_wall_s` carries it. The table is removed from the cache first; the stub `fit` looks."""
    monkeypatch.delitem(sagp.r2d2._TABLES, float(R2D2_K), raising=False)
    seen = []

    def first_fit(*args, **kwargs):
        seen.append(float(R2D2_K) in sagp.r2d2._TABLES)
        raise _FitReached

    monkeypatch.setattr(replay_module, "fit", first_fit)
    with pytest.raises(_FitReached):
        run_replay(runs, tmp_path / "replay", family="aligned3", seed=1, method=method, ts=(6,))
    assert seen == [True]


def test_a_dry_run_builds_no_table(runs, tmp_path, monkeypatch):
    monkeypatch.delitem(sagp.r2d2._TABLES, float(R2D2_K), raising=False)
    run_replay(runs, tmp_path / "dry", family="aligned3", seed=1,
               method="product/lengthscale_r2d2", ts=(6,), dry_run=True)
    assert float(R2D2_K) not in sagp.r2d2._TABLES


def test_the_warm_up_logs_the_r2d2_maps_values_at_full_precision(replayed):
    """On Triton the R2-D2 table is built by GPU arithmetic, which its accuracy certificate (CPU
    only) does not cover. The warm-up line therefore logs the map at five z, by repr, so the GPU
    log can be compared with a CPU evaluation of the same map: here, that evaluation itself."""
    log = (replayed[1] / "log.txt").read_text().splitlines()
    (line,) = [entry for entry in log if "R2-D2 table" in entry]
    logged = re.findall(r"y\((-?\d+)\) = (\S+?)(?:,|$)", line)
    assert [int(z) for z, _ in logged] == [-30, -5, 0, 5, 30]
    with jax.default_device(jax.devices("cpu")[0]):
        on_cpu = np.asarray(sagp.r2d2.log_gamma_icdf(np.array([-30.0, -5.0, 0.0, 5.0, 30.0]),
                                                     R2D2_K))
    assert [float(value) for _, value in logged] == on_cpu.tolist()


# A source run under an explicit alpha: twice the half-Cauchy lengthscale prior's global scale.
_SOURCE_ALPHA = 0.2


@pytest.fixture(scope="module")
def alpha_runs(tmp_path_factory) -> Path:
    """A stored runs root holding one product/lengthscale run at alpha = 0.2, rows t = 5-6."""
    root = tmp_path_factory.mktemp("alpha_runs")
    run(_objective(), "product/lengthscale", seed=1, T=7, n_init=5, out_dir=root,
        nuts=NUTSConfig(16, 16, 4), alpha=_SOURCE_ALPHA, **_LOOP_KW)
    return root


@pytest.mark.parametrize("dry_run", [False, True], ids=["fit", "dry_run"])
def test_a_twin_replay_refuses_a_source_run_with_its_own_alpha(
    alpha_runs, tmp_path, monkeypatch, dry_run
):
    """Ruling R27: alpha is the half-Cauchy global scale in the source's prior family and k on
    an R2-D2 cell, so a twin replay refuses to carry it over -- before writing anything."""
    monkeypatch.setattr(replay_module, "fit", _fit_reached)
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="half-Cauchy global scale"):
        run_replay(alpha_runs, out, family="aligned3", seed=1,
                   method="product/lengthscale_r2d2", ts=(6,), dry_run=dry_run)
    assert not out.exists()


def test_a_control_replay_passes_the_source_alpha_through(alpha_runs, tmp_path, monkeypatch):
    """Ruling R27's other branch: the cell refitted on its own run takes the run's alpha (D9)."""
    passed = []

    def first_fit(*args, **kwargs):
        passed.append(kwargs["alpha"])
        raise _FitReached

    monkeypatch.setattr(replay_module, "fit", first_fit)
    with pytest.raises(_FitReached):
        run_replay(alpha_runs, tmp_path / "replay", family="aligned3", seed=1,
                   method="product/lengthscale", ts=(6,))
    assert passed == [_SOURCE_ALPHA]


def test_each_r2d2_cell_is_replayed_on_its_half_cauchy_twin():
    """`TWINS` pairs each R2-D2 cell with the cell whose kernel and native site it shares."""
    assert len(TWINS) == 4 and REPLAY_T == (50, 100, 199)
    assert set(TWINS) | set(TWINS.values()) == {"/".join(key) for key in CELLS}
    for method, twin in TWINS.items():
        cell, other = CELLS[tuple(method.split("/"))], CELLS[tuple(twin.split("/"))]
        assert (cell.structure, cell.native_site) == (other.structure, other.native_site)
        assert cell.kernel is other.kernel and cell.pyro_model is not other.pyro_model


# --- compare ---

_DIAG_FIELDS = (
    "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
    "num_steps_mean",
    "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell", "n_eff_min_ell",
    "r_hat_max_global", "n_eff_min_global",
)
# The fake problem: D = 5, S = {0, 2}. The replayed fits rank S first (AP 1); the stored ones put
# coordinate 1 above both, so their AP is (1/2 + 2/3) / 2 = 7/12 and a swapped join would show.
_S = [0, 2]
_REPLAY_SCORES = [0.9, 0.1, 0.8, 0.2, 0.3]
_STORED_SCORES = [0.9, 0.95, 0.8, 0.1, 0.1]
# Every fake fit's budget (warmup/samples/thinning/depth), the stored runs' as the replays'.
_BUDGET = "16/16/4/6"
_COMMIT = "c0ffee"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _diagnostics(marker: float) -> dict[str, object]:
    """Twelve diagnostics tagged by `marker`, so a joined column says which fit it came from."""
    return {field: marker for field in _DIAG_FIELDS} | {"divergences": 0, "num_steps_mean": 63.0}


def _coords(ts: tuple[int, ...], scores: list[float], sobol: bool) -> list[dict[str, object]]:
    return [
        {"t": t, "i": i, "native_median": score, "p_active": float(i in _S),
         "sobol_hat": score if sobol else float("nan")}
        for t in ts
        for i, score in enumerate(scores)
    ]


def _stored_hash(family: str, method: str, seed: int) -> str:
    """A fake stored run's config_hash: any string that names the run will do."""
    return f"stored {family} {method} {seed}"


def _write_stored_run(root: Path, family: str, method: str, seed: int,
                      rows: dict[int, dict[str, object]], scores: list[float]) -> None:
    """A stored run as `compare` reads it: rows at the given t, coordinates, manifest and log."""
    run_dir = root / family / method.replace("/", "-") / f"seed{seed:02d}"
    rows_out = [{"t": t, "method": method, **row, "y_mean": 0.0, "y_std": 1.0}
                for t, row in rows.items()]
    _write_csv(run_dir / "iterations.csv", list(rows_out[0]), rows_out)
    _write_csv(run_dir / "coords.csv", ["t", "i", "native_median", "p_active", "sobol_hat"],
               _coords(tuple(rows), scores, sobol=False))
    (run_dir / "manifest.json").write_text(json.dumps({
        "family": family, "seed": seed, "method": method,
        "config_hash": _stored_hash(family, method, seed),
        "nuts": {"num_warmup": 16, "num_samples": 16, "thinning": 4, "max_tree_depth": 6,
                 "num_chains": 1},
        "objective": {"S": _S, "D": 5}, "env": {"jax_device": "stored-gpu"}, "resumed": [],
    }))
    (run_dir / "log.txt").write_text(
        "".join(f"t={t} method={method} status={row['status']}\n" for t, row in rows.items())
    )


def _write_replay(root: Path, family: str, method: str, source: str, seed: int,
                  rows: dict[int, dict[str, object]], scores: list[float], *,
                  commit: str = _COMMIT, dirty: bool | None = False,
                  resumed: list[dict[str, object]] | None = None) -> Path:
    """A replay directory as `run_replay` leaves it: `replay.csv`, coordinates and a manifest
    naming its source run by that run's config_hash, and the session's commit and checkout."""
    replay_dir = root / family / method.replace("/", "-") / f"seed{seed:02d}"
    rows_out = [
        {"t": t, "n": t, "method": method, "source_method": source, "family": family,
         "seed": seed, "nuts_seed": 7, "budget": _BUDGET, **row, "readout_wall_s": 1.0,
         "y_mean": 0.0, "y_std": 1.0, "r2d2_r2": 0.4, "first_order_r2": 0.6}
        for t, row in rows.items()
    ]
    _write_csv(replay_dir / "replay.csv", list(replay_module._REPLAY_FIELDS), rows_out)
    _write_csv(replay_dir / "coords.csv", ["t", "i", "native_median", "p_active", "sobol_hat"],
               _coords(tuple(rows), scores, sobol=True))
    (replay_dir / "manifest.json").write_text(json.dumps({
        "config_hash": f"replay {family} {method} {seed}", "created": "2026-10-05T12:00:00",
        "git": {"commit": commit, "dirty": dirty}, "resumed": resumed or [],
        "source": {"config_hash": _stored_hash(family, source, seed), "path": "runs",
                   "commit": "a51a4b9", "devices": ["stored-gpu"]},
    }))
    return replay_dir


def _fake_stored_run(root: Path, method: str, seed: int, ts: tuple[int, ...]) -> None:
    _write_stored_run(root, "aligned3", method, seed, {
        t: {"status": "excluded", "reason": "r_hat_max 1.5 > 1.1",
            **_diagnostics(1.0 + seed / 10 + t / 1000), "fit_wall_s": 10.0 + t}
        for t in ts
    }, _STORED_SCORES)


def _fake_replay(root: Path, method: str, source: str, seed: int, ts: tuple[int, ...],
                 marker: float, *, status: str = "ok", scores: list[float] = _REPLAY_SCORES,
                 fit_wall_s: dict[int, float] | None = None) -> None:
    _write_replay(root, "aligned3", method, source, seed, {
        t: {"status": status, "reason": "", **_diagnostics(marker + seed / 10 + t / 1000),
            "fit_wall_s": marker + t if fit_wall_s is None else fit_wall_s[t],
            "device": "replay-gpu"}
        for t in ts
    }, scores)


def test_compare_joins_one_row_per_cell_family_seed_and_t(tmp_path):
    """Two tiny trees: a stored half-Cauchy cell at seeds 0 and 1, its R2-D2 twin replayed on
    both, and its control refit on seed 0 only. Each replayed fit is one joined row, beside its
    source's stored fit of the same (seed, t) and, for the R2-D2 cell, the control where present."""
    runs, replay, out = tmp_path / "runs", tmp_path / "replay", tmp_path / "report"
    ts = (5, 6)
    for seed in (0, 1):
        _fake_stored_run(runs, "additive/amplitude", seed, ts)
        _fake_replay(replay, "additive/amplitude_r2d2", "additive/amplitude", seed, ts, 2.0)
    _fake_replay(replay, "additive/amplitude", "additive/amplitude", 0, ts, 3.0)

    assert main(["compare", "--runs", str(runs), "--replay", str(replay), "--out", str(out)]) == 0
    assert compare(runs, replay, tmp_path / "again") == tmp_path / "again"

    joined = _read_rows(out / "tables" / "replay_vs_stored.csv")
    keys = [(r["cell"], r["family"], int(r["seed"]), int(r["t"])) for r in joined]
    assert sorted(keys) == sorted(
        [("additive/amplitude_r2d2", "aligned3", s, t) for s in (0, 1) for t in ts]
        + [("additive/amplitude", "aligned3", 0, t) for t in ts]
    )
    for r in joined:
        seed, t = int(r["seed"]), int(r["t"])
        assert float(r["stored_r_hat_max"]) == 1.0 + seed / 10 + t / 1000
        assert r["stored_status"] == "excluded" and r["stored_device"] == "stored-gpu"
        assert float(r["ap_native"]) == 1.0
        assert float(r["stored_ap_native"]) == pytest.approx(7 / 12)
        assert math.isnan(float(r["stored_ap_sobol"]))  # no Sobol readout at these t
        if r["cell"] == "additive/amplitude":
            assert r["role"] == "control" and r["control_status"] == ""
            assert float(r["r_hat_max"]) == 3.0 + seed / 10 + t / 1000
        else:
            assert r["role"] == "twin" and r["source_method"] == "additive/amplitude"
            assert float(r["r_hat_max"]) == 2.0 + seed / 10 + t / 1000
            if seed == 0:
                assert float(r["control_r_hat_max"]) == 3.0 + t / 1000
                assert r["control_device"] == "replay-gpu"
            else:
                assert math.isnan(float(r["control_r_hat_max"])) and r["control_status"] == ""
        # The budgets the harness and the cost pairs are formed at: the stored fit's and the replay's.
        assert r["budget"] == r["stored_budget"] == _BUDGET

    identification = _read_rows(out / "tables" / "identification.csv")
    assert len(identification) == 6 + 4  # six replayed fits, four stored ones
    gate = _read_rows(out / "tables" / "gate_by_cell_t.csv")
    r2d2_all = next(
        r for r in gate if r["cell"] == "additive/amplitude_r2d2" and r["t"] == "all"
    )
    assert int(r2d2_all["fits"]) == 4 and float(r2d2_all["exclusion_rate"]) == 0.0
    assert float(r2d2_all["stored_exclusion_rate"]) == 1.0
    assert float(r2d2_all["frac_at_tree_cap"]) == 1.0
    assert (out / "tables" / "cost.csv").exists()
    report = (out / "REPORT.md").read_text()
    assert "G2" in report and "4 of 960" in report and "2 of 240" in report


@pytest.mark.parametrize("damage", ["another_source", "no_manifest"])
def test_compare_refuses_a_replay_not_tied_to_the_stored_run_it_joins(tmp_path, damage):
    """A replay directory names its source run by config_hash; joined to a stored run under
    `--runs` with another, its fits would be compared with a fit of other data."""
    runs, replay = tmp_path / "runs", tmp_path / "replay"
    _fake_stored_run(runs, "additive/amplitude", 0, (5, 6))
    _fake_replay(replay, "additive/amplitude_r2d2", "additive/amplitude", 0, (5, 6), 2.0)
    manifest = replay / "aligned3" / "additive-amplitude_r2d2" / "seed00" / "manifest.json"
    if damage == "another_source":
        _edit_json(manifest, lambda data: data["source"].update(config_hash="another run"))
        match = "not refits of that run's data"
    else:
        manifest.unlink()
        match = "no manifest.json"
    with pytest.raises(ValueError, match=match):
        compare(runs, replay, tmp_path / "report")
    assert not (tmp_path / "report").exists()


# --- G2's verdict on a whole tree ---

# The smallest complete tree G2 reads: the controls' two families, which then are the R2-D2 cells'
# too (their twins' stored runs are in those two only), ten seeds, three t -- 60 fits per cell.
_G2_FAMILIES = ("aligned10", "decoupled")
_G2_FITS = [(family, seed, t) for family in _G2_FAMILIES for seed in range(10) for t in REPLAY_T]
# The stored twins at the replay's t, as the stored runs have them (final review, P2): E* = 0.7
# and N* = 14 come from the additive lengthscale twin.
_STORED_TWINS = {
    "additive/amplitude": {"excluded": 60, "n_eff_min": 5.8, "r_hat_max": 1.5},
    "product/amplitude": {"excluded": 60, "n_eff_min": 6.2, "r_hat_max": 1.4},
    "additive/lengthscale": {"excluded": 42, "n_eff_min": 14.0, "r_hat_max": 1.2},
    "product/lengthscale": {"excluded": 35, "n_eff_min": 22.7, "r_hat_max": 1.1},
}
# Every R2-D2 cell at 0.5 and 20, inside the bar; t = 199 at 11 s against its control's 10 s.
_R2D2_FITS = {"excluded": 30, "n_eff_min": 20.0, "r_hat_max": 1.05,
              "fit_wall_s": {50: 30.0, 100: 30.0, 199: 11.0}}
_CONTROL_WALL_S = {50: 10.0, 100: 10.0, 199: 10.0}


def _g2_fit(spec: dict[str, object], index: int, t: int) -> dict[str, object]:
    """The `index`-th fit of a method in `_G2_FITS` order, at t: the first `excluded` fits are
    excluded; every r_hat_max and n_eff_min is the spec's, so its medians are exactly those."""
    r_hat, n_eff = spec["r_hat_max"], spec["n_eff_min"]
    wall = spec.get("fit_wall_s", _CONTROL_WALL_S)
    return {
        "status": "excluded" if index < spec["excluded"] else "ok", "reason": "",
        "r_hat_max": r_hat, "r_hat_median": 1.0, "frac_r_hat_below_1_05": 0.5, "n_eff_min": n_eff,
        "divergences": 0, "num_steps_mean": 63.0,
        "r_hat_max_native": r_hat, "n_eff_min_native": n_eff, "r_hat_max_ell": r_hat,
        "n_eff_min_ell": n_eff, "r_hat_max_global": r_hat, "n_eff_min_global": n_eff,
        "fit_wall_s": wall[t] if isinstance(wall, dict) else wall,
    }


def _g2_tree(root: Path, *, stored=None, fits=None, drop=()) -> tuple[Path, Path]:
    """A complete G2 tree under `root`: (the stored root, the replay root).

    The four half-Cauchy twins stored in `_G2_FAMILIES`, seeds 0-9; each R2-D2 cell replayed on
    its twin's runs and each twin on its own (the control, reproducing its stored fits exactly),
    at every t of `REPLAY_T`, one device, one budget, one clean commit. `stored[twin]` and
    `fits[method]` update a method's spec (`excluded` of its 60 fits, `n_eff_min`, `r_hat_max`,
    `fit_wall_s`, and for a replay `device`, `budget`, `commit`, `dirty`, `resumed`); `drop`
    lists replay directories, (method, family, seed), left out.
    """
    runs, replay = root / "runs", root / "replay"
    stored_specs = {twin: {**spec, **(stored or {}).get(twin, {})}
                    for twin, spec in _STORED_TWINS.items()}
    for twin, spec in stored_specs.items():
        for family in _G2_FAMILIES:
            for seed in range(10):
                _write_stored_run(runs, family, twin, seed, {
                    t: _g2_fit(spec, _G2_FITS.index((family, seed, t)), t) for t in REPLAY_T
                }, _STORED_SCORES)
    methods = [(cell, twin, {**_R2D2_FITS}) for cell, twin in TWINS.items()]
    methods += [(twin, twin, {**spec, "fit_wall_s": _CONTROL_WALL_S})
                for twin, spec in stored_specs.items()]
    for method, source, spec in methods:
        spec |= (fits or {}).get(method, {})
        for family in _G2_FAMILIES:
            for seed in range(10):
                if (method, family, seed) in drop:
                    continue
                rows = {
                    t: _g2_fit(spec, _G2_FITS.index((family, seed, t)), t)
                    | {"device": spec.get("device", "replay-gpu")}
                    for t in REPLAY_T
                }
                replay_dir = _write_replay(
                    replay, family, method, source, seed, rows, _STORED_SCORES,
                    commit=spec.get("commit", _COMMIT), dirty=spec.get("dirty", False),
                    resumed=spec.get("resumed"),
                )
                if "budget" in spec:  # a refit at another budget than the stored fits'
                    path = replay_dir / "replay.csv"
                    replayed_rows = _read_rows(path)
                    for row in replayed_rows:
                        row["budget"] = spec["budget"]
                    _write_csv(path, list(replayed_rows[0]), replayed_rows)
    return runs, replay


def _verdict_line(report: str) -> str:
    return next(line for line in report.splitlines() if line.startswith("**G2: "))


_ALL_SEEDS_OF = [(family, seed) for family in _G2_FAMILIES for seed in range(10)]

# Each verdict branch, and each INCONCLUSIVE cause, on the complete tree changed in one place:
# (stored, fits, drop, the verdict's first word, a fragment it must hold). Several sit exactly on
# a threshold: 54 of 60 fits is an exclusion rate of 0.90, 42 of 60 is E*, 36 of 60 the bar's
# floor of 0.60, 9 of 60 against 0 a |delta| of 0.15, and 15 s against 10 s a ratio of 1.5.
_G2_CASES = {
    "pass": ({}, {}, (), "PASS", "PASS: proceed to stage 3"),
    "pass_exactly_at_e_star_and_n_star": (
        {}, {"additive/lengthscale_r2d2": {"excluded": 42, "n_eff_min": 14.0}}, (), "PASS", ""),
    "pass_bar_floor_when_the_twins_run_below_it": (
        {"additive/lengthscale": {"excluded": 30, "n_eff_min": 20.0},
         "product/lengthscale": {"excluded": 24, "n_eff_min": 25.0}},
        {"additive/lengthscale_r2d2": {"excluded": 36, "n_eff_min": 16.0}}, (), "PASS", ""),
    "above_the_bar_floor": (
        {"additive/lengthscale": {"excluded": 30, "n_eff_min": 20.0},
         "product/lengthscale": {"excluded": 24, "n_eff_min": 25.0}},
        {"additive/lengthscale_r2d2": {"excluded": 37}}, (), "INCONCLUSIVE",
        "additive/lengthscale_r2d2 between the PASS bar and FAIL"),
    "fail_at_exclusion_0_90": (
        {}, {"product/amplitude_r2d2": {"excluded": 54}}, (), "FAIL", "amplitude signature"),
    "fail_at_n_eff_10": (
        {}, {"additive/amplitude_r2d2": {"n_eff_min": 10.0}}, (), "FAIL", "amplitude signature"),
    "between_bar_and_fail": (
        {}, {"additive/amplitude_r2d2": {"excluded": 48, "n_eff_min": 12.0}}, (),
        "INCONCLUSIVE", "additive/amplitude_r2d2 between the PASS bar and FAIL"),
    "stop_harness": (
        {}, {"additive/amplitude": {"excluded": 50}}, (), "STOP", "harness -- the control"),
    "harness_holds_at_0_15": (
        {"product/lengthscale": {"excluded": 0}}, {"product/lengthscale": {"excluded": 9}}, (),
        "PASS", ""),
    "stop_harness_past_0_15": (
        {"product/lengthscale": {"excluded": 0}}, {"product/lengthscale": {"excluded": 10}}, (),
        "STOP", "harness -- the control"),
    "stop_cost": (
        {}, {"product/lengthscale_r2d2": {"fit_wall_s": {50: 30.0, 100: 30.0, 199: 16.0}}}, (),
        "STOP", "cost -- at t = 199"),
    "cost_holds_at_1_5": (
        {}, {"product/lengthscale_r2d2": {"fit_wall_s": {50: 30.0, 100: 30.0, 199: 15.0}}}, (),
        "PASS", ""),
    "missing_r2d2_fits": (
        {}, {}, (("product/amplitude_r2d2", "decoupled", 3),), "INCONCLUSIVE",
        "product/amplitude_r2d2 lacks 3 of its 60 fits (first: decoupled/seed03/t=50, "
        "decoupled/seed03/t=100, decoupled/seed03/t=199)"),
    "a_missing_r2d2_cell_is_no_fail": (
        {}, {"product/amplitude_r2d2": {"excluded": 60}},
        tuple(("additive/amplitude_r2d2", family, seed) for family, seed in _ALL_SEEDS_OF),
        "INCONCLUSIVE", "additive/amplitude_r2d2 lacks 60 of its 60 fits"),
    "missing_control_cell": (
        {}, {}, tuple(("additive/lengthscale", family, seed) for family, seed in _ALL_SEEDS_OF),
        "INCONCLUSIVE",
        "additive/lengthscale (control) lacks 60 of its 60 fits at the stored budget"),
    "no_same_device_cost_pair": (
        {}, {"product/amplitude": {"device": "other-gpu"}}, (), "INCONCLUSIVE",
        "product/amplitude_r2d2 has no cost pair at t = 199 with its control on one device"),
    "control_at_another_budget": (
        {}, {"additive/lengthscale": {"budget": "16/16/4/8"}}, (), "INCONCLUSIVE",
        "additive/lengthscale (control) lacks 60 of its 60 fits at the stored budget"),
    "two_commits": (
        {}, {"product/amplitude_r2d2": {"commit": "deadbeef"}}, (), "INCONCLUSIVE",
        "the replays ran at 2 commits (c0ffee, deadbeef)"),
    "a_resume_at_another_commit": (
        {}, {"additive/amplitude": {"resumed": [{"time": "t", "commit": "deadbeef",
                                                  "dirty": False}]}}, (), "INCONCLUSIVE",
        "the replays ran at 2 commits (c0ffee, deadbeef)"),
    "a_dirty_session": (
        {}, {"additive/amplitude": {"dirty": True}}, (), "INCONCLUSIVE",
        "20 session(s) ran on a dirty checkout"),
    "a_resume_with_no_dirty_state": (
        {}, {"product/lengthscale": {"resumed": [{"time": "t", "commit": _COMMIT}]}}, (),
        "INCONCLUSIVE", "20 session(s) recorded no dirty state"),
}


@pytest.mark.parametrize("stored, fits, drop, kind, fragment", list(_G2_CASES.values()),
                         ids=list(_G2_CASES))
def test_g2_reads_pass_or_fail_only_on_a_complete_tree_at_one_clean_commit(
    tmp_path, stored, fits, drop, kind, fragment
):
    """G2's verdict (plan, Task 11; rulings R28, R31a, R31b) on the smallest complete tree,
    changed in one place per case: a stop first (harness, then cost), then INCONCLUSIVE for
    anything missing -- an R2-D2 cell's fits, a control cell, a same-device cost pair, a control
    at its stored budget -- or for sessions at two commits or on a dirty or unrecorded checkout;
    only then FAIL, PASS, or INCONCLUSIVE between the two. The tables print whatever there is."""
    runs, replay = _g2_tree(tmp_path, stored=stored, fits=fits, drop=drop)
    compare(runs, replay, tmp_path / "report")
    report = (tmp_path / "report" / "REPORT.md").read_text()
    verdict = _verdict_line(report)
    assert verdict.startswith(f"**G2: {kind}") and fragment in verdict
    assert "## Completeness" in report and "## Provenance" in report


def test_g2_prints_its_pass_bar_with_the_numbers_it_comes_from(tmp_path):
    """E* and N* (ruling R31b) are computed from the stored fits the lengthscale R2-D2 cells
    are joined to, and printed with that derivation."""
    runs, replay = _g2_tree(tmp_path)
    compare(runs, replay, tmp_path / "report")
    report = (tmp_path / "report" / "REPORT.md").read_text()
    assert "E* = max(0.6, the half-Cauchy lengthscale twins' stored exclusion rates) = 0.7" in report
    assert "N* = min(16, their stored median n_eff_min) = 14" in report
    assert ("additive/lengthscale: exclusion rate 0.7, median n_eff_min 14, over 60 fits; "
            "product/lengthscale: exclusion rate 0.583, median n_eff_min 22.7, over 60 fits"
            in report)


def test_the_cost_criterion_reads_t_199_alone(tmp_path):
    """Ruling R28: G2's cost criterion pairs each R2-D2 fit with its twin's control refit at
    t = 199 only, when every replay process has already fitted its cell once. Here the t = 50 and
    t = 100 pairs cost 3 times the control (a process's one-time start-up, in the replay) and the
    t = 199 pair 1.1 times: read over every t, the median ratio of 3 would stop the gate on cost.
    The control reproduces its stored fits exactly, so the harness holds and the verdict turns on
    the cost reading and the R2-D2 cell alone (its median n_eff_min, about 2, reads FAIL). The
    tree is complete, as a FAIL needs: twenty t = 199 pairs per cell."""
    runs, replay = _g2_tree(tmp_path, fits={"additive/amplitude_r2d2": {"n_eff_min": 2.0}})
    out = tmp_path / "report"

    compare(runs, replay, out)

    lines = (out / "REPORT.md").read_text().splitlines()
    verdict = next(line for line in lines if line.startswith("**G2: "))
    assert verdict.startswith("**G2: FAIL") and "cost" not in verdict
    (cost_row,) = [
        line for line in lines if line.startswith("| additive/amplitude_r2d2 | replay-gpu")
    ]
    cell, device, pairs, mine, theirs, ratio, holds = (
        column.strip() for column in cost_row.strip("|").split("|")
    )
    assert (pairs, mine, theirs, ratio, holds) == ("20", "11", "10", "1.1", "yes")
    # cost.csv keeps every t for the record; only the criterion reads t = 199 alone.
    cost = _read_rows(out / "tables" / "cost.csv")
    assert {r["t"] for r in cost if (r["kind"], r["method"]) == ("replay", cell)} == {
        "50", "100", "199", "all"
    }


def _probe_tree(root: Path, fits=None) -> tuple[Path, Path]:
    """The GPU probe's tree: aligned10 seed 0, t = 100 and 199, all eight cells (plan, Task 11)."""
    runs, replay = root / "runs", root / "replay"
    ts = (100, 199)
    for twin, spec in _STORED_TWINS.items():
        _write_stored_run(runs, "aligned10", twin, 0,
                          {t: _g2_fit(spec, 0, t) for t in ts}, _STORED_SCORES)
    methods = [(cell, twin, {**_R2D2_FITS}) for cell, twin in TWINS.items()]
    methods += [(twin, twin, {**spec, "fit_wall_s": _CONTROL_WALL_S})
                for twin, spec in _STORED_TWINS.items()]
    for method, source, spec in methods:
        spec |= (fits or {}).get(method, {})
        _write_replay(replay, "aligned10", method, source, 0, {
            t: _g2_fit(spec, 0, t) | {"device": spec.get("device", "replay-gpu")} for t in ts
        }, _STORED_SCORES)
    return runs, replay


@pytest.mark.parametrize(
    "fits, reading",
    [
        ({}, "**Cost criterion: every R2-D2 cell at or below 1.5 times its control.**"),
        ({"product/amplitude_r2d2": {"fit_wall_s": {100: 30.0, 199: 16.0}}},
         "**Cost criterion: above 1.5 times its control for product/amplitude_r2d2 (1.6 on "
         "replay-gpu) -- ask before the bulk array.**"),
        ({"product/amplitude": {"device": "other-gpu"}},
         "**Cost criterion: no same-device pair at t = 199 for product/amplitude_r2d2, whose "
         "cost is unread.**"),
    ],
    ids=["holds", "exceeded", "unpaired"],
)
def test_the_probe_reads_the_cost_criterion_alone(tmp_path, fits, reading):
    """`compare --probe` (final review I-3): the probe's sixteen fits are read for the cost
    criterion only. No G2 headline -- PASS, FAIL or STOP -- is printed, whatever the tables hold,
    and every other section is labeled as information."""
    runs, replay = _probe_tree(tmp_path, fits)
    out = tmp_path / "report"
    argv = ["compare", "--probe", "--runs", str(runs), "--replay", str(replay), "--out", str(out)]
    assert main(argv) == 0
    report = (out / "REPORT.md").read_text()
    headlines = [line for line in report.splitlines() if line.startswith("**")]
    assert headlines == [reading]
    assert not any(word in reading for word in ("PASS", "FAIL", "STOP"))
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert sections[0] == "## The probe's reading: cost at t = 199"
    assert all(section.endswith("(information, not a G2 reading)")
               for section in sections[1:] if section != "## Tables")
    assert "## Completeness" not in report and "## Verdict" not in report


@pytest.mark.parametrize(
    "exclusion, n_eff, reading",
    [
        (0.90, 20.0, "FAIL"),  # FAIL's exclusion bound is inclusive
        (np.nextafter(0.90, 0.0), 20.0, "neither"),
        (0.5, 10.0, "FAIL"),  # and so is its n_eff_min bound
        (0.5, np.nextafter(10.0, 11.0), "neither"),
        (0.7, 20.0, "PASS"),  # E* = 0.7 passes
        (np.nextafter(0.7, 1.0), 20.0, "neither"),
        (0.5, 14.0, "PASS"),  # N* = 14 passes
        (0.5, np.nextafter(14.0, 0.0), "neither"),
        (float("nan"), 20.0, "neither"),
    ],
)
def test_each_cell_threshold_is_inclusive(exclusion, n_eff, reading):
    """An R2-D2 cell exactly at 0.90 or at 10 fails and exactly at E* or N* passes; one ulp on
    the other side does not. The bar here is the stored data's, E* = 0.7 and N* = 14."""
    assert replay_module._cell_reading(exclusion, n_eff, 0.7, 14.0) == reading


@pytest.mark.parametrize(
    "deltas, holds",
    [
        ((0.15, 0.0, 0.0), True),
        ((np.nextafter(0.15, 1.0), 0.0, 0.0), False),
        ((0.0, 0.10, 0.0), True),
        ((0.0, np.nextafter(0.10, 1.0), 0.0), False),
        ((0.0, 0.0, 0.10), True),
        ((0.0, 0.0, np.nextafter(0.10, 1.0)), False),
        ((0.0, float("nan"), 0.0), False),
    ],
    ids=["exclusion_0_15", "exclusion_past", "r_hat_0_10", "r_hat_past", "ap_0_10", "ap_past",
         "nan"],
)
def test_each_harness_tolerance_is_inclusive(deltas, holds):
    """Ruling R31a: |delta| of the exclusion rates at 0.15, and of the r_hat_max and AP_native
    medians at 0.10, still hold; one ulp more does not, and a statistic missing holds nothing."""
    assert replay_module._harness_holds(*deltas) is holds


def test_the_cost_ratio_holds_at_1_5():
    assert replay_module._cost_holds(1.5) is True
    assert replay_module._cost_holds(float(np.nextafter(1.5, 2.0))) is False


def test_g2s_design_is_the_replay_arrays():
    """The design `compare` completes against is the one slurm/r2d2_replay.sbatch runs: its
    control families, its ten seeds, its three t, and the full array's 960 and 240 fits."""
    sbatch = (Path(__file__).resolve().parent.parent / "slurm" / "r2d2_replay.sbatch").read_text()
    families = re.search(r"^FAMILIES=\((.*)\)$", sbatch, re.M).group(1).split()
    controls = re.search(r"^CONTROL_FAMILIES=\((.*)\)$", sbatch, re.M).group(1).split()
    seeds = re.search(r"^for seed in ([\d ]+); do$", sbatch, re.M).group(1).split()
    ts = re.search(r'--seed "\$seed" --cell "\$cell" --t ([\d,]+)', sbatch).group(1)
    assert tuple(controls) == replay_module._G2_CONTROL_FAMILIES
    assert tuple(int(seed) for seed in seeds) == replay_module._G2_SEEDS
    assert ts == ",".join(str(t) for t in REPLAY_T)
    fits = len(REPLAY_T) * len(replay_module._G2_SEEDS)
    assert replay_module._G2_R2D2_FITS == len(TWINS) * len(families) * fits
    assert replay_module._G2_CONTROL_FITS == len(TWINS) * len(controls) * fits
