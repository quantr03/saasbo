"""Tests for `experiments.run_bo`'s `resolve_config` and `experiments.runlog`'s manifest and lock.

What a run directory has to be able to prove, and what these tests check it proves:

- that the seven methods were run under the *same* settings, since a difference in any other
  setting would be an alternative explanation for every regret difference the thesis reports.
  `test_resolve_config_differs_only_in_method` builds all seven from one command line and
  compares each to `dataclasses.replace(reference, method=...)`, so a flag that leaked a
  per-method default would fail; `test_every_flag_reaches_the_resolved_config` is its mirror --
  every flag set away from its default at once, compared to one hand-written `RunConfig`, so a
  flag that reaches nothing fails too.
- that every method started from the same points *and* maximized the acquisition the same way.
  `initial_design` takes no method, so the end-to-end runs here are the check that `run` really
  calls it before the method matters, and `propose_for` is compared across all seven: the
  maximizer is the other half of "the methods differ in `method` and in nothing else".
- that the manifest describes the run rather than the reader's environment: it round-trips into
  the `RunConfig` the run used, its `acquisition_constants` name the whole operating point the
  maximizer ran at -- BoTorch's fixed choices as well as the five a flag reaches -- and a resume
  neither rewrites what it recorded at the start nor raises when the environment underneath it
  has moved.

Runs are `dsp_map` at D = 5 with the loop tests' own reduced acquisition budget: nothing here is
about the surrogate, only about what is written beside it.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import pytest
import torch

from experiments import run_bo as run_bo_module
from experiments.run_bo import METHODS, propose_for, resolve_config, run
from experiments.runlog import RunConfig, config_hash
from sagp.bo import initial_design, propose_ei
from sagp.diagnostics import DiagThresholds
from sagp.gp import NUTSConfig
from sagp.references import propose_sobol
from synthobj.families import make_family

_LOOP_KW = dict(raw_samples=64, num_restarts=1, sobol_every=5)
_T = 10
_N_INIT = 5
# The run seed is deliberately not the objective's: the manifest records both, and a manifest that
# confused them would still look right if the two were equal.
_SEED = 3
_OBJECTIVE_SEED = 0


def _objective():
    """The loop tests' fixed D = 5 problem, built at a seed different from the runs' own."""
    return make_family("aligned3", _OBJECTIVE_SEED, D=5)


def _parse(*argv: str):
    return run_bo_module._build_parser().parse_args(list(argv))


def _config_from_manifest(manifest: dict) -> RunConfig:
    """The `RunConfig` a manifest describes, with JSON's own lossiness undone by hand.

    JSON has neither tuples nor dataclasses, so exactly three fields come back in another shape.
    Written here rather than as a `from_manifest` in `runlog.py` because nothing in the study
    reads a manifest back -- resume compares the hash and never reconstructs -- and a constructor
    whose only caller was this test would be a claim about the format that only this test could
    break.
    """
    fields = {f.name: manifest[f.name] for f in dataclasses.fields(RunConfig)}
    fields["nuts"] = NUTSConfig(**fields["nuts"])
    fields["thresholds"] = DiagThresholds(**fields["thresholds"])
    fields["ell_prior"] = tuple(fields["ell_prior"])
    return RunConfig(**fields)


# --- the resolved configuration ---


def test_resolve_config_differs_only_in_method(tmp_path):
    base = [
        "--family", "aligned10", "--seed", "3", "--D", "100", "--T", "200",
        "--n-init", "20", "--out", str(tmp_path),
    ]
    configs = {method: resolve_config(_parse(*base, "--cell", method)) for method in METHODS}
    assert list(configs) == METHODS and len(configs) == 7

    reference = configs["sobol"]
    for method, cfg in configs.items():
        assert cfg == dataclasses.replace(reference, method=method)

    # And the defaults the study is run at, spelled out: these are the numbers a SLURM array gets
    # when it passes nothing but --family, --seed, --cell and --out.
    assert (reference.T, reference.n_init, reference.D) == (200, 20, 100)
    assert reference.nuts == NUTSConfig(512, 256, 16)
    assert reference.thresholds == DiagThresholds()
    assert (reference.sobol_every, reference.sobol_n) == (25, 2048)
    # Papenmeier et al. (2025)'s maximizer, at the operating point every method shares.
    assert (reference.raw_samples, reference.num_restarts) == (512, 5)
    assert reference.sample_around_best_sigma == 1e-3
    assert (reference.batch_limit, reference.maxiter) == (1, 200)
    assert reference.alpha is None and reference.fixed_noise is None
    assert reference.noiseless is False


def test_every_flag_reaches_the_resolved_config(tmp_path):
    cfg = resolve_config(_parse(
        "--family", "decoupled", "--seed", "7", "--cell", "additive/amplitude",
        "--out", str(tmp_path), "--T", "40", "--n-init", "8", "--D", "12",
        "--alpha", "0.25", "--fixed-noise", "0.01", "--noiseless",
        "--sobol-every", "3", "--nuts", "8,16,4",
        "--raw-samples", "64", "--num-restarts", "2", "--sample-around-best-sigma", "0.01",
        "--batch-limit", "4", "--maxiter", "50",
    ))
    # One equality rather than seventeen: a flag wired to the wrong field fails here too, and
    # `NUTSConfig(8, 16, 4)` is what pins --nuts to warmup/samples/thinning with tree depth 6.
    assert cfg == RunConfig(
        family="decoupled", seed=7, D=12, method="additive/amplitude", T=40, n_init=8,
        alpha=0.25, fixed_noise=0.01, noiseless=True, nuts=NUTSConfig(8, 16, 4),
        raw_samples=64, num_restarts=2, sample_around_best_sigma=0.01, batch_limit=4,
        maxiter=50, sobol_every=3, out_dir=str(tmp_path),
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("raw_samples", 1024),
        ("num_restarts", 9),
        ("sample_around_best_sigma", 1e-2),
        ("batch_limit", 4),
        ("maxiter", 400),
    ],
)
def test_the_config_hash_moves_with_every_maximizer_setting(field, value):
    # The five sizes are hashed like every other setting, which is what makes a run at a different
    # operating point a different run: resume refuses it rather than splicing the two halves.
    base = RunConfig(family="aligned10", seed=3, D=100, method="sobol", T=200, n_init=20)
    assert getattr(base, field) != value
    assert config_hash(dataclasses.replace(base, **{field: value})) != config_hash(base)


def test_every_method_gets_the_same_design_and_maximizer(tmp_path):
    """The two things a method may not choose for itself: where it starts, and how it maximizes.

    `propose_for` is compared across all seven rather than run, because what has to hold is that
    the six model-based methods hand `optimize_acqf` the *same* operating point -- a maximizer
    budget that varied by method would be an alternative explanation for every regret difference
    the thesis reports. `sobol` is the one exception by construction: it fits nothing, so it
    walks the design's own sequence instead of maximizing anything.
    """
    base = RunConfig(
        family="aligned3", seed=_SEED, D=5, method="sobol", T=_T, n_init=_N_INIT,
        out_dir=str(tmp_path), **_LOOP_KW,
    )
    shared = dict(
        raw_samples=64, num_restarts=1, sample_around_best_sigma=1e-3, batch_limit=1, maxiter=200
    )
    for method in METHODS:
        proposer = propose_for(dataclasses.replace(base, method=method))
        if method == "sobol":
            assert proposer.func is propose_sobol
            assert np.array_equal(proposer.keywords["sequence"], initial_design(5, _T, _SEED))
            continue
        assert proposer.func is propose_ei
        assert proposer.keywords == shared

    design = initial_design(5, _N_INIT, _SEED)
    # One method that fits nothing and the two MAP references, whose fits are milliseconds. The
    # remaining four would each buy a NUTS run and no new information: `initial_design` takes
    # (D, n_init, seed) and no method, and the loop calls it once, before `cfg.method` has been
    # used for anything at all.
    for method in ("sobol", "dsp_map", "oracle_S"):
        run_dir = run(
            _objective(), method, seed=_SEED, T=_T, n_init=_N_INIT,
            out_dir=tmp_path / method, **_LOOP_KW,
        )
        with np.load(run_dir / "checkpoint.npz") as data:
            assert np.array_equal(data["X"][:_N_INIT], design)
    assert "method" not in inspect.signature(initial_design).parameters


# --- the manifest and the lock file ---


def _manifest_run(out_dir: Path) -> Path:
    return run(
        _objective(), "dsp_map", seed=_SEED, T=_T, n_init=_N_INIT, out_dir=out_dir, **_LOOP_KW
    )


def test_the_manifest_round_trips_into_the_config_the_run_used(tmp_path):
    run_dir = _manifest_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    expected = RunConfig(
        family="aligned3", seed=_SEED, D=5, method="dsp_map", T=_T, n_init=_N_INIT,
        out_dir=str(tmp_path), **_LOOP_KW,
    )
    assert _config_from_manifest(manifest) == expected
    assert manifest["config_hash"] == config_hash(expected)


def test_the_manifest_records_the_code_the_environment_and_the_problem(tmp_path):
    objective = _objective()
    run_dir = _manifest_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for key in ("git", "versions", "env", "objective", "acquisition_constants"):
        assert key in manifest
    # Nothing is copied from a reference implementation any more, so there is no file digest to
    # record; a manifest that still carried one would be describing a reference path this run
    # never took.
    assert "reference_sha256" not in manifest

    assert set(manifest["git"]) == {"commit", "dirty"}
    assert manifest["versions"]["python"] == platform.python_version()
    assert manifest["versions"]["numpyro"] == importlib.metadata.version("numpyro")
    assert manifest["versions"]["botorch"] == importlib.metadata.version("botorch")
    assert set(manifest["versions"]) == {
        "python", "jax", "jaxlib", "numpyro", "numpy", "scipy",
        "torch", "gpytorch", "botorch", "linear_operator",
    }
    assert set(manifest["env"]) == {
        "XLA_FLAGS", "OMP_NUM_THREADS", "platform", "machine", "processor", "cpu_count",
        "torch_num_threads", "jax_device",
    }
    assert manifest["env"]["machine"] == platform.machine()
    assert manifest["env"]["cpu_count"] == os.cpu_count()
    assert manifest["env"]["OMP_NUM_THREADS"] == os.environ.get("OMP_NUM_THREADS")
    # torch's own thread count, because the acquisition and the MAP references run on it: BoTorch
    # is where this run's linear algebra happens and XLA's thread settings say nothing about it.
    assert manifest["env"]["torch_num_threads"] == torch.get_num_threads()
    # The device the chain runs on, since sagp takes a GPU when one is visible: a resume on the
    # other kind of device is not bit-identical to the run it continues.
    assert manifest["env"]["jax_device"] == jax.devices()[0].device_kind

    # What BoTorch holds fixed below every flag, plus the five sizes a flag does reach: together
    # the acquisition maximizer's whole operating point for this run. `prob_perturb` is 1.0 here
    # because BoTorch perturbs every coordinate below D = 20.
    assert manifest["acquisition_constants"] == {
        "acquisition": "LogExpectedImprovement", "q": 1, "sample_around_best": True,
        "best_pct": 5.0, "cholesky_max_tries": 9, "prob_perturb": 1.0,
        "raw_samples": 64, "num_restarts": 1, "sample_around_best_sigma": 1e-3,
        "batch_limit": 1, "maxiter": 200,
    }
    # The objective's seed is its own, not the run's: a manifest that confused the two would
    # describe a different problem than the one the regret column was computed against.
    assert manifest["objective"] == {
        "family": "aligned3",
        "seed": _OBJECTIVE_SEED,
        "D": 5,
        "S": list(objective.labels.S),
        "f_star": objective.f_star,
        "gamma": objective.labels.gamma,
        "noise_sd": objective.labels.noise_sd,
    }
    assert manifest["seed"] == _SEED != _OBJECTIVE_SEED
    assert datetime.fromisoformat(manifest["created"]).tzinfo is not None
    assert manifest["resumed"] == []


def test_a_resume_records_itself_and_warns_about_a_version_that_moved(tmp_path):
    run_dir = run(
        _objective(), "dsp_map", seed=_SEED, T=7, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    path = run_dir / "manifest.json"
    started = json.loads(path.read_text())
    started["versions"]["jax"] = "0.0.0+not-this-one"
    path.write_text(json.dumps(started))

    # T is outside `config_hash` precisely so a run killed by a wall clock can be resumed longer.
    run(
        _objective(), "dsp_map", seed=_SEED, T=9, n_init=_N_INIT, out_dir=tmp_path,
        resume=True, **_LOOP_KW,
    )

    after = json.loads(path.read_text())
    assert len(after["resumed"]) == 1
    entry = after["resumed"][0]
    assert set(entry) == {"time", "commit", "versions_changed", "T", "env"}
    assert entry["versions_changed"] == ["jax"]
    assert entry["commit"] == after["git"]["commit"]
    assert datetime.fromisoformat(entry["time"]).tzinfo is not None
    # The resume's own budget and environment, not the start's: `T` is outside `config_hash` so
    # every resume may raise it, and bit-identity is only claimed within one CPU and one thread
    # setting -- neither is recoverable from the manifest's top-level blocks, which are the run's.
    assert entry["T"] == 9 and after["T"] == 7  # the resume's budget; the manifest keeps the run's
    assert entry["env"] == after["env"]  # same machine here; the point is that it is recorded
    # The resume appends and does not rewrite: what the manifest says about the start of the run
    # is the whole point of having recorded it.
    assert after["versions"]["jax"] == "0.0.0+not-this-one"
    assert after["created"] == started["created"]
    assert "warning: jax is" in (run_dir / "log.txt").read_text()
    # And it is a warning, not an error: the run went on to T.
    with np.load(run_dir / "checkpoint.npz") as data:
        assert int(data["t_done"]) == 8


def test_the_environment_lock_lists_the_installed_distributions(tmp_path):
    run_dir = run(
        _objective(), "dsp_map", seed=_SEED, T=7, n_init=_N_INIT, out_dir=tmp_path, **_LOOP_KW
    )
    lock = run_dir / "environment.lock.txt"
    lines = lock.read_text().splitlines()
    assert "numpyro==0.21.0" in lines
    assert f"numpy=={importlib.metadata.version('numpy')}" in lines
    assert lines == sorted(lines)

    # It describes the environment the run *started* in, so a resume must leave it alone; the
    # sentinel is what a rewrite would destroy, which comparing two identical writes would not.
    lock.write_text("sentinel\n")
    run(
        _objective(), "dsp_map", seed=_SEED, T=9, n_init=_N_INIT, out_dir=tmp_path,
        resume=True, **_LOOP_KW,
    )
    assert lock.read_text() == "sentinel\n"
