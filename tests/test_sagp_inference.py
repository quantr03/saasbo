"""Tests for sagp.gp's NUTS inference: reference equivalence, the refit policy and diagnostics.

The thesis's claim that the four cells differ in nothing but the kernel/prior block rests on the
sampler being the vendored reference's. `_run_nuts` is `SAASGP.run_inference` copied -- a copy
nothing but a test can keep honest -- so the first test runs the reference class and `fit` on the
same data with the key `SAASGP.fit` derives and demands *bit-for-bit* equal draws: a difference of
one ulp would mean the copy is no longer the same sampler on the same model. It runs twice, at a
toy budget and at the production 512/256/16 -- both on every invocation of the suite (ruling R20):
the production pair of fits measures about 4.4 s against the slow marker's ten-second threshold,
and reference equivalence at the budget the study is actually run at is not a claim worth
deferring to a marker the default invocation deselects.

The refit policy is behaviour the reference does not have, so it is tested against stubbed
diagnostics rather than a chain that happens to converge: the verdict decides the path, and the
path must be a fresh chain with twice the warm-up and a different key.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
from sagp.gp import (
    CELLS,
    KERNELS,
    DiagThresholds,
    Diagnostics,
    NUTSConfig,
    fit,
)

import jax
import numpy as np
import pytest

import saasgp
import sagp.gp

# Diagnostics that can never fail, so a test that is about the sampler's draws is not also about
# whether a 64-draw chain converged: with these, `fit` always returns attempt 0's samples.
NEVER_FAILS = DiagThresholds(r_hat_max=float("inf"), n_eff_min=0.0, max_divergences=10**9)

CELL_KEYS = list(CELLS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]


def _data(n: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A design in [0,1]^D and standardized targets, in the float64 numpy the loop hands `fit`."""
    rng = np.random.default_rng(seed)
    return rng.random((n, D)), rng.standard_normal(n)


def _reference_key() -> jax.Array:
    """The key `SAASGP.fit(X, y, seed=0)` hands its HMC run -- the first half of the split."""
    return jax.random.split(jax.random.PRNGKey(0), 2)[0]


def _assert_matches_reference(ours, ref, thinning: int) -> None:
    """Every hyperparameter the reference retains, thinned as it thins, equal draw for draw."""
    assert ours.status == "ok"
    for site in ("kernel_var", "kernel_noise", "kernel_tausq", "kernel_inv_length_sq"):
        ours_site = np.asarray(ours.samples[site])
        assert ours_site.shape[0] == 16
        assert np.array_equal(ours_site, np.asarray(ref.flat_samples[site])[::thinning])


def test_nuts_reproduces_reference_bit_for_bit():
    X, y = _data(n=30, D=5, seed=0)
    ref = saasgp.SAASGP(
        alpha=0.1,
        num_warmup=64,
        num_samples=64,
        max_tree_depth=6,
        num_chains=1,
        thinning=4,
        verbose=False,
        observation_variance=0.0,
        kernel="matern",
    ).fit(X, y)
    ours = fit(
        X,
        y,
        key=_reference_key(),
        cell=("product", "lengthscale"),
        nuts=NUTSConfig(64, 64, 4),
        thresholds=NEVER_FAILS,
    )
    _assert_matches_reference(ours, ref, thinning=4)


def test_nuts_reproduces_reference_bit_for_bit_production_budget():
    X, y = _data(n=30, D=5, seed=0)
    ref = saasgp.SAASGP(
        alpha=0.1,
        num_warmup=512,
        num_samples=256,
        max_tree_depth=6,
        num_chains=1,
        thinning=16,
        verbose=False,
        observation_variance=0.0,
        kernel="matern",
    ).fit(X, y)
    ours = fit(
        X,
        y,
        key=_reference_key(),
        cell=("product", "lengthscale"),
        nuts=NUTSConfig(),
        thresholds=NEVER_FAILS,
    )
    _assert_matches_reference(ours, ref, thinning=16)


def _stub_diagnose(verdicts):
    """A `_diagnose` returning the given passed/failed verdicts in order, whatever the chain did.

    The refit policy branches on the verdict alone, so stubbing it is what makes the two paths
    reachable in a test at all: no toy chain reliably fails, and none reliably fails then passes.
    """
    remaining = iter(verdicts)

    def stub(flat_samples, extra, thresholds, wall_s):
        passed = next(remaining)
        return Diagnostics(
            r_hat_max=1.0,
            r_hat_median=1.0,
            frac_r_hat_below_1_05=1.0,
            n_eff_min=32.0,
            divergences=0,
            num_steps_mean=1.0,
            wall_s=wall_s,
            passed=passed,
            reason="" if passed else "stubbed failure",
        )

    return stub


def _spy_on_mcmc(monkeypatch) -> list[dict]:
    """Record every `MCMC(...)` construction's kwargs while still running the real sampler."""
    real = sagp.gp.MCMC
    calls: list[dict] = []

    class Spy(real):
        def __init__(self, kernel, **kwargs):
            calls.append(dict(kwargs))
            super().__init__(kernel, **kwargs)

    monkeypatch.setattr(sagp.gp, "MCMC", Spy)
    return calls


def _spy_on_run_nuts(monkeypatch) -> list[jax.Array]:
    """Record the key of every `_run_nuts` call while still running it."""
    real = sagp.gp._run_nuts
    keys: list[jax.Array] = []

    def spy(model, X, Y, key, nuts):
        keys.append(key)
        return real(model, X, Y, key, nuts)

    monkeypatch.setattr(sagp.gp, "_run_nuts", spy)
    return keys


def test_refit_and_excluded_paths(monkeypatch):
    X, y = _data(n=12, D=3, seed=1)
    key = jax.random.PRNGKey(0)
    nuts = NUTSConfig(64, 32, 4)
    mcmc_calls = _spy_on_mcmc(monkeypatch)
    keys = _spy_on_run_nuts(monkeypatch)

    monkeypatch.setattr(sagp.gp, "_diagnose", _stub_diagnose([False, True]))
    refit = fit(X, y, key=key, cell=("product", "lengthscale"), nuts=nuts)

    assert refit.status == "refit"
    assert len(refit.attempts) == 2
    # The refit is a fresh chain with twice the warm-up, driven by a key disjoint from attempt 0's.
    # Both keys are pinned to the values `fit` documents, not merely to being different: attempt 0
    # must use the caller's key *unchanged* (that is what lets a run reproduce the reference), and
    # the refit must use `fold_in(key, 1)` (disjoint from `bo.py`'s exception retry, `fold_in(2)`).
    assert [call["num_warmup"] for call in mcmc_calls] == [64, 128]
    assert np.array_equal(np.asarray(keys[0]), np.asarray(key))
    assert np.array_equal(np.asarray(keys[1]), np.asarray(jax.random.fold_in(key, 1)))
    assert refit.samples["kernel_inv_length_sq"].shape == (8, 3)

    monkeypatch.setattr(sagp.gp, "_diagnose", _stub_diagnose([False, False]))
    excluded = fit(X, y, key=key, cell=("product", "lengthscale"), nuts=nuts)

    assert excluded.status == "excluded"
    assert excluded.status_reason != ""
    assert len(excluded.attempts) == 2
    # An excluded fit still carries the second attempt's draws: the loop has to query with it.
    assert excluded.samples["kernel_inv_length_sq"].shape == (8, 3)


def test_diagnostics_fields():
    X, y = _data(n=15, D=3, seed=2)
    # The one place `cell` is given as a `Cell` rather than a `CellKey`; `fit` normalizes it back.
    fitted = fit(
        X,
        y,
        key=jax.random.PRNGKey(0),
        cell=CELLS[("additive", "amplitude")],
        nuts=NUTSConfig(32, 32, 4),
    )
    assert fitted.cell == ("additive", "amplitude")
    diag = fitted.attempts[0]

    assert np.isfinite(diag.r_hat_max)
    assert np.isfinite(diag.r_hat_median)
    assert 0.0 <= diag.frac_r_hat_below_1_05 <= 1.0
    assert diag.n_eff_min > 0.0
    assert isinstance(diag.divergences, int)
    assert diag.divergences >= 0
    assert diag.num_steps_mean > 0.0
    assert diag.wall_s > 0.0
    assert diag.passed == (diag.reason == "")


# The retained sites and their shapes at S = 32 // 4 = 8 draws and D = 3: the amplitude cells
# carry no `kernel_var`, and the whole point of the sparse parameterization is which sites are
# per-coordinate rather than scalar.
S, D = 8, 3
SAMPLE_SHAPES = {
    "kernel_var": (S,),
    "kernel_noise": (S,),
    "kernel_tausq": (S,),
    "_kernel_inv_length_sq": (S, D),
    "kernel_inv_length_sq": (S, D),
    "_a_sq": (S, D),
    "a_sq": (S, D),
    "kernel_ell": (S, D),
}


@pytest.mark.parametrize(
    "cell_key",
    [("additive", "amplitude"), ("product", "amplitude"), ("additive", "lengthscale")],
    ids=["additive/amplitude", "product/amplitude", "additive/lengthscale"],
)
def test_amplitude_cells_fit_smoke(cell_key):
    X, y = _data(n=15, D=D, seed=3)
    fitted = fit(X, y, key=jax.random.PRNGKey(0), cell=cell_key, nuts=NUTSConfig(32, 32, 4))

    assert set(fitted.samples) == set(CELLS[cell_key].sites)
    assert (cell_key[1] == "amplitude") == ("kernel_var" not in fitted.samples)
    for site, samples in fitted.samples.items():
        assert samples.shape == SAMPLE_SHAPES[site]
        assert np.all(np.asarray(samples) > 0.0)


@pytest.mark.parametrize("cell_key", CELL_KEYS, ids=CELL_IDS)
def test_cells_wiring(cell_key):
    """A registry entry paired with the wrong cell's kernel would pass every other test.

    Site names encode only the sparsity prior, so ("additive", "amplitude") holding the *product*
    amplitude kernel traces, samples and thins exactly as it should -- while silently comparing
    the wrong two cells.
    """
    cell = CELLS[cell_key]

    assert cell.key == cell_key
    assert cell.model.__name__ == f"model_{cell.structure}_{cell.prior}"
    assert cell.kernel is KERNELS[cell.key][0]
    assert cell.kernel_diag is KERNELS[cell.key][1]


def test_fit_with_fixed_noise():
    """A fixed observation variance removes `kernel_noise` from the model, not just from the draws.

    The reproduction test runs the whole loop at `fixed_noise=1e-6`, which is the reference
    driver's `observation_variance`; what makes that the same model is that the site is never
    sampled, so a fit that merely fixed the *value* would still have an extra latent dimension and
    a different chain. `fixed_noise` is kept on the fitted object because prediction needs it:
    `FittedGP._noises` has no `kernel_noise` to read.
    """
    X, y = _data(n=12, D=3, seed=5)
    fitted = fit(
        X,
        y,
        key=jax.random.PRNGKey(0),
        cell=("product", "lengthscale"),
        fixed_noise=1.0e-6,
        nuts=NUTSConfig(32, 32, 4),
    )

    assert "kernel_noise" not in fitted.samples
    assert set(fitted.samples) == {
        "kernel_var", "kernel_tausq", "_kernel_inv_length_sq", "kernel_inv_length_sq"
    }
    assert fitted.fixed_noise == 1.0e-6
    assert fitted.samples["kernel_inv_length_sq"].shape == (8, 3)


def test_fit_rejects_fixed_noise_zero():
    """`observation_variance=0.0` means "learn the noise" in the reference; here it would fix it.

    Silently reinterpreting the value either way would be a trap, so `fit` refuses it and names
    the spelling that learns the noise.
    """
    X, y = _data(n=8, D=2, seed=4)

    with pytest.raises(ValueError, match="None"):
        fit(
            X,
            y,
            key=jax.random.PRNGKey(0),
            cell=("product", "lengthscale"),
            fixed_noise=0.0,
            nuts=NUTSConfig(2, 2, 1),  # never reached: the check fires before any sampling
        )
