"""Tests for sagp.gp's NUTS inference: BoTorch equivalence, the gate and prediction.

The thesis's claim that the cells differ in nothing but the kernel/prior block rests on the
sampler being BoTorch's. `_run_nuts` is `fit_fully_bayesian_model_nuts`'s sampler lines copied -- a
copy nothing but a test can keep honest -- so the first tests run
`SaasFullyBayesianSingleTaskGP` + `fit_fully_bayesian_model_nuts` and `fit` on the same data at the
same seed and demand *bit-for-bit* equal hyperparameters in the loaded model: a difference of one
ulp would mean the copy is no longer the same sampler on the same model. They run at a toy budget
and at the production 512/256/16 -- both on every invocation of the suite (ruling R20): reference
equivalence at the budget the study is actually run at is not a claim worth deferring to a marker
the default invocation deselects.

The gate is behaviour BoTorch does not have. `fit` makes one attempt and records its verdict, so
what is tested is that the verdict is read off the *un-thinned* chain the sampler produced and that
a failed gate still returns its draws -- the loop has to keep querying.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
from sagp.diagnostics import DiagThresholds
from sagp.gp import (
    CELLS,
    KERNELS,
    NUTSConfig,
    fit,
    standardize,
    torch_device,
)

import botorch.settings
import gpytorch.settings
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import (
    MIN_INFERRED_NOISE_LEVEL,
    SaasFullyBayesianSingleTaskGP,
)
from numpyro import handlers
from numpyro.infer.util import constrain_fn, potential_energy, unconstrain_fn

import sagp.gp

# Diagnostics that can never fail, so a test that is about the sampler's draws is not also about
# whether a 64-draw chain converged.
NEVER_FAILS = DiagThresholds(r_hat_max=float("inf"), n_eff_min=0.0, max_divergences=10**9)

# Their negation: every criterion fails whatever the chain did, so the excluded path is reachable
# without a chain that happens to diverge.
NEVER_PASSES = DiagThresholds(r_hat_max=0.0, n_eff_min=float("inf"), max_divergences=-1)

CELL_KEYS = list(CELLS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]


def _data(n: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A design in [0,1]^D and standardized targets, in the float64 numpy the loop hands `fit`."""
    rng = np.random.default_rng(seed)
    return rng.random((n, D)), rng.standard_normal(n)


def _reference_fit(X, y, fixed_noise, nuts: NUTSConfig, seed: int):
    """BoTorch's own SAAS fit of the same data at the same seed: the pin's other side."""
    # On the device `fit` builds its model on, so the two sides load the same draws the same way.
    Xt = torch.as_tensor(X, device=torch_device())
    zt = torch.as_tensor(y, device=torch_device())[:, None]
    train_Yvar = None if fixed_noise is None else torch.full_like(zt, fixed_noise)
    with botorch.settings.validate_input_scaling(False):
        ref = SaasFullyBayesianSingleTaskGP(Xt, zt, train_Yvar)
    fit_fully_bayesian_model_nuts(
        ref,
        max_tree_depth=nuts.max_tree_depth,
        warmup_steps=nuts.num_warmup,
        num_samples=nuts.num_samples,
        thinning=nuts.thinning,
        disable_progbar=True,
        seed=seed,
    )
    return ref


def _assert_matches_botorch(ours, ref) -> None:
    """Every hyperparameter the loaded GPyTorch model carries, equal tensor for tensor."""
    assert torch.equal(
        ours.model.covar_module.base_kernel.lengthscale,
        ref.covar_module.base_kernel.lengthscale,
    )
    assert torch.equal(ours.model.covar_module.outputscale, ref.covar_module.outputscale)
    assert torch.equal(ours.model.mean_module.constant, ref.mean_module.constant)
    assert torch.equal(ours.model.likelihood.noise, ref.likelihood.noise)


def test_fit_reproduces_botorch_nuts_bit_for_bit():
    X, y = _data(n=30, D=5, seed=0)
    nuts = NUTSConfig(64, 64, 4)

    ref = _reference_fit(X, y, None, nuts, seed=7)
    ours = fit(
        X, y, 7, ("product", "lengthscale"), nuts=nuts, thresholds=NEVER_FAILS
    )

    _assert_matches_botorch(ours, ref)
    assert ours.samples["lengthscale"].shape[0] == 16


def test_fit_reproduces_botorch_nuts_bit_for_bit_production_budget():
    """The same equivalence at the budget the study is actually run at (ruling R20).

    The pair of fits measures a few seconds against the slow marker's ten-second threshold, and
    equivalence at 512/256/16 is not a claim worth deferring to a marker the default invocation
    deselects: warm-up length changes the step size and the mass matrix, so a 64-draw agreement
    does not imply this one.
    """
    X, y = _data(n=30, D=5, seed=0)
    nuts = NUTSConfig()

    ref = _reference_fit(X, y, None, nuts, seed=7)
    ours = fit(X, y, 7, ("product", "lengthscale"), nuts=nuts, thresholds=NEVER_FAILS)

    _assert_matches_botorch(ours, ref)
    assert ours.samples["lengthscale"].shape[0] == 16


def test_fit_reproduces_botorch_nuts_bit_for_bit_with_a_fixed_noise():
    """The same equivalence on the *other* model the study runs: `noise` not sampled at all.

    BoTorch spells a fixed observation variance `train_Yvar` and drops the site from the model;
    `fit` spells it `fixed_noise` and does the same. That is a different latent space and therefore
    a different chain, so the two tests above -- both on the learned-noise model -- say nothing
    about it.
    """
    X, y = _data(n=30, D=5, seed=0)
    nuts = NUTSConfig(64, 64, 4)

    ref = _reference_fit(X, y, 1.0e-4, nuts, seed=7)
    ours = fit(
        X, y, 7, ("product", "lengthscale"), fixed_noise=1.0e-4, nuts=nuts, thresholds=NEVER_FAILS
    )

    _assert_matches_botorch(ours, ref)
    assert "noise" not in ours.samples
    assert np.array_equal(np.asarray(ours.noises()), np.full(16, 1.0e-4))


@pytest.mark.parametrize(
    ("cell_key", "site"),
    [(("additive", "amplitude"), "kernel_tausq"), (("additive", "amplitude_r2d2"), "r2d2_z_lam")],
    ids=["additive/amplitude", "additive/amplitude_r2d2"],
)
def test_fit_is_deterministic_in_its_seed(cell_key, site):
    # `seed` is the whole of the fit's randomness: a run is reproducible from its record, and two
    # cells at the same seed are comparable because they saw the same stream. An R2-D2 cell adds
    # nothing random of its own: its map's table is built from the shape alone.
    X, y = _data(n=12, D=3, seed=1)
    nuts = NUTSConfig(32, 32, 8)

    first = fit(X, y, 4, cell_key, nuts=nuts, thresholds=NEVER_FAILS)
    again = fit(X, y, 4, cell_key, nuts=nuts, thresholds=NEVER_FAILS)
    other = fit(X, y, 5, cell_key, nuts=nuts, thresholds=NEVER_FAILS)

    assert set(first.samples) == set(again.samples)
    for name, draws in first.samples.items():
        assert np.array_equal(np.asarray(draws), np.asarray(again.samples[name]))
    assert not np.array_equal(
        np.asarray(first.samples[site]), np.asarray(other.samples[site])
    )


def test_failed_gate_is_excluded_with_its_draws():
    # The gate decides the *label*, not whether the fit is usable: the loop has to keep querying
    # with an excluded fit, so its draws come back and its posterior works.
    X, y = _data(n=15, D=3, seed=2)
    nuts = NUTSConfig(64, 64, 4)
    X_test = np.random.default_rng(3).random((5, 3))

    excluded = fit(X, y, 0, ("product", "lengthscale"), nuts=nuts, thresholds=NEVER_PASSES)

    assert excluded.status == "excluded"
    assert len(excluded.attempts) == 1
    for criterion in ("r_hat_max", "n_eff_min", "divergences"):
        assert criterion in excluded.status_reason
    assert excluded.samples["kernel_inv_length_sq"].shape == (16, 3)
    assert excluded.posterior(X_test)[0].shape == (16, 5)

    passed = fit(X, y, 0, ("product", "lengthscale"), nuts=nuts, thresholds=NEVER_FAILS)

    assert passed.status == "ok"
    assert passed.status_reason == ""


@pytest.mark.parametrize("fixed_noise", [0.0, 1.0e-6], ids=["zero", "below_the_floor"])
def test_fixed_noise_below_the_floor_is_rejected(fixed_noise):
    # BoTorch clamps `train_Yvar` at 1e-4 before the sampler sees it, so a smaller value would run
    # a different model than the record says and 0.0 would silently become 1e-4.
    X, y = _data(n=8, D=2, seed=4)

    with pytest.raises(ValueError, match=str(MIN_INFERRED_NOISE_LEVEL)):
        fit(
            X, y, 0, ("product", "lengthscale"),
            fixed_noise=fixed_noise,
            nuts=NUTSConfig(2, 2, 1),  # never reached: the check fires before any sampling
        )

    accepted = fit(
        X, y, 0, ("product", "lengthscale"), fixed_noise=1.0e-4, nuts=NUTSConfig(8, 8, 4)
    )
    assert accepted.fixed_noise == 1.0e-4


def test_diagnostics_read_the_unthinned_chain(monkeypatch):
    # The gate is a statement about the chain, not about the 16 draws prediction keeps: R-hat over
    # 16 draws is meaningless, and the divergence count is a property of the whole run. BoTorch
    # discards both the un-thinned draws and the counters, which is why `_run_nuts` returns the
    # sampler rather than calling BoTorch's fit function.
    X, y = _data(n=12, D=3, seed=5)
    real = sagp.gp.diagnose
    seen: dict = {}

    def recorder(flat_samples, extra, thresholds, wall_s):
        seen.update(flat=flat_samples, extra=extra, wall_s=wall_s)
        return real(flat_samples, extra, thresholds, wall_s)

    monkeypatch.setattr(sagp.gp, "diagnose", recorder)
    fitted = fit(X, y, 0, ("product", "lengthscale"), nuts=NUTSConfig(64, 64, 4))

    assert np.asarray(seen["flat"]["kernel_tausq"]).shape == (64,)
    assert set(seen["extra"]) == {"diverging", "num_steps"}
    assert np.asarray(seen["extra"]["diverging"]).shape == (64,)
    assert np.asarray(seen["extra"]["num_steps"]).shape == (64,)
    assert seen["wall_s"] > 0.0
    # And what prediction keeps is the thinned chain, not the one the gate read.
    assert fitted.samples["kernel_tausq"].shape == (16,)


def test_run_nuts_refuses_more_than_one_chain():
    # The plan pins one chain per fit and `diagnose` pools with `group_by_chain=False`, so a second
    # chain would be concatenated onto the first and split-R-hat would compare the halves of the
    # concatenation -- a number that looks like a diagnostic and is not one.
    X, y = _data(n=8, D=2, seed=6)
    model = sagp.gp.CellGP(
        torch.as_tensor(X), torch.as_tensor(y)[:, None], cell=CELLS[("product", "lengthscale")]
    )

    with pytest.raises(ValueError, match="num_chains must be 1"):
        sagp.gp._run_nuts(model, NUTSConfig(2, 2, 1, num_chains=2), 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 60, "thinning": 8},
        {"num_warmup": 0},
        {"thinning": -1},
    ],
    ids=["thinning_does_not_divide", "zero_warmup", "negative_thinning"],
)
def test_nuts_config_rejects_a_budget_that_would_fail_later(kwargs):
    # A budget is chosen once, on a command line, and then spends hours. `--nuts 64,60,8` retains
    # `flat[::8]` = 8 draws where `num_samples // thinning` -- the count every cost estimate and
    # every "16 retained draws" claim is written against -- says 7; a non-positive count fails
    # somewhere inside NumPyro long after the run started.
    with pytest.raises(ValueError):
        NUTSConfig(**{"num_warmup": 64, "num_samples": 64, "thinning": 4, **kwargs})


# --- prediction ---


def _assert_posterior_is_the_models(fitted, X_test) -> None:
    """`FittedGP.posterior` is the GPyTorch posterior reshaped, in both noise conventions."""
    for observation_noise in (False, True):
        mean, var = fitted.posterior(X_test, observation_noise=observation_noise)
        # `cholesky_max_tries(9)` because `FittedGP.posterior` factorizes under it: a different
        # number of tries is a different jitter, and this comparison is exact.
        with torch.no_grad(), gpytorch.settings.cholesky_max_tries(9):
            post = fitted.model.posterior(
                torch.as_tensor(X_test, device=fitted.device)[:, None, :],
                observation_noise=observation_noise,
            )
        expected_mean = post.mean.reshape(X_test.shape[0], -1).T
        expected_var = post.variance.reshape(X_test.shape[0], -1).T
        assert np.array_equal(np.asarray(mean), expected_mean.cpu().numpy())
        assert np.array_equal(np.asarray(var), expected_var.cpu().numpy())


def test_posterior_matches_botorch_on_the_same_model():
    # `FittedGP.posterior` may reshape BoTorch's answer and nothing else -- the acquisition
    # optimizes the GPyTorch posterior, and a readout that saw a different number would describe a
    # different surrogate. `alphas()` is then pinned as the weights that very mean contracts.
    X, y = _data(n=20, D=4, seed=7)
    X_test = np.random.default_rng(8).random((6, 4))
    ours = fit(
        X, y, 2, ("product", "lengthscale"), nuts=NUTSConfig(32, 32, 8), thresholds=NEVER_FAILS
    )

    _assert_posterior_is_the_models(ours, X_test)

    mean, _ = ours.posterior(X_test)
    alphas, means = ours.alphas(), ours.means()
    kernel = KERNELS[("product", "lengthscale")][0]
    assert alphas.shape == (4, 20)
    for s in range(alphas.shape[0]):
        k_star = kernel(jnp.asarray(X_test), jnp.asarray(X), ours.params(s))
        rebuilt = means[s] + k_star @ alphas[s]
        assert np.max(np.abs(np.asarray(rebuilt) - np.asarray(mean[s]))) < 1.0e-8


def test_posterior_equals_a_stock_botorch_model_loaded_with_the_same_draws():
    """The reference cell's prediction is `SaasFullyBayesianSingleTaskGP`'s, to the bit.

    `test_posterior_matches_botorch_on_the_same_model` pins `FittedGP.posterior` against the model
    `fit` built, which leaves the model itself unpinned: a `CellGP` that assembled the reference
    cell slightly differently -- another likelihood, another mean, a lengthscale loaded into the
    wrong place -- would agree with itself and nobody would notice. Here BoTorch's own class is
    handed our draws and asked the same question.
    """
    X, y = _data(n=20, D=4, seed=13)
    X_test = np.random.default_rng(14).random((6, 4))
    fitted = fit(
        X, y, 3, ("product", "lengthscale"), nuts=NUTSConfig(64, 64, 4), thresholds=NEVER_FAILS
    )

    # On `fitted`'s device: the same arithmetic on another device is not the same to the bit.
    device = fitted.device
    with botorch.settings.validate_input_scaling(False):
        stock = SaasFullyBayesianSingleTaskGP(
            torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)[:, None]
        )
    stock.load_mcmc_samples(
        {
            # `np.array`, not `np.asarray`: a jnp array is read-only, and torch warns on wrapping
            # one without copying.
            site: torch.as_tensor(np.array(draws), device=device)
            for site, draws in fitted.samples.items()
            if site in ("mean", "outputscale", "noise", "lengthscale")
        }
    )
    stock.eval()

    for observation_noise in (False, True):
        mean, var = fitted.posterior(X_test, observation_noise=observation_noise)
        with torch.no_grad(), gpytorch.settings.cholesky_max_tries(9):
            post = stock.posterior(
                torch.as_tensor(X_test, device=device)[:, None, :],
                observation_noise=observation_noise,
            )
        assert np.array_equal(
            np.asarray(mean), post.mean.reshape(X_test.shape[0], -1).T.cpu().numpy()
        )
        assert np.array_equal(
            np.asarray(var), post.variance.reshape(X_test.shape[0], -1).T.cpu().numpy()
        )


@pytest.mark.parametrize(
    "cell_key",
    [("product", "lengthscale"), ("additive", "amplitude")],
    ids=["product/lengthscale", "additive/amplitude"],
)
def test_posterior_is_right_when_draws_equal_points(cell_key):
    # Task 1 found that GPyTorch's `Kernel.__call__` collapses a (S, n) diagonal to (n,) when
    # S == n on a bare (n, D) input -- a silently wrong posterior at exactly the retained count
    # this study uses. `FittedGP.posterior` gives each test point its own batch entry, so the
    # heuristic cannot fire; both a square and a non-square test set are checked.
    n = 16
    X, y = _data(n=n, D=3, seed=9)
    rng = np.random.default_rng(10)
    fitted = fit(X, y, 1, cell_key, nuts=NUTSConfig(64, 64, 4), thresholds=NEVER_FAILS)

    assert fitted.samples["mean"].shape == (16,)
    for n_test in (n, 5):
        X_test = rng.random((n_test, 3))

        mean, var = fitted.posterior(X_test)

        assert mean.shape == var.shape == (16, n_test)
        assert np.all(np.asarray(var) > 0.0)
        _assert_posterior_is_the_models(fitted, X_test)


@pytest.mark.parametrize("fixed_noise", [None, 1.0e-3], ids=["learned_noise", "fixed_noise"])
@pytest.mark.parametrize("cell_key", CELL_KEYS, ids=CELL_IDS)
def test_all_cells_fit_and_predict(cell_key, fixed_noise):
    # One code path serves every cell and both noise conventions: what is pinned is the site
    # set each fit retains, the shapes prediction answers with, and that `noises()` reports the
    # variance prediction actually used -- BoTorch's floor, or the fixed value exactly.
    S, D, n_test = 4, 3, 5
    X, y = _data(n=12, D=D, seed=11)
    X_test = np.random.default_rng(12).random((n_test, D))

    fitted = fit(X, y, 0, cell_key, fixed_noise=fixed_noise, nuts=NUTSConfig(16, 16, 4))

    expected = tuple(
        site for site in CELLS[cell_key].sites if fixed_noise is None or site != "noise"
    )
    assert tuple(fitted.samples) == expected
    # Every site's shape per draw: the global block is scalar, the sparsity block per-coordinate.
    # A site that silently came back one draw wide, or D-wide where it should be scalar, would
    # load into the model without complaint and change every readout that reads it.
    per_site = {
        "outputscale": (), "mean": (), "noise": (), "kernel_tausq": (),
        "_kernel_inv_length_sq": (D,), "kernel_inv_length_sq": (D,), "lengthscale": (D,),
        "_a_sq": (D,), "a_sq": (D,), "kernel_ell": (D,),
        "r2d2_R2": (), "r2d2_z_xi": (), "r2d2_z_lam": (D,),
    }
    for site in expected:
        assert fitted.samples[site].shape == (S,) + per_site[site]
    mean, var = fitted.posterior(X_test)
    assert mean.shape == var.shape == (S, n_test)
    assert np.all(np.isfinite(np.asarray(mean))) and np.all(np.asarray(var) > 0.0)
    assert fitted.means().shape == (S,)
    assert fitted.noises().shape == (S,)
    assert np.all(np.asarray(fitted.noises()) >= 1.0e-4)
    if fixed_noise is not None:
        assert np.array_equal(np.asarray(fitted.noises()), np.full(S, fixed_noise))


R2D2_KEYS = [("additive", "amplitude_r2d2"), ("product", "amplitude_r2d2"),
             ("additive", "lengthscale_r2d2"), ("product", "lengthscale_r2d2")]


@pytest.mark.parametrize("fixed_noise", [None, 1.0e-3], ids=["learned_noise", "fixed_noise"])
@pytest.mark.parametrize("cell_key", R2D2_KEYS, ids=["/".join(key) for key in R2D2_KEYS])
def test_r2d2_log_density_gradient_is_finite_deep_in_the_tail(cell_key, fixed_noise):
    # A copula coordinate at z = -40 sends its log theta to about -1600, far below any double. Its
    # a_sq underflows to 0, which the amplitude kernels take in stride; its rho would too, and
    # long before that the derivative of rho ** -0.5 in the additive lengthscale kernel overflows
    # and turns the whole gradient NaN. The floor on log rho is what keeps it finite. The gradient
    # checked is the one NUTS follows: the potential energy's, on the unconstrained coordinates,
    # with and without a `noise` site.
    X, y = _data(n=12, D=3, seed=15)
    Xt, zt = torch.as_tensor(X), torch.as_tensor(y)[:, None]
    train_Yvar = None if fixed_noise is None else torch.full_like(zt, fixed_noise)
    model = sagp.gp.CellGP(Xt, zt, train_Yvar, cell=CELLS[cell_key]).pyro_model.sample
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
    constrained = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    params = unconstrain_fn(model, (), {}, constrained)
    params["r2d2_z_lam"] = params["r2d2_z_lam"].at[0].set(-40.0)

    native = constrain_fn(model, (), {}, params, return_deterministic=True)[
        CELLS[cell_key].native_site
    ]
    grads = jax.grad(lambda p: potential_energy(model, (), {}, p))(params)

    assert float(native[0]) < 1.0e-150  # the point is in the deep tail the test is about
    assert set(grads) == set(params)
    for name, grad in grads.items():
        assert np.all(np.isfinite(np.asarray(grad))), name
    # And compiled, as NUTS evaluates it: XLA may fuse and reorder what the eager trace computes
    # op by op, so the jitted gradient is checked in its own right.
    jitted = jax.jit(jax.grad(lambda p: potential_energy(model, (), {}, p)))(params)
    assert set(jitted) == set(params)
    for name, grad in jitted.items():
        assert np.all(np.isfinite(np.asarray(grad))), name


# --- standardize ---


def test_standardize_zero_mean_unit_std_and_no_sign_flip():
    rng = np.random.default_rng(7)
    y = rng.normal(loc=3.0, scale=2.0, size=50)

    z, mean, std = standardize(y)

    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0)
    assert z.argmax() == y.argmax()  # no sign flip: the loop maximizes what the objective does
    assert mean == pytest.approx(y.mean())
    assert std == pytest.approx(y.std())
    assert type(mean) is float and type(std) is float
