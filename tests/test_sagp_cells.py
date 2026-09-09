"""Tests for sagp.gp's four NumPyro cells and their prediction: models, sites and `FittedGP`.

Two properties carry the 2x2 comparison. ("product", "lengthscale") must be the vendored
`SAASGP.model` -- its body is copied into `model_product_lengthscale` rather than imported, so
nothing but a test can keep the copy honest; comparing the two log joints on the same parameter
values checks the priors, the kernel and the likelihood in one number. And every cell must expose
exactly the sites the plan names, in the plan's order and shapes, because `Cell.sites` is what
inference retains and every readout downstream indexes those samples by name.

Prediction is held to the same standard: `FittedGP.posterior` is the reference's
`compute_choleskys` + `predict` with the kernel and its diagonal generalized to the cell, so on
("product", "lengthscale") it must reproduce `SAASGP.posterior` bit for bit, and the
generalizations it adds -- the other three cells, the oracle's `active` restriction, the
memory-driven row chunking -- must not change what any of the four cells predicts.

The same four cells now also exist as BoTorch `PyroModel`s, and the last section holds them to the
same two standards: ("product", "lengthscale") must be BoTorch's own `SaasPyroModel` -- checked as
an equal log density, since `sample()` is inherited and only the kernel/prior block may differ --
and every cell must declare exactly the sites named here, in this order, because those names are
what the retained draws are keyed by. Their kernels are pinned to the JAX ones a second time,
through the likelihood term `sample` builds and through the GPyTorch posterior
`load_mcmc_samples` produces.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp
import sagp.readouts
from sagp.gp import (
    CELLS,
    ELL_PRIOR,
    KERNELS,
    FittedGP,
    model_product_lengthscale,
)

from functools import partial

import botorch.settings
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
import scipy.stats
import torch
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from numpyro.infer.util import log_density

import saasgp

N = 12
P = 4

CELL_KEYS = list(CELLS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]

# The plan's site list per sparsity prior: every site inference retains, sampled and
# deterministic, in the order the model declares them. Stated here as literals rather than read
# off `Cell.sites` so that the trace pins the *names*, not merely the registry's agreement with
# itself; the registry is then checked against the same literals.
SITES = {
    "lengthscale": (
        "kernel_var",
        "kernel_noise",
        "kernel_tausq",
        "_kernel_inv_length_sq",
        "kernel_inv_length_sq",
    ),
    "amplitude": ("kernel_noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"),
}

# Scalar hyperparameters versus the (P,) per-coordinate ones -- the distinction the whole sparse
# parameterization rests on, and the one a mis-shaped prior would silently break.
SHAPES = {
    "kernel_var": (),
    "kernel_noise": (),
    "kernel_tausq": (),
    "_kernel_inv_length_sq": (P,),
    "kernel_inv_length_sq": (P,),
    "_a_sq": (P,),
    "a_sq": (P,),
    "kernel_ell": (P,),
}


def _data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """N points uniform on [0,1]^P with standard-normal responses: the shape a fit sees."""
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(0.0, 1.0, (N, P))), jnp.asarray(rng.normal(size=N))


@pytest.mark.parametrize("fixed_noise", [None, 1.0e-6], ids=["learned_noise", "fixed_noise"])
def test_product_lengthscale_log_joint_matches_reference(fixed_noise):
    # Both noise branches, because `fixed_noise` replaces the reference's `learn_noise` /
    # `observation_variance` pair and a wrong branch would change the site set and the kernel.
    X, Y = _data()
    params = {
        "kernel_var": 1.3,
        "kernel_noise": 0.05,
        "kernel_tausq": 0.2,
        "_kernel_inv_length_sq": jnp.asarray(np.random.default_rng(1).uniform(0.1, 5.0, P)),
    }
    if fixed_noise is not None:
        del params["kernel_noise"]
    reference = saasgp.SAASGP(alpha=0.1, observation_variance=fixed_noise or 0.0).model
    ours = partial(model_product_lengthscale, alpha=0.1, fixed_noise=fixed_noise)

    # `log_density` substitutes the sampled sites and recomputes the deterministic ones, so this
    # is the full log joint -- priors plus likelihood -- at one point of the parameter space.
    ours_log_joint, _ = log_density(ours, (X, Y), {}, params)
    reference_log_joint, _ = log_density(reference, (X, Y), {}, params)

    assert abs(float(ours_log_joint) - float(reference_log_joint)) < 1.0e-10


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
@pytest.mark.parametrize("fixed_noise", [None, 1.0e-6], ids=["learned_noise", "fixed_noise"])
def test_trace_sites_per_cell(key, fixed_noise):
    cell = CELLS[key]
    X, Y = _data()
    kwargs = {"alpha": cell.alpha_default, "fixed_noise": fixed_noise}
    if cell.prior == "amplitude":
        kwargs["ell_prior"] = ELL_PRIOR
    model = numpyro.handlers.seed(partial(cell.model, **kwargs), rng_seed=0)

    trace = numpyro.handlers.trace(model).get_trace(X, Y)

    assert cell.sites == SITES[cell.prior]
    # `kernel_noise` is the only site the noise branch adds or removes; everything else is the
    # cell's own parameterization and must be there whatever the noise is.
    expected = [site for site in cell.sites if fixed_noise is None or site != "kernel_noise"]
    latent = [
        name
        for name, site in trace.items()
        if site["type"] in {"sample", "deterministic"} and name != "Y"
    ]
    assert latent == expected
    assert [trace[name]["value"].shape for name in latent] == [SHAPES[name] for name in latent]
    assert trace["Y"]["is_observed"]  # the one site excluded above is the likelihood, not a latent


# --- prediction ---

# The reference's `posterior` thins its own draws and hands `util.chunk_vmap` a fixed
# `chunk_size=8`; `util.get_chunks` raises a latent `NameError` (it calls `np.arange` without
# importing numpy) whenever the retained count is not a multiple of 8. 32 // 4 = 8 keeps every
# reference call on the working path -- never give the vendored class fewer retained draws.
REFERENCE_RETAINED = 8
REFERENCE_THINNING = 4


def _hand_made_samples(prior: str, S: int, D: int, seed: int) -> dict[str, jnp.ndarray]:
    """Positive kernel parameters with the shapes a fit's `samples` has: leading dimension S.

    Hand-made rather than sampled, because these tests are about prediction's plumbing -- shapes,
    the `active` restriction, chunking -- and running NUTS to obtain draws would make them slow
    and make a failure ambiguous between the sampler and the code under test.
    """
    rng = np.random.default_rng(seed)
    if prior == "amplitude":
        samples = {
            "a_sq": rng.uniform(0.1, 2.0, (S, D)),
            "kernel_ell": rng.uniform(0.2, 2.0, (S, D)),
        }
    else:
        samples = {
            "kernel_var": np.full(S, 1.3),
            "kernel_inv_length_sq": rng.uniform(0.1, 5.0, (S, D)),
        }
    samples["kernel_noise"] = rng.uniform(0.01, 0.1, S)
    return {site: jnp.asarray(draws) for site, draws in samples.items()}


def _fitted(cell, X, y, samples, *, fixed_noise=None, active=None) -> FittedGP:
    """A `FittedGP` with the fit-status fields a prediction test does not care about filled in."""
    return FittedGP(
        cell=cell,
        X_train=X,
        Y_train=y,
        samples=samples,
        fixed_noise=fixed_noise,
        active=active,
        status="ok",
        status_reason="",
        attempts=(),
    )


@pytest.mark.parametrize("fixed_noise", [None, 1.0e-6], ids=["learned_noise", "fixed_noise"])
def test_posterior_matches_reference(fixed_noise):
    # ("product", "lengthscale") *is* SAASBO, so its prediction must be the vendored class's to
    # the last bit -- not merely close. Both noise branches, because ours uses the configured
    # variance in the predictive diagonal where the reference hard-codes 1e-6; 1e-6 is the value
    # at which the two agree, and the only fixed variance this study uses.
    rng = np.random.default_rng(2)
    X = jnp.asarray(rng.uniform(0.0, 1.0, (20, 4)))
    y = jnp.asarray(rng.normal(size=20))
    X_test = jnp.asarray(rng.uniform(0.0, 1.0, (7, 4)))

    reference = saasgp.SAASGP(
        num_warmup=16,
        num_samples=REFERENCE_RETAINED * REFERENCE_THINNING,
        thinning=REFERENCE_THINNING,
        verbose=False,
        kernel="matern",
        observation_variance=0.0 if fixed_noise is None else fixed_noise,
    ).fit(X, y)
    sites = ("kernel_var", "kernel_tausq", "kernel_inv_length_sq")
    if fixed_noise is None:
        sites = sites + ("kernel_noise",)
    fitted = _fitted(
        ("product", "lengthscale"),
        X,
        y,
        {site: reference.flat_samples[site][::REFERENCE_THINNING] for site in sites},
        fixed_noise=fixed_noise,
    )

    mean, var = fitted.posterior(X_test)
    reference_mean, reference_var = reference.posterior(X_test)

    assert mean.shape == (REFERENCE_RETAINED, 7)
    assert np.array_equal(np.asarray(mean), np.asarray(reference_mean))
    assert np.array_equal(np.asarray(var), np.asarray(reference_var))


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_posterior_shapes_all_cells(key):
    # The other three cells have no reference to compare against, so what is pinned here is that
    # one code path serves all four: every cell's parameters reach its kernel and its (in three
    # cases non-constant) diagonal, and the predictive variance stays a variance.
    S, n_test = 3, 5
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(3).uniform(0.0, 1.0, (n_test, P)))
    fitted = _fitted(key, X, y, _hand_made_samples(key[1], S, P, seed=4))

    mean, var = fitted.posterior(X_test)

    assert mean.shape == var.shape == (S, n_test)
    assert np.all(np.isfinite(np.asarray(mean)))
    assert np.all(np.asarray(var) > 0.0)


def test_oracle_active_restricts_columns():
    # Task 7's oracle fits on X[:, S] alone but keeps the full-D X_train, because the BO loop and
    # `saasbo.optimize_ei` read it. Restricting inside prediction must therefore be exactly the
    # same computation as having been handed the narrow matrix in the first place. The `cell`
    # string also pins the MAP references onto ("product", "lengthscale")'s kernel.
    active = np.asarray([0, 2])
    S = 2
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(5).uniform(0.0, 1.0, (6, P)))
    samples = _hand_made_samples("lengthscale", S, len(active), seed=6)

    restricted = _fitted("oracle_S", X, y, samples, active=active).posterior(X_test)
    narrow = _fitted(("product", "lengthscale"), X[:, active], y, samples).posterior(
        X_test[:, active]
    )

    assert np.array_equal(np.asarray(restricted[0]), np.asarray(narrow[0]))
    assert np.array_equal(np.asarray(restricted[1]), np.asarray(narrow[1]))


def test_posterior_chunked_equals_unchunked(monkeypatch):
    # Row-chunking (ruling R19) is a memory measure, not a modelling one: each test point's mean
    # and variance depend on no other test point, so blocking the test axis may move the last ulp
    # and nothing else. Forced on a small problem, since the size that triggers it in production
    # (5000 x 200 x 100) is far too large for a test.
    S, D, n, n_test = 2, 6, 30, 600
    rng = np.random.default_rng(7)
    X = jnp.asarray(rng.uniform(0.0, 1.0, (n, D)))
    y = jnp.asarray(rng.normal(size=n))
    X_test = jnp.asarray(rng.uniform(0.0, 1.0, (n_test, D)))
    fitted = _fitted(("product", "amplitude"), X, y, _hand_made_samples("amplitude", S, D, seed=8))

    unchunked = fitted.posterior(X_test)  # n_test * n * D = 1.1e5, far below the threshold
    monkeypatch.setattr(sagp.gp, "_CHUNK_THRESHOLD", 0)
    chunked = fitted.posterior(X_test)

    assert chunked[0].shape == (S, n_test)  # 600 rows is 2 full blocks of 256 and a partial one
    assert np.max(np.abs(np.asarray(chunked[0]) - np.asarray(unchunked[0]))) < 1.0e-12
    assert np.max(np.abs(np.asarray(chunked[1]) - np.asarray(unchunked[1]))) < 1.0e-12


@pytest.mark.parametrize(
    ("retained", "expected"), [(16, 8), (8, 8), (1, 1), (12, 6), (22, 2), (9, 3)]
)
def test_chunk_size_is_the_largest_divisor_below_the_references_eight(retained, expected):
    # Ruling R44. `util.get_chunks` builds its ragged final chunk with an `np.arange` in a module
    # that never imports numpy, so anything that leaves a remainder raises `NameError` deep inside
    # a prediction. The rule keeps the reference's own 8 wherever it divides -- 16 and 8 are the
    # study's retained counts and 1 is a MAP reference's -- and takes a divisor everywhere else.
    assert sagp.gp._chunk_size(retained) == expected
    assert retained % sagp.gp._chunk_size(retained) == 0


def test_posterior_works_at_a_retained_count_that_eight_does_not_divide():
    # The case R44 is about, end to end: `--nuts 512,252,21` retains 12 draws, and under the old
    # `min(8, S)` the first prediction of the run died in `util.get_chunks` rather than in
    # anything a reader could attribute to a budget.
    S, n_test = 12, 5
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(11).uniform(0.0, 1.0, (n_test, P)))
    fitted = _fitted(
        ("additive", "amplitude"), X, y, _hand_made_samples("amplitude", S, P, seed=12)
    )

    mean, var = fitted.posterior(X_test)

    assert mean.shape == var.shape == (S, n_test)
    assert np.all(np.isfinite(np.asarray(mean))) and np.all(np.asarray(var) > 0.0)


def test_unknown_string_cell_is_rejected():
    # "dsp_map" and "oracle_S" are the only non-CellKey cells; anything else would silently be
    # served the product/lengthscale kernel and reported under its own name.
    with pytest.raises(ValueError, match="unknown cell"):
        _fitted("map", *_data(), _hand_made_samples("lengthscale", 1, P, seed=0)).posterior(
            np.zeros((1, P))
        )


@pytest.mark.parametrize("prior", ["amplitude", "lengthscale"], ids=["amplitude", "lengthscale"])
def test_additive_posterior_mean_is_sum_of_component_means(prior):
    # The additive cells' posterior mean is the sum of its components, because the kernel is the
    # sum of one-coordinate kernels: that identity is what makes `component_means` the components
    # of the surrogate the loop optimizes, and what makes the exact Sobol index of decision 6
    # available at all. Evaluated with every coordinate at the same grid value at once, so the
    # sum on the left is a whole posterior mean rather than one coordinate's slice; there is no
    # constant term to account for, every component being centered under the reference measure.
    S, G = 3, 9
    X, y = _data()
    grid = np.linspace(0.05, 0.95, G)
    fitted = _fitted((("additive", prior)), X, y, _hand_made_samples(prior, S, P, seed=14))

    components = sagp.readouts.component_means(fitted, grid)
    mean, _ = fitted.posterior(np.tile(grid[:, None], (1, P)))

    assert components.shape == (S, P, G)
    assert np.max(np.abs(np.asarray(components.sum(axis=1)) - np.asarray(mean))) < 1.0e-12


def test_alphas_are_the_weights_the_mean_contracts():
    # Task 5's readouts contract the kernel with `alphas()` instead of calling `posterior`, so it
    # has to be exactly the vector `_predict` uses -- otherwise a component mean or a Sobol index
    # would describe a slightly different surrogate than the loop optimizes.
    S = 3
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(9).uniform(0.0, 1.0, (5, P)))
    key = ("additive", "amplitude")
    fitted = _fitted(key, X, y, _hand_made_samples(key[1], S, P, seed=10))

    mean, _ = fitted.posterior(X_test)
    kernel, _ = fitted._kernel()
    alphas = fitted.alphas()
    rebuilt = jnp.stack(
        [kernel(X_test, X, fitted.params(s), 0.0, False) @ alphas[s] for s in range(S)]
    )

    assert np.allclose(np.asarray(rebuilt), np.asarray(mean), rtol=0.0, atol=1.0e-12)


# --- the same four cells as BoTorch PyroModels ---

# Every site each cell's `PyroModel.sample` declares, in declaration order, ending at the
# likelihood. Literals again rather than `Cell.sites`, which lists the old path's names: these are
# the npz schema and the diagnostics' vocabulary on the BoTorch path, and BoTorch's own order for
# the shared prefix is what makes ("product", "lengthscale") its `SaasPyroModel` bit for bit.
PYRO_SITES = {
    "lengthscale": (
        "outputscale",
        "mean",
        "noise",
        "kernel_tausq",
        "_kernel_inv_length_sq",
        "kernel_inv_length_sq",
        "lengthscale",
        "Y",
    ),
    "amplitude": ("mean", "noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell", "Y"),
}

# BoTorch's `sample_noise` returns MIN_INFERRED_NOISE_LEVEL + the sampled value, so the variance
# the likelihood carries is the sampled `noise` shifted by this floor.
NOISE_FLOOR = 1.0e-4


def _torch_data() -> tuple[torch.Tensor, torch.Tensor]:
    """`_data()`'s X with its y standardized (ddof 0), in the (n, D)/(n, 1) shapes BoTorch takes."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (N, P))
    y = rng.normal(size=N)
    z = (y - y.mean()) / y.std()
    return torch.as_tensor(X), torch.as_tensor(z)[:, None]


def _constrained_params(prior: str, seed: int) -> dict[str, jnp.ndarray]:
    """One point of a cell's constrained parameter space: every *sampled* site, no deterministic."""
    rng = np.random.default_rng(seed)
    params = {
        "mean": jnp.asarray(0.2),
        "noise": jnp.asarray(0.05),
        "kernel_tausq": jnp.asarray(0.4),
    }
    if prior == "lengthscale":
        params["outputscale"] = jnp.asarray(1.3)
        params["_kernel_inv_length_sq"] = jnp.asarray(rng.uniform(0.5, 2.0, P))
    else:
        params["_a_sq"] = jnp.asarray(rng.uniform(0.2, 2.0, P))
        params["kernel_ell"] = jnp.asarray(rng.uniform(0.3, 2.0, P))
    return params


def _kernel_params(prior: str, params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """The deterministic kernel arguments `params` implies, in the names `KERNELS` takes."""
    if prior == "lengthscale":
        return {
            "kernel_var": params["outputscale"],
            "kernel_inv_length_sq": params["kernel_tausq"] * params["_kernel_inv_length_sq"],
        }
    return {
        "a_sq": params["kernel_tausq"] * params["_a_sq"],
        "kernel_ell": params["kernel_ell"],
    }


@pytest.mark.parametrize("fixed_noise", [None, 1.0e-3], ids=["learned_noise", "fixed_noise"])
def test_product_lengthscale_pyro_model_is_botorchs_saas_model(fixed_noise):
    # ("product", "lengthscale") *is* SAASBO, and on this path that means BoTorch's
    # `SaasPyroModel` with `sample()` inherited untouched -- so the two log densities at the same
    # parameters must agree bit for bit, which is what lets a later test pin our NUTS run against
    # `fit_fully_bayesian_model_nuts`. Both noise branches, because `train_Yvar` removes the
    # `noise` site and changes the likelihood.
    Xt, zt = _torch_data()
    train_Yvar = None if fixed_noise is None else torch.full_like(zt, fixed_noise)
    ours = sagp.gp.CellGP(Xt, zt, train_Yvar, cell=CELLS[("product", "lengthscale")])
    with botorch.settings.validate_input_scaling(False):
        reference = SaasFullyBayesianSingleTaskGP(Xt, zt, train_Yvar)

    params = _constrained_params("lengthscale", seed=20)
    if fixed_noise is not None:
        del params["noise"]
    ours_density, _ = log_density(ours.pyro_model.sample, (), {}, params)
    reference_density, _ = log_density(reference.pyro_model.sample, (), {}, params)

    difference = abs(float(ours_density) - float(reference_density))
    assert difference < 1.0e-10, f"log densities differ by {difference}"
    # And `alpha` is live: the study sets it, so a different value must move the prior.
    ours.pyro_model.alpha = 0.05
    moved, _ = log_density(ours.pyro_model.sample, (), {}, params)
    assert abs(float(moved) - float(reference_density)) > 1.0e-6


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
@pytest.mark.parametrize("fixed_noise", [None, 1.0e-3], ids=["learned_noise", "fixed_noise"])
def test_pyro_model_trace_sites_per_cell(key, fixed_noise):
    # The site names and their order are part of the specification: they are what the retained
    # draws are keyed by and what the diagnostics report on. `train_Yvar` drops exactly one of
    # them, `noise`; everything else is the cell's own parameterization.
    cell = CELLS[key]
    Xt, zt = _torch_data()
    train_Yvar = None if fixed_noise is None else torch.full_like(zt, fixed_noise)
    gp = sagp.gp.CellGP(Xt, zt, train_Yvar, cell=cell)

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(gp.pyro_model.sample, rng_seed=0)
    ).get_trace()

    expected = tuple(
        site for site in PYRO_SITES[cell.prior] if fixed_noise is None or site != "noise"
    )
    assert tuple(trace) == expected
    # The priors themselves, not merely their sites: alpha is the calibration that makes the four
    # cells comparable, and `ell_prior` the amplitude cells' only lengthscale prior.
    assert float(trace["kernel_tausq"]["fn"].scale) == cell.alpha_default
    if cell.prior == "amplitude":
        assert np.all(np.asarray(trace["kernel_ell"]["fn"].loc) == ELL_PRIOR[0])
        assert np.all(np.asarray(trace["kernel_ell"]["fn"].scale) == ELL_PRIOR[1])


@pytest.mark.parametrize(
    "key",
    [key for key in CELL_KEYS if key != ("product", "lengthscale")],
    ids=[ident for key, ident in zip(CELL_KEYS, CELL_IDS) if key != ("product", "lengthscale")],
)
def test_pyro_model_observation_term_is_the_cells_kernel(key):
    # The three cells whose `sample` is written here rather than inherited must put exactly the
    # old path's kernel into the likelihood -- no jitter of their own, the noise floor BoTorch's
    # `sample_noise` adds and nothing else -- or the BoTorch path would be fitting a different
    # model than the JAX path predicts with.
    cell = CELLS[key]
    Xt, zt = _torch_data()
    X, z = np.asarray(Xt), np.asarray(zt)[:, 0]
    gp = sagp.gp.CellGP(Xt, zt, cell=cell)
    params = _constrained_params(cell.prior, seed=22)

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(
            numpyro.handlers.substitute(gp.pyro_model.sample, data=params), rng_seed=0
        )
    ).get_trace()

    site = trace["Y"]
    K = np.asarray(KERNELS[key][0](X, X, _kernel_params(cell.prior, params), 0.0, False))
    noise = NOISE_FLOOR + float(params["noise"])
    expected = scipy.stats.multivariate_normal(
        mean=np.full(N, float(params["mean"])), cov=K + noise * np.eye(N)
    ).logpdf(z)
    assert abs(float(site["fn"].log_prob(site["value"])) - expected) < 1.0e-8
    assert site["is_observed"]
    assert np.array_equal(np.asarray(site["value"]), z)


def _fake_draws(prior: str, S: int, seed: int) -> dict[str, torch.Tensor]:
    """S draws in the postprocessed form `load_mcmc_samples` takes: torch, leading dimension S.

    Hand-made rather than sampled, so a failure is the loading code's and not the sampler's. The
    deterministic entries are built from the sampled ones so the dict is a consistent draw.
    """
    rng = np.random.default_rng(seed)
    draws = {
        "mean": rng.normal(size=S),
        "noise": rng.uniform(0.01, 0.1, S),
        "kernel_tausq": rng.uniform(0.2, 1.0, S),
    }
    if prior == "lengthscale":
        draws["outputscale"] = rng.uniform(0.5, 2.0, S)
        draws["_kernel_inv_length_sq"] = rng.uniform(0.5, 2.0, (S, P))
        draws["kernel_inv_length_sq"] = (
            draws["_kernel_inv_length_sq"] * draws["kernel_tausq"][:, None]
        )
        draws["lengthscale"] = draws["kernel_inv_length_sq"] ** -0.5
    else:
        draws["_a_sq"] = rng.uniform(0.2, 2.0, (S, P))
        draws["a_sq"] = draws["_a_sq"] * draws["kernel_tausq"][:, None]
        draws["kernel_ell"] = rng.uniform(0.3, 2.0, (S, P))
    return {site: torch.as_tensor(value) for site, value in draws.items()}


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_load_mcmc_samples_round_trip(key):
    # What `load_mcmc_samples` builds has to be the cell the JAX kernel describes, draw by draw:
    # the GPyTorch posterior is what acquisition optimizes, and the JAX kernel is what the model
    # was sampled under. Checked through `posterior` rather than on the parameter tables, so the
    # mean module, the kernel and the likelihood are all pinned at once.
    cell = CELLS[key]
    S, n_test = 3, 5
    Xt, zt = _torch_data()
    X, z = np.asarray(Xt), np.asarray(zt)[:, 0]
    X_test = np.random.default_rng(23).uniform(0.0, 1.0, (n_test, P))
    draws = _fake_draws(cell.prior, S, seed=24)
    gp = sagp.gp.CellGP(Xt, zt, cell=cell)

    gp.load_mcmc_samples(draws)
    gp.eval()
    with torch.no_grad():
        post = gp.posterior(torch.as_tensor(X_test)[:, None, :], observation_noise=True)

    assert post.mean.shape == (n_test, S, 1, 1)
    kernel = KERNELS[key][0]
    for s in range(S):
        if cell.prior == "lengthscale":
            params = {
                "kernel_var": np.asarray(draws["outputscale"][s]),
                "kernel_inv_length_sq": np.asarray(draws["lengthscale"][s]) ** -2.0,
            }
        else:
            params = {
                "a_sq": np.asarray(draws["a_sq"][s]),
                "kernel_ell": np.asarray(draws["kernel_ell"][s]),
            }
        mean_s, noise_s = float(draws["mean"][s]), float(draws["noise"][s])
        K = np.asarray(kernel(X, X, params, 0.0, False)) + noise_s * np.eye(N)
        k_star = np.asarray(kernel(X_test, X, params, 0.0, False))  # (n_test, n)
        k_ss = np.diag(np.asarray(kernel(X_test, X_test, params, 0.0, False)))
        solved = np.linalg.solve(K, k_star.T)  # (n, n_test)
        mean = mean_s + k_star @ np.linalg.solve(K, z - mean_s)
        var = k_ss + noise_s - np.einsum("ij,ji->i", k_star, solved)
        assert np.max(np.abs(np.asarray(post.mean[:, s, 0, 0]) - mean)) < 1.0e-8
        assert np.max(np.abs(np.asarray(post.variance[:, s, 0, 0]) - var)) < 1.0e-8

    # Postprocessing is what produces such a dict out of a NUTS run: every site survives it, as
    # torch float64, because the readouts and the diagnostics read sites BoTorch itself drops.
    raw = {site: jnp.asarray(np.asarray(value)) for site, value in draws.items()}
    if cell.prior == "lengthscale":
        del raw["lengthscale"]  # the deterministic postprocessing recomputes
    processed = gp.pyro_model.postprocess_mcmc_samples(raw)
    assert set(processed) == set(draws)
    assert all(value.dtype is torch.float64 for value in processed.values())


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_gp_binds_the_cells_pyro_model(key):
    # `CellGP` differs from `SaasFullyBayesianSingleTaskGP` in one thing: which `PyroModel` it
    # instantiates, chosen per instance rather than per class, so one class serves all four cells.
    cell = CELLS[key]
    Xt, zt = _torch_data()

    gp = sagp.gp.CellGP(Xt, zt, cell=cell)

    assert isinstance(gp.pyro_model, cell.pyro_model)
    assert gp.cell is cell
    assert gp.pyro_model.alpha == cell.alpha_default  # alpha=None takes the cell's calibration
    assert sagp.gp.CellGP(Xt, zt, cell=cell, alpha=0.07).pyro_model.alpha == 0.07
