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

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp
import sagp.readouts
from sagp.gp import (
    CELLS,
    ELL_PRIOR,
    FittedGP,
    model_product_lengthscale,
)

from functools import partial

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
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
        [kernel(X_test, X, fitted._params(s), 0.0, False) @ alphas[s] for s in range(S)]
    )

    assert np.allclose(np.asarray(rebuilt), np.asarray(mean), rtol=0.0, atol=1.0e-12)
