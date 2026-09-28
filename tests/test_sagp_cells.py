"""Tests for sagp.gp's BoTorch cells and their prediction: models, sites and `FittedGP`.

Three properties carry the 2x2 comparison under its two prior families. ("product",
"lengthscale") must be BoTorch's own `SaasPyroModel` -- checked as an equal log density, since
`sample()` is inherited and only the kernel/prior block may differ; every R2-D2 cell must be its
half-Cauchy twin with another prior block and nothing else; and every cell must declare exactly
the sites named here, in this order, because those names are what the retained draws are keyed by
and what the diagnostics report on. Their kernels are pinned to the JAX ones twice over, through
the likelihood term `sample` builds and through the GPyTorch posterior `load_mcmc_samples`
produces.

Prediction is held to the same standard: `FittedGP.posterior` is a reshape of that GPyTorch
posterior, so what is tested here is that one code path serves every cell, that the oracle's
`active` restriction is the same computation as having been handed the narrow design, and that the
two identities the readouts rest on -- the additive mean is its components plus the intercept, and
`alphas()` are the weights that mean contracts -- hold against the posterior itself.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp
import sagp.readouts
from sagp.gp import (
    CELLS,
    ELL_PRIOR,
    KERNELS,
    R2D2_A,
    R2D2_B,
    R2D2_FORM,
    R2D2_K,
    R2D2_RHO_SCALE,
    FittedGP,
)

import botorch.settings
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import scipy.stats
import torch
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from numpyro.infer.util import log_density
from scipy.special import gammainccinv, gammaincinv

N = 12
P = 4

CELL_KEYS = list(CELLS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]

# Each R2-D2 cell and the half-Cauchy cell it must differ from in the prior block only.
R2D2_TWINS = {("additive", "amplitude_r2d2"): ("additive", "amplitude"),
              ("product", "amplitude_r2d2"): ("product", "amplitude"),
              ("additive", "lengthscale_r2d2"): ("additive", "lengthscale"),
              ("product", "lengthscale_r2d2"): ("product", "lengthscale")}
_SHARED = ("sample", "load_mcmc_samples", "get_dummy_mcmc_samples", "sample_mean",
           "sample_noise", "sample_observations", "sample_ell", "ell_prior",
           "_jax_kernel", "_torch_kernel", "sample_outputscale")

# The R2-D2 prior's sampled sites in each form, as literals: the forms differ in the global site
# only, and `sagp.r2d2.SAMPLED_SITES` is what these pin.
R2D2_PRIOR_SITES = {"reference": ("r2d2_R2", "r2d2_z_lam"), "tied": ("r2d2_z_xi", "r2d2_z_lam")}


def _data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """N points uniform on [0,1]^P with standardized responses: the shape a fit sees."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=N)
    return jnp.asarray(rng.uniform(0.0, 1.0, (N, P))), jnp.asarray((y - y.mean()) / y.std())


# --- prediction ---


def _hand_made_samples(native_site: str, S: int, D: int, seed: int) -> dict[str, jnp.ndarray]:
    """Positive kernel parameters with the shapes a fit's `samples` has: leading dimension S.

    Hand-made rather than sampled, because these tests are about prediction's plumbing -- shapes,
    the `active` restriction, the two readout identities -- and running NUTS to obtain draws would
    make them slow and make a failure ambiguous between the sampler and the code under test. The
    dict carries what `load_mcmc_samples` reads, so it is a draw the GPyTorch model can be built
    from and not merely a bag of kernel arguments. Keyed by the cell's parameterization, its
    `native_site`, which is all prediction reads.
    """
    rng = np.random.default_rng(seed)
    if native_site == "a_sq":
        samples = {
            "a_sq": rng.uniform(0.1, 2.0, (S, D)),
            "kernel_ell": rng.uniform(0.2, 2.0, (S, D)),
        }
    else:
        samples = {
            "outputscale": np.full(S, 1.3),
            "kernel_inv_length_sq": rng.uniform(0.1, 5.0, (S, D)),
        }
        samples["lengthscale"] = samples["kernel_inv_length_sq"] ** -0.5
    samples["mean"] = rng.normal(size=S)
    samples["noise"] = rng.uniform(0.01, 0.1, S)
    return {site: jnp.asarray(draws) for site, draws in samples.items()}


def _fitted(cell, X, y, samples, *, fixed_noise=None, active=None) -> FittedGP:
    """A `FittedGP` with the fit-status fields a prediction test does not care about filled in."""
    return FittedGP.from_draws(cell, X, y, samples, fixed_noise=fixed_noise, active=active)


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_posterior_shapes_all_cells(key):
    # One code path serves every cell: every cell's draws reach its torch kernel and its
    # likelihood, and the predictive variance stays a variance.
    S, n_test = 3, 5
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(3).uniform(0.0, 1.0, (n_test, P)))
    fitted = _fitted(key, X, y, _hand_made_samples(CELLS[key].native_site, S, P, seed=4))

    mean, var = fitted.posterior(X_test)

    assert mean.shape == var.shape == (S, n_test)
    assert np.all(np.isfinite(np.asarray(mean)))
    assert np.all(np.asarray(var) > 0.0)


def test_oracle_active_restricts_columns():
    # Task 7's oracle fits on X[:, S] alone but keeps the full-D X_train, because the BO loop reads
    # it. Restricting through the model's own `FilterFeatures` transform must therefore be the same
    # computation as having been handed the narrow matrix in the first place -- the gather changes
    # the tensor's layout and so the last bits, and nothing else. The `cell` string also pins the
    # MAP references onto ("product", "lengthscale")'s kernel.
    active = np.asarray([0, 2])
    S = 2
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(5).uniform(0.0, 1.0, (6, P)))
    samples = _hand_made_samples("kernel_inv_length_sq", S, len(active), seed=6)

    restricted = _fitted("oracle_S", X, y, samples, active=active).posterior(X_test)
    narrow = _fitted(("product", "lengthscale"), X[:, active], y, samples).posterior(
        X_test[:, active]
    )

    for ours, theirs in zip(restricted, narrow):
        assert np.max(np.abs(np.asarray(ours) - np.asarray(theirs))) < 1.0e-12


def test_unknown_string_cell_is_rejected():
    # "dsp_map" and "oracle_S" are the only non-CellKey cells; anything else would silently be
    # served the product/lengthscale kernel and reported under its own name.
    with pytest.raises(ValueError, match="unknown cell"):
        _fitted("map", *_data(), _hand_made_samples("kernel_inv_length_sq", 1, P, seed=0))


ADDITIVE_KEYS = [key for key in CELL_KEYS if key[0] == "additive"]


@pytest.mark.parametrize("key", ADDITIVE_KEYS, ids=["/".join(key) for key in ADDITIVE_KEYS])
def test_additive_posterior_mean_equals_sum_of_component_means(key):
    # The additive cells' posterior mean is the constant mean plus the sum of its components,
    # because the kernel is the sum of one-coordinate kernels: that identity is what makes
    # `component_means` the components of the surrogate the loop optimizes, and what makes the
    # exact Sobol index of decision 6 available at all. Evaluated with every coordinate at the same
    # grid value at once, so the sum on the left is a whole posterior mean rather than one
    # coordinate's slice; every component is centered, so the intercept is the whole of the
    # constant term.
    S, G = 3, 9
    X, y = _data()
    grid = np.linspace(0.05, 0.95, G)
    fitted = _fitted(key, X, y, _hand_made_samples(CELLS[key].native_site, S, P, seed=14))

    components = sagp.readouts.component_means(fitted, grid)
    mean, _ = fitted.posterior(np.tile(grid[:, None], (1, P)))
    rebuilt = components.sum(axis=1) + fitted.means()[:, None]

    assert components.shape == (S, P, G)
    assert np.max(np.abs(np.asarray(rebuilt) - np.asarray(mean))) < 1.0e-10


def test_alphas_are_the_weights_the_mean_contracts():
    # Task 5's readouts contract the kernel with `alphas()` instead of calling `posterior`, so it
    # has to be exactly the vector the posterior mean uses -- otherwise a component mean or a Sobol
    # index would describe a slightly different surrogate than the loop optimizes.
    S = 3
    X, y = _data()
    X_test = jnp.asarray(np.random.default_rng(9).uniform(0.0, 1.0, (5, P)))
    key = ("additive", "amplitude")
    fitted = _fitted(key, X, y, _hand_made_samples(CELLS[key].native_site, S, P, seed=10))

    mean, _ = fitted.posterior(X_test)
    kernel = KERNELS[key][0]
    alphas, means = fitted.alphas(), fitted.means()
    rebuilt = jnp.stack(
        [means[s] + kernel(X_test, X, fitted.params(s)) @ alphas[s] for s in range(S)]
    )

    assert np.allclose(np.asarray(rebuilt), np.asarray(mean), rtol=0.0, atol=1.0e-10)


# --- the cells as BoTorch PyroModels ---

# Every site each R2-D2 cell's `PyroModel.sample` declares, per form: its twin's sites with the
# half-Cauchy block replaced by the form's sampled sites, and the same deterministic native.
R2D2_PYRO_SITES = {
    form: {
        "lengthscale_r2d2": (
            "outputscale", "mean", "noise", *sites, "kernel_inv_length_sq", "lengthscale", "Y",
        ),
        "amplitude_r2d2": ("mean", "noise", *sites, "a_sq", "kernel_ell", "Y"),
    }
    for form, sites in R2D2_PRIOR_SITES.items()
}

# Every site each cell's `PyroModel.sample` declares, in declaration order, ending at the
# likelihood. Literals rather than `Cell.sites`, so the trace pins the *names*: these are the npz
# schema and the diagnostics' vocabulary, and BoTorch's own order for the shared prefix is what
# makes ("product", "lengthscale") its `SaasPyroModel` bit for bit. The registry is checked against
# the same literals; the R2-D2 cells' are those of the form `sagp.gp` runs.
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
    **R2D2_PYRO_SITES[R2D2_FORM],
}

# BoTorch's `sample_noise` returns MIN_INFERRED_NOISE_LEVEL + the sampled value, so the variance
# the likelihood carries is the sampled `noise` shifted by this floor.
NOISE_FLOOR = 1.0e-4


def _torch_data() -> tuple[torch.Tensor, torch.Tensor]:
    """`_data()`'s X and y in the (n, D)/(n, 1) shapes BoTorch takes."""
    X, z = _data()
    return torch.as_tensor(np.array(X)), torch.as_tensor(np.array(z))[:, None]


def _set_r2d2_form(pyro_model, form: str) -> None:
    """Run `pyro_model`'s R2-D2 block in `form`: what switching `R2D2_FORM` does, per instance.

    The tied form fixes a = k * dim, which `r2d2_a` None states; the reference form keeps the
    class's `a`.
    """
    pyro_model.r2d2_form = form
    if form == "tied":
        pyro_model.r2d2_a = None


def _gamma_quantile(z: np.ndarray, k: float) -> np.ndarray:
    """F_k^-1(Phi(z)), the Gamma(k, 1) quantile of a standard normal z, from the smaller tail."""
    z = np.asarray(z)
    return np.where(z < 0.0, gammaincinv(k, scipy.stats.norm.cdf(z)),
                    gammainccinv(k, scipy.stats.norm.sf(z)))


def _r2d2_theta(draws: dict[str, np.ndarray], form: str) -> np.ndarray:
    """theta = omega * phi from an R2-D2 draw's sampled sites, in SciPy: no code shared with sagp.

    lam_i is the Gamma(R2D2_K) quantile of each copula coordinate. "reference": omega = R2 / (1 -
    R2) and phi = lam / sum lam; "tied": theta = lam / xi, xi the Gamma(R2D2_B) quantile of
    `r2d2_z_xi`. Coordinates last, any leading shape.
    """
    lam = _gamma_quantile(draws["r2d2_z_lam"], R2D2_K)
    if form == "reference":
        R2 = np.asarray(draws["r2d2_R2"])[..., None]
        return R2 / (1.0 - R2) * lam / lam.sum(axis=-1, keepdims=True)
    return lam / _gamma_quantile(draws["r2d2_z_xi"], R2D2_B)[..., None]


def _constrained_params(key, seed: int) -> dict[str, jnp.ndarray]:
    """One point of a cell's constrained parameter space: every *sampled* site, no deterministic."""
    rng = np.random.default_rng(seed)
    params = {"mean": jnp.asarray(0.2), "noise": jnp.asarray(0.05)}
    if key in R2D2_TWINS:
        # The running form's global site (a valid R2, or a copula coordinate), then its D shares.
        global_site, lam_site = R2D2_PRIOR_SITES[R2D2_FORM]
        params[global_site] = jnp.asarray(0.6 if global_site == "r2d2_R2" else 0.3)
        params[lam_site] = jnp.asarray(rng.normal(size=P))
    else:
        params["kernel_tausq"] = jnp.asarray(0.4)
    if CELLS[key].native_site == "kernel_inv_length_sq":
        params["outputscale"] = jnp.asarray(1.3)
        if key not in R2D2_TWINS:
            params["_kernel_inv_length_sq"] = jnp.asarray(rng.uniform(0.5, 2.0, P))
    else:
        if key not in R2D2_TWINS:
            params["_a_sq"] = jnp.asarray(rng.uniform(0.2, 2.0, P))
        params["kernel_ell"] = jnp.asarray(rng.uniform(0.3, 2.0, P))
    return params


def _kernel_params(key, params: dict[str, jnp.ndarray], trace) -> dict[str, jnp.ndarray]:
    """The deterministic kernel arguments `params` implies, in the names `KERNELS` takes.

    A half-Cauchy native is tausq times its local scale, recomputed here; an R2-D2 native is the
    whole prior block's output, read off the trace's deterministic site (`_r2d2_theta` checks that
    site against SciPy elsewhere).
    """
    native_site = CELLS[key].native_site
    if key in R2D2_TWINS:
        native = trace[native_site]["value"]
        if native_site == "kernel_inv_length_sq":
            return {"outputscale": params["outputscale"], "kernel_inv_length_sq": native}
        return {"a_sq": native, "kernel_ell": params["kernel_ell"]}
    if native_site == "kernel_inv_length_sq":
        return {
            "outputscale": params["outputscale"],
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
    # parameters must agree bit for bit, which is what lets `test_sagp_inference.py` pin our NUTS
    # run against `fit_fully_bayesian_model_nuts`. Both noise branches, because `train_Yvar`
    # removes the `noise` site and changes the likelihood.
    Xt, zt = _torch_data()
    train_Yvar = None if fixed_noise is None else torch.full_like(zt, fixed_noise)
    ours = sagp.gp.CellGP(Xt, zt, train_Yvar, cell=CELLS[("product", "lengthscale")])
    with botorch.settings.validate_input_scaling(False):
        reference = SaasFullyBayesianSingleTaskGP(Xt, zt, train_Yvar)

    params = _constrained_params(("product", "lengthscale"), seed=20)
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
    # The registry says the same thing: `Cell.sites` is that trace minus the likelihood.
    assert cell.sites == tuple(site for site in PYRO_SITES[cell.prior] if site != "Y")
    # The priors themselves, not merely their sites: alpha is the calibration that makes the
    # cells comparable, and `ell_prior` the amplitude cells' only lengthscale prior.
    if key in R2D2_TWINS:
        # An R2-D2 cell's alpha is k, the Dirichlet concentration per coordinate; its R2 prior
        # is Beta(a, b) at the calibrated constants.
        assert gp.pyro_model.alpha == cell.alpha_default == R2D2_K
        if R2D2_FORM == "reference":
            beta = trace["r2d2_R2"]["fn"]
            assert isinstance(beta, dist.Beta)
            assert (float(beta.concentration1), float(beta.concentration0)) == (R2D2_A, R2D2_B)
    else:
        assert float(trace["kernel_tausq"]["fn"].scale) == cell.alpha_default
    if cell.native_site == "a_sq":
        assert np.all(np.asarray(trace["kernel_ell"]["fn"].loc) == ELL_PRIOR[0])
        assert np.all(np.asarray(trace["kernel_ell"]["fn"].scale) == ELL_PRIOR[1])


@pytest.mark.parametrize("form", ["reference", "tied"])
@pytest.mark.parametrize("key", list(R2D2_TWINS), ids=["/".join(k) for k in R2D2_TWINS])
def test_both_r2d2_forms_trace_the_same_natives(key, form):
    # Switching the form is one constant, read through the class attribute `r2d2_form`: set on
    # the instance, it must swap the prior's sampled sites for that form's and leave every other
    # site, and its place, as it was -- and the deterministic native must be theta = omega * phi
    # (times the rho scale on the lengthscale cells), as SciPy computes it from the same sites.
    cell = CELLS[key]
    gp = sagp.gp.CellGP(*_torch_data(), cell=cell)
    _set_r2d2_form(gp.pyro_model, form)

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(gp.pyro_model.sample, rng_seed=0)
    ).get_trace()

    assert tuple(trace) == R2D2_PYRO_SITES[form][cell.prior]
    for site in R2D2_PRIOR_SITES[form]:
        assert trace[site]["type"] == "sample"
    theta = _r2d2_theta(
        {site: np.asarray(trace[site]["value"]) for site in R2D2_PRIOR_SITES[form]}, form
    )
    scale = 1.0 if cell.native_site == "a_sq" else R2D2_RHO_SCALE
    np.testing.assert_allclose(
        np.asarray(trace[cell.native_site]["value"]), scale * theta, rtol=1.0e-9, atol=0.0
    )


@pytest.mark.parametrize("key", list(R2D2_TWINS), ids=["/".join(k) for k in R2D2_TWINS])
def test_an_r2d2_cell_differs_from_its_twin_in_the_prior_block_only(key):
    cell, twin = CELLS[key], CELLS[R2D2_TWINS[key]]
    assert (cell.structure, cell.native_site) == (twin.structure, twin.native_site)
    assert cell.kernel is twin.kernel and cell.kernel_diag is twin.kernel_diag
    assert KERNELS[key][0] is KERNELS[twin.key][0] and KERNELS[key][1] is KERNELS[twin.key][1]
    assert issubclass(cell.pyro_model, twin.pyro_model)
    for name in _SHARED:
        if hasattr(twin.pyro_model, name):
            assert getattr(cell.pyro_model, name) is getattr(twin.pyro_model, name), name
    block = "sample_amplitudes" if cell.native_site == "a_sq" else "sample_inv_length_sq"
    assert getattr(cell.pyro_model, block) is not getattr(twin.pyro_model, block)


@pytest.mark.parametrize(
    "key",
    [key for key in CELL_KEYS if key != ("product", "lengthscale")],
    ids=[ident for key, ident in zip(CELL_KEYS, CELL_IDS) if key != ("product", "lengthscale")],
)
def test_pyro_model_observation_term_is_the_cells_kernel(key):
    # The cells whose `sample` is written here rather than inherited must put exactly the
    # registry's kernel into the likelihood -- no jitter of their own, the noise floor BoTorch's
    # `sample_noise` adds and nothing else -- or the model NUTS samples would not be the model the
    # readouts describe. This is also what pins each registry entry to its own kernel: site names
    # encode only the sparsity prior, so ("additive", "amplitude") holding the *product* kernel
    # would trace and sample exactly as it should while silently comparing the wrong two cells.
    # ("product", "lengthscale_r2d2") is included although its `sample` is BoTorch's: its prior
    # block is written here, and the native that block hands the ARD kernel is what is checked.
    cell = CELLS[key]
    Xt, zt = _torch_data()
    X, z = np.asarray(Xt), np.asarray(zt)[:, 0]
    gp = sagp.gp.CellGP(Xt, zt, cell=cell)
    params = _constrained_params(key, seed=22)

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(
            numpyro.handlers.substitute(gp.pyro_model.sample, data=params), rng_seed=0
        )
    ).get_trace()

    site = trace["Y"]
    assert cell.kernel is KERNELS[key][0] and cell.kernel_diag is KERNELS[key][1]
    K = np.asarray(KERNELS[key][0](X, X, _kernel_params(key, params, trace)))
    noise = NOISE_FLOOR + float(params["noise"])
    expected = scipy.stats.multivariate_normal(
        mean=np.full(N, float(params["mean"])), cov=K + noise * np.eye(N)
    ).logpdf(z)
    assert abs(float(site["fn"].log_prob(site["value"])) - expected) < 1.0e-8
    assert site["is_observed"]
    assert np.array_equal(np.asarray(site["value"]), z)


def _fake_draws(key, S: int, seed: int) -> dict[str, torch.Tensor]:
    """S draws in the postprocessed form `load_mcmc_samples` takes: torch, leading dimension S.

    Hand-made rather than sampled, so a failure is the loading code's and not the sampler's. The
    deterministic entries are built from the sampled ones so the dict is a consistent draw: an
    R2-D2 draw carries its form's sampled sites and the native `_r2d2_theta` makes of them.
    """
    rng = np.random.default_rng(seed)
    draws = {"mean": rng.normal(size=S), "noise": rng.uniform(0.01, 0.1, S)}
    if key in R2D2_TWINS:
        for site in R2D2_PRIOR_SITES[R2D2_FORM]:
            if site == "r2d2_R2":
                draws[site] = rng.uniform(0.2, 0.8, S)
            else:
                draws[site] = rng.normal(size=(S, P) if site == "r2d2_z_lam" else S)
        theta = _r2d2_theta(draws, R2D2_FORM)
    else:
        draws["kernel_tausq"] = rng.uniform(0.2, 1.0, S)
    if CELLS[key].native_site == "kernel_inv_length_sq":
        draws["outputscale"] = rng.uniform(0.5, 2.0, S)
        if key in R2D2_TWINS:
            draws["kernel_inv_length_sq"] = R2D2_RHO_SCALE * theta
        else:
            draws["_kernel_inv_length_sq"] = rng.uniform(0.5, 2.0, (S, P))
            draws["kernel_inv_length_sq"] = (
                draws["_kernel_inv_length_sq"] * draws["kernel_tausq"][:, None]
            )
        draws["lengthscale"] = draws["kernel_inv_length_sq"] ** -0.5
    else:
        if key in R2D2_TWINS:
            draws["a_sq"] = theta
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
    draws = _fake_draws(key, S, seed=24)
    gp = sagp.gp.CellGP(Xt, zt, cell=cell)

    gp.load_mcmc_samples(draws)
    gp.eval()
    with torch.no_grad():
        post = gp.posterior(torch.as_tensor(X_test)[:, None, :], observation_noise=True)

    assert post.mean.shape == (n_test, S, 1, 1)
    kernel = KERNELS[key][0]
    for s in range(S):
        if cell.native_site == "kernel_inv_length_sq":
            params = {
                "outputscale": np.asarray(draws["outputscale"][s]),
                "kernel_inv_length_sq": np.asarray(draws["lengthscale"][s]) ** -2.0,
            }
        else:
            params = {
                "a_sq": np.asarray(draws["a_sq"][s]),
                "kernel_ell": np.asarray(draws["kernel_ell"][s]),
            }
        mean_s, noise_s = float(draws["mean"][s]), float(draws["noise"][s])
        K = np.asarray(kernel(X, X, params)) + noise_s * np.eye(N)
        k_star = np.asarray(kernel(X_test, X, params))  # (n_test, n)
        k_ss = np.diag(np.asarray(kernel(X_test, X_test, params)))
        solved = np.linalg.solve(K, k_star.T)  # (n, n_test)
        mean = mean_s + k_star @ np.linalg.solve(K, z - mean_s)
        var = k_ss + noise_s - np.einsum("ij,ji->i", k_star, solved)
        assert np.max(np.abs(np.asarray(post.mean[:, s, 0, 0]) - mean)) < 1.0e-8
        assert np.max(np.abs(np.asarray(post.variance[:, s, 0, 0]) - var)) < 1.0e-8

    # Postprocessing is what produces such a dict out of a NUTS run: every site survives it, as
    # torch float64, because the readouts and the diagnostics read sites BoTorch itself drops.
    raw = {site: jnp.asarray(np.asarray(value)) for site, value in draws.items()}
    if cell.native_site == "kernel_inv_length_sq":
        del raw["lengthscale"]  # the deterministic postprocessing recomputes
    processed = gp.pyro_model.postprocess_mcmc_samples(raw)
    assert set(processed) == set(draws)
    assert all(value.dtype is torch.float64 for value in processed.values())
    if cell.native_site == "kernel_inv_length_sq":
        # Recomputed from the draw's own rho, whichever sites that rho is a function of.
        assert torch.allclose(processed["lengthscale"], draws["lengthscale"], rtol=1e-14, atol=0)


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_gp_binds_the_cells_pyro_model(key):
    # `CellGP` differs from `SaasFullyBayesianSingleTaskGP` in one thing: which `PyroModel` it
    # instantiates, chosen per instance rather than per class, so one class serves every cell.
    cell = CELLS[key]
    Xt, zt = _torch_data()

    gp = sagp.gp.CellGP(Xt, zt, cell=cell)

    assert isinstance(gp.pyro_model, cell.pyro_model)
    assert gp.cell is cell
    assert gp.pyro_model.alpha == cell.alpha_default  # alpha=None takes the cell's calibration
    assert sagp.gp.CellGP(Xt, zt, cell=cell, alpha=0.07).pyro_model.alpha == 0.07
