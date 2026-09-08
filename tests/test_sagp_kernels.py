"""Tests for sagp.gp's kernels: the centered Matern-5/2 component and the four cell kernels.

Three things have to hold for the thesis's 2x2 comparison to mean anything. The centered
component must reproduce `synthobj.kernel` -- and through it Gate 1's check script,
`Research Context/research direction/Kernel normalization check.py` -- in float64, so
the surrogate and the synthetic objectives share one definition of a component. Normalization
must actually deliver unit marginal variance under U[0,1], since that is the only thing that
makes an amplitude threshold well defined. And ("product", "lengthscale") must be
`saasgp.matern_kernel` bit-for-bit, because that cell is the SAASBO baseline the other three are
measured against.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
from sagp.gp import (
    ACTIVE_EPS,
    ALPHA_AMPLITUDE,
    ALPHA_LENGTHSCALE,
    ELL_EPS,
    GL_NODES,
    GL_WEIGHTS,
    KERNELS,
    RHO_EPS,
    cell_kernel_diag,
    centered_matern52_1d,
    kernel_product_lengthscale,
    v_of_ell,
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import saasgp
from synthobj.kernel import centered, normalized
from synthobj.kernel import v as v_numpy

D = 7
CELL_KEYS = list(KERNELS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]

# The three cells built out of the centered component; ("product", "lengthscale") is pinned
# against `saasgp.matern_kernel` instead.
CENTERED_CELL_KEYS = [key for key in CELL_KEYS if key != ("product", "lengthscale")]
CENTERED_CELL_IDS = ["/".join(key) for key in CENTERED_CELL_KEYS]

# The lengthscales the brief exercises the centered component at: rough, middling and so smooth
# that v(ell) is down at 0.015 and the centering is a near-total cancellation.
ELLS_CENTERED = [0.06, 0.5, 3.0]

# Table 1 of the check script, printed to five decimals.
ELLS_TABLE = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
V_TABLE = [0.88574, 0.78149, 0.60296, 0.28218, 0.10680, 0.03183, 0.01478, 0.00546, 0.00138, 0.00015]

# Every cell's `cell_kernel_diag` agrees with its own matrix diagonal to float64 round-off,
# except ("product", "lengthscale"): the vendored `matern_kernel` clips dsq to 1e-12 in the sqrt
# that makes `exponent` but leaves it at 0 in `poly`, so its diagonal is var * (1 - 2.5e-12),
# not var. That floor is the reference's own, and reproducing it in the diagonal would break
# `cell_kernel_diag + noise + 1e-6 == saasgp.kernel_diag(var, noise)`, so the diagonal stays
# var * ones and the tolerance absorbs the floor instead (1.3 * 2.5e-12 = 3.3e-12, measured).
DIAG_ATOL = dict.fromkeys(CELL_KEYS, 1.0e-12)
DIAG_ATOL[("product", "lengthscale")] = 1.0e-11

# Plan decision D2's simulation, at the size it was run during planning: 10^4 prior draws of the
# half-Cauchy scale mixture over D_ALPHA coordinates.
N_DRAWS = 10**4
D_ALPHA = 100


def _params() -> dict[str, jax.Array | float]:
    """The brief's random positive parameters: a_sq ~ U(0,1), ell ~ U(0.1,2), var = 1.3, rho ~ U(0.1,5).

    One dict carrying all four cells' parameters, so the same numbers drive every kernel; each
    kernel reads only the two keys it is parameterized by and ignores the rest.
    """
    rng = np.random.default_rng(0)
    return {
        "a_sq": jnp.asarray(rng.uniform(0.0, 1.0, D)),
        "kernel_ell": jnp.asarray(rng.uniform(0.1, 2.0, D)),
        "kernel_var": 1.3,
        "kernel_inv_length_sq": jnp.asarray(rng.uniform(0.1, 5.0, D)),
    }


def _points(n: int, seed: int) -> jax.Array:
    """n points drawn uniformly from the unit cube [0,1]^D, the domain every cell assumes."""
    return jnp.asarray(np.random.default_rng(seed).uniform(0.0, 1.0, (n, D)))


def _numpy_oracle(key, X, Z, params) -> np.ndarray:
    """Cell `key` rebuilt one coordinate at a time out of `synthobj.kernel`, in numpy.

    This is the file's only independent statement of *which* centering each cell uses:
    `normalized` for the amplitude cells, where a_sq_i has to be component i's variance under
    U[0,1]; plain `centered` at ell_i = rho_i^-0.5 for the lengthscale cell, whose shrinkage
    lives in v(ell_i) and would be undone by dividing it out. Nothing else here can see those
    two choices -- the JAX kernel and its `_diag_*` helper share them, so a flag flipped in both
    cancels -- and nothing else pins the *sign* of the rho -> ell exponent.
    """
    X, Z = np.asarray(X), np.asarray(Z)
    if key[1] == "amplitude":
        a_sq, ell = np.asarray(params["a_sq"]), np.asarray(params["kernel_ell"])
        terms = np.array([a_sq[i] * normalized(X[:, i], Z[:, i], ell[i]) for i in range(D)])
        return terms.sum(axis=0) if key[0] == "additive" else (1.0 + terms).prod(axis=0)
    ell = np.asarray(params["kernel_inv_length_sq"]) ** -0.5
    terms = np.array([centered(X[:, i], Z[:, i], ell[i]) for i in range(D)])
    return params["kernel_var"] * terms.sum(axis=0)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("ell", ELLS_CENTERED)
def test_centered_component_integrates_to_zero_under_the_reference_measure(ell, normalize):
    # The defining property of the centering: every component is orthogonal to the constants
    # under U[0,1], so no component can carry any of the function's mean.
    x = jnp.asarray(np.random.default_rng(0).uniform(0.0, 1.0, 20))
    k_tilde = centered_matern52_1d(GL_NODES, x, ell, normalize=normalize)  # (64, 20)

    assert float(jnp.max(jnp.abs(GL_WEIGHTS @ k_tilde))) < 1.0e-10


def test_v_of_ell_matches_the_numpy_reference():
    expected = np.array([v_numpy(ell) for ell in ELLS_TABLE])

    # The (D,) path is what the cell kernels use; the scalar path is what a single component
    # uses. Both have to be the same number synthobj centered its objectives with.
    np.testing.assert_allclose(
        np.asarray(v_of_ell(jnp.asarray(ELLS_TABLE))), expected, atol=1.0e-12, rtol=0.0
    )
    for ell, want in zip(ELLS_TABLE, expected):
        got = v_of_ell(ell)
        assert got.shape == ()
        assert abs(float(got) - want) < 1.0e-12


def test_v_of_ell_matches_the_check_script_table():
    np.testing.assert_allclose(
        np.asarray(v_of_ell(jnp.asarray(ELLS_TABLE))), V_TABLE, atol=1.0e-3, rtol=0.0
    )


@pytest.mark.parametrize("ell", ELLS_CENTERED)
def test_normalized_component_has_unit_quadrature_mean_variance(ell):
    # E_nu[kbar(x, x)] = 1 whatever the lengthscale is what makes a_sq_i a component's variance
    # under the reference measure rather than a lengthscale-dependent quantity.
    k_bar = centered_matern52_1d(GL_NODES, GL_NODES, ell, normalize=True)

    assert abs(float(GL_WEIGHTS @ jnp.diagonal(k_bar)) - 1.0) < 1.0e-12


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_kernel_is_positive_semidefinite(key):
    X = _points(50, seed=1)

    eigenvalues = jnp.linalg.eigvalsh(KERNELS[key][0](X, X, _params(), 0.0, False))

    assert float(eigenvalues.min()) > -1.0e-10


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_include_noise_adds_noise_plus_jitter_to_the_diagonal(key):
    X, params = _points(50, seed=1), _params()
    kernel = KERNELS[key][0]

    added = kernel(X, X, params, 0.1, True) - kernel(X, X, params, 0.1, False)

    np.testing.assert_allclose(
        np.asarray(added), (0.1 + 1.0e-6) * np.eye(50), atol=1.0e-12, rtol=0.0
    )


@pytest.mark.parametrize("include_noise", [False, True])
def test_product_lengthscale_cell_is_the_reference_matern_kernel(include_noise):
    # This cell is SAASBO; anything but bit-for-bit equality would make the baseline a different
    # model from the reference. Z is square because matern_kernel's noise term assumes it.
    rng = np.random.default_rng(2)
    X, Z = jnp.asarray(rng.uniform(size=(9, D))), jnp.asarray(rng.uniform(size=(9, D)))
    var, rho, noise = 1.3, jnp.asarray(rng.uniform(0.1, 5.0, D)), 0.1
    params = {"kernel_var": var, "kernel_inv_length_sq": rho}

    ours = kernel_product_lengthscale(X, Z, params, noise, include_noise)

    assert np.array_equal(
        np.asarray(ours), np.asarray(saasgp.matern_kernel(X, Z, var, rho, noise, include_noise))
    )


@pytest.mark.parametrize("key", CENTERED_CELL_KEYS, ids=CENTERED_CELL_IDS)
def test_centered_cells_match_the_numpy_oracle(key):
    # X and Z differ in length: the only test here that exercises the n != m path the acquisition
    # optimizer always takes.
    rng = np.random.default_rng(5)
    X = jnp.asarray(rng.uniform(0.0, 1.0, (11, D)))
    Z = jnp.asarray(rng.uniform(0.0, 1.0, (9, D)))
    params = _params()

    K = KERNELS[key][0](X, Z, params, 0.0, False)

    np.testing.assert_allclose(
        np.asarray(K), _numpy_oracle(key, X, Z, params), atol=1.0e-12, rtol=0.0
    )


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_kernel_diag_matches_the_kernel_diagonal(key):
    X, params = _points(12, seed=3), _params()

    diagonal = cell_kernel_diag(key, X, params)

    np.testing.assert_allclose(
        np.asarray(diagonal),
        np.asarray(jnp.diagonal(KERNELS[key][0](X, X, params, 0.0, False))),
        atol=DIAG_ATOL[key],
        rtol=0.0,
    )


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_kernel_gradients_are_finite(key):
    # NUTS differentiates every kernel at every leapfrog step, and X always has zero distances on
    # its own diagonal; row 1 repeats row 0 so zero distance also occurs off the diagonal, where
    # a sqrt(sum of squares) formulation would produce NaN rather than the true derivative 0.
    X = np.random.default_rng(4).uniform(0.0, 1.0, (12, D))
    X[1] = X[0]
    X = jnp.asarray(X)
    kernel = KERNELS[key][0]

    grads = jax.grad(lambda p: kernel(X, X, p, 0.0, False).sum())(_params())

    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads))


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_kernel_gradients_match_central_differences(key):
    """Finite and *correct*: one entry of the gradient NUTS follows, against the kernel's values.

    Every other kernel test here reads the kernel's values, and `..._are_finite` above reads only
    the gradient's shape and finiteness -- so a centering term differentiated with the wrong sign,
    or a `v(ell)` left out of the derivative of a normalized component, would pass all of them
    while sending the sampler in a direction the model does not have. Central differences on the
    same summed kernel are the independent statement of what that derivative is; the coordinate is
    the cell's own sparsity parameter, which is the one the study's conclusions rest on.
    """
    X, params = _points(12, seed=6), _params()
    kernel = KERNELS[key][0]
    site = "a_sq" if key[1] == "amplitude" else "kernel_inv_length_sq"
    theta = params[site]

    def total(value):
        return kernel(X, X, params | {site: value}, 0.0, False).sum()

    analytic = float(jax.grad(total)(theta)[0])

    step = 1.0e-6 * float(theta[0])
    shifted = jnp.zeros_like(theta).at[0].set(step)
    finite_difference = float(total(theta + shifted) - total(theta - shifted)) / (2.0 * step)

    assert analytic == pytest.approx(finite_difference, rel=1.0e-6)


def _active_counts(rng: np.random.Generator, alpha: float, cutoff: float) -> np.ndarray:
    """#{i : tausq * lam_i > cutoff} per draw, for tausq ~ HalfCauchy(alpha), lam_i ~ HalfCauchy(1).

    The prior-predictive active count of either cell: `alpha` and `cutoff` are (0.1, RHO_EPS) on
    the reference's rho scale and (ALPHA_AMPLITUDE, ACTIVE_EPS) on the amplitude scale. Drawn as
    |standard Cauchy| times the scale, which is exactly `dist.HalfCauchy(scale)`, in numpy so the
    check is independent of the JAX code it calibrates.
    """
    tausq = alpha * np.abs(rng.standard_cauchy(N_DRAWS))[:, None]
    lam = np.abs(rng.standard_cauchy((N_DRAWS, D_ALPHA)))
    return np.count_nonzero(tausq * lam > cutoff, axis=1)


def test_alpha_matches_reference_count():
    # ALPHA_AMPLITUDE is derived in closed form (plan D2) from the fact that the count
    # distribution depends on (alpha, cutoff) only through cutoff / alpha; this is the
    # independent numerical check that it really transports the reference prior's sparsity onto
    # the a^2 scale, so a difference between an amplitude cell and SAASBO is the parameterization
    # and not a differently sparse prior. Two independent sets of draws from one seeded generator,
    # so agreement is a property of the priors and not of shared random numbers.
    rng = np.random.default_rng(0)

    reference = _active_counts(rng, ALPHA_LENGTHSCALE, RHO_EPS)
    amplitude = _active_counts(rng, ALPHA_AMPLITUDE, ACTIVE_EPS)

    # Medians to within a coordinate, quartiles to within three: the whole count distribution
    # matches, not just its center. Monte-Carlo scatter at 10^4 draws is about +-1 coordinate.
    assert abs(np.median(reference) - np.median(amplitude)) <= 1
    for q in (0.25, 0.75):
        assert abs(np.quantile(reference, q) - np.quantile(amplitude, q)) <= 3

    # The constants the plan quotes, pinned to the precision it quotes them at.
    assert abs(ALPHA_AMPLITUDE - 0.01312) < 2.0e-4
    assert abs(ELL_EPS - 2.5613) < 2.0e-3
