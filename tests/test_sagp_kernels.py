"""Tests for sagp.gp's kernels: the centered Matern-5/2 component and the four cell kernels.

Three things have to hold for the thesis's 2x2 comparison to mean anything. The centered
component must reproduce `synthobj.kernel` -- and through it Gate 1's check script,
`Research Context/research direction/Kernel normalization check.py` -- in float64, so
the surrogate and the synthetic objectives share one definition of a component. Normalization
must actually deliver unit marginal variance under U[0,1], since that is the only thing that
makes an amplitude threshold well defined. And ("product", "lengthscale") must be BoTorch's own
`matern52_kernel` bit-for-bit, because that cell is the SAASBO baseline the other three are
measured against and the kernel NUTS samples the model under.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp
from sagp.gp import (
    ACTIVE_EPS,
    ALPHA_AMPLITUDE,
    ALPHA_LENGTHSCALE,
    CELLS,
    ELL_EPS,
    GL_NODES,
    GL_WEIGHTS,
    KERNELS,
    R2D2_A,
    R2D2_B,
    R2D2_K,
    R2D2_RHO_SCALE,
    RHO_EPS,
    cell_kernel_diag,
    centered_matern52_1d,
    kernel_product_lengthscale,
    v_of_ell,
)

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from botorch.models.fully_bayesian import matern52_kernel
from scipy.special import logsumexp
from synthobj.kernel import centered, normalized
from synthobj.kernel import v as v_numpy

D = 7
CELL_KEYS = list(KERNELS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]

# The cells built out of the centered component; those on ("product", "lengthscale")'s ARD
# kernel are pinned against BoTorch's `matern52_kernel` instead. By kernel identity rather than by
# name, so an R2-D2 cell is tested against its own kernel's oracle.
CENTERED_CELL_KEYS = [key for key in CELL_KEYS if KERNELS[key][0] is not kernel_product_lengthscale]
CENTERED_CELL_IDS = ["/".join(key) for key in CENTERED_CELL_KEYS]

# The lengthscales the brief exercises the centered component at: rough, middling and so smooth
# that v(ell) is down at 0.015 and the centering is a near-total cancellation.
ELLS_CENTERED = [0.06, 0.5, 3.0]

# Table 1 of the check script, printed to five decimals.
ELLS_TABLE = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
V_TABLE = [0.88574, 0.78149, 0.60296, 0.28218, 0.10680, 0.03183, 0.01478, 0.00546, 0.00138, 0.00015]

# Every cell's `cell_kernel_diag` agrees with its own matrix diagonal to float64 round-off,
# except on ("product", "lengthscale")'s kernel, which its R2-D2 twin shares: BoTorch floors the
# squared distance at 1e-30 before the square root, so the matrix's own diagonal is var * (1 -
# sqrt(5) * 1e-15) rather than var. That floor is BoTorch's, and reproducing it in an O(n D Q)
# diagonal would state it twice, so the diagonal stays var * ones and the tolerance absorbs the
# floor instead.
DIAG_ATOL = {
    key: 1.0e-11 if KERNELS[key][0] is kernel_product_lengthscale else 1.0e-12 for key in CELL_KEYS
}

# Plan decision D2's simulation, at the size it was run during planning: 10^4 prior draws of the
# half-Cauchy scale mixture over D_ALPHA coordinates.
N_DRAWS = 10**4
D_ALPHA = 100

# G0's calibration of the R2-D2 cells (sagp_analysis/2026-09-25-r2d2-prior/tables/
# tier_d_calibration.csv): the active-count median matched to the half-Cauchy cells', and each
# quartile off by this many coordinates at D = 100 -- 14 and 61 against 17 and 64.
Q25_RESIDUAL = 3
Q75_RESIDUAL = 3
# The exact active-count quartiles at D = 100 that G0 compared (plan D2; that table): the
# half-Cauchy cells', and the R2-D2 cells' at the residuals above, 14 / 37 / 61.
HALF_CAUCHY_QUARTILES = (17, 37, 64)
R2D2_QUARTILES = (HALF_CAUCHY_QUARTILES[0] - Q25_RESIDUAL, HALF_CAUCHY_QUARTILES[1],
                  HALF_CAUCHY_QUARTILES[2] - Q75_RESIDUAL)
# The calibration test's own draw count. The R2-D2 count's CDF crosses 0.25, 0.5 and 0.75 within
# 0.002-0.005 of an integer step (P(N <= 60) = 0.741, P(N <= 61) = 0.750), so a sample quartile
# can sit one coordinate off its exact value; at 4 * 10^4 draws each lies within one coordinate
# of it by 4.5 standard errors or more (200 of 200 seeds pass; k x 1.2 and k / 1.2 fail all 200).
N_CALIBRATION_DRAWS = 4 * 10**4


def _params() -> dict[str, jax.Array | float]:
    """The brief's random positive parameters: a_sq ~ U(0,1), ell ~ U(0.1,2), var = 1.3, rho ~ U(0.1,5).

    One dict carrying all four cells' parameters, so the same numbers drive every kernel; each
    kernel reads only the two keys it is parameterized by and ignores the rest.
    """
    rng = np.random.default_rng(0)
    return {
        "a_sq": jnp.asarray(rng.uniform(0.0, 1.0, D)),
        "kernel_ell": jnp.asarray(rng.uniform(0.1, 2.0, D)),
        "outputscale": 1.3,
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
    if CELLS[key].native_site == "a_sq":
        a_sq, ell = np.asarray(params["a_sq"]), np.asarray(params["kernel_ell"])
        terms = np.array([a_sq[i] * normalized(X[:, i], Z[:, i], ell[i]) for i in range(D)])
        return terms.sum(axis=0) if key[0] == "additive" else (1.0 + terms).prod(axis=0)
    ell = np.asarray(params["kernel_inv_length_sq"]) ** -0.5
    terms = np.array([centered(X[:, i], Z[:, i], ell[i]) for i in range(D)])
    return params["outputscale"] * terms.sum(axis=0)


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

    eigenvalues = jnp.linalg.eigvalsh(KERNELS[key][0](X, X, _params()))

    assert float(eigenvalues.min()) > -1.0e-10


def test_product_lengthscale_cell_is_botorchs_matern_kernel():
    # This cell is SAASBO, and on this path that means BoTorch's own `matern52_kernel` scaled by
    # the outputscale: anything but bit-for-bit equality would have the readouts describe a
    # different kernel from the one `SaasPyroModel` samples the model under. The oracle is jitted
    # because ours is: XLA fuses the multiply into the polynomial, which moves the last bit of 37
    # of these 81 entries, so an eager oracle would compare a differently *compiled* expression
    # rather than a different one.
    rng = np.random.default_rng(2)
    X = jnp.asarray(rng.uniform(size=(9, D)))
    var, rho = 1.3, jnp.asarray(rng.uniform(0.1, 5.0, D))
    params = {"outputscale": var, "kernel_inv_length_sq": rho}
    oracle = jax.jit(lambda X, rho: var * matern52_kernel(X, rho**-0.5))

    ours = kernel_product_lengthscale(X, X, params)

    assert np.array_equal(np.asarray(ours), np.asarray(oracle(X, rho)))


@pytest.mark.parametrize("key", CENTERED_CELL_KEYS, ids=CENTERED_CELL_IDS)
def test_centered_cells_match_the_numpy_oracle(key):
    # X and Z differ in length: the only test here that exercises the n != m path the acquisition
    # optimizer always takes.
    rng = np.random.default_rng(5)
    X = jnp.asarray(rng.uniform(0.0, 1.0, (11, D)))
    Z = jnp.asarray(rng.uniform(0.0, 1.0, (9, D)))
    params = _params()

    K = KERNELS[key][0](X, Z, params)

    np.testing.assert_allclose(
        np.asarray(K), _numpy_oracle(key, X, Z, params), atol=1.0e-12, rtol=0.0
    )


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_cell_kernel_diag_matches_the_kernel_diagonal(key):
    X, params = _points(12, seed=3), _params()

    diagonal = cell_kernel_diag(key, X, params)

    np.testing.assert_allclose(
        np.asarray(diagonal),
        np.asarray(jnp.diagonal(KERNELS[key][0](X, X, params))),
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

    grads = jax.grad(lambda p: kernel(X, X, p).sum())(_params())

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
    site = CELLS[key].native_site
    theta = params[site]

    def total(value):
        return kernel(X, X, params | {site: value}).sum()

    analytic = float(jax.grad(total)(theta)[0])

    step = 1.0e-6 * float(theta[0])
    shifted = jnp.zeros_like(theta).at[0].set(step)
    finite_difference = float(total(theta + shifted) - total(theta - shifted)) / (2.0 * step)

    assert analytic == pytest.approx(finite_difference, rel=1.0e-6)


def _active_counts(
    rng: np.random.Generator, alpha: float, cutoff: float, n: int = N_DRAWS
) -> np.ndarray:
    """#{i : tausq * lam_i > cutoff} per draw, for tausq ~ HalfCauchy(alpha), lam_i ~ HalfCauchy(1).

    The prior-predictive active count of either cell: `alpha` and `cutoff` are (0.1, RHO_EPS) on
    the reference's rho scale and (ALPHA_AMPLITUDE, ACTIVE_EPS) on the amplitude scale. Drawn as
    |standard Cauchy| times the scale, which is exactly `dist.HalfCauchy(scale)`, in numpy so the
    check is independent of the JAX code it calibrates.
    """
    tausq = alpha * np.abs(rng.standard_cauchy(n))[:, None]
    lam = np.abs(rng.standard_cauchy((n, D_ALPHA)))
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


def _r2d2_counts(rng, a, b, k, cutoff, scale=1.0, n=N_DRAWS):
    """#{i : scale * omega * phi_i > cutoff} per draw, in NumPy only: R2 ~ Beta(a, b), log-space
    Gamma(k) draws (log G' + log(U)/k, G' ~ Gamma(k + 1)), phi = softmax(log G)."""
    R2 = rng.beta(a, b, n)
    log_g = (np.log(rng.gamma(k + 1.0, size=(n, D_ALPHA)))
             + np.log(rng.random((n, D_ALPHA))) / k)
    log_phi = log_g - logsumexp(log_g, axis=1, keepdims=True)
    log_theta = np.log(scale) + (np.log(R2) - np.log1p(-R2))[:, None] + log_phi
    return np.count_nonzero(log_theta > np.log(cutoff), axis=1)


def test_r2d2_calibration_matches_reference_count():
    # G0's calibration: the median matched to the half-Cauchy cells' count, the quartiles off by
    # the residual G0 recorded (confounding prior family with the dispersion of sparsity, not
    # with its level), and the rho scale transported exactly, draw for draw. Both counts are read
    # against G0's exact quartiles, within one coordinate, at the test's own draw count; and the
    # constants the Triton runs use are pinned at G0's full precision.
    assert (R2D2_A, R2D2_B, R2D2_K) == (1.577790731256835, 0.8043916352200708, 0.49942145555129697)
    assert R2D2_QUARTILES == (14, 37, 61)
    rng = np.random.default_rng(0)
    reference = _active_counts(rng, ALPHA_LENGTHSCALE, RHO_EPS, n=N_CALIBRATION_DRAWS)
    a = R2D2_K * D_ALPHA if R2D2_A is None else R2D2_A
    state = rng.bit_generator.state
    amplitude = _r2d2_counts(rng, a, R2D2_B, R2D2_K, ACTIVE_EPS, n=N_CALIBRATION_DRAWS)
    rng.bit_generator.state = state
    rho = _r2d2_counts(
        rng, a, R2D2_B, R2D2_K, RHO_EPS, scale=R2D2_RHO_SCALE, n=N_CALIBRATION_DRAWS
    )
    assert np.array_equal(amplitude, rho)
    for q, half_cauchy, r2d2 in zip((0.25, 0.5, 0.75), HALF_CAUCHY_QUARTILES, R2D2_QUARTILES):
        assert abs(np.quantile(reference, q) - half_cauchy) <= 1
        assert abs(np.quantile(amplitude, q) - r2d2) <= 1
    assert abs(R2D2_RHO_SCALE - 7.6218) < 1e-4


def test_gp_constants_are_bit_identical_to_synthobjs():
    """`sagp.gp`'s own grid and cutoff reproduce `synthobj`'s, and `sagp.gp` never imports it."""
    from synthobj import kernel
    from synthobj.objective import ACTIVE_EPS as EPS

    assert np.array_equal(np.asarray(GL_NODES), kernel.GL_NODES)
    assert np.array_equal(np.asarray(GL_WEIGHTS), kernel.GL_WEIGHTS)
    assert ACTIVE_EPS == EPS

    tree = ast.parse(Path(sagp.gp.__file__).read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("synthobj") for name in imported)
