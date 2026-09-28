"""Tests for sagp.r2d2: the R2-D2 prior's Gaussian-copula map, its reference solver and its sites.

`log_gamma_icdf_newton` is checked against SciPy's inverse regularized incomplete gamma in log
space; `log_gamma_icdf`, the loop-free table the cells run, against the solver over the whole
table, and for finiteness, monotonicity and gradient beyond it. Both forms' sampled sites are
checked by name, support and shape, and their forward draws against a NumPy sampler of the same
law that shares no code with the module.
"""
import sagp  # float64 before any array
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.distributions import constraints
from numpyro.infer import Predictive
from scipy.special import gammaincinv, gammainccinv, gammaln, log_ndtr
from scipy.stats import ks_2samp, norm

from sagp.r2d2 import (
    SAMPLED_SITES, log_gamma_icdf, log_gamma_icdf_newton, sample_log_theta,
)

SHAPES = (0.005, 0.01, 0.0333, 0.0892, 0.2093, 0.44, 0.5, 0.75, 1.0, 20.93)


def _scipy_log_icdf(z, k):
    with np.errstate(divide="ignore"):  # SciPy's x underflows to 0 where log x < -745
        return np.where(z < 1.2816, np.log(gammaincinv(k, norm.cdf(z))),
                        np.log(gammainccinv(k, norm.sf(z))))


@pytest.mark.parametrize("k", SHAPES)
def test_newton_solver_matches_scipy_in_log_space(k):
    z = np.linspace(-2.4, 6.0, 200)
    y = np.asarray(log_gamma_icdf_newton(jnp.asarray(z), k))
    reference = _scipy_log_icdf(z, k)
    # SciPy's x is a normal double: not 0 (log x < -745) and not subnormal (log x < -708.4), where
    # its log loses digits -- only z < -1.93 at k = 0.005 (ledger rulings R10, R13).
    finite = reference > np.log(np.finfo(float).tiny)
    assert finite.sum() >= 150
    assert np.all(np.abs(y[finite] - reference[finite])
                  <= 1e-11 * np.maximum(1.0, np.abs(y[finite])))


@pytest.mark.parametrize("k", SHAPES)
def test_runtime_map_matches_the_solver(k):
    rng = np.random.default_rng(0)  # dense where the prior lives, and across the whole table
    z = jnp.asarray(np.concatenate([rng.uniform(-8.0, 8.0, 5_000), rng.uniform(-38.0, 36.0, 5_000)]))
    y, reference = np.asarray(log_gamma_icdf(z, k)), np.asarray(log_gamma_icdf_newton(z, k))
    assert np.all(np.abs(y - reference) <= 1e-10 * np.maximum(1.0, np.abs(reference)))


@pytest.mark.parametrize("k", SHAPES)
def test_runtime_map_is_finite_monotone_and_differentiable_far_out(k):
    z = jnp.linspace(-40.0, 45.0, 4251)  # both asymptotic pieces and the table
    y = np.asarray(log_gamma_icdf(z, k))
    g = np.asarray(jax.vmap(jax.grad(lambda t: log_gamma_icdf(t, k)))(z))
    assert np.all(np.isfinite(y)) and np.all(np.diff(y) > 0)
    assert np.all(np.isfinite(g)) and np.all(g > 0)


@pytest.mark.parametrize("k", SHAPES)
def test_runtime_map_tails_are_the_asymptote_and_the_tangent_line(k):
    # Beyond the table the map is two formulas: below z = -38 the small-x asymptote
    # y = (log Phi(z) + r(-38)) / k, r(-38) = k y(-38) - log Phi(-38), and above z = 36 the
    # tangent line y(36) + y'(36) (z - 36). Each is checked against its formula with y(-38),
    # y(36) and y'(36) from the solver (y' from its closed-form custom_jvp), so an off-by-one knot
    # or a wrong constant shows; the lower tail against the solver itself too, which stays valid
    # there, and r(-38) against lgamma(k + 1), its limit. Monotone, with a finite positive
    # gradient, out to |z| = 40.
    lower = np.linspace(-40.0, -38.0, 201)[:-1]
    upper = np.linspace(36.0, 40.0, 201)[1:]
    y_lower = np.asarray(log_gamma_icdf(jnp.asarray(lower), k))
    y_upper = np.asarray(log_gamma_icdf(jnp.asarray(upper), k))

    r_edge = k * float(log_gamma_icdf_newton(-38.0, k)) - log_ndtr(-38.0)
    asymptote = (log_ndtr(lower) + r_edge) / k
    solver = np.asarray(log_gamma_icdf_newton(jnp.asarray(lower), k))
    y_edge = float(log_gamma_icdf_newton(36.0, k))
    slope_edge = float(jax.grad(lambda z: log_gamma_icdf_newton(z, k))(36.0))
    tangent = y_edge + slope_edge * (upper - 36.0)

    def relative(a, b):
        return np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))

    assert relative(y_lower, asymptote) <= 1e-14
    assert relative(y_lower, solver) <= 1e-14
    assert abs(r_edge - gammaln(k + 1.0)) <= 1e-12
    assert relative(y_upper, tangent) <= 1e-14
    for z, y in ((lower, y_lower), (upper, y_upper)):
        g = np.asarray(jax.vmap(jax.grad(lambda t: log_gamma_icdf(t, k)))(jnp.asarray(z)))
        assert np.all(np.diff(y) > 0)
        assert np.all(np.isfinite(g)) and np.all(g > 0)


@pytest.mark.parametrize("k", (0.0892, 0.44, 20.93))
def test_runtime_gradient_is_the_maps_derivative(k):
    z, h = jnp.linspace(-7.9, 35.9, 300), 1e-6
    g = jax.vmap(jax.grad(lambda t: log_gamma_icdf(t, k)))(z)
    fd = (log_gamma_icdf(z + h, k) - log_gamma_icdf(z - h, k)) / (2 * h)
    assert np.allclose(np.asarray(g), np.asarray(fd), rtol=1e-5, atol=0.0)


def test_runtime_map_has_no_while_loop_even_when_first_built_inside_a_trace():
    fresh_shape = 0.08931  # not built by any earlier test
    jaxpr = jax.make_jaxpr(jax.value_and_grad(
        lambda z: jnp.sum(log_gamma_icdf(z, fresh_shape))))(jnp.zeros(100))
    assert "while" not in str(jaxpr)


@pytest.mark.parametrize("form", ["reference", "tied"])
def test_sampled_sites_names_supports_and_shapes(form):
    def model():
        numpyro.deterministic("theta", jnp.exp(sample_log_theta(5, form=form, k=0.44, b=0.75)))
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, rng_seed=0)).get_trace()
    sampled = tuple(name for name, site in trace.items() if site["type"] == "sample")
    assert sampled == SAMPLED_SITES[form]
    assert trace["r2d2_z_lam"]["value"].shape == (5,)
    assert trace["theta"]["value"].shape == (5,)
    # The Beta site is the only one off the real line: every copula site is N(0, 1).
    supports = {"r2d2_R2": constraints.unit_interval, "r2d2_z_xi": constraints.real,
                "r2d2_z_lam": constraints.real}
    assert all(trace[name]["fn"].support is supports[name] for name in sampled)


def test_tied_form_rejects_an_untied_a():
    with pytest.raises(ValueError, match="tied"):
        numpyro.handlers.seed(
            lambda: sample_log_theta(5, form="tied", k=0.1, b=0.5, a=2.0), rng_seed=0)()


def _logsumexp_rows(a):
    m = a.max(axis=1)
    return m + np.log(np.exp(a - m[:, None]).sum(axis=1))


def _numpy_log_theta(a, b, k, *, dim, n, seed):
    """log theta_i = logit R2 + log phi_i with R2 ~ Beta(a, b), phi ~ Dir(k, ..., k); NumPy only.

    The shares come from log-space Gamma(k) draws, log G = log G' + log(U)/k with G' ~ Gamma(k + 1)
    and U ~ U(0, 1) (G itself underflows at small k), normalized by a softmax in log space.
    """
    rng = np.random.default_rng(seed)
    r2 = rng.beta(a, b, size=n)
    log_g = np.log(rng.gamma(k + 1.0, size=(n, dim))) + np.log(rng.uniform(size=(n, dim))) / k
    return (np.log(r2) - np.log1p(-r2))[:, None] + log_g - _logsumexp_rows(log_g)[:, None]


# The untied form at the calibration candidate (a, b, k * dim) = (1.5, 0.75, 44), and the tied
# form at the paper's b = 0.5, where its law is the reference law at a = k * dim.
@pytest.mark.parametrize(
    "form, a, b, k", [("reference", 1.5, 0.75, 0.44), ("tied", None, 0.5, 0.0892)]
)
def test_forward_draws_of_both_forms_match_a_numpy_ground_truth(form, a, b, k):
    dim, n = 100, 4_000

    def model():
        numpyro.deterministic("log_theta", sample_log_theta(dim, form=form, k=k, b=b, a=a))

    draws = np.asarray(Predictive(model, num_samples=n)(jax.random.PRNGKey(0))["log_theta"])
    truth = _numpy_log_theta(k * dim if a is None else a, b, k, dim=dim, n=n, seed=0)
    statistics = {
        "one coordinate": lambda log_theta: log_theta[:, 0],
        "log omega": _logsumexp_rows,
        "one share": lambda log_theta: log_theta[:, 0] - _logsumexp_rows(log_theta),
    }
    for name, statistic in statistics.items():
        assert ks_2samp(statistic(draws), statistic(truth)).pvalue > 1e-3, name
