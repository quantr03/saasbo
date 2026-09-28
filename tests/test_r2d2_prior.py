"""Tests for the prior-only R2-D2 harness: `experiments/r2d2_prior.py` and its study.

The harness certifies three NumPyro implementations of one R2-D2 prior against a NumPy ground truth
that shares no code with them, so these tests check the checker. The closed-form identities are
checked against a Monte Carlo built differently from the ground truth, and the ground truth against
the identities and its own log-space construction. The count mixture is checked against a
brute-force count, and the gate's NaN semantics directly. The Tier A checks see exact draws and a
deliberately wrong law, so they cannot pass vacuously. The forward-sampling property separates I1
from I2 and I3, and every implementation gets a smoke NUTS run. The slow test runs the study's
fixed budget once per implementation at p = 30.
"""
import sagp  # float64 before any array
import math

import numpy as np
import pytest
from numpyro import handlers
from numpyro.distributions import constraints
from scipy.special import logsumexp
from scipy.stats import kurtosis

from experiments.r2d2_prior import (
    RHO_SCALE, SITE_SUPPORTS, active_count, closed_forms, count_pmf, count_quartiles, gate,
    ground_truth, half_cauchy_count_pmf, p_active_tied, pmf_quantile, prior_model,
)
from experiments.r2d2_prior_study import (
    hyperparameters, main, median_matched_k, run_nuts, tier_a_statistics, tier_b_row, tier_c_row,
)
from sagp.gp import ACTIVE_EPS, ALPHA_AMPLITUDE, NUTSConfig, RHO_EPS

GROUND_TRUTH_KEYS = {
    "logit_R2", "log_phi_1", "log_a_sq_1", "log_max_phi", "n_active", "n_active_rho",
}


def _se_mean(x):
    return x.std(ddof=1) / math.sqrt(x.size)


def _se_var(x):
    """Standard error of a sample variance, var sqrt((kappa - 1) / n), kappa the plain kurtosis."""
    return x.var(ddof=1) * math.sqrt((kurtosis(x, fisher=False) - 1.0) / x.size)


def _log_gamma(rng, shape, size):
    """log Gamma(shape) in log space, log G' + log(U) / shape with G' ~ Gamma(shape + 1)."""
    return np.log(rng.gamma(shape + 1.0, size=size)) + np.log(rng.uniform(size=size)) / shape


def _monte_carlo(a, b, k, dim, n, seed):
    """log R2, log omega and (log phi_1, log phi_2) by NumPy, built differently from `ground_truth`.

    R2 = X / (X + Y) with X ~ Gamma(a), Y ~ Gamma(b) rather than a Beta draw, and the two shares
    from G_1, G_2 ~ Gamma(k) and the other dim - 2 coordinates' total, one Gamma(k (dim - 2)) in
    law, rather than from a full (n, dim) array.
    """
    rng = np.random.default_rng(seed)
    log_x, log_y = np.log(rng.gamma(a, size=n)), np.log(rng.gamma(b, size=n))
    log_g = _log_gamma(rng, k, (n, 2))
    log_rest = _log_gamma(rng, k * (dim - 2), n)
    log_s = logsumexp(np.column_stack([log_g, log_rest]), axis=1)
    return log_x - np.logaddexp(log_x, log_y), log_x - log_y, log_g - log_s[:, None]


@pytest.mark.parametrize(
    "a, b, k, dim", [(2.0, 0.5, 0.2, 10), (1.5, 0.75, 0.05, 20), (1.0, 1.0, 0.01, 100)]
)
def test_every_closed_form_identity_matches_a_numpy_monte_carlo(a, b, k, dim):
    log_r2, log_omega, log_phi = _monte_carlo(a, b, k, dim, n=400_000, seed=0)
    samples = {
        "R2": np.exp(log_r2), "log_R2": log_r2, "log_omega": log_omega,
        "phi": np.exp(log_phi[:, 0]), "log_phi": log_phi[:, 0],
        "log_a_sq": log_omega + log_phi[:, 0],
    }
    estimates = {f"mean_{name}": (x.mean(), _se_mean(x)) for name, x in samples.items()}
    estimates |= {f"var_{name}": (x.var(ddof=1), _se_var(x)) for name, x in samples.items()}
    diff = log_phi[:, 0] - log_phi[:, 1]
    estimates["var_log_phi_diff"] = (diff.var(ddof=1), _se_var(diff))
    exact = closed_forms(a, b, k, dim=dim)
    assert set(exact) == set(estimates)  # every identity is checked, and nothing unchecked is added
    for name, (estimate, se) in estimates.items():
        assert abs(estimate - exact[name]) <= 5.0 * se, (name, estimate, exact[name], se)


@pytest.mark.parametrize("a, b, k", [(20.93, 0.5, 0.2093), (1.0, 1.0, 0.01), (1.5, 0.75, 0.44)])
def test_ground_truth_follows_the_closed_forms(a, b, k):
    truth = ground_truth(a, b, k, dim=100, n=50_000, seed=3)
    exact = closed_forms(a, b, k, dim=100)
    for name, x in (("log_omega", truth["logit_R2"]), ("log_phi", truth["log_phi_1"]),
                    ("log_a_sq", truth["log_a_sq_1"])):
        assert abs(x.mean() - exact[f"mean_{name}"]) <= 5.0 * _se_mean(x), name
        assert abs(x.var(ddof=1) - exact[f"var_{name}"]) <= 5.0 * _se_var(x), name
    assert np.all(truth["log_max_phi"] <= 0.0) and np.all(truth["log_max_phi"] >= -math.log(100))
    assert np.all((truth["n_active"] >= 0) & (truth["n_active"] <= 100))


def test_ground_truth_has_no_minus_inf_at_small_k():
    n = 50_000
    truth = ground_truth(1.0, 1.0, 0.01, dim=100, n=n, seed=0)
    assert set(truth) == GROUND_TRUTH_KEYS
    for name, x in truth.items():
        assert x.shape == (n,) and np.all(np.isfinite(x)), name
    # The construction the log-space draw replaces does underflow here: a plain Gamma(0.01) draw is
    # exactly 0 about 6e-4 of the time, which would make log phi -inf.
    assert np.any(np.random.default_rng(0).gamma(0.01, size=100_000) == 0.0)


def test_rho_transport_counts_equal_amplitude_counts_draw_for_draw():
    assert RHO_SCALE == RHO_EPS / ACTIVE_EPS and RHO_SCALE > 7.0  # not the trivial scale 1
    truth = ground_truth(1.5, 0.75, 0.44, dim=100, n=20_000, seed=0)
    assert np.array_equal(truth["n_active_rho"], truth["n_active"])
    # A value between the two cutoffs is counted only when the scale is applied.
    log_theta = np.log(np.array([[0.03, 0.01]]))
    assert active_count(log_theta, ACTIVE_EPS)[0] == 1
    assert active_count(log_theta, RHO_EPS, scale=RHO_SCALE)[0] == 1
    assert active_count(log_theta, RHO_EPS)[0] == 0


@pytest.mark.parametrize(
    "a, b, k", [(8.92, 0.5, 0.0892), (1.5, 0.75, 0.44), (2.0, 2.0, 0.2093), (0.5, 0.5, 0.01)]
)
def test_count_quartiles_match_a_brute_force_count(a, b, k):
    n = 200_000
    brute = ground_truth(a, b, k, dim=100, n=n, seed=1)["n_active"]
    expected = np.quantile(brute, [0.25, 0.5, 0.75], method="inverted_cdf")
    assert np.all(np.abs(np.array(count_quartiles(a, b, k, ACTIVE_EPS, dim=100)) - expected) <= 1)
    # The mean has teeth the quartiles lack: mixing over S / omega with S drawn apart from the
    # shares widens log omega by 2 psi'(k dim), and at (0.5, 0.5, 0.01) that moves the mean 0.07,
    # 10 se.
    pmf = count_pmf(a, b, k, ACTIVE_EPS, dim=100)
    assert abs(np.arange(101) @ pmf - brute.mean()) <= 5.0 * _se_mean(brute) + 0.005


def test_count_mixture_is_exact_in_the_tied_mean_and_refuses_a_above_k_dim():
    pmf = count_pmf(8.92, 0.5, 0.0892, ACTIVE_EPS, dim=100)
    assert abs(pmf.sum() - 1.0) < 1e-12
    assert abs(np.arange(101) @ pmf - 100 * p_active_tied(0.0892, 0.5, ACTIVE_EPS)) < 0.01
    with pytest.raises(ValueError, match="k \\* dim"):
        count_quartiles(5.0, 0.5, 0.01, ACTIVE_EPS, dim=100)  # a = 5 > k dim = 1


def test_the_half_cauchy_target_and_the_tied_median_match():
    target = half_cauchy_count_pmf(ALPHA_AMPLITUDE, ACTIVE_EPS, dim=100)
    # The four cells' prior count, as `test_alpha_matches_reference_count` states it.
    assert pmf_quantile(target, 0.5) == 37
    assert pmf_quantile(target, 0.25) in (16, 17) and pmf_quantile(target, 0.75) in (64, 65)
    k = median_matched_k(None, 0.5, target, dim=100, eps=ACTIVE_EPS)  # tied: a = k * dim
    assert 0.085 < k < 0.095
    assert pmf_quantile(count_pmf(100 * k, 0.5, k, ACTIVE_EPS, dim=100), 0.5) == 37


def test_forward_draws_miss_the_prior_for_i1_only():
    dim = 100
    a, b, alpha = hyperparameters("P2", dim)
    truth = ground_truth(a, b, alpha / dim, dim=dim, n=50_000, seed=0)
    rows = {impl: tier_c_row(impl, "P2", dim, truth=truth) for impl in ("I1", "I2", "I3")}
    # I1's factor is an observed site that `Predictive` never reads, so its shares are the N(0, 1)
    # proposal's; its Beta site is a plain sample site and is forward-sampled correctly.
    assert rows["I1"]["has_factor"] and not rows["I1"]["matches"]
    assert rows["I1"]["ks_p_log_phi_1"] < 1e-6 and rows["I1"]["ks_p_logit_R2"] >= 0.01
    for impl in ("I2", "I3"):
        assert rows[impl]["matches"] and not rows[impl]["has_factor"], impl


_GATE_SUPPORTS = {"x": "real", "r2": "unit_interval", "scale": "positive"}
_QUIET = {"diverging": np.zeros(256, dtype=bool), "num_steps": np.full(256, 7)}


def test_gate_passes_iid_draws_and_fails_nan_and_every_broken_criterion():
    rng = np.random.default_rng(0)
    draws = {"x": rng.normal(size=256), "r2": rng.beta(2.0, 2.0, size=256),
             "scale": rng.lognormal(size=(256, 3))}
    passed = gate(draws, _QUIET, _GATE_SUPPORTS)
    assert passed["passed"] and passed["reason"] == "" and passed["divergences"] == 0
    # A chain holding a NaN draw has NaN R-hat and ESS, and a NaN must fail rather than pass.
    holed = draws["x"].copy()
    holed[100] = np.nan
    nan_chain = gate(dict(draws, x=holed), _QUIET, _GATE_SUPPORTS)
    assert math.isnan(nan_chain["r_hat_max"]) and math.isnan(nan_chain["n_eff_min"])
    assert not nan_chain["passed"]
    assert "r_hat_max" in nan_chain["reason"] and "n_eff_min" in nan_chain["reason"]
    # So does a site that never moved, when its variance is exactly 0 (0.25 sums without
    # rounding; a value like 0.3 leaves a rounding-level variance and an ESS of 1 instead).
    stuck = gate(dict(draws, x=np.full(256, 0.25)), _QUIET, _GATE_SUPPORTS)
    assert math.isnan(stuck["r_hat_max"]) and not stuck["passed"]
    # A unit-interval draw at exactly 1 has an infinite logit, so its statistics are NaN too.
    edge = draws["r2"].copy()
    edge[10] = 1.0
    assert not gate(dict(draws, r2=edge), _QUIET, _GATE_SUPPORTS)["passed"]
    six = gate(draws, dict(_QUIET, diverging=np.arange(256) < 6), _GATE_SUPPORTS)
    assert not six["passed"] and six["reason"] == "divergences 6 > 5"
    with pytest.raises(ValueError, match="support"):
        gate(draws, _QUIET, dict(_GATE_SUPPORTS, x="simplex"))


_SUPPORT_CATEGORY = [
    (constraints.real, "real"), (constraints.positive, "positive"),
    (constraints.unit_interval, "unit_interval"),
]


@pytest.mark.parametrize("impl", ["I1", "I2", "I3", "HC"])
def test_each_implementation_samples_exactly_its_documented_sites(impl):
    model = prior_model(impl, 5, a=1.0, b=0.5, alpha=1.0)  # tied, so I3 represents it
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
    latent = [name for name, site in trace.items()
              if site["type"] == "sample" and not site["is_observed"]]
    assert latent == list(SITE_SUPPORTS[impl])
    for name in latent:
        support = trace[name]["fn"].support
        assert [c for s, c in _SUPPORT_CATEGORY if s is support] == [SITE_SUPPORTS[impl][name]]
    assert trace["log_a_sq"]["type"] == "deterministic"
    assert trace["log_a_sq"]["value"].shape == (5,)
    observed = [name for name, site in trace.items()
                if site["type"] == "sample" and site["is_observed"]]
    assert observed == (["r2d2_raw_log_density"] if impl == "I1" else [])


def test_the_tied_implementation_refuses_an_untied_point():
    with pytest.raises(ValueError, match="tied"):
        handlers.seed(prior_model("I3", 5, a=2.0, b=2.0, alpha=1.0), rng_seed=0)()


def test_run_nuts_returns_every_unthinned_draw_and_a_blocked_wall_time():
    run = run_nuts(prior_model("I2", 5, a=1.0, b=0.5, alpha=1.0), NUTSConfig(32, 32, 4), seed=0)
    assert run.samples["log_a_sq"].shape == (32, 5)
    assert np.all(np.isfinite(run.samples["log_a_sq"]))
    assert run.samples["r2d2_R2"].shape == (32,) and run.samples["r2d2_z_lam"].shape == (32, 5)
    assert run.extra["diverging"].shape == (32,) and run.extra["num_steps"].shape == (32,)
    assert run.wall_s > 0.0


@pytest.mark.parametrize("impl, point", [("I1", "P1"), ("I2", "P1"), ("I3", "P1"), ("HC", "-")])
def test_smoke_nuts_run_of_each_implementation_at_p5(impl, point):
    nuts = NUTSConfig(32, 32, 4)
    row = tier_b_row(impl, point, 5, 0, nuts)
    identity = (row["impl"], row["point"], row["p"], row["seed"], row["depth"])
    assert identity == (impl, point, 5, 0, 6)
    assert row["wall_s"] > 0.0 and 0 <= row["divergences"] <= 32
    assert 1.0 <= row["num_steps_mean"] <= 63.0 and 0.0 <= row["frac_at_cap"] <= 1.0
    assert row["gate"] in ("pass", "fail") and row["gate_log_a_sq"] in ("pass", "fail")
    assert row["limiting_site"] in SITE_SUPPORTS[impl]


def _exact_log_a_sq(a, b, k, dim, n, seed):
    """n exact i.i.d. draws of log a_sq by NumPy: the prior itself rather than a sampler of it."""
    rng = np.random.default_rng(seed)
    r2 = rng.beta(a, b, size=n)
    log_g = _log_gamma(rng, k, (n, dim))
    return (np.log(r2) - np.log1p(-r2))[:, None] + log_g - logsumexp(log_g, axis=1, keepdims=True)


def test_tier_a_checks_pass_exact_draws_and_fail_a_shifted_law():
    dim, n = 100, 4_000
    a, b, alpha = hyperparameters("P1", dim)
    truth = ground_truth(a, b, alpha / dim, dim=dim, n=50_000, seed=0)
    quiet = {"diverging": np.zeros(n, dtype=bool), "num_steps": np.full(n, 7)}
    exact = _exact_log_a_sq(a, b, alpha / dim, dim, n, seed=11)
    good = tier_a_statistics(exact, quiet, a=a, b=b, alpha=alpha, truth=truth)
    assert good["evaluable"] and good["verdict"] == "pass", good["failed_checks"]
    # The global scale off by a factor e moves every coordinate's mean by 12 standard errors.
    shifted = tier_a_statistics(exact + 1.0, quiet, a=a, b=b, alpha=alpha, truth=truth)
    assert shifted["verdict"] == "fail" and shifted["frac_outside_mean"] > 0.9
    # 41 divergences in 4,000 draws is more than 1 %: neither a pass nor a fail.
    noisy = dict(quiet, diverging=np.arange(n) < 41)
    assert tier_a_statistics(exact, noisy, a=a, b=b, alpha=alpha, truth=truth)["verdict"] == (
        "not evaluable")


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--tiers" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.parametrize("impl", ["I1", "I2", "I3"])
def test_the_fixed_budget_at_p30_depth6(impl):
    """The study's own NUTS budget, once per implementation at p = 30, P1, seed 0.

    I2 and I3 must pass the gate; I1 must only complete with finite statistics, its verdict being
    a result of the study rather than a property the harness requires.
    """
    row = tier_b_row(impl, "P1", 30, 0, NUTSConfig())
    assert math.isfinite(row["r_hat_max"]) and math.isfinite(row["n_eff_min"])
    assert math.isfinite(row["r_hat_max_log_a_sq"]) and math.isfinite(row["n_eff_min_log_a_sq"])
    if impl != "I1":
        assert row["gate"] == "pass", row["gate_reason"]
