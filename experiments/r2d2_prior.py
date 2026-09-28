"""experiments.r2d2_prior: the models and the ground truth of the prior-only R2-D2 study.

The study (`docs/superpowers/specs/2026-09-25-r2d2-prior-parametrizations-brief.md`, run by
`experiments.r2d2_prior_study`) asks which NumPyro parameterization of one R2-D2 prior NUTS can
sample exactly at the study's fixed budget. With p = dim coordinates, hyperparameters
(a, b, alpha) and k = alpha / p the per-coordinate concentration, the prior is

    R2 ~ Beta(a, b),  phi ~ Dir(k, ..., k),  omega = R2 / (1 - R2),  a_sq_i = omega * phi_i.

This module holds the implementations the study runs and everything it checks them against:

- I1, `sample_i1`: the supervisor's reference with its `z_beta` site removed. I2 and I3 are the
  two forms of `sagp.r2d2`, reached through `sample_log_theta`, the entry point the cells use, so
  the code the study certifies is the code the cells run;
- `half_cauchy_control`: the amplitude cells' own half-Cauchy prior, the study's control;
- `prior_model`: the harness wrapper that adds the one deterministic site the statistics read;
- `ground_truth`: the NumPy sampler of the prior, which shares no code with any model;
- `closed_forms` and `p_active_tied`: the closed-form identities;
- `count_pmf` and `count_quartiles`: the law of the active count, exact given one scalar;
- `gate`: the study's own convergence verdict.

Every a_sq statistic is taken in log space or through quantiles: E omega is infinite for b <= 1,
and at k = 0.01 a share is below 1e-300 often enough that a_sq itself underflows.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.nn import logsumexp
from jax.scipy.special import gammaln
from numpyro.diagnostics import summary
from scipy import special
from scipy.stats import qmc

from sagp.diagnostics import DiagThresholds
from sagp.gp import ACTIVE_EPS, ALPHA_AMPLITUDE, RHO_EPS
from sagp.r2d2 import sample_log_theta

# rho_i = RHO_SCALE * theta_i carries a calibration to the inverse squared lengthscales:
# #{rho_i > RHO_EPS} = #{theta_i > ACTIVE_EPS} draw for draw. Task 6 names this ratio
# `R2D2_RHO_SCALE` (ledger ruling R4).
RHO_SCALE: float = RHO_EPS / ACTIVE_EPS

# Every sampled site of each implementation, in trace order, with the scale the gate reads it on:
# log for a positive site, natural for a real one, logit for a unit-interval one. The logit is
# NumPyro's own unconstrained coordinate for a Beta site, a category `sagp.diagnostics` does not
# have. On the prior alone every site of I2 but `r2d2_R2`, and every site of I3, is exactly N(0, 1).
SITE_SUPPORTS: dict[str, dict[str, str]] = {
    "I1": {"r2d2_R2": "unit_interval", "r2d2_raw": "real"},
    "I2": {"r2d2_R2": "unit_interval", "r2d2_z_lam": "real"},
    "I3": {"r2d2_z_xi": "real", "r2d2_z_lam": "real"},
    "HC": {"kernel_tausq": "positive", "_a_sq": "positive"},
}


# --- the models ---


# Verbatim from the appendix of the 2026-09-25 brief (the supervisor's reference, 2026-09-25).
def _log_gamma_values(
    concentration: float,
    rate: float,
    p: int,
    *,
    site_name: str,
):
    """Sample independent Gamma values through standardized log sites."""
    concentration = jnp.asarray(concentration)
    rate = jnp.asarray(rate)

    location = jnp.log(concentration) - jnp.log(rate)
    proposal = dist.Normal(0.0, 1.0).expand([p])
    raw = numpyro.sample(site_name, proposal)

    log_values = location + raw / concentration

    # Transformed Gamma density, including the Jacobian.
    target_log_prob = (
        concentration * log_values
        - rate * jnp.exp(log_values)
        + concentration * jnp.log(rate)
        - gammaln(concentration)
        - jnp.log(concentration)
    )
    numpyro.factor(
        f"{site_name}_log_density",
        jnp.sum(target_log_prob - proposal.log_prob(raw)),
    )
    return log_values


def sample_i1(dim: int, *, a: float, b: float, alpha: float) -> Array:
    """I1, the control: the supervisor's `r2d2_prior` with `z_beta` removed; log a_sq, (dim,).

    The sites are `r2d2_R2` ~ Beta(a, b) (the reference's `R2_genetic`) and `r2d2_raw`, the
    standardized log-Gamma(alpha / dim) site of `_log_gamma_values`. That function is kept
    verbatim, with its factor `r2d2_raw_log_density` and the factor's Jacobian term. NUTS, which
    reads the factor through the log density, therefore targets the exact prior. `Predictive`,
    which never reads a factor, draws `r2d2_raw` from its N(0, 1) proposal.

    The return value is log a_sq = log omega + log phi, whose exponential is the brief's a_sq. It
    is returned in log space, as I2 and I3 return theirs, because a_sq underflows at small k.
    With `z_beta` gone, the reference's `phi` and `beta` deterministic sites go too; the caller
    names the deterministic sites.
    """
    R2 = numpyro.sample("r2d2_R2", dist.Beta(a, b))

    log_allocation = _log_gamma_values(
        concentration=alpha / dim,
        rate=1.0,
        p=dim,
        site_name="r2d2_raw",
    )
    log_phi = log_allocation - logsumexp(log_allocation)

    log_global_variance = jnp.log(R2) - jnp.log1p(-R2)
    return log_global_variance + log_phi


def half_cauchy_control(dim: int, *, alpha: float = ALPHA_AMPLITUDE) -> Array:
    """The control: `AdditiveAmplitudePyroModel.sample_amplitudes` (`sagp/gp.py:380-384`) alone.

    It has the same sites: `kernel_tausq` ~ HalfCauchy(alpha), `_a_sq` ~ HalfCauchy(1) per
    coordinate, both positive, and the deterministic `a_sq` = kernel_tausq * _a_sq, which it
    returns, (dim,).
    """
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(jnp.array(alpha)))
    a_sq = numpyro.sample("_a_sq", dist.HalfCauchy(jnp.ones(dim)))
    return numpyro.deterministic("a_sq", tausq * a_sq)


# log theta, (dim,), per implementation. I2 and I3 go through `sample_log_theta`, as the cells will:
# the reference form at any (a, b, k), and the tied form, which raises unless a = k * dim.
_LOG_THETA: dict[str, Callable[[int, float, float, float], Array]] = {
    "I1": lambda dim, a, b, alpha: sample_i1(dim, a=a, b=b, alpha=alpha),
    "I2": lambda dim, a, b, alpha: sample_log_theta(dim, form="reference", k=alpha / dim, b=b, a=a),
    "I3": lambda dim, a, b, alpha: sample_log_theta(dim, form="tied", k=alpha / dim, b=b, a=a),
    "HC": lambda dim, a, b, alpha: jnp.log(half_cauchy_control(dim)),
}


def prior_model(
    impl: str, dim: int, *, a: float | None = None, b: float | None = None,
    alpha: float | None = None,
) -> Callable[[], None]:
    """The zero-argument prior-only model that NUTS and `Predictive` run.

    `impl` is "I1", "I2", "I3" or "HC". The model's sampled sites are exactly
    `SITE_SUPPORTS[impl]`, plus I1's factor, which is an observed site. Its one addition is the
    deterministic `log_a_sq` (dim,), from which every statistic of the study is read. (a, b, alpha)
    are the R2-D2 hyperparameters; the control ignores them. It runs at `ALPHA_AMPLITUDE`, the
    cells' own scale. For "I3" at a point with a != alpha the model raises `ValueError` when traced.
    """
    if impl not in _LOG_THETA:
        raise ValueError(
            f"unknown implementation {impl!r}; the implementations are {tuple(_LOG_THETA)}")
    log_theta = _LOG_THETA[impl]

    def model() -> None:
        numpyro.deterministic("log_a_sq", log_theta(dim, a, b, alpha))

    return model


# --- the closed forms ---


def closed_forms(a: float, b: float, k: float, *, dim: int) -> dict[str, float]:
    """The prior's closed-form moments at (a, b, k), alpha = k * dim: the brief's identity sheet.

    omega = X / Y with X ~ Gamma(a), Y ~ Gamma(b). log phi_i = log G_i - log S with G_i ~ Gamma(k)
    and S = sum_j G_j ~ Gamma(alpha), independent of phi (Lukacs). Every log moment is therefore a
    sum of digammas psi and trigammas psi'. Var(log phi_i - log phi_j) = 2 psi'(k) involves neither
    alpha nor S, which is why it is the identity to use when k is tiny. E omega is finite only for
    b > 1, so nothing here is a moment of a_sq itself.
    """
    alpha = k * dim
    psi, tri = special.digamma, lambda x: special.polygamma(1, x)
    moments = {
        "mean_R2": a / (a + b),
        "var_R2": a * b / ((a + b) ** 2 * (a + b + 1.0)),
        "mean_log_R2": psi(a) - psi(a + b),
        "var_log_R2": tri(a) - tri(a + b),
        "mean_log_omega": psi(a) - psi(b),
        "var_log_omega": tri(a) + tri(b),
        "mean_phi": 1.0 / dim,
        "var_phi": (1.0 / dim) * (1.0 - 1.0 / dim) / (alpha + 1.0),
        "mean_log_phi": psi(k) - psi(alpha),
        "var_log_phi": tri(k) - tri(alpha),
        "var_log_phi_diff": 2.0 * tri(k),
        "mean_log_a_sq": psi(a) - psi(b) + psi(k) - psi(alpha),
        "var_log_a_sq": tri(a) + tri(b) + tri(k) - tri(alpha),
    }
    return {name: float(value) for name, value in moments.items()}


def p_active_tied(k: float, b: float, eps: float) -> float:
    """P(a_sq_i > eps) = 1 - I_{eps / (1 + eps)}(k, b), exact only under the tie a = k * dim.

    With a = k * dim, a_sq_i = lam_i / xi with lam_i ~ Gamma(k) and xi ~ Gamma(b), so a_sq_i is
    Beta-prime(k, b) and a_sq_i / (1 + a_sq_i) is Beta(k, b).
    """
    return float(special.betaincc(k, b, eps / (1.0 + eps)))


# --- the ground truth ---

# Draws per chunk of the ground truth's share arrays. The chunking is part of the random stream,
# so changing it changes the draws.
_TRUTH_CHUNK = 10_000


def active_count(log_theta: np.ndarray, eps: float, *, scale: float = 1.0) -> np.ndarray:
    """#{i : scale * theta_i > eps} per draw, from log theta (..., dim) -> (...)."""
    with np.errstate(over="ignore"):
        return np.count_nonzero(scale * np.exp(log_theta) > eps, axis=-1)


def ground_truth(
    a: float, b: float, k: float, *, dim: int = 100, n: int = 200_000, seed: int = 0,
) -> dict[str, np.ndarray]:
    """n exact draws of the prior by NumPy alone, kept as per-draw scalars: the ground truth.

    R2 ~ Beta(a, b) by `Generator.beta`. Each share's Gamma(k) is drawn in log space as
    log G = log G' + log(U) / k, with G' ~ Gamma(k + 1) and log U taken as -Exp(1). A plain
    Gamma(k) draw is exactly 0 about 6e-4 of the time at k = 0.01, and a uniform U can be exactly
    0; either would make a log statistic -inf. Then phi = softmax(log G) and
    log a_sq = logit R2 + log phi. Nothing here touches NumPyro, JAX or `sagp.r2d2`, so it can
    check them.

    Kept per draw, never the (n, dim) arrays:
    - `logit_R2`, which is log omega;
    - `log_phi_1`, `log_a_sq_1` and `log_max_phi`;
    - `n_active` = #{a_sq_i > ACTIVE_EPS};
    - `n_active_rho` = #{RHO_SCALE a_sq_i > RHO_EPS}, the same count on the rho scale.
    Fixed `seed`, so a point's ground truth is one fixed set of draws.
    """
    rng = np.random.default_rng(seed)
    r2 = rng.beta(a, b, size=n)
    with np.errstate(divide="ignore"):
        logit_r2 = np.log(r2) - np.log1p(-r2)
    truth = {name: np.empty(n) for name in ("log_phi_1", "log_a_sq_1", "log_max_phi")}
    truth |= {name: np.empty(n, dtype=np.int64) for name in ("n_active", "n_active_rho")}
    for start in range(0, n, _TRUTH_CHUNK):
        stop = min(start + _TRUTH_CHUNK, n)
        shape = (stop - start, dim)
        log_g = np.log(rng.gamma(k + 1.0, size=shape)) - rng.standard_exponential(size=shape) / k
        log_phi = log_g - special.logsumexp(log_g, axis=1, keepdims=True)
        log_theta = logit_r2[start:stop, None] + log_phi
        truth["log_phi_1"][start:stop] = log_phi[:, 0]
        truth["log_a_sq_1"][start:stop] = log_theta[:, 0]
        truth["log_max_phi"][start:stop] = log_phi.max(axis=1)
        truth["n_active"][start:stop] = active_count(log_theta, ACTIVE_EPS)
        truth["n_active_rho"][start:stop] = active_count(log_theta, RHO_EPS, scale=RHO_SCALE)
    return {"logit_R2": logit_r2, **truth}


# --- the active count ---

# 2^14 scrambled Sobol points carry the one scalar the count law mixes over; a search may use
# fewer.
QMC_LOG2 = 14
_BINOMIAL_CHUNK = 4_096


@lru_cache(maxsize=4)
def _qmc_uniforms(log2_n: int) -> np.ndarray:
    """2^log2_n scrambled Sobol points in [0, 1)^2, seed 0, so the count law is deterministic."""
    points = qmc.Sobol(d=2, scramble=True, seed=0).random_base2(m=log2_n)
    points.setflags(write=False)
    return points


@lru_cache(maxsize=64)
def _gamma_quantiles(b: float, log2_n: int) -> np.ndarray:
    """xi = F_Gamma(b)^-1(u) at the first Sobol coordinate: fixed per b while a search moves k."""
    xi = special.gammaincinv(b, _qmc_uniforms(log2_n)[:, 0])
    xi.setflags(write=False)
    return xi


def _binomial_mixture(p: np.ndarray, dim: int) -> np.ndarray:
    """The Binomial(dim, p) pmf averaged over the values p, in log space, 0 log 0 = 0 exactly."""
    n = np.arange(dim + 1)
    log_choose = (
        special.gammaln(dim + 1.0) - special.gammaln(n + 1.0) - special.gammaln(dim - n + 1.0))
    with np.errstate(divide="ignore"):
        log_p, log_q = np.log(p), np.log1p(-p)
    pmf = np.zeros(dim + 1)
    for start in range(0, p.size, _BINOMIAL_CHUNK):
        chunk = slice(start, start + _BINOMIAL_CHUNK)
        lp, lq = log_p[chunk, None], log_q[chunk, None]
        with np.errstate(invalid="ignore"):  # 0 * -inf, discarded by the where
            terms = (log_choose + np.where(n > 0, n * lp, 0.0)
                     + np.where(n < dim, (dim - n) * lq, 0.0))
        pmf += np.exp(terms).sum(axis=0)
    return pmf / p.size


def count_pmf(
    a: float, b: float, k: float, eps: float, dim: int = 100, *, qmc_log2: int = QMC_LOG2,
) -> np.ndarray:
    """P(N = n), n = 0..dim, for N = #{i : theta_i > eps}: exact given one scalar, QMC over it.

    Take G_i ~ Gamma(k) i.i.d. and a scalar T independent of them. Then theta_i = G_i / T has the
    prior's law exactly when omega = S / T is Beta-prime(a, b), where S = sum_j G_j ~ Gamma(k dim)
    is independent of phi = G / S (Lukacs). Two cases have such a T:
    - under the tie a = k dim, T = xi ~ Gamma(b);
    - for a < k dim, T = xi / B with B ~ Beta(a, k dim - a), because S B ~ Gamma(a) (beta-gamma
      algebra).
    Given T the coordinates are independent, so N | T ~ Binomial(dim, Q(k, eps T)), with Q the
    upper regularized gamma function. The pmf is that Binomial averaged over T, taken over
    2^`qmc_log2` scrambled Sobol points pushed through the inverse CDFs of xi and B. The points
    are fixed, so the pmf is deterministic and smooth in (a, b, k).

    For a > k dim no independent T exists (S B would need a shape above S's own), and the function
    raises rather than return an approximation. Planning §0 wrote T = S / omega with S drawn apart
    from the shares; that is not exact, because it widens log omega by the variance 2 psi'(k dim)
    of log(S / S'). The error is negligible at k dim = 44 but moves the mean by 10 standard errors
    of a 2e5-draw count at (0.5, 0.5, k dim = 1).
    """
    total = k * dim
    xi = _gamma_quantiles(float(b), qmc_log2)
    if math.isclose(a, total, rel_tol=1e-12):
        t = xi
    elif a < total:
        with np.errstate(divide="ignore"):
            t = xi / special.betaincinv(a, total - a, _qmc_uniforms(qmc_log2)[:, 1])
    else:
        raise ValueError(
            f"a = {a} > k * dim = {total}: the active count has no exact one-scalar mixture there"
        )
    return _binomial_mixture(special.gammaincc(k, eps * t), dim)


def pmf_quantile(pmf: np.ndarray, q: float, *, continuous: bool = False) -> float:
    """The q-quantile of a pmf on 0, 1, ..., len(pmf) - 1.

    By default it is the least n with P(N <= n) >= q, the integer rule NumPy's "inverted_cdf"
    applies to samples. With `continuous`, the same point is refined inside its unit cell to
    n - 1/2 + (q - P(N <= n - 1)) / P(N = n). That is the q-quantile of N + U(-1/2, 1/2): it rounds
    to the integer quantile and moves continuously with the pmf, which a search needs.
    """
    cdf = np.cumsum(pmf)
    n = min(int(np.searchsorted(cdf, q)), len(pmf) - 1)
    if not continuous:
        return n
    below = cdf[n - 1] if n > 0 else 0.0
    return n - 0.5 + (q - below) / pmf[n]


def count_quartiles(
    a: float, b: float, k: float, eps: float, dim: int = 100,
) -> tuple[int, int, int]:
    """(q25, median, q75) of the active count, from `count_pmf`, by the integer quantile rule."""
    pmf = count_pmf(a, b, k, eps, dim)
    return tuple(int(pmf_quantile(pmf, q)) for q in (0.25, 0.5, 0.75))


def half_cauchy_count_pmf(alpha: float, eps: float, dim: int = 100) -> np.ndarray:
    """P(N = n) for N = #{i : tausq lam_i > eps}, tausq ~ HC(alpha), lam_i ~ HC(1).

    The four half-Cauchy cells' prior count. It is exact given tausq, the one scalar:
    P(lam_i > eps / tausq) = (2 / pi) arctan(tausq / eps), so
    N | tausq ~ Binomial(dim, (2 / pi) arctan(tausq / eps)). The average over tausq is taken
    at tausq = alpha tan(pi u / 2), the half-Cauchy quantile, for u at the midpoints of 2^16 equal
    cells of (0, 1).
    """
    u = (np.arange(2**16) + 0.5) / 2**16
    p = (2.0 / np.pi) * np.arctan((alpha / eps) * np.tan(0.5 * np.pi * u))
    return _binomial_mixture(p, dim)


# --- the gate ---

_SCALES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "positive": np.log,
    "real": np.asarray,
    "unit_interval": lambda x: np.log(x) - np.log1p(-x),
}


def gate(
    samples: Mapping[str, np.ndarray],
    extra: Mapping[str, np.ndarray],
    supports: Mapping[str, str],
    thresholds: DiagThresholds = DiagThresholds(),
) -> dict[str, object]:
    """The study's convergence verdict for one chain, the rule of `sagp/diagnostics.py:102-162`.

    Each site of `supports` is read on its scale: the log of a positive site, a real one as it is,
    the logit of a unit-interval one. `numpyro.diagnostics.summary(..., prob=0.9,
    group_by_chain=False)` is taken over the un-thinned draws, and its split-R-hat and ESS are
    pooled over every coordinate of every site. The verdict passes iff r_hat_max <= 1.1,
    n_eff_min >= 16 and divergences <= 5 (`thresholds`), where divergences is the `diverging`
    counter summed.

    Each criterion is written as the negation of its pass condition, so a NaN statistic fails
    rather than passes. A NaN comes from a site that never moved, or from an infinite logit or log
    of a draw on its support's edge. `sagp.diagnostics.diagnose` is not called: it reads only the
    cells' site names, so on these models it would find nothing, or only `mean` and `noise`.
    """
    unknown = {site: support for site, support in supports.items() if support not in _SCALES}
    if unknown:
        raise ValueError(
            f"unknown support categories {unknown}; the categories are {tuple(_SCALES)}")
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = {
            site: _SCALES[support](np.asarray(samples[site], dtype=float))
            for site, support in supports.items()
        }
        stats = summary(scaled, prob=0.9, group_by_chain=False)
    per_site = {
        site: (np.ravel(site_stats["r_hat"]), np.ravel(site_stats["n_eff"]))
        for site, site_stats in stats.items()
    }
    r_hat_max = float(np.max(np.concatenate([r_hat for r_hat, _ in per_site.values()])))
    n_eff_min = float(np.min(np.concatenate([n_eff for _, n_eff in per_site.values()])))
    divergences = int(np.asarray(extra["diverging"]).sum())

    failures = []
    if not r_hat_max <= thresholds.r_hat_max:
        failures.append(f"r_hat_max {r_hat_max:.3f} > {thresholds.r_hat_max}")
    if not n_eff_min >= thresholds.n_eff_min:
        failures.append(f"n_eff_min {n_eff_min:.1f} < {thresholds.n_eff_min}")
    if not divergences <= thresholds.max_divergences:
        failures.append(f"divergences {divergences} > {thresholds.max_divergences}")
    return {
        "r_hat_max": r_hat_max,
        "n_eff_min": n_eff_min,
        "divergences": divergences,
        "num_steps_mean": float(np.mean(extra["num_steps"])),
        "r_hat_max_by_site": {site: float(np.max(r_hat)) for site, (r_hat, _) in per_site.items()},
        "n_eff_min_by_site": {site: float(np.min(n_eff)) for site, (_, n_eff) in per_site.items()},
        "passed": not failures,
        "reason": "; ".join(failures),
    }
