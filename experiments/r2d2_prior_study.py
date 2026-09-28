"""experiments.r2d2_prior_study: the prior-only R2-D2 study -- Tiers A to D, acceptance, the CLI.

It runs the protocol of `docs/superpowers/specs/2026-09-25-r2d2-prior-parametrizations-brief.md`
on the models of `experiments.r2d2_prior`: I1, the supervisor's reference; I2 and I3, the two
forms of `sagp.r2d2`; and the half-Cauchy control. Task 3 of the 2026-09-27 plan adds four
extensions:
- Tier D searches the calibration, and its two points C_tied and C_untied join Tiers A to C;
- the rho-scale transport identity, with its Tier D row;
- wall times read after `jax.block_until_ready`;
- this command line:

    python -m experiments.r2d2_prior_study --tiers A,B,C,D \\
        --out sagp_analysis/2026-09-25-r2d2-prior/ --cache runs_smoke/r2d2_prior_study/

The tiers:
- Tier A, exactness, at p = 100: one long chain per implementation and point (512 warm-up, 4,000
  draws, dense mass, depth 10). It is checked against the closed forms (per-coordinate moments of
  log a_sq; log R2, log omega, share differences), against the ground truth (KS on ESS_min
  equally spaced draws; the active count's quartiles) and, at a tied point, against the exact
  P(a_sq_i > eps). A run with ESS_min < 100 or more than 1 % divergent draws is "not evaluable".
- Tier B, the fixed budget: `_run_nuts`'s protocol at depth 6 and, for context, 10, per
  implementation, point, p in {30, 100} and seed in {0, 1, 2}, gated as the study gates a fit,
  with `log a_sq` gated separately. The half-Cauchy control runs once per seed, p and depth,
  first. If it fails the depth-6 gate at p = 100 the harness is wrong, and the command stops with
  exit status 3.
- Tier C, forward sampling: `Predictive` draws against the ground truth, a property, not a gate.
- Tier D, the calibration readout at p = 100: the active count from the ground truth, from the
  exact mixture and from each Tier A chain, at every point, at the brief's tied k = 0.0892 and on
  the calibration profile, each also on the rho scale. The calibration table holds the D2 search:
  the half-Cauchy target, the tied median-matched k at b = 0.5, the untied profile (the best b for
  each a), and the untied (a, b, k) that matches the median and minimizes the quartile residual
  over a, b >= 0.5.

Every verdict comes from the harness alone. The scouting's numbers are compared afterwards, by
hand, in the blank `agrees_with_scouting` column every tier table carries: the blind comparison.
Raw draws and ground truths are cached under `--cache`, so a rerun reuses every finished run;
tables go to `--out`/tables/, as CSV and Markdown, beside `run_config.json`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import numpy as np
import numpyro
import scipy
from numpyro import handlers
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, Predictive
from scipy import optimize, special
from scipy.stats import beta as beta_distribution
from scipy.stats import ks_2samp, kurtosis

from experiments.r2d2_prior import (
    QMC_LOG2, RHO_SCALE, SITE_SUPPORTS, active_count, closed_forms, count_pmf, gate, ground_truth,
    half_cauchy_count_pmf, p_active_tied, pmf_quantile, prior_model,
)
from sagp.gp import ACTIVE_EPS, ALPHA_AMPLITUDE, RHO_EPS, NUTSConfig
from sagp.r2d2 import sample_log_theta

A_PI = 0.2093  # the 2026-09-21 per-coordinate concentration of P1 and P3
BRIEF_TIED_K = 0.0892  # the 2026-09-25 brief's median-matched tied point, a Tier D row
IMPLEMENTATIONS = ("I1", "I2", "I3")
CONTROL = "HC"
POINTS = ("P1", "P2", "P3", "C_tied", "C_untied")

KS_MIN_P = 0.01
MIN_EVALUABLE_ESS = 100.0
MAX_DIVERGENT_FRACTION = 0.01
MAX_FRACTION_OUTSIDE = 0.01
PHI_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
_QUANTILES = {"q05": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}
# Every cache key carries a digest of the code its entries depend on: the prior's map and forms
# (`sagp/r2d2.py`), the models and ground truth (`experiments/r2d2_prior.py`), and this harness.
# An edit to any of them therefore invalidates the cache, instead of letting a rerun reuse draws
# of the old code.
SOURCE_DIGEST = hashlib.sha256(b"".join(
    Path(path).read_bytes()
    for path in (
        inspect.getsourcefile(sample_log_theta), inspect.getsourcefile(prior_model), __file__)
)).hexdigest()[:16]

DEFAULT_OUT = Path("sagp_analysis/2026-09-25-r2d2-prior")
DEFAULT_CACHE = Path("runs_smoke/r2d2_prior_study")


def hyperparameters(
    point: str, dim: int, calibration: dict | None = None,
) -> tuple[float, float, float]:
    """(a, b, alpha) of a study point at p = dim.

    P1 = (A_PI p, 0.5, A_PI p) is tied. P2 = (1, 1, 1) is tied, with k = 1/p. P3 = (2, 2, A_PI p) is
    untied. The calibration's two points hold their per-coordinate concentration k across p:
    C_tied = (k p, b, k p) and C_untied = (a, b, k p).
    """
    if point == "P1":
        return A_PI * dim, 0.5, A_PI * dim
    if point == "P2":
        return 1.0, 1.0, 1.0
    if point == "P3":
        return 2.0, 2.0, A_PI * dim
    if point in ("C_tied", "C_untied"):
        if calibration is None:
            raise ValueError(f"{point} is set by Tier D's calibration search; pass `calibration`")
        c = calibration[point]
        return (c["k"] * dim if point == "C_tied" else c["a"]), c["b"], c["k"] * dim
    raise ValueError(f"unknown point {point!r}; the points are {POINTS}")


def represents(impl: str, a: float, alpha: float) -> bool:
    """I3, the tied form, represents a point only when a = alpha; I1 and I2 represent all."""
    return impl != "I3" or math.isclose(a, alpha, rel_tol=1e-12)


# --- NUTS ---


@dataclass(frozen=True)
class NutsRun:
    """One chain: every site's un-thinned draws, the `diverging`/`num_steps` counters, the time."""

    samples: dict[str, np.ndarray]
    extra: dict[str, np.ndarray]
    wall_s: float


def run_nuts(model: Callable[[], None], nuts: NUTSConfig, seed: int) -> NutsRun:
    """`sagp.gp._run_nuts`'s sampler lines on a prior-only model, the clock read after the draws.

    The lines are reproduced here because `_run_nuts` is private and takes a `CellGP`:
    - `NUTS(model, dense_mass=True, max_tree_depth=nuts.max_tree_depth)`, NumPyro's defaults
      otherwise (`init_to_uniform`, `target_accept_prob` 0.8);
    - `MCMC(kernel, num_warmup, num_samples, progress_bar=False)`, one chain;
    - `mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))`.
    The wall time runs from building the kernel to `jax.block_until_ready` on the draws and
    counters, so it includes compilation, as `_run_nuts`'s does. JAX dispatch is asynchronous, and
    a clock read as soon as `MCMC.run` returns can stop before the chain has run.
    """
    if nuts.num_chains != 1:
        raise ValueError(f"num_chains must be 1, as in `_run_nuts`; got {nuts.num_chains}")
    start = time.perf_counter()
    kernel = NUTS(model, dense_mass=True, max_tree_depth=nuts.max_tree_depth)
    mcmc = MCMC(
        kernel, num_warmup=nuts.num_warmup, num_samples=nuts.num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
    samples, extra = jax.block_until_ready((mcmc.get_samples(), mcmc.get_extra_fields()))
    wall_s = time.perf_counter() - start
    return NutsRun(
        samples={site: np.asarray(value) for site, value in samples.items()},
        extra={name: np.asarray(value) for name, value in extra.items()},
        wall_s=wall_s,
    )


# --- caching ---


def _digest(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write through a temporary file, so an interrupted run leaves no truncated cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def _cached_run(cache: Path | None, key: dict, compute: Callable[[], NutsRun]) -> NutsRun:
    """The run `key` names: loaded from `cache` if stored there, else computed and stored."""
    if cache is None:
        return compute()
    path = cache / (
        f"run_{key['tier']}_{key['impl']}_{key['point']}_p{key['dim']}_s{key['seed']}"
        f"_d{key['depth']}_{_digest(key)}.npz"
    )
    if path.exists():
        with np.load(path) as stored:
            return NutsRun(
                samples={name[3:]: stored[name] for name in stored.files if name.startswith("s__")},
                extra={name[3:]: stored[name] for name in stored.files if name.startswith("e__")},
                wall_s=float(stored["wall_s"]),
            )
    run = compute()
    _save_npz(path, {
        **{f"s__{site}": value for site, value in run.samples.items()},
        **{f"e__{name}": value for name, value in run.extra.items()},
        "wall_s": np.asarray(run.wall_s),
        "key": np.asarray(json.dumps(key, sort_keys=True)),
    })
    return run


def _cached_truth(cache: Path | None, a: float, b: float, k: float, dim: int, n: int) -> dict:
    """`ground_truth(a, b, k, dim=dim, n=n, seed=0)`, stored once per point in `cache`."""
    if cache is None:
        return ground_truth(a, b, k, dim=dim, n=n)
    key = {"a": a, "b": b, "k": k, "dim": dim, "n": n, "seed": 0, "code": SOURCE_DIGEST}
    path = cache / f"truth_p{dim}_{_digest(key)}.npz"
    if path.exists():
        with np.load(path) as stored:
            return {name: stored[name] for name in stored.files}
    truth = ground_truth(a, b, k, dim=dim, n=n)
    _save_npz(path, truth)
    return truth


# --- statistics ---


def draw_statistics(log_a_sq: np.ndarray) -> dict[str, np.ndarray]:
    """The ground truth's per-draw scalars, plus log R2 and log phi, from model draws of log a_sq.

    Every implementation returns log theta = log omega + log phi, and the shares sum to 1, so
    log omega = logsumexp_i log a_sq_i (= logit R2) whatever sites the implementation samples. The
    statistics are read off that one array, so they cannot favour one parameterization. For the
    control, whose shares are not a Dirichlet, only the counts mean anything.
    """
    log_a_sq = np.asarray(log_a_sq, dtype=float)
    log_omega = special.logsumexp(log_a_sq, axis=-1)
    log_phi = log_a_sq - log_omega[..., None]
    return {
        "logit_R2": log_omega,
        "log_R2": -np.logaddexp(0.0, -log_omega),
        "log_phi": log_phi,
        "log_phi_1": log_phi[..., 0],
        "log_a_sq_1": log_a_sq[..., 0],
        "log_max_phi": log_phi.max(axis=-1),
        "n_active": active_count(log_a_sq, ACTIVE_EPS),
        "n_active_rho": active_count(log_a_sq, RHO_EPS, scale=RHO_SCALE),
    }


def count_summary(counts: np.ndarray) -> dict[str, float]:
    """Mean, sd and the 5/25/50/75/95 % quantiles of per-draw counts, by the "inverted_cdf" rule."""
    counts = np.asarray(counts)
    summary = {"mean": float(counts.mean()), "sd": float(counts.std(ddof=1))}
    return summary | {
        name: float(np.quantile(counts, q, method="inverted_cdf")) for name, q in _QUANTILES.items()
    }


def pmf_summary(pmf: np.ndarray) -> dict[str, float]:
    """`count_summary`'s statistics of a pmf on 0, 1, ..., by the same integer quantile rule."""
    n = np.arange(len(pmf))
    mean = float(n @ pmf)
    summary = {"mean": mean, "sd": float(math.sqrt((n - mean) ** 2 @ pmf))}
    return summary | {name: float(pmf_quantile(pmf, q)) for name, q in _QUANTILES.items()}


def _moment_check(x: np.ndarray, mean: float, var: float) -> dict[str, np.ndarray]:
    """Per-column ESS and |deviation| / sigma of the mean and variance from the closed forms.

    sigma is sqrt(var / ESS) for the mean and var sqrt((kappa - 1) / ESS) for the variance, with
    kappa the column's plain (non-excess) kurtosis. The Gaussian sqrt(2 / ESS) would be far too
    tight at small k, where the law is nearly -Exponential(1) / k and kappa - 1 = 8. A column is
    inside when both are at most 3; a NaN is outside.
    """
    columns = np.asarray(x, dtype=float).reshape(len(x), -1)
    ess = effective_sample_size(columns[None])
    kappa = kurtosis(columns, axis=0, fisher=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_mean = np.abs(columns.mean(axis=0) - mean) / np.sqrt(var / ess)
        z_var = np.abs(columns.var(axis=0, ddof=1) - var) / (var * np.sqrt((kappa - 1.0) / ess))
    return {"ess": ess, "z_mean": z_mean, "z_var": z_var}


def tier_a_statistics(
    log_a_sq: np.ndarray, extra: dict[str, np.ndarray], *, a: float, b: float, alpha: float,
    truth: dict[str, np.ndarray],
) -> dict[str, object]:
    """Tier A's checks of one long chain of log a_sq (n, p) against the closed forms and `truth`.

    - Moments, per coordinate: mean(log a_sq_i) within 3 sqrt(Var / ESS_i), and var(log a_sq_i)
      within 3 Var sqrt((kappa_i - 1) / ESS_i); at most 1 % of the p coordinates outside. The same
      for log R2 and log omega on their own ESS, and Var(log phi_i - log phi_j) = 2 psi'(k) on the
      pairs `PHI_PAIRS` (log omega cancels in the difference), none outside.
    - Distribution: two-sample KS of logit R2, log phi_1, log a_sq_1 and log max phi, each on
      n = ESS_min equally spaced draws (KS assumes independent draws) against the ground truth,
      every p >= 0.01.
    - Count: the median of N_eps = #{a_sq_i > ACTIVE_EPS} within 1 and its quartiles within 3 of
      the ground truth's; at a tied point also the exact P(a_sq_i > eps) against the empirical
      frequency, within 3 standard errors on the count series' ESS.

    The verdict is "not evaluable" when the ESS_min of log a_sq is below 100 or more than 1 % of the
    draws diverged. Such a run is neither a pass nor a fail; its ESS and `need_k_*` are kept. The
    `need_k_*` fields are the multiple of the standard error that would put every coordinate
    inside. Otherwise the verdict is "pass" when every check holds and "fail" when one does not;
    `failed_checks` names the failures.
    """
    log_a_sq = np.asarray(log_a_sq, dtype=float)
    n_draws, dim = log_a_sq.shape
    k = alpha / dim
    exact = closed_forms(a, b, k, dim=dim)
    stats = draw_statistics(log_a_sq)
    divergences = int(np.asarray(extra["diverging"]).sum())

    coords = _moment_check(log_a_sq, exact["mean_log_a_sq"], exact["var_log_a_sq"])
    ess_min, ess_median = float(np.min(coords["ess"])), float(np.median(coords["ess"]))
    evaluable = bool(
        ess_min >= MIN_EVALUABLE_ESS and divergences <= MAX_DIVERGENT_FRACTION * n_draws)
    outside_mean = float(np.mean(~(coords["z_mean"] <= 3.0)))
    outside_var = float(np.mean(~(coords["z_var"] <= 3.0)))
    scalars = {
        name: _moment_check(stats[series], exact[f"mean_{name}"], exact[f"var_{name}"])
        for name, series in (("log_R2", "log_R2"), ("log_omega", "logit_R2"))
    }
    pairs = [(i, j) for i, j in PHI_PAIRS if j < dim]
    pair_check = _moment_check(
        np.column_stack([log_a_sq[:, i] - log_a_sq[:, j] for i, j in pairs]),
        0.0, exact["var_log_phi_diff"],
    )
    pairs_outside = int(np.sum(~(pair_check["z_var"] <= 3.0)))

    n_ks = int(min(n_draws, math.floor(ess_min))) if ess_min >= 1.0 else 0
    thinned = np.round(np.linspace(0, n_draws - 1, n_ks)).astype(int)
    ks = {
        f"ks_p_{name}":
            float(ks_2samp(stats[name][thinned], truth[name]).pvalue) if n_ks else math.nan
        for name in ("logit_R2", "log_phi_1", "log_a_sq_1", "log_max_phi")
    }

    quartiles = np.quantile(stats["n_active"], [0.25, 0.5, 0.75], method="inverted_cdf")
    truth_quartiles = np.quantile(truth["n_active"], [0.25, 0.5, 0.75], method="inverted_cdf")
    gaps = np.abs(quartiles - truth_quartiles)

    checks = {
        "mean_log_a_sq": outside_mean <= MAX_FRACTION_OUTSIDE,
        "var_log_a_sq": outside_var <= MAX_FRACTION_OUTSIDE,
        **{name: bool(c["z_mean"][0] <= 3.0 and c["z_var"][0] <= 3.0)
           for name, c in scalars.items()},
        "phi_pairs": pairs_outside == 0,
        "ks": all(p >= KS_MIN_P for p in ks.values()),
        "count": bool(gaps[1] <= 1 and gaps[0] <= 3 and gaps[2] <= 3),
    }
    p_exact = p_emp = p_z = math.nan
    if math.isclose(a, alpha, rel_tol=1e-12):
        share = stats["n_active"] / dim
        p_exact, p_emp = p_active_tied(k, b, ACTIVE_EPS), float(share.mean())
        se = float(share.std(ddof=1) / np.sqrt(effective_sample_size(share[None].astype(float))))
        p_z = abs(p_emp - p_exact) / se if se > 0 else math.nan
        checks["p_active"] = bool(p_z <= 3.0)
    failed = [name for name, ok in checks.items() if not ok]
    verdict = "not evaluable" if not evaluable else ("fail" if failed else "pass")
    return {
        "verdict": verdict,
        "evaluable": evaluable,
        "failed_checks": ", ".join(failed),
        "divergences": divergences,
        "ess_log_a_sq_median": ess_median,
        "ess_log_a_sq_min": ess_min,
        "frac_outside_mean": outside_mean,
        "frac_outside_var": outside_var,
        "need_k_mean": float(np.max(coords["z_mean"])),
        "need_k_var": float(np.max(coords["z_var"])),
        **{f"{name}_{field}": float(c[key][0]) for name, c in scalars.items()
           for field, key in (("ess", "ess"), ("z_mean", "z_mean"), ("z_var", "z_var"))},
        "pairs_outside": pairs_outside,
        "pairs_z_var_max": float(np.max(pair_check["z_var"])),
        "ks_n": n_ks,
        **ks,
        **{f"count_{name}": float(value)
           for name, value in zip(("q25", "q50", "q75"), quartiles)},
        **{f"truth_{name}": float(value)
           for name, value in zip(("q25", "q50", "q75"), truth_quartiles)},
        "p_active_exact": p_exact,
        "p_active_emp": p_emp,
        "p_active_z": p_z,
    }


# --- the tiers ---


def _point_columns(impl: str, point: str, dim: int, a: float, b: float, alpha: float) -> dict:
    return {
        "impl": impl, "point": point, "p": dim, "a": a, "b": b, "alpha": alpha, "k": alpha / dim}


def tier_a_row(
    impl: str, point: str, *, dim: int, nuts: NUTSConfig, truth: dict[str, np.ndarray],
    calibration: dict | None = None, cache: Path | None = None,
) -> tuple[dict, NutsRun]:
    """One Tier A chain (seed 0) of `impl` at `point`, its row and the run itself."""
    a, b, alpha = hyperparameters(point, dim, calibration)
    key = {"tier": "A", "impl": impl, "point": point, "dim": dim, "seed": 0,
           "depth": nuts.max_tree_depth, "num_warmup": nuts.num_warmup,
           "num_samples": nuts.num_samples, "a": a, "b": b, "alpha": alpha, "code": SOURCE_DIGEST}
    run = _cached_run(
        cache, key, lambda: run_nuts(prior_model(impl, dim, a=a, b=b, alpha=alpha), nuts, 0))
    stats = tier_a_statistics(
        run.samples["log_a_sq"], run.extra, a=a, b=b, alpha=alpha, truth=truth)
    num_steps = run.extra["num_steps"]
    return {
        **_point_columns(impl, point, dim, a, b, alpha),
        "seed": 0, "depth": nuts.max_tree_depth, "num_warmup": nuts.num_warmup,
        "num_samples": nuts.num_samples, **stats,
        "num_steps_mean": float(num_steps.mean()),
        "frac_at_cap": float(np.mean(num_steps >= 2**nuts.max_tree_depth - 1)),
        "wall_s": run.wall_s, "agrees_with_scouting": "",
    }, run


def tier_b_row(
    impl: str, point: str, dim: int, seed: int, nuts: NUTSConfig, *,
    calibration: dict | None = None, cache: Path | None = None,
) -> dict:
    """One Tier B chain at the budget `nuts` and its gate, on the sampled sites and on log a_sq.

    `point` is "-" for the control, which has no R2-D2 hyperparameters. `limiting_site` is the
    sampled site with the smallest ESS (a NaN counts as smallest), the one a failed gate points at.
    """
    a, b, alpha = (math.nan,) * 3 if impl == CONTROL else hyperparameters(point, dim, calibration)
    key = {"tier": "B", "impl": impl, "point": point, "dim": dim, "seed": seed,
           "depth": nuts.max_tree_depth, "num_warmup": nuts.num_warmup,
           "num_samples": nuts.num_samples, "a": a, "b": b, "alpha": alpha, "code": SOURCE_DIGEST}
    run = _cached_run(
        cache, key, lambda: run_nuts(prior_model(impl, dim, a=a, b=b, alpha=alpha), nuts, seed))
    supports = SITE_SUPPORTS[impl]
    sampled = gate({site: run.samples[site] for site in supports}, run.extra, supports)
    on_log_a_sq = gate({"log_a_sq": run.samples["log_a_sq"]}, run.extra, {"log_a_sq": "real"})
    by_site = sampled["n_eff_min_by_site"]
    num_steps = run.extra["num_steps"]
    return {
        **_point_columns(impl, point, dim, a, b, alpha),
        "seed": seed, "depth": nuts.max_tree_depth,
        "r_hat_max": sampled["r_hat_max"], "n_eff_min": sampled["n_eff_min"],
        "r_hat_max_log_a_sq": on_log_a_sq["r_hat_max"],
        "n_eff_min_log_a_sq": on_log_a_sq["n_eff_min"],
        "divergences": sampled["divergences"],
        "num_steps_mean": float(num_steps.mean()),
        "frac_at_cap": float(np.mean(num_steps >= 2**nuts.max_tree_depth - 1)),
        "wall_s": run.wall_s,
        "gate": "pass" if sampled["passed"] else "fail",
        "gate_reason": sampled["reason"],
        "gate_log_a_sq": "pass" if on_log_a_sq["passed"] else "fail",
        "gate_log_a_sq_reason": on_log_a_sq["reason"],
        "limiting_site": min(
            by_site, key=lambda site: -math.inf if math.isnan(by_site[site]) else by_site[site]),
        "n_eff_min_by_site": by_site,
        "agrees_with_scouting": "",
    }


def _has_factor(model: Callable[[], None]) -> bool:
    """Whether the prior-only model has an observed site; here that can only be a factor."""
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
    return any(site["type"] == "sample" and site["is_observed"] for site in trace.values())


def tier_c_row(
    impl: str, point: str, dim: int, *, truth: dict[str, np.ndarray],
    calibration: dict | None = None, num_samples: int = 4_000, seed: int = 0,
) -> dict:
    """Forward draws, `Predictive(model, num_samples)`, against the ground truth: not a gate.

    KS of log phi_1 and of logit R2 against `truth`; `matches` when both p >= 0.01. `has_factor`
    records the mechanism of a mismatch: `Predictive` samples every latent site from its own
    distribution and never reads a factor, so a factor-corrected site comes out as its proposal.
    """
    a, b, alpha = hyperparameters(point, dim, calibration)
    model = prior_model(impl, dim, a=a, b=b, alpha=alpha)
    draws = Predictive(model, num_samples=num_samples)(jax.random.PRNGKey(seed))
    stats = draw_statistics(np.asarray(draws["log_a_sq"]))
    ks_phi = ks_2samp(stats["log_phi_1"], truth["log_phi_1"])
    ks_r2 = ks_2samp(stats["logit_R2"], truth["logit_R2"])
    matches = bool(ks_phi.pvalue >= KS_MIN_P and ks_r2.pvalue >= KS_MIN_P)
    has_factor = _has_factor(model)
    note = "" if matches else (
        "Predictive draws the factor-corrected site from its N(0, 1) proposal and never reads the "
        "factor" if has_factor else "no factor in the model: unexplained")
    return {
        **_point_columns(impl, point, dim, a, b, alpha),
        "num_samples": num_samples,
        "ks_d_log_phi_1": float(ks_phi.statistic), "ks_p_log_phi_1": float(ks_phi.pvalue),
        "ks_d_logit_R2": float(ks_r2.statistic), "ks_p_logit_R2": float(ks_r2.pvalue),
        "matches": matches, "has_factor": has_factor, "note": note,
        "agrees_with_scouting": "",
    }


# --- the calibration (Tier D, plan D2) ---


@dataclass(frozen=True)
class CalibrationConfig:
    """The D2 search: the tied median match at `tied_b`; the untied profile over `a_grid`."""

    dim: int = 100
    tied_b: float = 0.5
    a_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0,
                                 20.0, 30.0)
    refine: bool = True
    search_qmc_log2: int = 12  # the search's resolution; every reported row is re-solved at 2^14


B_MIN = 0.5  # D2's proper region is a, b >= 0.5
_INFEASIBLE = 1e3  # the search's residual where no k matches the median


def median_matched_k(
    a: float | None, b: float, target: np.ndarray, *, dim: int, eps: float,
    qmc_log2: int = QMC_LOG2, k_hint: float | None = None,
) -> float | None:
    """The k at which the R2-D2 active count's median equals `target`'s, as continuous quantiles.

    `a` None means the tie a = k * dim, searched from k = 1e-3. Otherwise k runs up from a / dim,
    the tie, which is the lower edge of `count_pmf`'s exact domain. The first upward crossing on a
    log-spaced scan up to k = 20 is refined by Brent's method. `k_hint`, a nearby solution, is
    tried first as a bracket [k_hint / 1.2, 1.2 k_hint]. The result is None when the median never
    reaches the target, or already exceeds it at the lower edge. Matching the continuous
    quantiles makes the integer medians equal and fixes k uniquely.
    """
    goal = pmf_quantile(target, 0.5, continuous=True)
    k_lo = 1e-3 if a is None else a / dim

    def gap(k: float) -> float:
        pmf = count_pmf(k * dim if a is None else a, b, k, eps, dim, qmc_log2=qmc_log2)
        return pmf_quantile(pmf, 0.5, continuous=True) - goal

    def solve(lo: float, hi: float) -> float:
        return float(optimize.brentq(gap, lo, hi, xtol=1e-12, rtol=1e-10))

    if k_hint is not None and k_hint / 1.2 > k_lo and gap(k_hint / 1.2) < 0.0 <= gap(1.2 * k_hint):
        return solve(k_hint / 1.2, 1.2 * k_hint)
    grid = np.geomspace(k_lo, 20.0, 32)
    if not gap(float(grid[0])) < 0.0:
        return None
    for lo, hi in zip(grid[:-1], grid[1:]):
        if gap(float(hi)) >= 0.0:
            return solve(float(lo), float(hi))
    return None


def _calibration_row(
    kind: str, a: float, b: float, k: float | None, target: np.ndarray, *, dim: int,
) -> dict:
    """One point of the search: its count quartiles (integer and continuous) and residuals, at 2^14.

    The residual is D2's max(|q25 - q25_HC|, |q75 - q75_HC|), in integers and in continuous
    quantiles; the search minimizes the continuous one. Two closed forms describe the point's
    regime beside k itself (shares near-uniform once k >= 1): sd(log omega) =
    sqrt(psi'(a) + psi'(b)), the global scale's spread, and P(R2 < 0.6), the D1 argument's mass
    below R2 = 0.6.
    """
    row = {"kind": kind, "a": a, "b": b, "k": math.nan if k is None else k,
           "alpha": math.nan if k is None else k * dim,
           "sd_log_omega": math.sqrt(special.polygamma(1, a) + special.polygamma(1, b)),
           "P_R2_below_0.6": float(beta_distribution.cdf(0.6, a, b))}
    names = ("q25", "q50", "q75")
    if k is None:
        return row | {name: math.nan for name in names} | {f"{n}_c": math.nan for n in names} | {
            "residual": math.nan, "residual_c": math.nan,
            "note": f"no k (with k dim >= a) matches the median at any b >= {B_MIN}"}
    pmf = target if kind == "target" else count_pmf(a, b, k, ACTIVE_EPS, dim)
    for name, q in zip(names, (0.25, 0.5, 0.75)):
        row[name] = pmf_quantile(pmf, q)
        row[f"{name}_c"] = pmf_quantile(pmf, q, continuous=True)
    outer = (("q25", 0.25), ("q75", 0.75))
    row["residual"] = max(abs(row[n] - pmf_quantile(target, q)) for n, q in outer)
    row["residual_c"] = max(
        abs(row[f"{n}_c"] - pmf_quantile(target, q, continuous=True)) for n, q in outer)
    return row | {"note": ""}


class _UntiedSearch:
    """The continuous residual at (a, b), k matched, at the search's resolution; its min over b."""

    def __init__(self, target: np.ndarray, *, dim: int, qmc_log2: int) -> None:
        self.target, self.dim, self.qmc_log2 = target, dim, qmc_log2
        self.goals = {q: pmf_quantile(target, q, continuous=True) for q in (0.25, 0.75)}
        self.last_k: float | None = None
        self.evaluations = 0

    def residual(self, a: float, b: float) -> tuple[float, float | None]:
        """(continuous residual, k), or (_INFEASIBLE, None) where no k matches the median."""
        self.evaluations += 1
        k = median_matched_k(a, b, self.target, dim=self.dim, eps=ACTIVE_EPS,
                             qmc_log2=self.qmc_log2, k_hint=self.last_k)
        if k is None:
            return _INFEASIBLE, None
        self.last_k = k
        pmf = count_pmf(a, b, k, ACTIVE_EPS, self.dim, qmc_log2=self.qmc_log2)
        return max(abs(pmf_quantile(pmf, q, continuous=True) - goal)
                   for q, goal in self.goals.items()), k

    def best_b(self, a: float) -> tuple[float, float]:
        """(min over b >= B_MIN of the residual at a, its b); (_INFEASIBLE, nan) if none matches.

        For fixed a, raising b lowers omega. The b at which some k matches the median form an
        interval. Below it, when a is large, the median already exceeds the target at the tie;
        above it, no k reaches the target. Inside it, raising b raises the matched k and spreads
        the count (q25 falls, q75 rises), so the residual has a narrow valley. A log-spaced scan
        over [B_MIN, 4 a + 4] finds the interval and the best scanned b. Bounded Brent then refines
        between that b's two neighbours, the infeasible side reading as a plateau at _INFEASIBLE.
        """
        grid = np.geomspace(B_MIN, 4.0 * a + 4.0, 24)
        values = [self.residual(a, float(b))[0] for b in grid]
        i = int(np.argmin(values))
        if values[i] >= _INFEASIBLE:
            return _INFEASIBLE, math.nan
        lo, hi = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, len(grid) - 1)])
        result = optimize.minimize_scalar(lambda b: self.residual(a, b)[0], bounds=(lo, hi),
                                          method="bounded", options={"xatol": 1e-4})
        if result.fun < values[i]:
            return float(result.fun), float(result.x)
        return float(values[i]), float(grid[i])


def calibrate(config: CalibrationConfig) -> dict:
    """The D2 rule's two calibration points at D = `config.dim`, against the half-Cauchy count.

    The target is the count distribution of the four half-Cauchy cells at `ALPHA_AMPLITUDE`,
    exact given tausq. C_tied is the tie a = k D at b = `tied_b`, with k matching the target's
    median; it has one knob. C_untied matches the median exactly and minimizes the continuous
    quartile residual over a, b >= 0.5, in three steps:
    - a profile: for each a of `a_grid`, the best b (k matched at every b). The residual has a
      narrow valley in b and is shallow along a, which a grid over (a, b) misses;
    - with `refine`, a bounded Brent search for a between the best profile point's two
      neighbours, each evaluation being a profile point;
    - every reported row (the profile's and the refined point's) re-solved for k at the full 2^14
      resolution, and C_untied the best of them there, since the valley is shallow enough along a
      that the two resolutions can rank neighbouring points differently.
    The search stays in the exact domain k D >= a of `count_pmf`, whose edge is the tie. Returns
    the two points, the target's quartiles, every row, the search's evaluation count, and
    `optimum_at_profile_edge`: whether the best profile point is the grid's largest a, in which
    case the minimum may lie beyond the grid.
    """
    dim = config.dim
    target = half_cauchy_count_pmf(ALPHA_AMPLITUDE, ACTIVE_EPS, dim)
    search = _UntiedSearch(target, dim=dim, qmc_log2=config.search_qmc_log2)

    def reported(kind: str, a: float, b: float, k_hint: float | None) -> dict:
        k = median_matched_k(a, b, target, dim=dim, eps=ACTIVE_EPS, k_hint=k_hint)
        return _calibration_row(kind, a, b, k, target, dim=dim)

    k_tied = median_matched_k(None, config.tied_b, target, dim=dim, eps=ACTIVE_EPS)
    if k_tied is None:
        raise RuntimeError(f"no tied k at b = {config.tied_b} matches the half-Cauchy median")
    tied = _calibration_row("C_tied", k_tied * dim, config.tied_b, k_tied, target, dim=dim)

    profile = []
    for a in config.a_grid:
        residual, b = search.best_b(a)
        profile.append((residual, a, b, search.last_k if residual < _INFEASIBLE else None))
    feasible = [point for point in profile if point[0] < _INFEASIBLE]
    if not feasible:
        raise RuntimeError("no a of the calibration profile matches the half-Cauchy median")
    best = min(feasible)
    rows = [tied]
    rows += [reported("profile", a, b, k) if k is not None
             else _calibration_row("profile", a, math.nan, None, target, dim=dim)
             for _, a, b, k in profile]
    candidates = [row for row in rows[1:] if math.isfinite(row["residual_c"])]
    i = config.a_grid.index(best[1])
    lo, hi = config.a_grid[max(i - 1, 0)], config.a_grid[min(i + 1, len(config.a_grid) - 1)]
    if config.refine and hi > lo:
        result = optimize.minimize_scalar(lambda a: search.best_b(a)[0], bounds=(lo, hi),
                                          method="bounded", options={"xatol": 1e-3})
        b_refined = search.best_b(float(result.x))[1]
        rows.append(reported("refined", float(result.x), b_refined, search.last_k))
        candidates.append(rows[-1])
    # The search's resolution decides where to look; the reported resolution decides the choice.
    untied = min((row for row in candidates if math.isfinite(row["residual_c"])),
                 key=lambda row: row["residual_c"]) | {"kind": "C_untied"}
    rows.append(untied)
    return {
        "config": asdict(config),
        "target": _calibration_row("target", math.nan, math.nan, math.nan, target, dim=dim),
        "C_tied": {"a": tied["a"], "b": tied["b"], "k": tied["k"]},
        "C_untied": {"a": untied["a"], "b": untied["b"], "k": untied["k"]},
        "rows": rows,
        "search_evaluations": search.evaluations,
        "optimum_at_profile_edge": best[1] == config.a_grid[-1],
    }


def _cached_calibration(cache: Path | None, config: CalibrationConfig) -> dict:
    if cache is None:
        return calibrate(config)
    path = cache / f"calibration_{_digest(asdict(config) | {'code': SOURCE_DIGEST})}.json"
    if path.exists():
        return json.loads(path.read_text())
    calibration = calibrate(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(calibration, indent=1))
    os.replace(tmp, path)
    return calibration


# --- Tier D ---


def _half_cauchy_counts(dim: int, n: int, seed: int = 0) -> np.ndarray:
    """#{i : tausq lam_i > ACTIVE_EPS} per draw, tausq ~ HC(ALPHA_AMPLITUDE), lam_i ~ HC(1)."""
    rng = np.random.default_rng(seed)
    tausq = ALPHA_AMPLITUDE * np.abs(rng.standard_cauchy(n))
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, 10_000):
        stop = min(start + 10_000, n)
        lam = np.abs(rng.standard_cauchy((stop - start, dim)))
        counts[start:stop] = np.count_nonzero(tausq[start:stop, None] * lam > ACTIVE_EPS, axis=1)
    return counts


def _profile_name(row: dict) -> str:
    return f"profile_a{row['a']:g}"


def tier_d(
    config: StudyConfig, calibration: dict, tier_a_runs: dict[str, dict[str, tuple[NutsRun, bool]]],
    cache: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Tier D's two tables: the active-count readout and the calibration search.

    The readout is at p = `config.dim_a`. It covers the half-Cauchy target (mixture and NumPy),
    every study point, the brief's tied k = 0.0892 and the calibration profile (where the search
    ran at this p). Each has its ground truth on both scales (rho = RHO_SCALE a_sq counted at
    RHO_EPS, with the transport identity checked draw for draw), its exact mixture where one
    exists, and each Tier A chain of `tier_a_runs` ({point: {impl: (run, evaluable)}}) as a
    cross-check. The calibration rows gain the ground truth's quartiles at their point.
    """
    dim, n = config.dim_a, config.n_truth
    rows: list[dict] = []

    def add(point: str, a: float, b: float, alpha: float, scale: str, source: str,
            summary: dict, **fields: object) -> None:
        rows.append({"point": point, "a": a, "b": b, "alpha": alpha, "k": alpha / dim,
                     "scale": scale, "source": source, **summary, **fields,
                     "agrees_with_scouting": ""})

    nan = math.nan
    add("half-Cauchy", nan, nan, nan, "amplitude", "mixture",
        pmf_summary(half_cauchy_count_pmf(ALPHA_AMPLITUDE, ACTIVE_EPS, dim)))
    add("half-Cauchy", nan, nan, nan, "amplitude", "ground_truth",
        count_summary(_half_cauchy_counts(dim, n)))

    points = {name: hyperparameters(name, dim, calibration) for name in POINTS}
    points["tied_k0.0892"] = (BRIEF_TIED_K * dim, 0.5, BRIEF_TIED_K * dim)
    on_calibration_p = dim == calibration["config"]["dim"]
    if on_calibration_p:
        points |= {_profile_name(r): (r["a"], r["b"], r["alpha"]) for r in calibration["rows"]
                   if r["kind"] == "profile" and math.isfinite(r["k"])}
    truth_quartiles = {}
    for name, (a, b, alpha) in points.items():
        k = alpha / dim
        truth = (_cached_truth(cache, a, b, k, dim, n) if name in POINTS
                 else ground_truth(a, b, k, dim=dim, n=n))
        amplitude = count_summary(truth["n_active"])
        truth_quartiles[name] = amplitude
        add(name, a, b, alpha, "amplitude", "ground_truth", amplitude)
        add(name, a, b, alpha, "rho", "ground_truth", count_summary(truth["n_active_rho"]),
            transport_identity=bool(np.array_equal(truth["n_active_rho"], truth["n_active"])))
        if a <= alpha or math.isclose(a, alpha, rel_tol=1e-12):
            add(name, a, b, alpha, "amplitude", "mixture",
                pmf_summary(count_pmf(a, b, k, ACTIVE_EPS, dim)))
        for impl, (run, evaluable) in tier_a_runs.get(name, {}).items():
            stats = draw_statistics(run.samples["log_a_sq"])
            add(name, a, b, alpha, "amplitude", f"{impl} Tier A", count_summary(stats["n_active"]),
                evaluable=evaluable)
            add(name, a, b, alpha, "rho", f"{impl} Tier A", count_summary(stats["n_active_rho"]),
                evaluable=evaluable,
                transport_identity=bool(np.array_equal(stats["n_active_rho"], stats["n_active"])))

    calibration_rows = [calibration["target"] | {"kind": "target"}]
    for row in calibration["rows"]:
        name = _profile_name(row) if row["kind"] == "profile" else row["kind"]
        truth = truth_quartiles.get(name) if on_calibration_p else None
        calibration_rows.append(row | {
            f"truth_{q}": truth[q] if truth else math.nan for q in ("q25", "q50", "q75")})
    return rows, calibration_rows


# --- acceptance ---


def acceptance(
    tier_a_rows: list[dict], tier_b_rows: list[dict], *, seeds: tuple[int, ...],
    calibration: dict, dim: int = 100, depth: int = 6,
) -> list[dict]:
    """The acceptance rule per implementation, and the same two checks at P2 and the C points.

    An implementation passes if (i) Tier A is within tolerance at P1 and, where it represents
    it, at P3, and (ii) the depth-6 Tier B gate passes for every seed at p = 100 at P1. P2 is
    reported, not required. The C points carry G0's check of the calibrated point. A Tier A entry
    is its verdict, "n/a" where the implementation cannot represent the point, or "missing" when
    the tier did not run. A Tier B entry is "passed/seeds".
    """
    verdicts = {(r["impl"], r["point"]): r["verdict"] for r in tier_a_rows if r["p"] == dim}

    def gate_passes(impl: str, point: str) -> tuple[int, int]:
        rows = [r for r in tier_b_rows
                if (r["impl"], r["point"], r["p"], r["depth"]) == (impl, point, dim, depth)]
        ran = {r["seed"] for r in rows} & set(seeds)
        return len({r["seed"] for r in rows if r["gate"] == "pass"} & ran), len(ran)

    table = []
    for impl in IMPLEMENTATIONS:
        row: dict[str, object] = {"impl": impl}
        for point in ("P1", "P3", "P2", "C_tied", "C_untied"):
            a, b, alpha = hyperparameters(point, dim, calibration)
            if not represents(impl, a, alpha):
                row[f"tier_a_{point}"] = row[f"tier_b_{point}"] = "n/a"
                continue
            row[f"tier_a_{point}"] = verdicts.get((impl, point), "missing")
            passed, ran = gate_passes(impl, point)
            row[f"tier_b_{point}"] = f"{passed}/{len(seeds)}" if ran else "missing"
        b_passed, b_ran = gate_passes(impl, "P1")
        row["passes"] = bool(
            row["tier_a_P1"] == "pass" and row["tier_a_P3"] in ("pass", "n/a")
            and b_ran == len(seeds) and b_passed == len(seeds))
        table.append(row | {"agrees_with_scouting": ""})
    return table


def harness_check(control_rows: list[dict], *, dim: int = 100, depth: int = 6) -> bool | None:
    """Whether the half-Cauchy control passes the depth-6 gate at p = 100 for every seed.

    None when no such control row ran.
    """
    rows = [r for r in control_rows if r["p"] == dim and r["depth"] == depth]
    return all(r["gate"] == "pass" for r in rows) if rows else None


# --- the command line ---


@dataclass(frozen=True)
class StudyConfig:
    """The study's settings: the defaults are the protocol's; `SMOKE` is a short check."""

    dim_a: int = 100  # Tier A and Tier D
    dims_b: tuple[int, ...] = (30, 100)  # Tier B and Tier C
    seeds: tuple[int, ...] = (0, 1, 2)
    depths: tuple[int, ...] = (6, 10)
    tier_a_nuts: NUTSConfig = NUTSConfig(num_warmup=512, num_samples=4_000, max_tree_depth=10)
    tier_b_nuts: NUTSConfig = NUTSConfig()  # its depth is replaced per row
    n_truth: int = 200_000
    tier_c_samples: int = 4_000
    calibration: CalibrationConfig = CalibrationConfig()
    smoke: bool = False


SMOKE = StudyConfig(
    dim_a=5, dims_b=(5,), seeds=(0,), depths=(6,),
    tier_a_nuts=NUTSConfig(num_warmup=32, num_samples=64, thinning=4, max_tree_depth=6),
    tier_b_nuts=NUTSConfig(num_warmup=32, num_samples=32, thinning=4),
    n_truth=5_000, tier_c_samples=200,
    calibration=CalibrationConfig(a_grid=(1.5,), refine=False), smoke=True,
)


def _cell(value: object) -> str:
    """A table cell for Markdown: 4 significant digits, a dict as "key value; ..."."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (float, np.floating)):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    if isinstance(value, dict):
        return "; ".join(f"{key} {_cell(item)}" for key, item in value.items())
    return str(value)


def write_table(rows: list[dict], stem: Path) -> None:
    """`stem`.csv (full precision, a dict cell as JSON) and `stem`.md, columns as first seen."""
    if not rows:
        return
    columns = list(dict.fromkeys(name for row in rows for name in row))
    with open(stem.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(row[name]) if isinstance(row.get(name), dict)
                             else row.get(name, "") for name in columns})
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(_cell(row.get(name, "")) for name in columns) + " |" for row in rows]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def _log(tier: str, row: dict, *fields: str) -> None:
    head = f"{tier} {row.get('impl', '')} {row.get('point', '')} p={row.get('p', '')}"
    print(head + "".join(f" {name}={_cell(row[name])}" for name in fields), flush=True)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """Run the requested tiers and write their tables; exit status 3 if the control fails."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.r2d2_prior_study",
        description="The prior-only R2-D2 study: Tiers A-D of the 2026-09-25 brief, with the "
                    "calibration search, the rho-scale identity and the acceptance table.")
    parser.add_argument("--tiers", default="A,B,C,D",
                        help="comma-separated subset of A,B,C,D (default: all four)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="results directory; tables go to OUT/tables/ "
                             f"(default: {DEFAULT_OUT})")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="raw draws and ground truths, reused on rerun "
                             f"(default: {DEFAULT_CACHE})")
    parser.add_argument("--smoke", action="store_true",
                        help="a minutes-long end-to-end check at p = 5 with tiny budgets; its "
                             "tables are not the study's, so it refuses the default --out")
    args = parser.parse_args(argv)
    tiers = [tier.strip().upper() for tier in args.tiers.split(",") if tier.strip()]
    if not tiers or not set(tiers) <= set("ABCD"):
        parser.error(f"--tiers must be a comma-separated subset of A,B,C,D; got {args.tiers!r}")
    config = SMOKE if args.smoke else StudyConfig()
    if args.smoke and args.out.resolve() == DEFAULT_OUT.resolve():
        parser.error("--smoke writes throwaway tables; give it its own --out")
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)

    started, walls, exit_status = time.time(), {}, 0
    clock = time.perf_counter()
    calibration = _cached_calibration(cache, config.calibration)
    walls["calibration"] = time.perf_counter() - clock
    print(f"calibration: C_tied {calibration['C_tied']}, C_untied {calibration['C_untied']}",
          flush=True)
    written: dict[str, list[dict]] = {}

    tier_b_rows: list[dict] = []
    control_ok = None
    if "B" in tiers:  # the control first: if it fails, nothing else is worth reading
        clock = time.perf_counter()
        for depth in config.depths:
            for dim in config.dims_b:
                for seed in config.seeds:
                    row = tier_b_row(CONTROL, "-", dim, seed,
                                     replace(config.tier_b_nuts, max_tree_depth=depth), cache=cache)
                    tier_b_rows.append(row)
                    _log("B", row, "seed", "depth", "gate", "n_eff_min", "wall_s")
        walls["B_control"] = time.perf_counter() - clock
        protocol = config.tier_b_nuts == NUTSConfig()
        control_ok = harness_check(tier_b_rows) if protocol else None
        if control_ok is False:
            print("HARNESS CHECK FAILED: the half-Cauchy control fails the depth-6 gate at "
                  "p = 100; fix the harness before reading any other row.", flush=True)
            exit_status = 3

    tier_a_rows: list[dict] = []
    tier_a_runs: dict[str, dict[str, tuple[NutsRun, bool]]] = {}
    if "A" in tiers and exit_status == 0:
        clock = time.perf_counter()
        for point in POINTS:
            a, b, alpha = hyperparameters(point, config.dim_a, calibration)
            truth = _cached_truth(cache, a, b, alpha / config.dim_a, config.dim_a, config.n_truth)
            for impl in IMPLEMENTATIONS:
                if not represents(impl, a, alpha):
                    continue
                row, run = tier_a_row(impl, point, dim=config.dim_a, nuts=config.tier_a_nuts,
                                      truth=truth, calibration=calibration, cache=cache)
                tier_a_rows.append(row)
                tier_a_runs.setdefault(point, {})[impl] = (run, row["evaluable"])
                _log("A", row, "verdict", "failed_checks", "ess_log_a_sq_min", "divergences",
                     "wall_s")
        walls["A"] = time.perf_counter() - clock

    if "B" in tiers and exit_status == 0:
        clock = time.perf_counter()
        for depth in config.depths:
            nuts = replace(config.tier_b_nuts, max_tree_depth=depth)
            for dim in config.dims_b:
                for impl in IMPLEMENTATIONS:
                    for point in POINTS:
                        a, b, alpha = hyperparameters(point, dim, calibration)
                        if not represents(impl, a, alpha):
                            continue
                        for seed in config.seeds:
                            row = tier_b_row(impl, point, dim, seed, nuts,
                                             calibration=calibration, cache=cache)
                            tier_b_rows.append(row)
                            _log("B", row, "seed", "depth", "gate", "n_eff_min",
                                 "divergences", "wall_s")
        walls["B"] = time.perf_counter() - clock

    tier_c_rows: list[dict] = []
    if "C" in tiers and exit_status == 0:
        clock = time.perf_counter()
        for dim in config.dims_b:
            for point in POINTS:
                a, b, alpha = hyperparameters(point, dim, calibration)
                truth = _cached_truth(cache, a, b, alpha / dim, dim, config.n_truth)
                for impl in IMPLEMENTATIONS:
                    if represents(impl, a, alpha):
                        row = tier_c_row(impl, point, dim, truth=truth, calibration=calibration,
                                         num_samples=config.tier_c_samples)
                        tier_c_rows.append(row)
                        _log("C", row, "matches", "ks_p_log_phi_1", "ks_p_logit_R2")
        walls["C"] = time.perf_counter() - clock

    if "D" in tiers and exit_status == 0:
        clock = time.perf_counter()
        written["tier_d_counts"], written["tier_d_calibration"] = tier_d(
            config, calibration, tier_a_runs, cache)
        walls["D"] = time.perf_counter() - clock

    written |= {"tier_a": tier_a_rows, "tier_b": tier_b_rows, "tier_c": tier_c_rows}
    if "A" in tiers and "B" in tiers and exit_status == 0:
        written["acceptance"] = acceptance(tier_a_rows, tier_b_rows, seeds=config.seeds,
                                           calibration=calibration, dim=config.dim_a)
    for name, rows in written.items():
        write_table(rows, tables / name)
    (tables / "run_config.json").write_text(json.dumps({
        "tiers": tiers, "config": asdict(config), "calibration": {
            name: calibration[name] for name in ("C_tied", "C_untied", "target")},
        "control_passes_depth6_p100": control_ok, "exit_status": exit_status,
        "wall_s": walls, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": _git_head(),
        "source_digest": SOURCE_DIGEST,
        "versions": {"python": platform.python_version(), "jax": jax.__version__,
                     "numpyro": numpyro.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__},
    }, indent=1, default=str))
    print(f"tables written to {tables}; wall s {walls}", flush=True)
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
