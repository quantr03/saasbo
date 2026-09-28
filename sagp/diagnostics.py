"""sagp.diagnostics: the NUTS convergence verdict for one fit attempt.

Pools every sampled site's R-hat and effective sample size against `DiagThresholds` -- the
positive ones on the log scale and the unit-interval one on the logit scale the sampler moves in,
over the attempt's un-thinned draws, not the thinned draws `fit` retains -- into one pass/fail
`Diagnostics` record.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpyro.diagnostics import summary


@dataclass(frozen=True)
class DiagThresholds:
    """The pass rule for one NUTS attempt, fixed for every cell.

    Split-R-hat on the two halves of 128 draws resolves to about +-0.02, so 1.1 is the classical
    bound, not a tuned one; 16 is the number of draws retained for prediction; 5 is 2 % of 256.
    """

    r_hat_max: float = 1.1
    n_eff_min: float = 16.0
    max_divergences: int = 5


@dataclass(frozen=True)
class Diagnostics:
    """One attempt's convergence summary, on its un-thinned draws.

    `passed` decides `status`: ok or excluded. Only `r_hat_max`, `n_eff_min` and `divergences`
    gate; the rest is recorded for reporting. The six per-group fields split those same per-site
    statistics three ways (`_DIAG_GROUPS`) so a trigger can be *attributed* rather than only
    counted, and read NaN for a group this cell has no site in. `reason` names each failed
    criterion and is "" exactly when `passed`.
    """

    r_hat_max: float
    r_hat_median: float
    frac_r_hat_below_1_05: float
    n_eff_min: float
    divergences: int
    num_steps_mean: float
    r_hat_max_native: float
    n_eff_min_native: float
    r_hat_max_ell: float
    n_eff_min_ell: float
    r_hat_max_global: float
    n_eff_min_global: float
    wall_s: float
    passed: bool
    reason: str


# Every *sampled* positive site across the cells; the deterministic `kernel_inv_length_sq` and
# `a_sq` are left out, so the gate reads the sampler's own coordinates. R-hat and ESS are taken on
# the logs: on the constrained half-Cauchy sites both are dominated by single draws.
_POSITIVE_SAMPLED_SITES: tuple[str, ...] = (
    "outputscale",
    "noise",
    "kernel_tausq",
    "_kernel_inv_length_sq",
    "_a_sq",
    "kernel_ell",
)

# The sampled sites that are real-valued rather than positive: `mean` is a Normal draw, and so are
# the R2-D2 prior's copula coordinates in either form, so their R-hat and ESS are taken on their
# natural scale and a log would be NaN half the time.
_REAL_SAMPLED_SITES: tuple[str, ...] = ("mean", "r2d2_z_xi", "r2d2_z_lam")

# The sampled sites on the unit interval: the R2-D2 reference form's Beta site, read on the logit
# scale log(x) - log1p(-x), NumPyro's own unconstrained coordinate for it.
_UNIT_SAMPLED_SITES: tuple[str, ...] = ("r2d2_R2",)

# The three blocks the per-group fields report over, together exactly the three tuples above:
# `native` is whichever site the cell's own sparsity lives on (R2-D2's D copula coordinates where
# the half-Cauchy cells have their D local scales), `ell` the shape parameter only the amplitude
# cells carry (2D + 3 pooled statistics against a lengthscale cell's D + 4), and `global` the
# scalars every cell has. Both R2-D2 forms' sites are listed, so switching the form edits nothing
# here.
_DIAG_GROUPS: dict[str, tuple[str, ...]] = {
    "native": ("_kernel_inv_length_sq", "_a_sq", "r2d2_z_lam"),
    "ell": ("kernel_ell",),
    "global": ("outputscale", "noise", "kernel_tausq", "mean", "r2d2_R2", "r2d2_z_xi"),
}


def _group_extremes(
    per_site: dict[str, tuple[np.ndarray, np.ndarray]], group: tuple[str, ...]
) -> tuple[float, float]:
    """(max R-hat, min ESS) over one group's sites, or (NaN, NaN) when this cell has none of them.

    NaN rather than an infinity, so a group's absence is visibly missing in a log rather than a
    value that would compare as "worst" or "best" against a cell that does carry it.
    """
    present = [per_site[site] for site in group if site in per_site]
    if not present:
        return float("nan"), float("nan")
    return (
        float(np.max(np.concatenate([r_hat for r_hat, _ in present]))),
        float(np.min(np.concatenate([n_eff for _, n_eff in present]))),
    )


def diagnose(
    flat_samples: dict[str, Array],
    extra: dict[str, Array],
    thresholds: DiagThresholds,
    wall_s: float,
) -> Diagnostics:
    """The convergence verdict for one attempt, from its un-thinned draws.

    `numpyro.diagnostics.summary` is the reference's own, and with one chain its split-R-hat
    compares the chain's two halves. Every sampled site is read -- the positive ones on the log
    scale, the real ones on their own, the unit-interval one on the logit scale -- and every
    per-coordinate site contributes D statistics, all of them pooled, so the rule reads "every
    scalar the sampler moved converged", not "the average did". The criteria are negations of the
    pass conditions, so a NaN R-hat -- what a site that never moved produces -- fails rather than
    silently passing; the per-group fields are read off these very statistics and so cannot
    disagree with the verdict.
    """
    stats_input = (
        {
            site: jnp.log(flat_samples[site])
            for site in _POSITIVE_SAMPLED_SITES
            if site in flat_samples
        }
        | {site: flat_samples[site] for site in _REAL_SAMPLED_SITES if site in flat_samples}
        | {
            site: jnp.log(flat_samples[site]) - jnp.log1p(-flat_samples[site])
            for site in _UNIT_SAMPLED_SITES
            if site in flat_samples
        }
    )
    stats = summary(stats_input, prob=0.9, group_by_chain=False)
    per_site = {
        site: (np.ravel(site_stats["r_hat"]), np.ravel(site_stats["n_eff"]))
        for site, site_stats in stats.items()
    }
    r_hat = np.concatenate([site_r_hat for site_r_hat, _ in per_site.values()])
    n_eff = np.concatenate([site_n_eff for _, site_n_eff in per_site.values()])
    groups = {
        name: _group_extremes(per_site, group) for name, group in _DIAG_GROUPS.items()
    }

    r_hat_max = float(np.max(r_hat))
    n_eff_min = float(np.min(n_eff))
    divergences = int(np.asarray(extra["diverging"]).sum())

    failures = []
    if not r_hat_max <= thresholds.r_hat_max:
        failures.append(f"r_hat_max {r_hat_max:.3f} > {thresholds.r_hat_max}")
    if not n_eff_min >= thresholds.n_eff_min:
        failures.append(f"n_eff_min {n_eff_min:.1f} < {thresholds.n_eff_min}")
    if not divergences <= thresholds.max_divergences:
        failures.append(f"divergences {divergences} > {thresholds.max_divergences}")

    return Diagnostics(
        r_hat_max=r_hat_max,
        r_hat_median=float(np.median(r_hat)),
        frac_r_hat_below_1_05=float(np.mean(r_hat < 1.05)),
        n_eff_min=n_eff_min,
        divergences=divergences,
        num_steps_mean=float(np.asarray(extra["num_steps"]).mean()),
        r_hat_max_native=groups["native"][0],
        n_eff_min_native=groups["native"][1],
        r_hat_max_ell=groups["ell"][0],
        n_eff_min_ell=groups["ell"][1],
        r_hat_max_global=groups["global"][0],
        n_eff_min_global=groups["global"][1],
        wall_s=wall_s,
        passed=not failures,
        reason="; ".join(failures),
    )
