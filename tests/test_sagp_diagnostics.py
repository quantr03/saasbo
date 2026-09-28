"""Tests for sagp.diagnostics: dataclass fields, per-group splits and the sampled-site tuples.

Exercises the `Diagnostics` verdict `fit` returns: its fields are finite and well-typed, the
three per-group extremes (`native`, `ell`, `global`) partition the pooled `r_hat_max`/`n_eff_min`
exactly, and the three site tuples are exactly the union of every cell's sampled sites in either
R2-D2 form, split by whether the site's support is positive, the unit interval or the real line.
"""
import numpy as np
import numpyro.distributions as dist
import pytest
import torch
from numpyro import handlers
from numpyro.diagnostics import summary

from sagp.diagnostics import (
    _DIAG_GROUPS,
    _POSITIVE_SAMPLED_SITES,
    _REAL_SAMPLED_SITES,
    _UNIT_SAMPLED_SITES,
    DiagThresholds,
    Diagnostics,
    diagnose,
)
from sagp.gp import CELLS, NUTSConfig, CellGP, fit


def _data(n: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A design in [0,1]^D and standardized targets, in the float64 numpy the loop hands `fit`."""
    rng = np.random.default_rng(seed)
    return rng.random((n, D)), rng.standard_normal(n)


def test_diagnostics_fields():
    X, y = _data(n=15, D=3, seed=2)
    # The one place `cell` is given as a `Cell` rather than a `CellKey`; `fit` normalizes it back.
    fitted = fit(X, y, 0, CELLS[("additive", "amplitude")], nuts=NUTSConfig(32, 32, 4))
    assert fitted.cell == ("additive", "amplitude")
    diag = fitted.attempts[0]

    assert np.isfinite(diag.r_hat_max)
    assert np.isfinite(diag.r_hat_median)
    assert 0.0 <= diag.frac_r_hat_below_1_05 <= 1.0
    assert diag.n_eff_min > 0.0
    assert isinstance(diag.divergences, int)
    assert diag.divergences >= 0
    assert diag.num_steps_mean > 0.0
    assert diag.wall_s > 0.0
    assert diag.passed == (diag.reason == "")
    assert isinstance(diag, Diagnostics)


@pytest.mark.parametrize(
    "cell_key",
    [
        ("additive", "amplitude"), ("additive", "lengthscale"),
        ("additive", "amplitude_r2d2"), ("additive", "lengthscale_r2d2"),
    ],
    ids=[
        "additive/amplitude", "additive/lengthscale",
        "additive/amplitude_r2d2", "additive/lengthscale_r2d2",
    ],
)
def test_per_group_diagnostics_split_the_pooled_ones(cell_key):
    """Ruling R42: the gate is unchanged, but a fit that fails it has to say *which* block failed.

    An amplitude cell pools 2D + 3 statistics against a lengthscale cell's D + 4, the extra D
    being `kernel_ell` -- so whether the two cells' exclusion rates are comparable at all turns on
    whether `kernel_ell` is what fires. The `_ell` fields are therefore finite exactly where that
    site exists, and the three groups partition the pooled statistics rather than recomputing
    them: their extremes have to reproduce the pooled ones exactly, not approximately. The R2-D2
    cells pool the same counts, their D copula coordinates standing where the D local scales do.
    """
    X, y = _data(n=15, D=3, seed=6)
    fitted = fit(X, y, 0, cell_key, nuts=NUTSConfig(32, 32, 4))
    diag = fitted.attempts[0]

    has_ell = CELLS[cell_key].native_site == "a_sq"
    assert np.isfinite(diag.r_hat_max_ell) == has_ell
    assert np.isfinite(diag.n_eff_min_ell) == has_ell
    # `native` and `global` exist in every cell, whichever site carries the sparsity.
    for value in (diag.r_hat_max_native, diag.n_eff_min_native, diag.r_hat_max_global,
                  diag.n_eff_min_global):
        assert np.isfinite(value)

    groups_r_hat = [diag.r_hat_max_native, diag.r_hat_max_ell, diag.r_hat_max_global]
    groups_n_eff = [diag.n_eff_min_native, diag.n_eff_min_ell, diag.n_eff_min_global]
    assert diag.r_hat_max == float(np.nanmax(groups_r_hat))
    assert diag.n_eff_min == float(np.nanmin(groups_n_eff))


def test_sampled_site_tuples_are_exactly_the_cells_sampled_sites():
    """What `diagnose` reads, and on which scale, pinned against the cells' own traces.

    The tuples are hand-written lists of site names, and every way of getting them wrong is
    silent: a site left out is a coordinate the gate never looks at; a *deterministic* site put in
    (`a_sq` and `kernel_inv_length_sq` are one transform away from sites that are listed) would
    have the gate diagnose a function of the draws rather than the geometry NUTS moves in; a
    real-valued site listed as positive would be logged, which is NaN half the time; and a
    unit-interval site read on any scale but the logit is not the coordinate NUTS moves. The R2-D2
    cells are traced in both forms, since the tuples list both forms' sites and a switch of form
    must never touch them.
    """
    n, D = 8, 3
    X, y = _data(n, D, seed=7)
    Xt = torch.as_tensor(X)
    zt = torch.as_tensor((y - y.mean()) / y.std())[:, None]
    positive_union, unit_union, real_union = set(), set(), set()
    half_cauchy_positive, half_cauchy_real = set(), set()

    for cell in CELLS.values():
        is_r2d2 = hasattr(cell.pyro_model, "r2d2_form")
        for form in ("reference", "tied") if is_r2d2 else (None,):
            gp = CellGP(Xt, zt, cell=cell)
            if form is not None:
                gp.pyro_model.r2d2_form = form
                if form == "tied":
                    gp.pyro_model.r2d2_a = None  # the tied form fixes a = k * dim
            traced = handlers.trace(handlers.seed(gp.pyro_model.sample, rng_seed=0)).get_trace()
            sampled = {
                name: site
                for name, site in traced.items()
                if site["type"] == "sample" and not site.get("is_observed")
            }
            support = {name: getattr(site["fn"], "support", None) for name, site in sampled.items()}
            positive = {name for name, s in support.items() if s is dist.constraints.positive}
            unit = {name for name, s in support.items() if s is dist.constraints.unit_interval}
            real = set(sampled) - positive - unit

            assert all(support[name] is dist.constraints.real for name in real)
            assert positive <= set(_POSITIVE_SAMPLED_SITES)
            assert unit <= set(_UNIT_SAMPLED_SITES)
            assert real <= set(_REAL_SAMPLED_SITES)
            # And every listed site this cell has is one NUTS samples, not a `deterministic`.
            for name in _POSITIVE_SAMPLED_SITES + _UNIT_SAMPLED_SITES + _REAL_SAMPLED_SITES:
                if name in traced:
                    assert traced[name]["type"] == "sample"
            positive_union |= positive
            unit_union |= unit
            real_union |= real
            if not is_r2d2:
                assert unit == set()
                half_cauchy_positive |= positive
                half_cauchy_real |= real

    # Across the eight cells in both forms, nothing sampled is missing from any tuple.
    assert positive_union == set(_POSITIVE_SAMPLED_SITES)
    assert unit_union == set(_UNIT_SAMPLED_SITES)
    assert real_union == set(_REAL_SAMPLED_SITES)
    # And the four half-Cauchy cells alone read exactly what they read before R2-D2 existed.
    assert half_cauchy_positive == set(_POSITIVE_SAMPLED_SITES)
    assert half_cauchy_real == {"mean"}


def test_the_groups_partition_the_three_tuples():
    # The per-group fields are only an attribution of the pooled ones if the groups cover every
    # site the gate reads and overlap in none: `mean` is a global scalar like the outputscale and
    # the noise, so it belongs to `global` and not to a fourth, unreported block. R2-D2's D copula
    # coordinates carry each coordinate's share, as the half-Cauchy local scales do, so they are
    # `native`; its one scalar site in either form is `global`, as `kernel_tausq` is.
    grouped = [site for group in _DIAG_GROUPS.values() for site in group]

    assert sorted(grouped) == sorted(
        _POSITIVE_SAMPLED_SITES + _UNIT_SAMPLED_SITES + _REAL_SAMPLED_SITES
    )
    assert len(grouped) == len(set(grouped))
    assert "mean" in _DIAG_GROUPS["global"]
    assert "r2d2_z_lam" in _DIAG_GROUPS["native"]
    assert {"r2d2_R2", "r2d2_z_xi"} <= set(_DIAG_GROUPS["global"])


def test_the_unit_interval_site_is_read_on_the_logit_scale():
    # NUTS moves a Beta site on logit(R2) = log(R2) - log1p(-R2), NumPyro's unconstrained
    # coordinate for it, so that is where the gate takes its R-hat and ESS -- as it takes the
    # positive sites' on their logs. A chain that wanders over most of (0, 1) tells the scales
    # apart: on R2 itself the squashed tails would give other numbers.
    logit = 2.5 * np.sin(np.linspace(0.0, 9.0, 128)) + np.random.default_rng(8).normal(size=128)
    r2 = 1.0 / (1.0 + np.exp(-logit))
    extra = {"diverging": np.zeros(128, dtype=bool), "num_steps": np.full(128, 7)}

    diag = diagnose({"r2d2_R2": r2}, extra, DiagThresholds(), wall_s=1.0)

    on_logit = summary({"x": np.log(r2) - np.log1p(-r2)}, prob=0.9, group_by_chain=False)["x"]
    on_r2 = summary({"x": r2}, prob=0.9, group_by_chain=False)["x"]
    assert diag.r_hat_max == pytest.approx(float(on_logit["r_hat"]), rel=1e-12)
    assert diag.n_eff_min == pytest.approx(float(on_logit["n_eff"]), rel=1e-12)
    assert diag.n_eff_min != pytest.approx(float(on_r2["n_eff"]), rel=1e-3)
    # One scalar site, and it is `global`.
    assert (diag.r_hat_max_global, diag.n_eff_min_global) == (diag.r_hat_max, diag.n_eff_min)
