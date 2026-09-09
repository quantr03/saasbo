"""Tests for sagp.diagnostics: dataclass fields, per-group splits and the sampled-site tuples.

Exercises the `Diagnostics` verdict `fit` returns: its fields are finite and well-typed, the
three per-group extremes (`native`, `ell`, `global`) partition the pooled `r_hat_max`/`n_eff_min`
exactly, and the two site tuples are exactly the union of every cell's sampled sites, split by
whether the site's support is positive.
"""
import numpy as np
import numpyro.distributions as dist
import pytest
import torch
from numpyro import handlers

from sagp.diagnostics import (
    _DIAG_GROUPS,
    _POSITIVE_SAMPLED_SITES,
    _REAL_SAMPLED_SITES,
    Diagnostics,
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
    [("additive", "amplitude"), ("additive", "lengthscale")],
    ids=["additive/amplitude", "additive/lengthscale"],
)
def test_per_group_diagnostics_split_the_pooled_ones(cell_key):
    """Ruling R42: the gate is unchanged, but a fit that fails it has to say *which* block failed.

    An amplitude cell pools 2D + 3 statistics against a lengthscale cell's D + 4, the extra D
    being `kernel_ell` -- so whether the two cells' exclusion rates are comparable at all turns on
    whether `kernel_ell` is what fires. The `_ell` fields are therefore finite exactly where that
    site exists, and the three groups partition the pooled statistics rather than recomputing
    them: their extremes have to reproduce the pooled ones exactly, not approximately.
    """
    X, y = _data(n=15, D=3, seed=6)
    fitted = fit(X, y, 0, cell_key, nuts=NUTSConfig(32, 32, 4))
    diag = fitted.attempts[0]

    has_ell = cell_key[1] == "amplitude"
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

    Both tuples are hand-written lists of site names, and every way of getting them wrong is
    silent: a site left out is a coordinate the gate never looks at; a *deterministic* site put in
    (`a_sq` and `kernel_inv_length_sq` are one transform away from sites that are listed) would
    have the gate diagnose a function of the draws rather than the geometry NUTS moves in; and a
    real-valued site listed as positive would be logged, which is NaN half the time.
    """
    n, D = 8, 3
    X, y = _data(n, D, seed=7)
    Xt = torch.as_tensor(X)
    zt = torch.as_tensor((y - y.mean()) / y.std())[:, None]
    positive_union, real_union = set(), set()

    for cell in CELLS.values():
        gp = CellGP(Xt, zt, cell=cell)
        traced = handlers.trace(handlers.seed(gp.pyro_model.sample, rng_seed=0)).get_trace()
        sampled = {
            name: site
            for name, site in traced.items()
            if site["type"] == "sample" and not site.get("is_observed")
        }
        positive = {
            name
            for name, site in sampled.items()
            if getattr(site["fn"], "support", None) is dist.constraints.positive
        }
        real = set(sampled) - positive

        assert positive <= set(_POSITIVE_SAMPLED_SITES)
        assert real <= set(_REAL_SAMPLED_SITES)
        # And every listed site this cell has is one NUTS samples, not a `deterministic`.
        for name in _POSITIVE_SAMPLED_SITES + _REAL_SAMPLED_SITES:
            if name in traced:
                assert traced[name]["type"] == "sample"
        positive_union |= positive
        real_union |= real

    # Across the four cells, nothing sampled is missing from either tuple.
    assert positive_union == set(_POSITIVE_SAMPLED_SITES)
    assert real_union == set(_REAL_SAMPLED_SITES) == {"mean"}


def test_the_groups_partition_the_two_tuples():
    # The per-group fields are only an attribution of the pooled ones if the groups cover every
    # site the gate reads and overlap in none: `mean` is a global scalar like the outputscale and
    # the noise, so it belongs to `global` and not to a fourth, unreported block.
    grouped = [site for group in _DIAG_GROUPS.values() for site in group]

    assert sorted(grouped) == sorted(_POSITIVE_SAMPLED_SITES + _REAL_SAMPLED_SITES)
    assert len(grouped) == len(set(grouped))
    assert "mean" in _DIAG_GROUPS["global"]
