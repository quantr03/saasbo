"""Tests for experiments.identify, the study's offline identification runner (SQ1).

`identify` is one Sobol design, one fit, one `readouts` call and one `manipulation_checks` call,
flattened into a single record with no acquisition loop involved. What is pinned here is therefore
not any one readout's correctness -- that is `test_sagp_readouts.py`'s job -- but that the record
itself is complete (every key the brief lists, at the right shape), that the whole pipeline is
reproducible from `(objective, cell, n, seed)` alone, and that neither `sagp.gp` nor
`experiments.identify` pulls in `sagp.bo`.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp as gp
from experiments.identify import identify
from sagp.gp import NUTSConfig

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synthobj.families import make_family

# A budget small enough for a non-slow test (a few seconds per fit) while still exercising a real
# NUTS chain rather than stubbed diagnostics, per the task's global test constraints.
_FAST_NUTS = NUTSConfig(32, 32, 4)

# The retained sites of each cell the record test runs, spelled out rather than read off `CELLS`:
# BoTorch's own names for the half-Cauchy cell's, and for its R2-D2 twin the sampled sites of the
# prior's form in place of `kernel_tausq` and `_a_sq`.
_R2D2_PRIOR_SITES = {"reference": ("r2d2_R2", "r2d2_z_lam"), "tied": ("r2d2_z_xi", "r2d2_z_lam")}
_SAMPLES = {
    "additive/amplitude": {"mean", "noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"},
    "additive/amplitude_r2d2": {
        "mean", "noise", *_R2D2_PRIOR_SITES[gp.R2D2_FORM], "a_sq", "kernel_ell",
    },
}


def _identify_aligned3(cell=("additive", "amplitude")):
    """`identify` on the fixed design the brief pins its tests to: aligned3 at D=6, n=25."""
    return identify(make_family("aligned3", 0, D=6), cell, n=25, seed=0, nuts=_FAST_NUTS)


@pytest.mark.parametrize("method", list(_SAMPLES))
def test_identify_returns_every_key_at_the_right_shape(method):
    D = 6
    cell = tuple(method.split("/"))
    out = _identify_aligned3(cell)

    expected_keys = {
        "family", "seed", "cell", "n", "D", "alpha", "status", "status_reason", "nuts_attempts",
        "diag_r_hat_max", "diag_r_hat_median", "diag_frac_r_hat_below_1_05", "diag_n_eff_min",
        "diag_divergences", "diag_num_steps_mean", "diag_wall_s", "diag_passed", "diag_reason",
        "diag_r_hat_max_native", "diag_n_eff_min_native", "diag_r_hat_max_ell",
        "diag_n_eff_min_ell", "diag_r_hat_max_global", "diag_n_eff_min_global",
        "wall_s", "y_mean", "y_std",
        "native_median", "p_active", "sobol_hat", "total_var_hat", "active_neutral",
        "active_native",
        "labels_s", "labels_g", "labels_active",
        "spearman_native_vs_s", "spearman_native_vs_g", "spearman_sobol_vs_s",
        "spearman_sobol_vs_g", "spearman_native_vs_sobol", "amplitude_vs_realized",
        "samples",
    }
    assert len(expected_keys) == 43
    assert set(out) == expected_keys

    assert out["family"] == "aligned3"
    assert out["seed"] == 0
    assert out["cell"] == method
    assert out["n"] == 25
    assert out["D"] == D
    assert out["alpha"] == gp.CELLS[cell].alpha_default
    assert out["status"] in {"ok", "excluded"}
    assert isinstance(out["status_reason"], str)
    assert out["nuts_attempts"] == 1

    # Every Diagnostics field of the attempt whose draws were retained, prefixed diag_.
    for field in (
        "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
        "num_steps_mean", "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell",
        "n_eff_min_ell", "r_hat_max_global", "n_eff_min_global", "wall_s", "passed", "reason",
    ):
        assert f"diag_{field}" in out
    assert out["diag_wall_s"] > 0.0

    assert out["wall_s"] > 0.0
    assert isinstance(out["y_mean"], float) and isinstance(out["y_std"], float)

    for key in ("native_median", "p_active", "sobol_hat", "active_neutral", "active_native"):
        assert np.asarray(out[key]).shape == (D,)
    assert isinstance(out["total_var_hat"], float)

    for key in ("labels_s", "labels_g", "labels_active"):
        assert np.asarray(out[key]).shape == (D,)

    for key in (
        "spearman_native_vs_s", "spearman_native_vs_g", "spearman_sobol_vs_s",
        "spearman_sobol_vs_g", "spearman_native_vs_sobol",
    ):
        assert isinstance(out[key], float)
    assert out["amplitude_vs_realized"].shape == (3, 2)  # aligned3 has |S| = 3

    # The cell's `sites`, under the model's own names: BoTorch's, and for an R2-D2 cell its
    # prior's `r2d2_*` sites.
    assert set(out["samples"]) == _SAMPLES[method]
    s = out["samples"]["a_sq"].shape[0]
    for draws in out["samples"].values():
        assert isinstance(draws, np.ndarray)
        assert draws.shape[0] == s


def test_identify_is_reproducible():
    # Two independent calls -- fresh objective, fresh Sobol design, fresh NUTS chain -- must fit
    # the same data, since the design and the HMC key are both pure functions of (seed, n).
    first = _identify_aligned3()
    second = _identify_aligned3()

    assert set(first["samples"]) == set(second["samples"])
    for site in first["samples"]:
        assert np.array_equal(first["samples"][site], second["samples"][site])


def test_identify_records_the_alpha_fit_actually_used():
    # `alpha=None` must resolve to the cell's default, exactly as `fit` itself resolves it
    # (`fit`'s own `if alpha is None: alpha = cell.alpha_default`) -- not report `None` verbatim,
    # which `fit_kwargs.get("alpha", default)` would do since the key is present even when its
    # value is `None` (fix-round finding).
    key = ("additive", "amplitude")
    default = identify(
        make_family("aligned3", 0, D=6), key, n=25, seed=0, alpha=None, nuts=_FAST_NUTS
    )
    explicit = identify(
        make_family("aligned3", 0, D=6), key, n=25, seed=0, alpha=0.05, nuts=_FAST_NUTS
    )

    assert default["alpha"] == gp.CELLS[key].alpha_default
    assert explicit["alpha"] == 0.05


def test_identify_does_not_import_sagp_bo(repo_root: Path):
    # `bo.py` imports `standardize` from `sagp.gp`; this is the guard that the reverse never
    # happens -- `gp.py` must stay usable without ever loading `bo.py`, and the study module that
    # builds on it must stay clear of the loop too.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, sagp.gp, experiments.identify; assert 'sagp.bo' not in sys.modules",
        ],
        check=True,
        cwd=repo_root,
        timeout=120,
    )
