"""Identity tests the thesis appendix's derivations cite, plus a currency check on its tables.

Task-2 brief: the first three tests are independent computations of an identity the appendix's
derivations rely on that the existing suite does not already cover -- none of them reads back
stored state or repeats a computation `synthobj` itself performs. The fourth runs
`regen_tables.py --check` as a subprocess and is marked `slow`.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synthobj import kernel
from synthobj.component import Component
from synthobj.draws import gp_draw
from synthobj.families import STUDY_GRID, FamilySpec, make_family
from synthobj.rotation import rotated_block_stats

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_rank_one_collapse_at_large_ell() -> None:
    """At ell=30 the centered Matern-5/2 kernel is close to its large-ell rank-one limit
    (5/(3 ell^2)) (z - 1/2)(z' - 1/2) -- the Design-brief's stated asymptotic for a very smooth
    component -- and v(ell) itself sits close to the same limit's own marginal-variance figure,
    5/(36 ell^2). Tolerances were set from a direct measurement before dispatch (relative
    Frobenius error 9.2e-4 on a 50-point grid; |v(30)/(5/(36*30**2)) - 1| = 5.4e-4), each with
    several-fold margin above the measured value.
    """
    ell = 30.0
    Z = kernel.make_grid(0.0, 1.0, 50)
    K = kernel.centered(Z, Z, ell)
    approx = (5.0 / (3.0 * ell**2)) * np.outer(Z - 0.5, Z - 0.5)
    rel_err = np.linalg.norm(K - approx) / np.linalg.norm(K)
    assert rel_err < 5e-3

    asymptotic = 5.0 / (36.0 * ell**2)
    assert kernel.v(ell) / asymptotic == pytest.approx(1.0, abs=0.02)


def test_raw_draw_slope_energy_matches_kernel_curvature() -> None:
    """The mean, over N=400 independent draws at ell=0.5, of the *unrescaled centered* draw's
    slope energy is within a few standard errors of the theoretical per-unit-variance slope
    energy 5/(3 ell^2 v(ell)).

    This isolates the exact-share rescale (`Component.from_raw`'s `sign * sqrt(share / var)`
    factor) as the source of the realized/theory ratio the appendix's slope-ratio table
    discusses: centering alone (no rescale) already matches theory closely, and the *rescaled*
    ratio is a materially different number -- see `tab_slope_ratio.tex`, whose ratio column
    reports the rescaled figure, not this one.

    Fixed rng (`default_rng(20260907)`, one draw grid of 1024 points -- the production
    resolution). A pre-dispatch measurement at this seed gave |z| = 0.92 (mean 22.82 against
    target 23.63, SE 0.87); the 4-SE bound below has several-fold margin above that, and the
    "use the rescaled draw instead of the raw one" mutant measures |z| = 10.6 at the same seed,
    decisively outside it.
    """
    ell = 0.5
    grid = kernel.make_grid(0.0, 1.0, 1024)
    F = kernel.eigen_factor(0.0, 1.0, 1024, ell)
    rng = np.random.default_rng(20260907)

    n_draws = 400
    energies = np.empty(n_draws)
    for k in range(n_draws):
        raw = gp_draw(F, rng)
        node_values = raw[-kernel.GL_NODES.size:]
        var_raw = kernel.nu_var(node_values)  # the draw's own quadrature variance, pre-rescale
        comp = Component.from_raw(0, ell, var_raw, grid, raw, sign=1.0)  # share=var_raw: no rescale
        energies[k] = comp.slope_energy()

    mean_energy = float(energies.mean())
    se = float(energies.std(ddof=1) / math.sqrt(n_draws))
    target = 5.0 / (3.0 * ell**2 * kernel.v(ell))
    assert abs(mean_energy - target) < 4.0 * se


def test_two_part_rotation_identity() -> None:
    """On `rotated_t45` seed 0 at D=20: `labels.s_axis.sum() + labels.gamma_axis == 1`, and
    `labels.gamma_axis` equals `labels.scale**2` times the summed ANOVA remainder
    `var_phi - var_a - var_b` over the objective's rotated pairs.

    The first half is vacuous by construction: `SyntheticObjective._compute_labels` literally
    defines `gamma_axis = 1.0 - s_axis.sum()` under a rotation, so this assertion checks
    self-consistent arithmetic on that one formula, not an independent property -- it is kept
    only because the brief asks for it explicitly. The second half is the real content: it
    recomputes `gamma_axis` from `rotation.rotated_block_stats`'s own per-pair quadrature (a
    function `_compute_labels` also calls, but this test does its own summation and its own
    `scale**2` multiplication, rather than reading `s_axis` back), so it is an independent check
    that the axis-misspecification remainder really is the rescaled ANOVA remainder the rotation
    leaves behind, not merely an internally-consistent bookkeeping identity.
    """
    obj = make_family("rotated_t45", seed=0, D=20)
    labels = obj.labels
    assert obj.rotation is not None

    # (a) vacuous by construction -- see docstring.
    assert float(labels.s_axis.sum()) + labels.gamma_axis == pytest.approx(1.0, abs=1e-12)

    # (b) the non-vacuous half.
    by_coord = {c.coord: c for c in obj.components}
    remainder_sum = 0.0
    for i, j in obj.rotation.pairs:
        var_phi, var_a, var_b = rotated_block_stats(by_coord[i], by_coord[j], obj.rotation)
        remainder_sum += var_phi - var_a - var_b
    assert labels.gamma_axis == pytest.approx(labels.scale**2 * remainder_sum, abs=1e-10)


@pytest.mark.slow
def test_dense_weak_x_star_is_a_corner_on_the_archived_grid() -> None:
    """`dense_weak`'s recorded maximizer is a cube corner at every seed of the archived grid.

    Pins the measurement A2's `dense_weak` entry states (fix round 1, review Minor 1). Monotone
    grid values do not *prove* an endpoint maximum -- `Component` interpolates with
    `CubicSpline(bc_type="not-a-knot")`, which is not monotonicity-preserving, and
    `draw_until_monotone` constrains only the grid values -- so the appendix states it as a
    property of the archived grid and this test is what makes that statement checkable.

    Both directions must occur, which is the observable consequence of `_draw_component`'s random
    per-coordinate sign: a build that collapsed every component to one direction would put every
    coordinate of `x_star` at the same endpoint and still pass the corner assertion alone.

    Seeds 0-9 and the study's own dimension: `D` is left at `make_family`'s default, which is
    `FamilySpec.D = 100`, the dimension the archived grid is built at and the one A2 quotes as
    `\\SOnumStudyD`. Builds ten dense D=100 objectives, hence `slow`.
    """
    variant = next(name for name, family, _ in STUDY_GRID if family == "dense_weak")
    for seed in range(10):
        obj = make_family(variant, seed=seed)
        assert obj.D == FamilySpec.D
        x_star = obj.labels.x_star
        assert x_star.shape == (obj.D,)
        assert np.all((x_star == 0.0) | (x_star == 1.0)), (
            f"seed {seed}: interior maximizer at coordinates "
            f"{np.flatnonzero((x_star != 0.0) & (x_star != 1.0)).tolist()}"
        )
        assert (x_star == 0.0).any() and (x_star == 1.0).any(), (
            f"seed {seed}: every coordinate at the same endpoint"
        )


@pytest.mark.slow
def test_tables_are_current() -> None:
    """`regen_tables.py --check` exits 0: every committed table under docs/thesis/tables/ still
    matches what the script would generate right now. Takes ~1-2 minutes (it rebuilds the full
    14 x 10 D=100 study grid and re-derives every table), hence `slow`."""
    result = subprocess.run(
        [sys.executable, "regen_tables.py", "--check"],
        cwd=str(REPO_ROOT / "docs" / "thesis"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
