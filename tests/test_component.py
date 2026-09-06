"""Tests for synthobj.component: exact-variance-share spline components with an exact 1-D argmax.

Three kinds of assertion live here.

*Analytic* tests build a component from a closed-form `raw_joint_values` instead of a random
draw, so every asserted number is exact arithmetic: the linear component sqrt(12)(x - 1/2) has
slope energy 12, the cubic x^3 has slope energy 9/5, and sin(2 pi x) has its maximum at exactly
x = 1/4. These carry the power against a wrong quadrature, a wrong derivative and a grid-search
argmax.

*Exactness* tests check the property the thesis depends on -- `nu_var()` returns the requested
share and `nu_mean()` returns 0 -- across the study's lengthscales and shares.

*Accuracy* tests measure spline interpolation error and slope-energy grid-convergence against
the tolerances measured before this task was dispatched and recorded in
`.superpowers/sdd/2026-09-06-synthetic-objectives/task-3-brief.md` (worst of 6 draws:
ell=0.08 max 1.5e-5 / rms 4.1e-6, ell=0.06 max 5.8e-5 / rms 1.5e-5; slope-energy relative
difference between the production 1024 resolution and a 2048-point reference 4e-7 to 8e-5).
They are marked `slow` -- their `eigh` calls dominate this file's runtime.
"""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from synthobj.component import Component
from synthobj.draws import gp_draw
from synthobj.kernel import (
    GL_NODES,
    eigen_factor,
    joint_points,
    make_grid,
    normalized,
    nu_mean,
    nu_var,
)

# Small enough that eigh is free; the exactness properties do not depend on resolution.
N_SMALL = 256
# The production grid (plan Global Constraints): 1024 points, spacing 1/1023.
N_PROD = 1024


def _gp_component(
    ell: float,
    share: float,
    n: int,
    seed: int,
    coord: int = 0,
    sign: float = 1.0,
) -> Component:
    """A component from one GP draw on `make_grid(0, 1, n)`; `eigen_factor` is cached by (n, ell)."""
    raw = gp_draw(eigen_factor(0.0, 1.0, n, ell), np.random.default_rng(seed))
    return Component.from_raw(coord, ell, share, make_grid(0.0, 1.0, n), raw, sign)


def _analytic_component(grid: np.ndarray, share: float, f) -> Component:
    """A component whose raw joint values are `f` evaluated at `joint_points(grid)`."""
    return Component.from_raw(0, np.inf, share, grid, f(joint_points(grid)))


def _psd_factor(Z: np.ndarray, ell: float) -> np.ndarray:
    """PSD square-root factor of `normalized(Z, Z, ell)`, same rule as `kernel.eigen_factor`.

    `eigen_factor` only accepts point sets of the form grid-union-GL-nodes, so the fine-patch
    and grid-refinement designs below cannot use it. Eigenvalues below 1e-12 * lambda_max are
    clipped to zero; no jitter is added to the covariance (plan decision D3).
    """
    eigvals, eigvecs = np.linalg.eigh(normalized(Z, Z, ell))
    eigvals = np.where(eigvals < 1e-12 * eigvals.max(), 0.0, eigvals)
    return eigvecs * np.sqrt(eigvals)


# --- exact variance shares -----------------------------------------------------------------

ELLS = [0.08, 0.5, 1.5, 3.0]
SHARES = [0.02, 0.25, 1.0]


@pytest.mark.parametrize("share", SHARES)
@pytest.mark.parametrize("ell", ELLS)
def test_centered_to_zero_mean_and_rescaled_to_exact_share(ell: float, share: float) -> None:
    comp = _gp_component(ell, share, N_SMALL, seed=0)
    assert comp.nu_mean() == pytest.approx(0.0, abs=1e-12)
    assert comp.nu_var() == pytest.approx(share, abs=1e-8)
    assert comp.ell == ell
    assert comp.share == share


def test_sign_flips_values_and_preserves_the_variance_share() -> None:
    ell, share, coord = 0.5, 0.25, 3
    grid = make_grid(0.0, 1.0, N_SMALL)
    raw = gp_draw(eigen_factor(0.0, 1.0, N_SMALL, ell), np.random.default_rng(5))
    up = Component.from_raw(coord, ell, share, grid, raw)
    down = Component.from_raw(coord, ell, share, grid, raw, sign=-1.0)

    # The rescale is a single multiply by sign * sqrt(share / var), and (-a) * b == -(a * b)
    # exactly in IEEE arithmetic, so the mirrored values are bit-for-bit negations.
    assert np.array_equal(down.values, -up.values)
    assert down.nu_mean() == pytest.approx(0.0, abs=1e-12)
    assert down.nu_var() == pytest.approx(share, abs=1e-8)
    assert down.slope_energy() == pytest.approx(up.slope_energy(), rel=1e-12)

    x = np.linspace(0.0, 1.0, 1000)
    assert np.allclose(down(x), -up(x), rtol=0.0, atol=1e-14)

    # Rescaling is a positive affine map of the raw draw, so the stored shape is the raw shape
    # (correlation +1) and sign=-1 mirrors it about zero (correlation -1).
    raw_on_grid = raw[:N_SMALL]
    assert np.corrcoef(up(grid), raw_on_grid)[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert np.corrcoef(down(grid), raw_on_grid)[0, 1] == pytest.approx(-1.0, abs=1e-12)


# --- analytic components: quadrature, derivative and argmax against closed forms -------------


def test_linear_component_has_slope_energy_twelve() -> None:
    # sqrt(12) (x - 1/2) is the ell -> infinity limit of the normalized kernel: Var_nu = 1 and
    # E_nu[(f')^2] = 12 exactly, so this pins the normalization convention the thesis intends.
    grid = make_grid(0.0, 1.0, N_PROD)
    comp = _analytic_component(grid, 1.0, lambda z: np.sqrt(12.0) * (z - 0.5))

    assert comp.nu_mean() == pytest.approx(0.0, abs=1e-12)
    assert comp.nu_var() == pytest.approx(1.0, abs=1e-10)
    assert comp.slope_energy() == pytest.approx(12.0, abs=1e-6)

    x = np.linspace(0.0, 1.0, 997)
    assert np.max(np.abs(comp(x) - np.sqrt(12.0) * (x - 0.5))) < 1e-12

    # A monotone increasing component attains its maximum at the right endpoint.
    x_star, f_star = comp.argmax()
    assert x_star == pytest.approx(1.0, abs=1e-12)
    assert f_star == pytest.approx(np.sqrt(3.0), abs=1e-12)


def test_slope_energy_of_an_exact_cubic_matches_its_closed_form() -> None:
    # Not-a-knot cubic interpolation reproduces a cubic exactly and 64-node Gauss-Legendre is
    # exact to degree 127, so every number below is analytic. x^3 has E_nu = 1/4 and
    # Var_nu = 1/7 - 1/16 = 9/112, so building with share = 9/112 makes the rescale factor 1:
    # f = x^3 - 1/4, f' = 3x^2, E_nu[(f')^2] = int_0^1 9 x^4 dx = 9/5. Unlike the linear
    # component the integrand 9x^4 is not constant, so this also pins the quadrature weights,
    # which any weights summing to 1 would satisfy for a constant integrand.
    grid = make_grid(0.0, 1.0, 512)
    comp = _analytic_component(grid, 9.0 / 112.0, lambda z: z**3)

    assert comp.nu_mean() == pytest.approx(0.0, abs=1e-12)
    assert comp.nu_var() == pytest.approx(9.0 / 112.0, rel=1e-12)
    assert comp.slope_energy() == pytest.approx(9.0 / 5.0, rel=1e-8)

    x = np.linspace(0.0, 1.0, 997)
    assert np.max(np.abs(comp(x) - (x**3 - 0.25))) < 1e-12


def test_argmax_finds_an_interior_maximum_off_the_knot_grid() -> None:
    # sin(2 pi x) has E_nu = 0 and Var_nu = 1/2, so share = 1/2 leaves it unscaled; its maximum
    # is 1 at x = 1/4, which is not a knot of make_grid(0, 1, 1024) (0.25 * 1023 = 255.75).
    # A grid search over the 1088 knots would be off in x by up to half a spacing, 4.9e-4,
    # which is 490x the tolerance asserted here -- this is what "exact" buys.
    grid = make_grid(0.0, 1.0, N_PROD)
    comp = _analytic_component(grid, 0.5, lambda z: np.sin(2.0 * np.pi * z))

    x_star, f_star = comp.argmax()
    assert x_star == pytest.approx(0.25, abs=1e-6)
    assert f_star == pytest.approx(1.0, abs=1e-9)
    assert comp.derivative(x_star) == pytest.approx(0.0, abs=1e-6)


# --- argmax on real draws -------------------------------------------------------------------


@pytest.mark.parametrize("ell", [0.08, 0.5])
def test_argmax_dominates_dense_and_random_sampling(ell: float) -> None:
    comp = _gp_component(ell, 1.0, N_PROD, seed=11)
    x_star, f_star = comp.argmax()

    assert 0.0 <= x_star <= 1.0
    assert f_star >= comp(np.linspace(0.0, 1.0, 100_000)).max() - 1e-12
    assert f_star >= comp(np.random.default_rng(3).uniform(0.0, 1.0, 1_000_000)).max() - 1e-12
    assert comp(x_star) == pytest.approx(f_star, abs=1e-12)

    # An interior maximizer of the exact answer is a true critical point of the spline; a grid
    # search would leave a derivative of order |f''| * h / 2, which is O(0.1) here.
    assert x_star in (0.0, 1.0) or abs(comp.derivative(x_star)) < 1e-8


def test_argmax_respects_lo_and_hi() -> None:
    comp = _gp_component(0.08, 1.0, N_PROD, seed=11)
    lo, hi = 0.2, 0.6
    x_star, f_star = comp.argmax(lo, hi)

    assert lo <= x_star <= hi
    assert f_star >= comp(np.linspace(lo, hi, 100_000)).max() - 1e-12
    assert comp(x_star) == pytest.approx(f_star, abs=1e-12)
    assert f_star <= comp.argmax()[1] + 1e-12


# --- knot vector and reconstruction ----------------------------------------------------------


def test_knots_are_the_merged_sorted_deduplicated_union() -> None:
    grid = make_grid(0.0, 1.0, N_PROD)
    comp = _analytic_component(grid, 1.0, lambda z: np.sin(3.0 * z) + 0.5 * z)

    assert comp.grid.shape == (N_PROD + 64,)
    assert np.all(np.diff(comp.grid) > 0.0)
    assert np.array_equal(comp.grid, np.sort(np.concatenate([grid, GL_NODES])))
    assert comp.values == pytest.approx(comp(comp.grid), abs=1e-12)


def test_knots_within_the_dedupe_tolerance_are_merged() -> None:
    # A uniform grid point that coincides with a GL node makes joint_points repeat a knot, and
    # CubicSpline rejects a non-strictly-increasing x outright, so these cases fail loudly
    # without the dedupe step.
    base = make_grid(0.0, 1.0, 49)
    shape = (lambda z: np.sin(3.0 * z) + 0.5 * z)

    exact_dup = np.sort(np.concatenate([base, GL_NODES[[10]]]))
    assert _analytic_component(exact_dup, 1.0, shape).grid.shape == (113,)

    near_dup = np.sort(np.concatenate([base, [GL_NODES[10] + 5e-13]]))
    assert _analytic_component(near_dup, 1.0, shape).grid.shape == (113,)

    # 1e-6 apart is a genuinely distinct knot and must survive.
    distinct = np.sort(np.concatenate([base, [GL_NODES[10] + 1e-6]]))
    assert _analytic_component(distinct, 1.0, shape).grid.shape == (114,)


def test_stored_grid_and_values_rebuild_the_component_without_a_draw() -> None:
    comp = _gp_component(0.5, 0.3, N_SMALL, seed=17, coord=4)
    rebuilt = Component(comp.coord, comp.ell, comp.share, comp.grid, comp.values)

    x = np.linspace(0.0, 1.0, 1000)
    assert np.array_equal(rebuilt(x), comp(x))
    assert rebuilt.nu_var() == comp.nu_var()
    assert rebuilt.slope_energy() == comp.slope_energy()


def test_component_uses_identity_equality_and_is_frozen() -> None:
    # Ruling R4: eq=True would compare the ndarray fields as a tuple and raise
    # "truth value of an array with more than one element is ambiguous".
    a = _gp_component(0.5, 0.3, N_SMALL, seed=17)
    b = _gp_component(0.5, 0.3, N_SMALL, seed=17)
    assert a == a
    assert a != b
    with pytest.raises(FrozenInstanceError):
        a.share = 0.5


# --- extended domain --------------------------------------------------------------------------


def test_extended_domain_component_evaluates_outside_the_unit_interval() -> None:
    # Rotated components live on a wider box (plan decision D2) but are still centered and
    # rescaled under U[0,1]: nu_mean, nu_var and slope_energy read the GL nodes only, so an
    # implementation that normalized against the whole extended grid would fail here.
    ell, share = 0.5, 0.4
    raw = gp_draw(eigen_factor(-0.25, 1.25, N_SMALL, ell), np.random.default_rng(7))
    comp = Component.from_raw(0, ell, share, make_grid(-0.25, 1.25, N_SMALL), raw)

    outside = np.array([-0.2, 1.2])
    assert np.all(np.isfinite(comp(outside)))
    assert np.all(np.isfinite(comp.derivative(outside)))
    assert comp.nu_mean() == pytest.approx(0.0, abs=1e-12)
    assert comp.nu_var() == pytest.approx(share, abs=1e-8)
    assert comp.slope_energy() > 0.0


def test_unit_domain_component_is_nan_outside_its_knot_range() -> None:
    comp = _gp_component(0.5, 1.0, N_SMALL, seed=17)
    assert np.all(np.isnan(comp(np.array([-0.2, 1.2]))))


# --- spline accuracy (measured tolerances, see the module docstring) --------------------------

FINE_PATCH = np.linspace(0.30, 0.40, 400)


@pytest.mark.slow
@pytest.mark.parametrize("ell,max_tol,rms_tol", [(0.08, 2e-4, 3e-5), (0.06, 5e-4, 1e-4)])
def test_spline_error_rough(ell: float, max_tol: float, rms_tol: float) -> None:
    """Interpolation error alone: the fine patch comes from the same draw, not a second one.

    One joint draw on grid(1024) union GL nodes union 400 evenly spaced points in [0.30, 0.40];
    the component is built from the grid-union-nodes part only and evaluated on the patch. The
    reference is the raw patch put through the same centering and rescaling `from_raw` applies,
    so the residual is purely the spline's.
    """
    n_draws = 6
    grid = make_grid(0.0, 1.0, N_PROD)
    Z = np.concatenate([joint_points(grid), FINE_PATCH])
    draws = _psd_factor(Z, ell) @ np.random.default_rng(20260906).standard_normal((Z.size, n_draws))

    worst_max = worst_rms = 0.0
    for k in range(n_draws):
        knot_raw, patch_raw = draws[: N_PROD + 64, k], draws[N_PROD + 64 :, k]
        comp = Component.from_raw(0, ell, 1.0, grid, knot_raw)
        node_raw = knot_raw[-64:]
        reference = (patch_raw - nu_mean(node_raw)) / np.sqrt(nu_var(node_raw))
        err = np.abs(comp(FINE_PATCH) - reference)
        worst_max = max(worst_max, float(err.max()))
        worst_rms = max(worst_rms, float(np.sqrt((err**2).mean())))

    assert worst_max < max_tol
    assert worst_rms < rms_tol


@pytest.mark.slow
@pytest.mark.parametrize("ell", [0.06, 0.08])
def test_slope_energy_converges_under_grid_refinement(ell: float) -> None:
    """The recorded g_i label must not depend on the storage resolution.

    Both components come from one draw on a 2047-point grid: the coarse one keeps every second
    point, which is exactly the production `make_grid(0, 1, 1024)`, spacing 1/1023.
    """
    n_fine, n_draws = 2047, 4
    fine_grid = make_grid(0.0, 1.0, n_fine)
    Z = joint_points(fine_grid)
    draws = _psd_factor(Z, ell) @ np.random.default_rng(4242).standard_normal((Z.size, n_draws))

    for k in range(n_draws):
        raw = draws[:, k]
        fine = Component.from_raw(0, ell, 1.0, fine_grid, raw)
        coarse = Component.from_raw(
            0, ell, 1.0, fine_grid[::2], np.concatenate([raw[:n_fine][::2], raw[n_fine:]])
        )
        rel = abs(coarse.slope_energy() - fine.slope_energy()) / fine.slope_energy()
        assert rel < 1e-3
