"""Tests for synthobj.draws: GP, sinusoid, and monotone-rejection 1-D draws.

The sinusoid ratio/slope-share tests below implement controller ruling R11 (see
`.superpowers/sdd/2026-09-06-synthetic-objectives/task-2-brief.md`), which replaces the plan's
original "sinusoid slope-energy/variance ratio within factor 1.5 of the Matern slope factor at
ell in {0.08, 0.5, 3}" assertion — that assertion is false at ell=0.5 and ell=3, because the
closed form omega_for_ell is derived assuming the sinusoid completes at least one cycle on
[0,1], which only holds for ell <~ 0.2-0.3.
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from synthobj.draws import draw_until_monotone, gp_draw, omega_for_ell, sinusoid_draw
from synthobj.kernel import (
    GL_NODES,
    GL_WEIGHTS,
    eigen_factor,
    joint_points,
    make_grid,
    nu_var,
    v,
)


# --- omega_for_ell -----------------------------------------------------------

OMEGA_REFERENCE = {
    0.06: 3.68395,
    0.08: 2.83284,
    0.10: 2.32426,
    0.20: 1.32304,
    0.50: 0.77359,
    1.50: 0.59128,
    3.00: 0.56335,
}


@pytest.mark.parametrize("ell,expected", sorted(OMEGA_REFERENCE.items()))
def test_omega_for_ell_matches_reference_table(ell, expected):
    assert omega_for_ell(ell) == pytest.approx(expected, rel=1e-4)


def test_omega_for_ell_decreasing_and_bounded_below_limit():
    ells = [0.06, 0.08, 0.1, 0.2, 0.5, 1.5, 3.0, 30.0, 300.0]
    omegas = [omega_for_ell(ell) for ell in ells]
    assert all(a > b for a, b in zip(omegas, omegas[1:])), "omega_for_ell must decrease in ell"
    assert all(o > np.sqrt(12.0) / (2.0 * np.pi) for o in omegas)


# --- gp_draw -------------------------------------------------------------------

def test_gp_draw_shape():
    F = eigen_factor(0.0, 1.0, 50, 0.5)
    values = gp_draw(F, np.random.default_rng(0))
    assert values.shape == (F.shape[0],) == (50 + 64,)


def test_gp_draw_mean_quadrature_variance_is_one_at_ell_half():
    # E[Var_nu f] = 1 exactly for a draw from the normalized centered kernel: reuse a single
    # cached eigen_factor and vectorize 2000 draws as one matmul, per the task's performance
    # guidance.
    F = eigen_factor(0.0, 1.0, 100, 0.5)
    rng = np.random.default_rng(0)
    n_draws = 2000
    values = F @ rng.standard_normal((F.shape[1], n_draws))
    node_vals = values[-64:, :]
    mean = (GL_WEIGHTS[:, None] * node_vals).sum(axis=0)
    var = (GL_WEIGHTS[:, None] * node_vals**2).sum(axis=0) - mean**2
    assert var.mean() == pytest.approx(1.0, rel=0.05)


# --- sinusoid_draw: R11 test 1, ratio test restricted to the derived regime --------

def _sinusoid_slope_variance_ratio(ell, rng, n_terms=3, h=1e-6):
    """Realized slope-energy-to-variance ratio of one sinusoid draw at GL_NODES.

    Slope energy is estimated by a central finite difference of `sinusoid_draw` itself
    (evaluated at GL_NODES +/- h with an independently-seeded rng holding identical draws),
    so this exercises the actual generator rather than a re-derived closed form.
    """
    seed = int(rng.integers(0, 2**32 - 1))
    values = sinusoid_draw(GL_NODES, ell, np.random.default_rng(seed), n_terms)
    plus = sinusoid_draw(GL_NODES + h, ell, np.random.default_rng(seed), n_terms)
    minus = sinusoid_draw(GL_NODES - h, ell, np.random.default_rng(seed), n_terms)
    deriv = (plus - minus) / (2.0 * h)
    slope_energy = float((GL_WEIGHTS * deriv**2).sum())
    variance = nu_var(values)
    return slope_energy / variance


@pytest.mark.parametrize("ell", [0.06, 0.08, 0.2])
def test_sinusoid_ratio_matches_slope_factor_in_derived_regime(ell):
    # R11 test 1: restricted to ell where omega_for_ell(ell) >= 1.32, i.e. the sinusoid
    # completes at least one cycle on [0,1] and the closed form's derivation actually applies.
    rng = np.random.default_rng(20260906)
    ratios = [_sinusoid_slope_variance_ratio(ell, rng) for _ in range(200)]
    target = 5.0 / (3.0 * ell**2 * v(ell))
    realized = float(np.mean(ratios))
    assert target / 1.2 <= realized <= target * 1.2


# --- sinusoid_draw: R11 test 2, slope-share test at a common lengthscale ----------

_SHARE_GRID = make_grid(0.0, 1.0, 200)
_SHARE_Z = joint_points(_SHARE_GRID)
_SHARE_SORT_IDX = np.argsort(_SHARE_Z)
_SHARE_Z_SORTED = _SHARE_Z[_SHARE_SORT_IDX]


def _mean_slope_energy(raw_draws, share):
    """Center each column of `raw_draws` by its quadrature mean, rescale to an exact
    quadrature-variance `share`, fit a cubic spline over grid+nodes (sorted), and return the
    mean quadrature slope energy Sum_q w_q * spline'(t_q)**2 across columns.

    This mirrors the centering/rescaling the next task's `Component` will do (raw draw -> zero
    quadrature mean -> exact variance share) and the spline-derivative slope-energy definition
    planned for `Component.slope_energy`, without depending on either (neither exists yet).
    """
    node_vals = raw_draws[-64:, :]
    mean = (GL_WEIGHTS[:, None] * node_vals).sum(axis=0)
    var = (GL_WEIGHTS[:, None] * node_vals**2).sum(axis=0) - mean**2
    scale = np.sqrt(share / var)
    scaled = (raw_draws - mean[None, :]) * scale[None, :]
    spline = CubicSpline(_SHARE_Z_SORTED, scaled[_SHARE_SORT_IDX, :], axis=0)
    deriv_at_nodes = spline.derivative()(GL_NODES)
    slope_energy = (GL_WEIGHTS[:, None] * deriv_at_nodes**2).sum(axis=0)
    return float(slope_energy.mean())


def test_sinusoid_and_matern_slope_shares_agree_at_common_lengthscale():
    # R11 test 2: the property the spec actually asks for. At one shared lengthscale, the
    # per-lengthscale bias (R11, measured 1.25-1.29x at ell=0.5) is a constant factor that
    # cancels in the *normalized* slope-share vector, so it should agree closely with the
    # Matern generator's even though the absolute slope energies do not.
    ell = 0.5
    shares = (0.4, 0.3, 0.2, 0.1)
    n_draws = 100

    F = eigen_factor(0.0, 1.0, 200, ell)
    rng_matern = np.random.default_rng(7)
    matern_energy = np.array(
        [
            _mean_slope_energy(F @ rng_matern.standard_normal((F.shape[1], n_draws)), share)
            for share in shares
        ]
    )

    rng_sin = np.random.default_rng(7)
    sinusoid_energy = np.array(
        [
            _mean_slope_energy(
                np.column_stack(
                    [sinusoid_draw(_SHARE_Z, ell, rng_sin, 3) for _ in range(n_draws)]
                ),
                share,
            )
            for share in shares
        ]
    )

    g_matern = matern_energy / matern_energy.sum()
    g_sinusoid = sinusoid_energy / sinusoid_energy.sum()
    assert np.max(np.abs(g_matern - g_sinusoid)) < 0.05


# --- draw_until_monotone -----------------------------------------------------

def test_draw_until_monotone_returns_monotone_values_at_ell_3():
    n_grid = 1024
    F = eigen_factor(0.0, 1.0, n_grid, 3.0)
    rng = np.random.default_rng(0)
    values = draw_until_monotone(lambda r: gp_draw(F, r), n_grid, rng)
    diffs = np.diff(values[:n_grid])
    assert np.all(diffs >= 0) or np.all(diffs <= 0)


def test_draw_until_monotone_raises_runtime_error_when_never_monotone():
    x = make_grid(0.0, 1.0, 100)

    def non_monotone_draw(rng):
        return np.sin(20.0 * np.pi * x)

    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        draw_until_monotone(non_monotone_draw, 100, rng, max_tries=5)
