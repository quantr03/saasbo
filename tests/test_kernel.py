"""Tests for synthobj.kernel: centered, normalized Matern-5/2 kernel and cached eigen-factor.

Reference values are Gate 1 of the thesis (see
`Research Context/research direction/Kernel normalization check.py`) and must reproduce that
script's own output.
"""
import numpy as np
import pytest

from synthobj.kernel import (
    GL_NODES,
    GL_WEIGHTS,
    centered,
    eigen_factor,
    joint_points,
    make_grid,
    normalized,
    nu_mean,
    nu_var,
    v,
)


def test_gl_quadrature_nodes_and_weights():
    assert GL_WEIGHTS.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(GL_NODES > 0.0) and np.all(GL_NODES < 1.0)


# v(ell) reference table from the check script, rel tol 1e-3 (task brief). v(30) is checked
# separately below against the large-ell asymptotic limit, since the table only gives it to
# 3 significant figures.
V_REFERENCE = {
    0.05: 0.885740,
    0.06: 0.864091,
    0.08: 0.821989,
    0.1: 0.781486,
    0.2: 0.602956,
    0.5: 0.282184,
    1: 0.106799,
    1.5: 0.053669,
    2: 0.031829,
    3: 0.014781,
    10: 0.001383,
}


@pytest.mark.parametrize("ell,expected", sorted(V_REFERENCE.items()))
def test_v_matches_reference_table(ell, expected):
    assert v(ell) == pytest.approx(expected, rel=1e-3)


def test_v_large_ell_matches_asymptotic_limit():
    ell = 30
    assert v(ell) == pytest.approx(5.0 / (36.0 * ell**2), rel=0.05)


@pytest.mark.parametrize("x", [0.0, 0.3, 1.0])
def test_centered_is_orthogonal_to_reference_measure(x):
    row = centered(np.array([x]), GL_NODES, 0.5)[0]
    assert abs((GL_WEIGHTS * row).sum()) < 1e-12


@pytest.mark.parametrize("ell", [0.08, 0.5, 3])
def test_normalized_average_marginal_variance_is_one(ell):
    diag = np.diag(normalized(GL_NODES, GL_NODES, ell))
    assert (GL_WEIGHTS * diag).sum() == pytest.approx(1.0, abs=1e-10)


GRID = make_grid(0.0, 1.0, 100)
Z = joint_points(GRID)


@pytest.mark.parametrize("ell", [0.08, 0.5, 3])
def test_eigen_factor_reconstructs_normalized_kernel(ell):
    F = eigen_factor(0.0, 1.0, 100, ell)
    K = normalized(Z, Z, ell)
    rel_err = np.linalg.norm(F @ F.T - K) / np.linalg.norm(K)
    assert rel_err < 1e-8


def test_eigen_factor_is_cached_by_identity():
    first = eigen_factor(0.0, 1.0, 100, 0.5)
    second = eigen_factor(0.0, 1.0, 100, 0.5)
    assert first is second


def test_nu_mean_and_nu_var_on_linear_function():
    # 64-node Gauss-Legendre quadrature integrates low-degree polynomials exactly:
    # E[x] = 1/2, Var[x] = 1/12 under U[0,1], with f(x) = x evaluated at the nodes themselves.
    assert nu_mean(GL_NODES) == pytest.approx(0.5, abs=1e-10)
    assert nu_var(GL_NODES) == pytest.approx(1.0 / 12.0, abs=1e-10)


def test_nu_mean_and_nu_var_on_constant():
    ones = np.ones_like(GL_NODES)
    assert nu_mean(ones) == pytest.approx(1.0, abs=1e-12)
    assert nu_var(ones) == pytest.approx(0.0, abs=1e-12)


def test_make_grid_and_joint_points():
    grid = make_grid(0.0, 1.0, 10)
    assert np.array_equal(grid, np.linspace(0.0, 1.0, 10))
    z = joint_points(grid)
    assert z.shape == (10 + 64,)
    assert np.array_equal(z[:10], grid)
    assert np.array_equal(z[10:], GL_NODES)
