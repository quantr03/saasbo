"""Tests for synthobj.interaction: product interactions and the exact maximum of a paired block.

Three kinds of assertion live here.

*Exactness* tests pin the algebra the study's non-additivity budget is written in. Because
`h_ij(x_i, x_j) = c u_i(x_i) u_j(x_j)` is a product of two components that were centered and
rescaled to unit variance at exactly the 64 Gauss-Legendre nodes the quadrature reads back, the
identities `E_nu[h | x_i] = 0`, `E_nu[h f_i] = 0` and `Var_nu h = c^2` hold to machine precision
rather than to quadrature accuracy. The orthogonality comes entirely from the centered factor
being integrated out, not from any relation between `u_i` and `f_i`: `E_nu[u_i f_i]` is a generic
non-zero number here (measured 0.3305 and 0.3911 for the two draws these tests use), and
`test_orthogonal_to_main_effects` asserts that too, so the orthogonality cannot pass for the
wrong reason.

*Optimality* tests carry the plan's headline property (decision D1): interaction pairs are
disjoint, so `f_star` is a sum of independent per-block maxima and `block_argmax` is what makes
it exact. `test_block_argmax_dominates_a_million_random_pairs` is the dominance half, and the
first-order optimality assertion beside it is the stationarity half. The attainment assertion is
neither: it pins that the returned value belongs to the returned point -- an `unravel_index` slip
or a refinement that updates the value without the point -- which per ruling R19's postmortem is
a real failure class no other assertion here covers, but it says nothing about global optimality.
`test_block_argmax_reduces_to_the_separable_maxima_when_c_is_zero` checks the two-dimensional
search against the exact one-dimensional `Component.argmax` oracle, and
`test_block_argmax_searches_only_the_unit_square` uses a component whose global peak sits at
z = 1.18, outside the cube, so that a search over unrestricted knots would report a maximum
2.29x too small.

*Gradient* tests check `block_value_and_grad`'s analytic gradient against central differences at
a step of 1e-6, at interior points so the stencil stays inside [0,1].
"""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from synthobj.component import Component
from synthobj.draws import gp_draw
from synthobj.interaction import Interaction, block_argmax, block_value_and_grad
from synthobj.kernel import GL_NODES, GL_WEIGHTS, eigen_factor, joint_points, make_grid

# Small enough that eigh is free; the exactness identities do not depend on resolution.
N_SMALL = 256
# The production grid (plan Global Constraints), which is what the brief's block_argmax timings
# and accuracy numbers were measured on: 1024 points, 1088 knots after the GL-node merge.
N_PROD = 1024

# The two interaction shares the study grid reaches: 0.375 is the gamma = 0.75 variant split over
# two pairs, 0.05 the smallest share any variant uses.
SHARES = [0.05, 0.375]

# The brief's measured configuration: ell = 0.5, main shares 0.1875, c = 0.5 (so s_ij = 0.25).
ELL = 0.5
MAIN_SHARE = 0.1875

EXT_LO, EXT_HI = -0.25, 1.25


def _gp_component(
    ell: float, share: float, n: int, seed: int, coord: int = 0, lo: float = 0.0, hi: float = 1.0
) -> Component:
    """A component from one GP draw on `make_grid(lo, hi, n)`; `eigen_factor` is cached by key."""
    raw = gp_draw(eigen_factor(lo, hi, n, ell), np.random.default_rng(seed))
    return Component.from_raw(coord, ell, share, make_grid(lo, hi, n), raw)


def _block(
    s_ij: float, n: int = N_SMALL, seed: int = 0, main_share: float = MAIN_SHARE
) -> tuple[Component, Component, Interaction]:
    """One paired block: two main effects, two unit-variance interaction factors, c = sqrt(s_ij).

    The four draws take consecutive seeds, so every block is built from four independent paths.
    """
    f_i = _gp_component(ELL, main_share, n, seed + 0, coord=0)
    f_j = _gp_component(ELL, main_share, n, seed + 1, coord=1)
    u_i = _gp_component(ELL, 1.0, n, seed + 2, coord=0)
    u_j = _gp_component(ELL, 1.0, n, seed + 3, coord=1)
    return f_i, f_j, Interaction(0, 1, float(np.sqrt(s_ij)), u_i, u_j)


def _analytic_component(grid: np.ndarray, coord: int, f) -> Component:
    """A unit-variance component whose raw joint values are `f` at `joint_points(grid)`."""
    return Component.from_raw(coord, np.inf, 1.0, grid, f(joint_points(grid)))


def _phi(f_i: Component, f_j: Component, inter: Interaction, x, y):
    """phi = f_i(x) + f_j(y) + h(x, y), assembled from the public evaluators only."""
    return f_i(x) + f_j(y) + inter(x, y)


# --- exact variance share ---------------------------------------------------------------------


@pytest.mark.parametrize("s_ij", SHARES)
def test_nu_var_is_the_exact_interaction_share(s_ij: float) -> None:
    _, _, inter = _block(s_ij)
    assert inter.c == pytest.approx(np.sqrt(s_ij), rel=1e-15)
    assert inter.nu_var() == pytest.approx(s_ij, abs=1e-8)

    # Independent check: the 64x64 product quadrature of h itself, which never touches
    # Component.nu_var. It fails for any c other than sqrt(s_ij), and for factors that are not
    # both unit variance under U[0,1].
    H = inter(GL_NODES[:, None], GL_NODES[None, :])
    w2 = GL_WEIGHTS[:, None] * GL_WEIGHTS[None, :]
    mean = float((w2 * H).sum())
    assert mean == pytest.approx(0.0, abs=1e-12)
    assert float((w2 * H**2).sum()) - mean**2 == pytest.approx(s_ij, abs=1e-8)


def test_call_is_the_scaled_product_of_the_two_factors() -> None:
    _, _, inter = _block(0.375)
    rng = np.random.default_rng(0)
    xi, xj = rng.uniform(0.0, 1.0, 50), rng.uniform(0.0, 1.0, 50)

    # Bounds rather than array_equal: the three factors may legitimately be multiplied in any
    # order, and float multiplication is commutative but not associative, so a reordering shifts
    # the last ulp (max|h| is 1.86 here, one ulp 2.2e-16). Any real defect misses by O(1).
    assert np.max(np.abs(inter(xi, xj) - inter.c * inter.u_i(xi) * inter.u_j(xj))) < 1e-15

    # It must also broadcast to the outer product block_argmax scans.
    outer = inter(xi[:, None], xj[None, :])
    assert outer.shape == (50, 50)
    expected = inter.c * inter.u_i(xi)[:, None] * inter.u_j(xj)[None, :]
    assert np.max(np.abs(outer - expected)) < 1e-15


# --- exact orthogonality ------------------------------------------------------------------------


@pytest.mark.parametrize("s_ij", SHARES)
def test_conditional_mean_zero(s_ij: float) -> None:
    _, _, inter = _block(s_ij)
    H = inter(GL_NODES[:, None], GL_NODES[None, :])

    # Sum_q w_q h(t, t_q) = c u_i(t) E_nu[u_j] = 0 at every node t, and the mirror image.
    assert np.max(np.abs(H @ GL_WEIGHTS)) < 1e-10
    assert np.max(np.abs(GL_WEIGHTS @ H)) < 1e-10

    # Symmetric in (i, j): exchanging the two roles transposes h. Not bitwise -- the swapped
    # product evaluates (c u_j) u_i where the original evaluates (c u_i) u_j, and floating-point
    # multiplication is commutative but not associative, so about a third of the 4096 entries
    # differ by exactly one ulp (measured max 2.2e-16 at max|h| = 1.86, i.e. 4.5x under the bound).
    # A factor used twice, or a sum in place of the product, misses by O(1).
    swapped = Interaction(inter.j, inter.i, inter.c, inter.u_j, inter.u_i)
    assert np.max(np.abs(swapped(GL_NODES[:, None], GL_NODES[None, :]) - H.T)) < 1e-15


@pytest.mark.parametrize("s_ij", SHARES)
def test_orthogonal_to_main_effects(s_ij: float) -> None:
    f_i, f_j, inter = _block(s_ij)
    H = inter(GL_NODES[:, None], GL_NODES[None, :])
    w2 = GL_WEIGHTS[:, None] * GL_WEIGHTS[None, :]
    fi, fj = f_i(GL_NODES), f_j(GL_NODES)

    assert abs(float((w2 * H * fi[:, None]).sum())) < 1e-10
    assert abs(float((w2 * H * fj[None, :]).sum())) < 1e-10

    # Neither identity says anything about u_i against f_i: E_nu[u_i f_i] is a generic non-zero
    # number (0.3305 and 0.3911 for these two draws), and the double sums above are zero only
    # because the *other* factor integrates to zero. Without this line the two assertions above
    # would also pass for an implementation that had accidentally made u_i proportional to f_i.
    assert abs(float((GL_WEIGHTS * inter.u_i(GL_NODES) * fi).sum())) > 0.1
    assert abs(float((GL_WEIGHTS * inter.u_j(GL_NODES) * fj).sum())) > 0.1


# --- gradient -------------------------------------------------------------------------------------


def test_block_value_and_grad_matches_central_differences() -> None:
    f_i, f_j, inter = _block(0.375, seed=40)
    step = 1e-6
    points = np.array([[0.20, 0.30], [0.50, 0.50], [0.37, 0.81], [0.90, 0.10], [0.64, 0.42]])

    for xy in points:
        value, grad = block_value_and_grad(f_i, f_j, inter, xy)
        assert isinstance(value, float)
        assert grad.shape == (2,)
        assert value == pytest.approx(float(_phi(f_i, f_j, inter, xy[0], xy[1])), abs=1e-14)

        for k in range(2):
            offset = np.zeros(2)
            offset[k] = step
            up = block_value_and_grad(f_i, f_j, inter, xy + offset)[0]
            down = block_value_and_grad(f_i, f_j, inter, xy - offset)[0]
            # The interaction term contributes O(c) = O(0.6) to each partial derivative here, so
            # dropping it, or mismatching which factor is differentiated, misses by ~1e6 x this
            # tolerance. The abs floor sits far below the O(1) gradient scale and well above the
            # ~2e-9 roundoff floor of a central difference at step 1e-6.
            assert grad[k] == pytest.approx((up - down) / (2.0 * step), rel=1e-6, abs=1e-7)


# --- block_argmax ------------------------------------------------------------------------------


@pytest.mark.parametrize("trial", [0, 1, 2, 3])
def test_block_argmax_dominates_a_million_random_pairs(trial: int) -> None:
    f_i, f_j, inter = _block(0.25, n=N_PROD, seed=100 * trial)
    xy_star, phi_star = block_argmax(f_i, f_j, inter)

    assert isinstance(xy_star, np.ndarray)
    assert xy_star.shape == (2,)
    assert np.all((xy_star >= 0.0) & (xy_star <= 1.0))

    rng = np.random.default_rng(20260906 + trial)
    x, y = rng.uniform(0.0, 1.0, 1_000_000), rng.uniform(0.0, 1.0, 1_000_000)
    assert phi_star >= float(_phi(f_i, f_j, inter, x, y).max()) - 1e-12

    # Refinement can only improve on the knot grid it starts from, so the returned value also
    # dominates the whole 1088 x 1088 scan, recomputed here independently.
    xs, ys = f_i.grid, f_j.grid
    grid_max = float(_phi(f_i, f_j, inter, xs[:, None], ys[None, :]).max())
    assert phi_star >= grid_max - 1e-15

    # Attained: a fresh evaluation of phi at the returned point through the public evaluators.
    # This pins pair consistency -- that the reported value belongs to the reported maximizer --
    # not optimality, which is what the two dominance assertions above are for (ruling R19).
    assert float(_phi(f_i, f_j, inter, xy_star[0], xy_star[1])) == pytest.approx(phi_star, abs=1e-6)

    # First-order optimality of the returned point: interior coordinates are stationary, and a
    # coordinate at a bound has its gradient pointing out of the box. An unrefined knot-grid
    # answer leaves |d phi / dx| of order h |phi''| ~ 1e-3 x 1e2 = 1e-1 here, 1e5 x this bound.
    grad = block_value_and_grad(f_i, f_j, inter, xy_star)[1]
    for k in range(2):
        if xy_star[k] <= 1e-12:
            assert grad[k] <= 1e-6
        elif xy_star[k] >= 1.0 - 1e-12:
            assert grad[k] >= -1e-6
        else:
            assert abs(grad[k]) < 1e-6


def test_block_argmax_reduces_to_the_separable_maxima_when_c_is_zero() -> None:
    # At gamma = 0 (the interaction_g0.00 variant) c is exactly zero and phi is separable, so the
    # block maximum must equal the sum of the two exact 1-D maxima -- an oracle that does not
    # share a line of code with block_argmax.
    f_i, f_j, inter = _block(0.0, n=N_PROD, seed=900)
    assert inter.c == 0.0

    xy_star, phi_star = block_argmax(f_i, f_j, inter)
    x_i, max_i = f_i.argmax()
    x_j, max_j = f_j.argmax()

    assert phi_star == pytest.approx(max_i + max_j, abs=1e-10)
    assert xy_star[0] == pytest.approx(x_i, abs=1e-5)
    assert xy_star[1] == pytest.approx(x_j, abs=1e-5)


def test_block_argmax_returns_the_right_corner_when_the_maximum_lies_on_a_knot() -> None:
    # The one case where the knot scan's own bookkeeping is load-bearing: the maximum sits exactly
    # on a knot, every refinement starts at a point where the projected gradient is already zero,
    # and the scan's answer is returned unrefined. f_i increases and f_j decreases, so the maximum
    # is the corner (1, 0) -- transposing the unravelled cell returns (0, 1), the block MINIMUM.
    # Every other block_argmax test here is refined before it returns and cannot see that slip.
    grid = make_grid(0.0, 1.0, N_SMALL)
    f_i = _analytic_component(grid, 0, lambda z: np.sqrt(12.0) * (z - 0.5))
    f_j = _analytic_component(grid, 1, lambda z: -np.sqrt(12.0) * (z - 0.5))
    u_i = _analytic_component(grid, 0, lambda z: np.sin(2.0 * np.pi * z))
    u_j = _analytic_component(grid, 1, lambda z: np.cos(2.0 * np.pi * z))
    inter = Interaction(0, 1, 0.0, u_i, u_j)

    xy_star, phi_star = block_argmax(f_i, f_j, inter)
    assert xy_star == pytest.approx(np.array([1.0, 0.0]), abs=1e-12)
    assert phi_star == pytest.approx(2.0 * np.sqrt(3.0), abs=1e-10)
    assert float(_phi(f_i, f_j, inter, xy_star[0], xy_star[1])) == pytest.approx(phi_star, abs=1e-12)


def test_block_argmax_searches_only_the_unit_square() -> None:
    # Rotated blocks (a later task) reuse block_argmax on components drawn on the wider box, where
    # searching the raw knot vector would look at points the objective never sees. This shape has
    # its global peak at z = 1.18, outside the cube, and its best point inside the cube at z = 0.5;
    # an unrestricted knot scan starts every refinement at the clipped corner (1, 1), which is a
    # local maximum of phi, and would report 1.9157 instead of 4.3787.
    grid = make_grid(EXT_LO, EXT_HI, 512)
    shape = lambda z: 3.0 * np.exp(-50.0 * (z - 1.18) ** 2) + np.exp(-50.0 * (z - 0.5) ** 2)
    f_i, f_j = _analytic_component(grid, 0, shape), _analytic_component(grid, 1, shape)
    u_i = _analytic_component(grid, 0, lambda z: np.sin(2.0 * np.pi * z))
    u_j = _analytic_component(grid, 1, lambda z: np.cos(2.0 * np.pi * z))
    inter = Interaction(0, 1, 0.0, u_i, u_j)

    # The bait: outside the cube the component really is 3.77x higher than its best point inside.
    assert float(f_i(1.18)) > 3.7 * float(f_i(0.5))

    xy_star, phi_star = block_argmax(f_i, f_j, inter)
    assert np.all((xy_star >= 0.0) & (xy_star <= 1.0))
    assert xy_star == pytest.approx(np.array([0.5, 0.5]), abs=1e-4)
    assert phi_star == pytest.approx(f_i.argmax(0.0, 1.0)[1] + f_j.argmax(0.0, 1.0)[1], abs=1e-10)

    rng = np.random.default_rng(7)
    x, y = rng.uniform(0.0, 1.0, 1_000_000), rng.uniform(0.0, 1.0, 1_000_000)
    assert phi_star >= float(_phi(f_i, f_j, inter, x, y).max()) - 1e-12


def test_block_argmax_honours_top_k() -> None:
    # top_k = 1 refines only the best knot cell, so it can only be worse than the default sweep;
    # both must still be consistent pairs and inside the cube.
    f_i, f_j, inter = _block(0.25, n=N_PROD, seed=100)
    xy_one, phi_one = block_argmax(f_i, f_j, inter, top_k=1)
    _, phi_many = block_argmax(f_i, f_j, inter)

    assert phi_one <= phi_many + 1e-12
    assert np.all((xy_one >= 0.0) & (xy_one <= 1.0))
    assert float(_phi(f_i, f_j, inter, xy_one[0], xy_one[1])) == pytest.approx(phi_one, abs=1e-6)


# --- dataclass conventions -----------------------------------------------------------------------


def test_interaction_is_frozen_and_compares_by_identity() -> None:
    _, _, a = _block(0.375, seed=700)
    _, _, b = _block(0.375, seed=700)

    # Ruling R4 and the branch convention: a frozen dataclass holding Components is eq=False.
    # NOTE ON POWER: unlike Component, Interaction would behave identically with eq=True -- its
    # scalar fields compare fine and Component already compares by identity -- so the two
    # behavioural assertions below pass either way. The declaration is pinned directly instead.
    assert Interaction.__eq__ is object.__eq__
    assert a == a
    assert a != b

    with pytest.raises(FrozenInstanceError):
        a.c = 1.0
