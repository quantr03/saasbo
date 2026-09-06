"""Tests for synthobj.rotation: planar rotation of active-coordinate pairs, and its objective wiring.

Four kinds of assertion live here.

*Geometry* tests pin `PairRotation.forward` and `R`: the identity at theta=0, that a rotated pair
never leaves `[EXT_LO, EXT_HI]` (ruling R7's domain-sufficiency claim, re-checked at 10**6 points),
and that `forward` returns a copy and leaves unpaired coordinates alone.

*Variance decomposition* tests pin `rotated_block_stats`. At theta in {0, 90} the rotation is exactly
additive (all 20 of the brief's dispatch seeds reproduced this exactly), so the total and both
first-order variances are asserted to 1e-8. At other angles, only the ANOVA inequality
`Var phi >= Var_a[E_b phi] + Var_b[E_a phi]` is asserted (a variance decomposition remainder cannot be
negative) -- **not** the specific total/scale/gamma_axis numbers, which the brief's 20-seed sweep
found range 4x across seeds at a fixed angle. That table is quoted in the module below only to
justify what is deliberately not asserted.

*Optimality* tests carry `rotated_block_argmax`'s exactness claim, mirroring `test_interaction.py`'s
pattern: dominance over a million random pairs, the coarse-grid comparison recomputed independently
through `Component.__call__`, attainment, and first-order stationarity. `test_..._refinement_gains`
is new relative to that pattern (carried from the brief's own six-trial measurement): unlike one of
Task 4's four trials, refinement here gains something at every trial, so the strict-improvement
assertion has power everywhere it runs.

*Objective* tests build a `SyntheticObjective` with a real `PairRotation` (D=20, two pairs, ell=0.5,
share 0.25 each, matching the brief's dispatch configuration) and check the label bookkeeping: `s`
stays the pre-rotation share, `s_axis`/`gamma_axis` sum to 1, `gamma` stays exactly 0 while
`gamma_axis` is positive (the family has no explicit `Interaction`), and `f_star` is bounded both
below (a million random points) and above (a dense grid through the public `__call__`, sharing no
code with `rotated_block_argmax` -- the Task 5 review's carried-over requirement that dominance alone
cannot catch a double-counted or mis-scaled `f_star`).
"""
import math

import numpy as np
import pytest
from scipy.stats import qmc

from synthobj.component import Component
from synthobj.draws import gp_draw
from synthobj.kernel import GL_NODES, GL_WEIGHTS, eigen_factor, make_grid
from synthobj.objective import SyntheticObjective
from synthobj.rotation import (
    EXT_HI,
    EXT_LO,
    PairRotation,
    rotated_block_argmax,
    rotated_block_mean,
    rotated_block_stats,
    rotated_block_value_and_grad,
)

# The brief's dispatch configuration: two pairs of ell=0.5 components, shares 0.25 each, drawn on
# the extended grid (ruling R7: n_ext = 1469, joint length 1533).
D = 20
ELL = 0.5
MAIN_SHARE = 0.25
N_EXT = 1469
COORDS: tuple[int, ...] = (3, 8, 12, 17)
PAIRS: tuple[tuple[int, int], ...] = ((3, 8), (12, 17))
INACTIVE: tuple[int, ...] = (0, 1, 19)
SEED = 42


def _component(coord: int, share: float, seed: int, ell: float = ELL, n: int = N_EXT) -> Component:
    """One GP-draw component on the extended grid; `eigen_factor` is cached by key."""
    raw = gp_draw(eigen_factor(EXT_LO, EXT_HI, n, ell), np.random.default_rng(seed))
    return Component.from_raw(coord, ell, share, make_grid(EXT_LO, EXT_HI, n), raw)


def _phi(f_a: Component, f_b: Component, rot: PairRotation, xa, xb):
    """`phi = f_a(z_a) + f_b(z_b)`, assembled from the public evaluators only -- independent of
    `rotated_block_argmax` and `rotated_block_stats`, which this file tests against it."""
    theta = np.deg2rad(rot.theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    za = 0.5 + c * (xa - 0.5) - s * (xb - 0.5)
    zb = 0.5 + s * (xa - 0.5) + c * (xb - 0.5)
    za = np.clip(za, f_a.grid[0], f_a.grid[-1])
    zb = np.clip(zb, f_b.grid[0], f_b.grid[-1])
    return f_a(za) + f_b(zb)


@pytest.fixture(scope="module")
def mains() -> tuple[Component, ...]:
    """The four main effects of the dispatch configuration, one independent draw each."""
    return tuple(_component(coord, MAIN_SHARE, 100 + k) for k, coord in enumerate(COORDS))


def _objective(theta_deg: float, mains: tuple[Component, ...], seed: int = SEED) -> SyntheticObjective:
    return SyntheticObjective(D, mains, (), PairRotation(theta_deg, PAIRS), None, seed)


@pytest.fixture(scope="module")
def obj_theta0(mains: tuple[Component, ...]) -> SyntheticObjective:
    return _objective(0.0, mains)


@pytest.fixture(scope="module")
def obj_theta45(mains: tuple[Component, ...]) -> SyntheticObjective:
    return _objective(45.0, mains)


@pytest.fixture(scope="module")
def obj_theta30(mains: tuple[Component, ...]) -> SyntheticObjective:
    # theta=0, 45 and 90 are each fixed points of theta -> 90 - theta under this fixture's equal
    # shares: swapping cos/sin inside rotated_block_stats maps 0 <-> 90 (both individually asserted
    # exactly elsewhere) and is a literal no-op at 45 (cos 45 = sin 45), so a mutant confined to that
    # one function is invisible to any objective-level test that only ever exercises those three
    # angles. theta=30 has no such symmetry.
    return _objective(30.0, mains)


# --- EXT_LO / EXT_HI / n_ext (ruling R7) -----------------------------------------------------------


def test_ext_bounds_and_n_ext_match_ruling_r7() -> None:
    assert EXT_LO == pytest.approx(0.5 - math.sqrt(2) / 2 - 0.01, abs=1e-15)
    assert EXT_HI == pytest.approx(1.0 - EXT_LO, abs=1e-15)
    # A point at norm sqrt(2)/2 from the cube centre (the unit square's corner) plus the 0.01 margin
    # lands exactly on EXT_LO/EXT_HI, so this is the domain claim itself, not just its arithmetic.
    assert EXT_LO == pytest.approx(-0.21710678118654758, abs=1e-15)
    assert EXT_HI == pytest.approx(1.2171067811865477, abs=1e-15)

    n_ext = math.ceil((EXT_HI - EXT_LO) * (1024 - 1)) + 1
    assert n_ext == 1469
    assert n_ext + 64 == 1533  # joint length: draw grid + the 64 GL nodes


# --- R() and forward ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_deg", [0.0, 15.0, 45.0, 90.0, 180.0])
def test_R_is_orthogonal_and_matches_the_closed_form(theta_deg: float) -> None:
    R = PairRotation(theta_deg, ()).R()
    theta = np.deg2rad(theta_deg)
    expected = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert R.shape == (2, 2)
    assert R == pytest.approx(expected, abs=1e-15)
    assert R.T @ R == pytest.approx(np.eye(2), abs=1e-12)


def test_forward_at_theta_zero_is_identity() -> None:
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(1000, 4))
    Z = PairRotation(0.0, ((0, 1), (2, 3))).forward(X)
    assert Z == pytest.approx(X, abs=1e-12)


def test_forward_returns_a_copy_and_leaves_unpaired_coordinates_alone() -> None:
    X = np.random.default_rng(1).uniform(0.0, 1.0, size=(100, 5))
    original = X.copy()
    Z = PairRotation(30.0, ((0, 1),)).forward(X)

    assert np.array_equal(X, original)  # not mutated
    assert Z is not X
    assert np.array_equal(Z[:, 2:], X[:, 2:])  # coordinates 2, 3, 4 are untouched


@pytest.mark.parametrize("theta_deg", [15.0, 45.0])
def test_forward_of_a_million_points_stays_inside_the_extended_domain(theta_deg: float) -> None:
    X = np.random.default_rng(2026).uniform(0.0, 1.0, size=(1_000_000, 4))
    Z = PairRotation(theta_deg, ((0, 1), (2, 3))).forward(X)
    assert np.all(Z >= EXT_LO)
    assert np.all(Z <= EXT_HI)


# --- rotated_block_stats -------------------------------------------------------------------------


# An unequal-share pair for the theta in {0, 90} exactness tests: with both factors at the same
# share (as an earlier version of this file had it), var_a and var_b are numerically identical at
# these two angles and a mutant that returns (var_phi, var_b, var_a) is invisible. 0.25 vs 0.10
# breaks that symmetry.
SHARE_A, SHARE_B = 0.25, 0.10


def test_rotated_block_stats_at_theta_zero_matches_the_unrotated_shares() -> None:
    # theta=0: z_a = x_a, z_b = x_b (forward is the identity), so phi = f_a(x_a) + f_b(x_b) is
    # already separable and var_a/var_b are exactly f_a's/f_b's own nu_var.
    f_a, f_b = _component(0, SHARE_A, 10), _component(1, SHARE_B, 11)
    var_phi, var_a, var_b = rotated_block_stats(f_a, f_b, PairRotation(0.0, ((0, 1),)))
    assert var_phi == pytest.approx(f_a.nu_var() + f_b.nu_var(), abs=1e-8)
    assert var_a == pytest.approx(f_a.nu_var(), abs=1e-8)
    assert var_b == pytest.approx(f_b.nu_var(), abs=1e-8)


def test_rotated_block_stats_at_theta_ninety_is_also_exactly_additive() -> None:
    # theta=90: z_a = 1 - x_b, z_b = x_a. The variance attributable to x_a is Var[f_b(x_a)] (since
    # z_b = x_a), i.e. f_b's own nu_var, and the variance attributable to x_b is Var[f_a(1 - x_b)] =
    # f_a's own nu_var (1 - x_b is uniform on [0,1] too) -- var_a and var_b are SWAPPED relative to
    # theta=0, not equal to their theta=0 values. (An earlier version of this test asserted the
    # theta=0 pairing here; it passed only because that fixture used equal shares, which cannot
    # distinguish "var_a is f_a's variance" from "var_a is f_b's variance.")
    f_a, f_b = _component(0, SHARE_A, 12), _component(1, SHARE_B, 13)
    var_phi, var_a, var_b = rotated_block_stats(f_a, f_b, PairRotation(90.0, ((0, 1),)))
    assert var_phi == pytest.approx(f_a.nu_var() + f_b.nu_var(), abs=1e-8)
    assert var_a == pytest.approx(f_b.nu_var(), abs=1e-8)
    assert var_b == pytest.approx(f_a.nu_var(), abs=1e-8)


@pytest.mark.parametrize("theta_deg", [15.0, 30.0, 45.0, 60.0])
def test_rotated_block_stats_first_order_never_exceeds_the_total(theta_deg: float) -> None:
    # ANOVA identity: Var phi splits into the two axis-aligned first-order variances plus a
    # remainder that is itself a variance, hence never negative -- so the first-order sum can never
    # exceed the total. This holds at every angle and every seed; it is what makes
    # gamma_axis = 1 - sum(s_axis) meaningful rather than an artifact of one draw. Per the task-6
    # brief's 20-seed sweep, the specific numbers here are seed noise (a 4x range at theta=45) and
    # are deliberately NOT pinned.
    f_a, f_b = _component(0, MAIN_SHARE, 20), _component(1, MAIN_SHARE, 21)
    var_phi, var_a, var_b = rotated_block_stats(f_a, f_b, PairRotation(theta_deg, ((0, 1),)))
    assert var_a >= 0.0
    assert var_b >= 0.0
    assert var_phi >= var_a + var_b - 1e-10


def test_rotated_block_stats_var_a_and_var_b_subtract_the_mean_correctly() -> None:
    # Regression for a mutant that drops "- mean_phi**2" when computing var_a/var_b. At a
    # non-symmetric angle with unequal shares, E_nu[phi] is measurably non-zero (the R30 finding),
    # so omitting the mean-square term inflates var_a/var_b by mean_phi**2 -- asserted directly
    # against an independent recomputation of the same 64x64 GL quadrature, rather than relying on
    # the ANOVA inequality above to notice the inflation incidentally.
    f_a, f_b = _component(0, SHARE_A, 40), _component(1, SHARE_B, 41)
    rot = PairRotation(30.0, ((0, 1),))
    var_phi, var_a, var_b = rotated_block_stats(f_a, f_b, rot)

    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    xa, xb = np.meshgrid(GL_NODES, GL_NODES, indexing="ij")
    za = 0.5 + c * (xa - 0.5) - s * (xb - 0.5)
    zb = 0.5 + s * (xa - 0.5) + c * (xb - 0.5)
    phi = f_a(za) + f_b(zb)
    w = GL_WEIGHTS[:, None] * GL_WEIGHTS[None, :]
    mean_phi = float((w * phi).sum())
    assert abs(mean_phi) > 1e-3  # non-trivial: dropping mean_phi**2 is not a rounding no-op here

    e_given_a = phi @ GL_WEIGHTS
    e_given_b = GL_WEIGHTS @ phi
    expected_var_a = float((GL_WEIGHTS * e_given_a**2).sum()) - mean_phi**2
    expected_var_b = float((GL_WEIGHTS * e_given_b**2).sum()) - mean_phi**2

    assert var_a == pytest.approx(expected_var_a, abs=1e-12)
    assert var_b == pytest.approx(expected_var_b, abs=1e-12)


# --- rotated_block_value_and_grad (chain rule) ----------------------------------------------------


def test_rotated_block_value_and_grad_matches_central_differences() -> None:
    f_a, f_b = _component(0, MAIN_SHARE, 940), _component(1, MAIN_SHARE, 941)
    rot = PairRotation(45.0, ((0, 1),))
    step = 1e-6
    points = np.array([[0.20, 0.30], [0.50, 0.50], [0.37, 0.81], [0.90, 0.10], [0.64, 0.42]])

    for xy in points:
        value, grad = rotated_block_value_and_grad(f_a, f_b, rot, xy)
        assert isinstance(value, float)
        assert grad.shape == (2,)
        assert value == pytest.approx(float(_phi(f_a, f_b, rot, xy[0], xy[1])), abs=1e-12)

        for k in range(2):
            offset = np.zeros(2)
            offset[k] = step
            up = rotated_block_value_and_grad(f_a, f_b, rot, xy + offset)[0]
            down = rotated_block_value_and_grad(f_a, f_b, rot, xy - offset)[0]
            assert grad[k] == pytest.approx((up - down) / (2.0 * step), rel=1e-6, abs=1e-7)


def test_theta_45_handles_the_cubes_corners_without_raising() -> None:
    # Integration hazard from Task 3: Component.argmax/evaluation raises or NaNs outside the knot
    # range, a rotated component's knots live on the extended domain, and rotated_block_argmax
    # searches x-space over [0,1]^2 -- so the rotation maps the corners of that square to z values
    # that must land inside the extended knots. rotated_block_value_and_grad clamps z into the knot
    # range before every evaluation; this pins that choice at the trickiest points and angle.
    f_a, f_b = _component(0, MAIN_SHARE, 960), _component(1, MAIN_SHARE, 961)
    rot = PairRotation(45.0, ((0, 1),))

    for xy in np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]):
        value, grad = rotated_block_value_and_grad(f_a, f_b, rot, xy)
        assert np.isfinite(value)
        assert np.all(np.isfinite(grad))

    xy_star, phi_star = rotated_block_argmax(f_a, f_b, rot)
    assert np.isfinite(phi_star)
    assert np.all((xy_star >= 0.0) & (xy_star <= 1.0))


# --- rotated_block_argmax -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta_deg,seed", [(15.0, 0), (15.0, 1), (15.0, 2), (45.0, 0), (45.0, 1), (45.0, 2)]
)
def test_rotated_block_argmax_dominates_and_matches_a_dense_grid(theta_deg: float, seed: int) -> None:
    f_a = _component(0, MAIN_SHARE, 2000 + 10 * seed)
    f_b = _component(1, MAIN_SHARE, 2001 + 10 * seed)
    rot = PairRotation(theta_deg, ((0, 1),))

    xy_star, phi_star = rotated_block_argmax(f_a, f_b, rot)
    assert xy_star.shape == (2,)
    assert np.all((xy_star >= 0.0) & (xy_star <= 1.0))

    # Attained: the reported value belongs to the reported point (ruling R19's postmortem).
    assert float(_phi(f_a, f_b, rot, xy_star[0], xy_star[1])) == pytest.approx(phi_star, abs=1e-6)

    # Dominance over a million random pairs, and over the coarse 1024x1024 grid recomputed
    # independently through the public evaluators.
    rng = np.random.default_rng(20260907 + seed)
    x, y = rng.uniform(0.0, 1.0, 1_000_000), rng.uniform(0.0, 1.0, 1_000_000)
    assert phi_star >= float(_phi(f_a, f_b, rot, x, y).max()) - 1e-12

    xs = np.linspace(0.0, 1.0, 1024)
    xa, xb = np.meshgrid(xs, xs, indexing="ij")
    coarse_max = float(_phi(f_a, f_b, rot, xa, xb).max())
    assert phi_star >= coarse_max - 1e-15

    # The refinement always gains something here (measured 5.9e-8 to 5.8e-6 across six trials,
    # never exactly 0), unlike Task 4's block_argmax where one of four trials gained exactly
    # nothing -- so this strict inequality has power at every trial, not just some.
    assert phi_star > coarse_max

    # First-order optimality: interior coordinates are stationary, a coordinate at a bound has its
    # gradient pointing out of the box. Per the task-6 brief, most of the six measured maximizers
    # sit on or within 0.0025 of the boundary, so only the one-sided condition is checked there.
    grad = rotated_block_value_and_grad(f_a, f_b, rot, xy_star)[1]
    for k in range(2):
        if xy_star[k] <= 1e-6:
            assert grad[k] <= 1e-4
        elif xy_star[k] >= 1.0 - 1e-6:
            assert grad[k] >= -1e-4
        else:
            assert abs(grad[k]) < 1e-4

    # Two-sided (Task 5 review carryover): dominance alone cannot catch an inflated maximum, so a
    # dense grid must bound phi_star from above too. 2048x2048, chunked by row per ruling R10's
    # spirit (a full 4096x4096 float64 array is 134 MB).
    dense_max = -np.inf
    xs_dense = np.linspace(0.0, 1.0, 2048)
    for start in range(0, xs_dense.size, 256):
        rows = xs_dense[start : start + 256]
        xa, xb = np.meshgrid(rows, xs_dense, indexing="ij")
        dense_max = max(dense_max, float(_phi(f_a, f_b, rot, xa, xb).max()))

    assert phi_star >= dense_max - 1e-9
    assert phi_star - dense_max <= 1e-5


# --- rotation acceptance guards on SyntheticObjective ---------------------------------------------


def test_a_rotation_pair_coordinate_needs_a_main_effect() -> None:
    f0 = _component(0, MAIN_SHARE, 950)
    with pytest.raises(ValueError, match="main effect"):
        SyntheticObjective(2, (f0,), (), PairRotation(30.0, ((0, 1),)), None, 0)


def test_overlapping_rotation_pairs_are_rejected() -> None:
    comps = tuple(_component(c, MAIN_SHARE, 951 + c) for c in (0, 1, 2))
    with pytest.raises(ValueError, match="disjoint"):
        SyntheticObjective(3, comps, (), PairRotation(30.0, ((0, 1), (1, 2))), None, 0)


def test_an_unpaired_component_under_rotation_keeps_its_own_argmax_and_share() -> None:
    # SyntheticObjective must not assume every component sits in a rotation pair -- the rotated
    # family itself pairs all ten active coordinates, but the mechanism is more general than that
    # one family. A coordinate outside every pair is untouched by rotation.forward, so its argmax
    # and axis-aligned share are just its own, rescaled by the (still global) scale.
    f0, f1 = _component(0, MAIN_SHARE, 970), _component(1, MAIN_SHARE, 971)
    plain = Component.from_raw(
        2, ELL, MAIN_SHARE, make_grid(0.0, 1.0, 1024),
        gp_draw(eigen_factor(0.0, 1.0, 1024, ELL), np.random.default_rng(972)),
    )
    obj = SyntheticObjective(3, (f0, f1, plain), (), PairRotation(45.0, ((0, 1),)), None, 0)

    x2, value2 = plain.argmax()
    assert obj.labels.x_star[2] == pytest.approx(x2, abs=1e-12)
    assert obj.labels.s_axis[2] == pytest.approx(plain.share * obj.scale**2, abs=1e-12)
    assert obj.labels.s[2] == MAIN_SHARE


# --- the assembled rotated objective ---------------------------------------------------------------


def test_theta_zero_gives_exact_no_rotation_labels(
    obj_theta0: SyntheticObjective, mains: tuple[Component, ...]
) -> None:
    # All 20 of the brief's dispatch seeds reproduced this exactly; asserted to 1e-8.
    labels = obj_theta0.labels
    assert labels.scale == pytest.approx(1.0, abs=1e-8)
    assert labels.rotation_deg == 0.0
    assert labels.rotation_pairs == PAIRS
    assert labels.gamma_axis == pytest.approx(0.0, abs=1e-8)

    expected_s = np.zeros(D)
    expected_s[list(COORDS)] = MAIN_SHARE
    assert np.array_equal(labels.s, expected_s)
    assert labels.s_axis == pytest.approx(expected_s, abs=1e-8)
    assert float(labels.s_axis.sum()) + labels.gamma_axis == pytest.approx(1.0, abs=1e-8)

    # rotated_t0 is still built on the extended grid, so it is NOT bit-identical to an aligned
    # objective built from the same seeds on [0,1] -- that is expected and not asserted here.


def test_theta_ninety_is_also_exactly_additive(mains: tuple[Component, ...]) -> None:
    labels = _objective(90.0, mains).labels
    assert labels.scale == pytest.approx(1.0, abs=1e-8)
    assert labels.gamma_axis == pytest.approx(0.0, abs=1e-8)


def test_s_is_the_prescribed_pre_rotation_share_regardless_of_rotation(
    obj_theta45: SyntheticObjective,
) -> None:
    expected_s = np.zeros(D)
    expected_s[list(COORDS)] = MAIN_SHARE
    assert np.array_equal(obj_theta45.labels.s, expected_s)


def test_gamma_stays_zero_while_gamma_axis_is_positive(obj_theta45: SyntheticObjective) -> None:
    # The rotated family has no explicit Interaction: rotation is its only source of non-additivity,
    # so gamma (the product-interaction share) stays exactly 0 while gamma_axis -- what a
    # first-order additive model actually faces -- is positive. Kept as its own test because a
    # reviewer checks this distinction directly.
    labels = obj_theta45.labels
    assert labels.gamma == 0.0
    assert labels.s_pairs == ()
    assert labels.pairs == ()
    assert labels.gamma_axis > 0.0


def test_s_axis_matches_rotated_block_stats_times_scale_squared(
    obj_theta30: SyntheticObjective, mains: tuple[Component, ...]
) -> None:
    # `gamma_axis = 1 - s_axis.sum()` (the brief's own formula) makes the sum-to-one check below
    # tautological with respect to s_axis's own values -- it cannot tell a correctly indexed s_axis
    # from one with i and j swapped, since the swap cancels in the sum. This test recomputes each
    # pair's two first-order variances independently via rotated_block_stats and pins them to the
    # matching coordinate, which the sum-to-one check cannot. theta=30 (not 45), because at 45
    # cos(theta) == sin(theta), so a mutant that swaps them inside rotated_block_stats is a literal
    # no-op there and this check would not be exercising anything at that particular angle.
    by_coord = {comp.coord: comp for comp in mains}
    labels = obj_theta30.labels
    rot = PairRotation(30.0, PAIRS)
    for i, j in PAIRS:
        _, var_a, var_b = rotated_block_stats(by_coord[i], by_coord[j], rot)
        assert labels.s_axis[i] == pytest.approx(var_a * labels.scale**2, abs=1e-10)
        assert labels.s_axis[j] == pytest.approx(var_b * labels.scale**2, abs=1e-10)


def test_intermediate_angle_scale_and_axis_shares_are_only_checked_qualitatively(
    obj_theta45: SyntheticObjective,
) -> None:
    # Per the task-6 brief's 20-seed sweep at theta=45, total Var ranges 0.81 to 1.89 and gamma_axis
    # ranges 0.16 to 0.73 -- the extended-domain energy is a free random quantity of the draw, and
    # scale absorbs it (the whole point of scale, not a bug). Only what held at all 20 seeds is
    # asserted: scale != 1, and the shares still sum to 1 after rescaling. NOT asserted: a numeric
    # value for scale/gamma_axis, or scale > 1 (its direction flips across seeds).
    labels = obj_theta45.labels
    assert labels.scale != pytest.approx(1.0, abs=1e-6)
    assert float(labels.s_axis.sum()) + labels.gamma_axis == pytest.approx(1.0, abs=1e-8)


def test_rotated_objective_qmc_variance_is_one_after_scaling(obj_theta30: SyntheticObjective) -> None:
    # theta=30 (not 45): the previous fixture-choice note applies here too (see
    # test_s_axis_matches_rotated_block_stats_times_scale_squared). Variance only, per the brief's
    # checklist. Mean is checked separately, exactly, by
    # test_e_nu_f_is_zero_after_recentering_at_a_non_symmetric_angle below -- ruling R30 recenters
    # by `mu` precisely so the QMC-sampled mean here is small-sample noise around 0, not a
    # structurally nonzero quantity the way it was before that fix.
    X = qmc.Sobol(D, scramble=True, seed=0).random(2**17)
    values = obj_theta30(X)
    assert float(values.var()) == pytest.approx(1.0, rel=0.01)


def test_e_nu_f_is_zero_after_recentering_at_a_non_symmetric_angle() -> None:
    # Ruling R30: SyntheticObjective recenters by `mu = Sum_p rotated_block_mean` so `E_nu[f] = 0`
    # despite rotation moving it (measured -0.006 to -0.14 across seeds/angles before this fix, per
    # the fix-round message). This integrates the objective's PUBLIC __call__ over the same 64x64
    # Gauss-Legendre grid `rotated_block_mean` itself uses, rather than re-deriving `mu`'s formula,
    # so it exercises the whole pipeline (forward, mu, scale together) and is not a tautological
    # restatement of `_compute_mu`. theta=30 is not a multiple of 90, where the mean is exactly 0
    # even before recentering and this test would have no power to distinguish "recentered" from
    # "never needed it."
    f0 = _component(0, MAIN_SHARE, 800)
    f1 = _component(1, MAIN_SHARE, 801)
    obj = SyntheticObjective(2, (f0, f1), (), PairRotation(30.0, ((0, 1),)), None, 0)
    assert obj.mu != 0.0  # the correction is doing real work at this angle, not a no-op

    xa, xb = np.meshgrid(GL_NODES, GL_NODES, indexing="ij")
    X = np.stack([xa.ravel(), xb.ravel()], axis=-1)
    values = obj(X).reshape(xa.shape)
    w = GL_WEIGHTS[:, None] * GL_WEIGHTS[None, :]
    mean_f = float((w * values).sum())

    assert mean_f == pytest.approx(0.0, abs=1e-10)


def test_f_star_dominates_a_million_uniform_points(obj_theta45: SyntheticObjective) -> None:
    rng = np.random.default_rng(20260907)
    chunk = 100_000
    best = -np.inf
    for _ in range(1_000_000 // chunk):
        best = max(best, float(obj_theta45(rng.uniform(0.0, 1.0, size=(chunk, D))).max()))
    assert obj_theta45.f_star >= best - 1e-12


def test_f_star_attained_and_x_star_inside_the_cube(obj_theta45: SyntheticObjective) -> None:
    x_star = obj_theta45.labels.x_star
    assert x_star.shape == (D,)
    # Closed cube, not a strict inequality: per the brief, most measured maximizers sit on or near
    # the boundary of their pair's square.
    assert np.all(x_star >= -1e-9)
    assert np.all(x_star <= 1.0 + 1e-9)
    assert np.all(x_star[list(INACTIVE)] == 0.5)

    assert float(obj_theta45(x_star)) == pytest.approx(obj_theta45.f_star, abs=1e-6)
    assert obj_theta45.f_star == obj_theta45.labels.f_star


def test_f_star_matches_a_dense_grid_through_the_public_call() -> None:
    # Required per the task-6 brief's carryover from the Task 5 review: f_star >= f(x) at random
    # points is one-sided and does not catch double-counting (a corrupted, inflated f_star still
    # dominates). A single-pair objective keeps the dense grid 2-D: the maximum is bounded above by
    # a grid evaluated through the objective's public __call__ (so rotation.forward and scale are
    # both exercised as a real caller would see them), which shares no code with
    # rotated_block_argmax.
    f0 = _component(0, MAIN_SHARE, 900)
    f1 = _component(1, MAIN_SHARE, 901)
    obj = SyntheticObjective(2, (f0, f1), (), PairRotation(45.0, ((0, 1),)), None, 0)

    xs = np.linspace(0.0, 1.0, 2048)
    brute = -np.inf
    for start in range(0, xs.size, 256):
        rows = xs[start : start + 256]
        X = np.stack(np.meshgrid(rows, xs, indexing="ij"), axis=-1).reshape(-1, 2)
        brute = max(brute, float(obj(X).max()))

    assert brute <= obj.f_star + 1e-9  # dominance: f_star must be at least as big as the grid found
    assert obj.f_star - brute <= 1e-5  # two-sided: catches an inflated f_star


# --- component_variances() under rotation (ruling R31) ---------------------------------------------


def test_component_variances_excludes_rotation_and_scale() -> None:
    # Ruling R31: component_variances()'s docstring promises pre-rotation Var_nu per block, not the
    # realized post-rotation axis variance (Labels.s_axis) or Var_nu f, which `scale` forces to
    # exactly 1 regardless. Shares deliberately do NOT sum to 1 pre-rotation here -- every other
    # fixture in this file uses shares that do, which is exactly why the brief's own note says the
    # old, wrong docstring's claim held "only by coincidence of this configuration": with an
    # unrescaled sum of 1.2, that coincidence cannot happen.
    f0 = _component(0, 0.6, 500)
    f1 = _component(1, 0.6, 501)
    obj = SyntheticObjective(2, (f0, f1), (), PairRotation(30.0, ((0, 1),)), None, 0)

    variances = obj.component_variances()
    assert variances == pytest.approx([f0.nu_var(), f1.nu_var()], abs=1e-10)
    assert float(variances.sum()) == pytest.approx(1.2, abs=1e-6)  # pre-rotation, un-rescaled
    assert not np.allclose(variances, obj.labels.s_axis)  # NOT the realized axis-aligned variance

    X = qmc.Sobol(2, scramble=True, seed=1).random(2**16)
    assert float(obj(X).var()) == pytest.approx(1.0, rel=0.02)  # the REAL Var_nu f, via scale


# --- R() bridged to forward -------------------------------------------------------------------------


@pytest.mark.parametrize("theta_deg", [15.0, 30.0])
def test_forward_matches_R_directly(theta_deg: float) -> None:
    # R() is normative per the brief but nothing in synthobj/ calls it -- forward computes cos/sin
    # itself rather than going through R(). Pin the two together directly so R() is load-bearing:
    # a future refactor that changes forward's formula without updating R() (or vice versa) is
    # caught here rather than by nothing at all. Non-symmetric angles, since at a multiple of 45 or
    # 90 several plausible-but-wrong formulas would coincidentally agree with the correct one.
    rot = PairRotation(theta_deg, ((0, 1),))
    x = np.array([0.2, 0.9])
    expected = 0.5 + rot.R() @ (x - 0.5)
    assert rot.forward(x) == pytest.approx(expected, abs=1e-14)
