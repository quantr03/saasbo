"""Tests for synthobj.objective: assembling components and interactions into one objective.

The objective under test is the plan's hand-built configuration (Task 5): D = 20, four main
effects at ell in {0.08, 0.5, 1.5, 3} with share 0.1875 each, and one product interaction on a
pair of those coordinates with s_ij = 0.25, so the variance budget is 4 x 0.1875 + 0.25 = 1.

Three kinds of assertion live here.

*Budget* tests separate the prescribed from the realized. `Labels.s` and `Labels.s_pairs` record
what was asked for (the component's `share` and the interaction's `c**2`), so
`test_shares_sum_to_one` only pins the bookkeeping. `component_variances` reads the realized
variance back by 64-node quadrature, and `test_total_variance_qmc` checks that quadrature against
a scrambled Sobol sample of the whole cube -- the one assertion here that would notice if the
components' variances did not actually add up under the product measure.

*Optimality* tests carry the plan's exact-`f_star` claim (decision D1). Pairs are disjoint, so
`f_star` separates into one 2-D block maximum plus two 1-D maxima. `test_f_star_dominates_...`
is the lower-bound half over 10^6 uniform points (evaluated in chunks per ruling R10: one
1e6 x 20 float64 array would be 160 MB), `test_f_star_attained` pins that the reported value
belongs to the reported `x_star`, and `test_f_star_matches_an_independent_brute_force` bounds it
from above with a plain grid scan that calls neither `Component.argmax` nor `block_argmax`. Per
Task 4's note, no test here assumes the maximizer is interior: three of four measured blocks put
it on the boundary of the unit square.

*Label* tests pin the ground truth every downstream identification metric is scored against.
`test_active_is_membership_in_S_not_a_cutoff` uses a second objective whose shares are all 0.01,
below `ACTIVE_EPS` = 0.02, so an implementation that defined `active` by the cutoff (the
dense-weak family's exact trap) fails it while every other test still passes.
"""
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

from synthobj.component import Component
from synthobj.draws import gp_draw
from synthobj.interaction import Interaction
from synthobj.kernel import eigen_factor, make_grid
from synthobj.objective import ACTIVE_EPS, Labels, SyntheticObjective
from synthobj.rotation import EXT_HI, EXT_LO, PairRotation

# The plan's hand-built objective: D = 20, four components, one pair, gamma = 0.25.
D = 20
N_GRID = 1024
COORDS: tuple[int, ...] = (2, 7, 11, 16)
ELLS: tuple[float, ...] = (0.08, 0.5, 1.5, 3.0)
MAIN_SHARE = 0.1875
S_IJ = 0.25
# The pair spans the roughest and a smooth main effect, so the block maximum is not a
# coordinate-wise one. Both coordinates carry a main effect, as every family the plan builds does.
PAIR: tuple[int, int] = (2, 11)
UNPAIRED: tuple[int, ...] = (7, 16)
# Coordinates with no component at all.
INACTIVE: tuple[int, ...] = (0, 5, 19)
SEED = 7

# Chunk size for the 10^6-point dominance sweep: 100k x 20 float64 is 16 MB (ruling R10).
CHUNK = 100_000


def _component(coord: int, ell: float, share: float, seed: int) -> Component:
    """One GP-draw component on the production grid; `eigen_factor` is cached by key."""
    raw = gp_draw(eigen_factor(0.0, 1.0, N_GRID, ell), np.random.default_rng(seed))
    return Component.from_raw(coord, ell, share, make_grid(0.0, 1.0, N_GRID), raw)


def _pieces() -> tuple[tuple[Component, ...], tuple[Interaction, ...]]:
    """The four main effects and the single interaction, each from its own independent draw."""
    mains = tuple(
        _component(coord, ell, MAIN_SHARE, 10 + k)
        for k, (coord, ell) in enumerate(zip(COORDS, ELLS))
    )
    u_i = _component(PAIR[0], 0.5, 1.0, 20)
    u_j = _component(PAIR[1], 0.5, 1.0, 21)
    return mains, (Interaction(PAIR[0], PAIR[1], float(np.sqrt(S_IJ)), u_i, u_j),)


@pytest.fixture(scope="module")
def pieces() -> tuple[tuple[Component, ...], tuple[Interaction, ...]]:
    return _pieces()


@pytest.fixture(scope="module")
def obj(pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]) -> SyntheticObjective:
    mains, inters = pieces
    return SyntheticObjective(D, mains, inters, None, None, SEED)


@pytest.fixture(scope="module")
def weak_obj() -> SyntheticObjective:
    """Two components whose shares sit below ACTIVE_EPS, as the dense-weak family's do."""
    mains = (_component(3, 3.0, 0.01, 30), _component(8, 3.0, 0.01, 31))
    return SyntheticObjective(D, mains, (), None, None, 0)


def _reference(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]], X: np.ndarray
) -> np.ndarray:
    """f(X) assembled from the public evaluators alone, sharing no code with SyntheticObjective."""
    mains, inters = pieces
    total = np.zeros(X.shape[0])
    for comp in mains:
        total += comp(X[:, comp.coord])
    for inter in inters:
        total += inter(X[:, inter.i], X[:, inter.j])
    return total


# --- the variance budget -------------------------------------------------------------------------


def test_shares_sum_to_one(obj: SyntheticObjective) -> None:
    labels = obj.labels
    # Bookkeeping only: s and s_pairs are the PRESCRIBED shares (`Component.share` and `c**2`),
    # so this catches a mis-placed or mis-scaled label, not a mis-scaled draw. The realized
    # variance is what test_component_variances_* and test_total_variance_qmc check.
    assert float(labels.s.sum()) + float(sum(labels.s_pairs)) == pytest.approx(1.0, abs=1e-12)
    assert labels.s_pairs == pytest.approx((S_IJ,), abs=1e-15)
    assert labels.gamma == pytest.approx(S_IJ, abs=1e-15)
    assert labels.pairs == (PAIR,)


def test_component_variances_are_the_realized_per_block_variances(
    obj: SyntheticObjective, pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    mains, inters = pieces
    variances = obj.component_variances()

    assert variances.shape == (len(mains) + len(inters),)
    # Quadrature of each block, recomputed here from the pieces: main effects first in the order
    # they were given, then the interactions.
    expected = np.array([comp.nu_var() for comp in mains] + [inter.nu_var() for inter in inters])
    assert variances == pytest.approx(expected, abs=1e-14)
    assert float(variances.sum()) == pytest.approx(1.0, abs=1e-8)


def test_component_variances_measure_rather_than_echo_the_share(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    # On a well-built component `nu_var()` and `share` are bit-identical -- that is the point of
    # `Component.from_raw` -- so the test above cannot tell a measurement from an echo. A
    # Component constructed field by field can: that is the save/load path (Task 8), where a
    # corrupted `share` in the npz is exactly the failure that must not pass silently.
    mains, inters = pieces
    comp = mains[0]
    mislabelled = Component(comp.coord, comp.ell, 0.5, comp.grid, comp.values)
    built = SyntheticObjective(D, (mislabelled,), (), None, None, 0)

    assert float(built.component_variances()[0]) == pytest.approx(MAIN_SHARE, abs=1e-12)
    assert built.labels.s[comp.coord] == 0.5

    # The same split for an interaction: `c**2` is its share only while both factors really are
    # unit variance under U[0,1]. Doubling one factor quadruples the block's variance while the
    # recorded s_pairs, which is c**2, does not move.
    inter = inters[0]
    loud = Component(inter.u_i.coord, inter.u_i.ell, 1.0, inter.u_i.grid, 2.0 * inter.u_i.values)
    with_loud = SyntheticObjective(
        D, (mains[0], mains[2]), (Interaction(inter.i, inter.j, inter.c, loud, inter.u_j),),
        None, None, 0,
    )
    assert float(with_loud.component_variances()[-1]) == pytest.approx(4.0 * S_IJ, abs=1e-12)
    assert with_loud.labels.s_pairs == pytest.approx((S_IJ,), abs=1e-15)


def test_total_variance_qmc(obj: SyntheticObjective) -> None:
    # The independent check that the per-block quadrature variances really add up to Var_nu f:
    # a scrambled Sobol sample of the whole cube, which knows nothing about the decomposition.
    X = qmc.Sobol(D, scramble=True, seed=0).random(2**17)
    values = obj(X)
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-2)
    assert float(values.var()) == pytest.approx(1.0, rel=0.01)


# --- f_star ----------------------------------------------------------------------------------------


def test_f_star_dominates_a_million_uniform_points(obj: SyntheticObjective) -> None:
    rng = np.random.default_rng(20260906)
    best = -np.inf
    for _ in range(1_000_000 // CHUNK):
        best = max(best, float(obj(rng.uniform(0.0, 1.0, size=(CHUNK, D))).max()))
    assert obj.f_star >= best - 1e-12


def test_f_star_attained(
    obj: SyntheticObjective, pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    x_star = obj.labels.x_star
    assert x_star.shape == (D,)
    assert np.all((x_star >= 0.0) & (x_star <= 1.0))
    assert np.all(x_star[list(INACTIVE)] == 0.5)

    # Independent recomputation through the public evaluators: pins that the reported value
    # belongs to the reported maximizer (ruling R19's postmortem), which dominance cannot see.
    assert float(_reference(pieces, x_star[None, :])[0]) == pytest.approx(obj.f_star, abs=1e-6)
    assert float(obj(x_star)) == pytest.approx(obj.f_star, abs=1e-6)
    assert obj.f_star == obj.labels.f_star


def test_f_star_matches_an_independent_brute_force(
    obj: SyntheticObjective, pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    # A plain grid scan that calls neither Component.argmax nor block_argmax. The paired block is
    # scanned jointly on a 1025 x 1025 uniform grid (spacing 9.8e-4, so the scan is within about
    # 1e-5 of the block maximum), the two unpaired components on 200001-point grids. It bounds
    # f_star from ABOVE, which the dominance test cannot: an assembly that maximized the paired
    # coordinates separately, or double-counted a component, lands well outside the 1e-3 window.
    mains, inters = pieces
    inter = inters[0]
    by_coord = {comp.coord: comp for comp in mains}

    z = np.linspace(0.0, 1.0, 1025)
    block = (
        by_coord[inter.i](z)[:, None]
        + by_coord[inter.j](z)[None, :]
        + inter(z[:, None], z[None, :])
    )
    brute = float(block.max())

    fine = np.linspace(0.0, 1.0, 200_001)
    for coord in UNPAIRED:
        brute += float(by_coord[coord](fine).max())

    assert brute <= obj.f_star + 1e-12
    assert obj.f_star - brute < 1e-3


# --- evaluation ------------------------------------------------------------------------------------


def test_call_is_the_sum_of_the_components_and_interactions(
    obj: SyntheticObjective, pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    X = np.random.default_rng(1).uniform(0.0, 1.0, size=(500, D))
    values = obj(X)
    assert values.shape == (500,)
    # Bounds rather than array_equal: the terms may legitimately be summed in any order, and
    # float addition is not associative, so a reordering shifts the last ulp. A dropped
    # component or a dropped interaction misses by O(0.4) here.
    assert np.max(np.abs(values - _reference(pieces, X))) < 1e-14


def test_a_single_point_returns_a_scalar_array(obj: SyntheticObjective) -> None:
    x = np.random.default_rng(2).uniform(0.0, 1.0, size=D)
    value = obj(x)
    assert isinstance(value, np.ndarray)
    assert value.shape == ()
    assert float(value) == pytest.approx(float(obj(x[None, :])[0]), abs=0.0)


def test_inactive_coordinates_do_not_change_f(obj: SyntheticObjective) -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(200, D))
    perturbed = X.copy()
    perturbed[:, list(INACTIVE)] = rng.uniform(0.0, 1.0, size=(200, len(INACTIVE)))

    assert np.array_equal(obj(X), obj(perturbed))
    # The perturbation is real, and an active coordinate does move f -- otherwise this test would
    # pass for an objective that ignored its input entirely.
    assert not np.array_equal(X, perturbed)
    moved = X.copy()
    moved[:, COORDS[0]] = rng.uniform(0.0, 1.0, size=200)
    assert not np.array_equal(obj(X), obj(moved))


@pytest.mark.parametrize("bad", [1.1, -0.5, 1.0 + 1e-6, -1e-6])
def test_call_rejects_points_outside_the_cube(obj: SyntheticObjective, bad: float) -> None:
    X = np.full((3, D), 0.5)
    X[1, COORDS[0]] = bad
    with pytest.raises(ValueError, match="outside the unit cube"):
        obj(X)


def test_call_accepts_points_inside_the_box_tolerance(obj: SyntheticObjective) -> None:
    # Within 1e-9 of a face the point is accepted and clipped onto it. Without the clip the
    # spline (extrapolate=False) would return NaN and f would be silently NaN rather than wrong.
    X = np.full((2, D), 0.5)
    X[0, COORDS[0]] = 1.0 + 1e-12
    X[1, COORDS[1]] = -1e-12
    values = obj(X)
    assert np.all(np.isfinite(values))

    on_face = np.full((2, D), 0.5)
    on_face[0, COORDS[0]] = 1.0
    on_face[1, COORDS[1]] = 0.0
    assert values == pytest.approx(obj(on_face), abs=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_call_rejects_non_finite_input(obj: SyntheticObjective, bad: float) -> None:
    X = np.full((3, D), 0.5)
    X[2, COORDS[1]] = bad
    with pytest.raises(ValueError, match="non-finite"):
        obj(X)


def test_an_empty_batch_returns_an_empty_result(obj: SyntheticObjective) -> None:
    # A zero-row batch is a legal request, not an error. It reaches the box test, where min()/max()
    # would raise "zero-size array to reduction operation minimum" -- an accurate message about
    # the wrong thing.
    empty = np.zeros((0, D))
    assert obj(empty).shape == (0,)
    assert obj.observe(empty, np.random.default_rng(0)).shape == (0,)


def test_call_rejects_the_wrong_number_of_coordinates(obj: SyntheticObjective) -> None:
    with pytest.raises(ValueError, match=re.escape(f"{D} coordinates")):
        obj(np.full((3, D + 1), 0.5))
    with pytest.raises(ValueError, match=re.escape(f"{D} coordinates")):
        obj(np.full(D - 1, 0.5))


# --- observe ---------------------------------------------------------------------------------------


def test_observe_adds_noise_of_the_right_sd(obj: SyntheticObjective) -> None:
    n = 100_000
    X = np.random.default_rng(4).uniform(0.0, 1.0, size=(n, D))
    residual = obj.observe(X, np.random.default_rng(5)) - obj(X)

    # The relative standard error of an sd estimate from n normals is 1/sqrt(2n) = 0.22 % here,
    # so the plan's 2 % band is a 9-sigma assertion and cannot fail by sampling luck. It misses
    # by a factor of 3 or more for every way of getting the scale wrong: one draw broadcast over
    # the whole call (sd 0), noise_sd squared (sd 0.01), noise_sd read as a variance (sd 0.316).
    assert float(residual.std()) == pytest.approx(0.1, rel=0.02)
    assert float(residual.mean()) == pytest.approx(0.0, abs=0.01)
    assert obj.noise_sd == 0.1


def test_observe_consumes_only_the_callers_generator(obj: SyntheticObjective) -> None:
    X = np.random.default_rng(6).uniform(0.0, 1.0, size=(64, D))

    assert np.array_equal(
        obj.observe(X, np.random.default_rng(11)), obj.observe(X, np.random.default_rng(11))
    )
    assert not np.array_equal(
        obj.observe(X, np.random.default_rng(11)), obj.observe(X, np.random.default_rng(12))
    )
    # A generator advanced by one call gives a different second draw: the objective holds no
    # stream of its own that could be re-seeded behind the caller's back.
    rng = np.random.default_rng(11)
    first, second = obj.observe(X, rng), obj.observe(X, rng)
    assert not np.array_equal(first, second)


def test_observe_keeps_the_shape_of_call(obj: SyntheticObjective) -> None:
    x = np.random.default_rng(7).uniform(0.0, 1.0, size=D)
    value = obj.observe(x, np.random.default_rng(0))
    assert isinstance(value, np.ndarray)
    assert value.shape == ()


# --- labels ------------------------------------------------------------------------------------------


def test_labels_record_the_shares_and_lengthscales(obj: SyntheticObjective) -> None:
    labels = obj.labels
    assert labels.D == D
    assert labels.S == tuple(sorted(COORDS))
    assert labels.seed == SEED

    expected_s = np.zeros(D)
    expected_s[list(COORDS)] = MAIN_SHARE
    assert np.array_equal(labels.s, expected_s)

    expected_ell = np.full(D, np.nan)
    expected_ell[list(COORDS)] = ELLS
    assert np.array_equal(labels.ell, expected_ell, equal_nan=True)
    assert np.all(np.isnan(labels.ell[list(INACTIVE)]))


def test_slope_shares_normalize_over_S(
    obj: SyntheticObjective, pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    mains, _ = pieces
    g = obj.labels.g
    S = list(obj.labels.S)

    assert g.shape == (D,)
    assert float(g[S].sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(g[list(INACTIVE)] == 0.0)
    assert float(g.sum()) == pytest.approx(1.0, abs=1e-12)

    # Independently recomputed from the components' own slope energies: g is E_nu[(f_i')^2]
    # normalized over S, and the interaction factors do NOT contribute (they are unit-variance
    # draws at ell = 0.5 whose slope energy would swamp the ell = 3 component if included).
    energies = np.zeros(D)
    for comp in mains:
        energies[comp.coord] = comp.slope_energy()
    assert g == pytest.approx(energies / energies.sum(), abs=1e-12)

    # What makes g a different label from s: the four shares are equal, so a g that merely echoed
    # s would be flat at 0.25. The Matern slope factor 5/(3 ell^2 v(ell)) is 316.8 at ell = 0.08
    # against 23.6, 13.8 and 12.5 at the other three, so the rough coordinate carries the bulk of
    # the slope energy. Measured on these four draws: energies 64.74, 18.33, 2.44, 2.99, i.e.
    # g = (0.7315, 0.2072, 0.0275, 0.0338) and a 3.5x realized margin, well short of the 13x the
    # expectations suggest -- realized slope energy at smooth lengthscales scatters widely. For
    # the same reason the ell = 1.5 against ell = 3 ordering is NOT asserted: their factors differ
    # by 10 % and this seed inverts them.
    assert int(np.argmax(g)) == COORDS[0]
    assert g[COORDS[0]] > 0.5
    assert not np.allclose(g[S], 0.25)


def test_active_is_membership_in_S_not_a_cutoff(weak_obj: SyntheticObjective) -> None:
    labels = weak_obj.labels
    assert ACTIVE_EPS == 0.02
    assert labels.active_eps == ACTIVE_EPS

    # Both shares are 0.01, below ACTIVE_EPS -- the dense-weak family's design. `active` marks
    # membership in S all the same; only downstream posterior scoring uses the cutoff.
    assert labels.S == (3, 8)
    assert np.all(labels.s[[3, 8]] < ACTIVE_EPS)
    assert labels.active.dtype == np.dtype(bool)
    assert labels.active.shape == (D,)
    assert list(np.flatnonzero(labels.active)) == [3, 8]


def test_labels_active_matches_S_on_the_main_objective(obj: SyntheticObjective) -> None:
    expected = np.zeros(D, dtype=bool)
    expected[list(COORDS)] = True
    assert np.array_equal(obj.labels.active, expected)


def test_labels_carry_the_no_rotation_defaults(obj: SyntheticObjective) -> None:
    labels = obj.labels
    assert labels.rotation_deg == 0.0
    assert labels.rotation_pairs == ()
    assert labels.scale == 1.0
    assert np.array_equal(labels.s_axis, labels.s)
    assert labels.gamma_axis == pytest.approx(labels.gamma, abs=1e-15)
    assert float(labels.s_axis.sum()) + labels.gamma_axis == pytest.approx(1.0, abs=1e-12)


def test_labels_is_frozen_and_compares_by_identity(obj: SyntheticObjective) -> None:
    # Ruling R4 and the branch convention: a frozen dataclass holding ndarrays is eq=False,
    # because the generated __eq__ compares the fields as a tuple and raises on an array.
    assert Labels.__eq__ is object.__eq__
    assert obj.labels == obj.labels
    with pytest.raises(FrozenInstanceError):
        obj.labels.f_star = 0.0


# --- spec, rotation and the assembly guards ------------------------------------------------------


def test_spec_none_gives_the_custom_defaults(obj: SyntheticObjective) -> None:
    assert obj.spec is None
    assert obj.labels.family == "custom"
    assert obj.labels.noise_sd == 0.1
    assert obj.noise_sd == 0.1
    assert obj.labels.generator == "matern"
    assert obj.labels.monotone is False


def test_spec_fields_are_read_when_a_spec_is_given(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    class _Spec:  # a stand-in for families.FamilySpec, which does not exist yet (ruling R15)
        name = "dense_weak"
        noise_sd = 0.25
        generator = "sinusoid"
        monotone = True

    mains, inters = pieces
    built = SyntheticObjective(D, mains, inters, None, _Spec(), 3)
    assert built.noise_sd == 0.25
    assert built.labels.noise_sd == 0.25
    assert built.labels.family == "dense_weak"
    assert built.labels.generator == "sinusoid"
    assert built.labels.monotone is True


@pytest.mark.parametrize("missing", ["name", "noise_sd", "generator", "monotone"])
def test_a_present_spec_missing_a_field_raises(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]], missing: str
) -> None:
    # Ruling R27. The test above can only confirm that objective.py reads the names objective.py
    # chose, since `FamilySpec` does not exist yet -- it is circular with respect to a rename.
    # This one is not: the four defaults apply to `spec=None` alone, so a spec that calls its
    # noise level `noise_sigma` fails loudly at construction instead of recording noise_sd = 0.1
    # on every regret curve and matern/False on two study variants, with `family` still correct
    # and the record therefore looking plausible.
    fields = {"name": "aligned10_sin", "noise_sd": 0.25, "generator": "sinusoid", "monotone": True}
    del fields[missing]
    partial = type("_PartialSpec", (), fields)()

    with pytest.raises(AttributeError, match=missing):
        SyntheticObjective(D, (pieces[0][0],), (), None, partial, 0)


def test_family_spec_annotation_is_deferred(obj: SyntheticObjective, repo_root: Path) -> None:
    # Ruling R15: families.py imports objective.py, so objective.py must not import families.py
    # at runtime. `from __future__ import annotations` keeps the annotation as a string, and the
    # import sits behind TYPE_CHECKING.
    assert SyntheticObjective.__init__.__annotations__["spec"] == "FamilySpec | None"

    # A hermetic subprocess, not this process's `sys.modules`: pytest collects every test file
    # (importing each one) before running any test, and `tests/test_families.py` necessarily
    # imports `synthobj.families` -- it is the module under test -- so by the time this assertion
    # would run in-process, `sys.modules` already carries that entry regardless of whether
    # `objective.py` itself ever imports it (fix-round finding: a mutant that appends a real
    # `import synthobj.families` to `objective.py` left the in-process check unaffected). A fresh
    # interpreter that imports only `synthobj.objective` is the only way to ask the question this
    # test is actually about.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys, synthobj.objective; print("synthobj.families" in sys.modules)',
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_rotation_is_accepted(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    # Task 6 implements what task 5 rejected. A rotated pair must live on the extended domain
    # (ruling R7): reusing an ordinary [0,1] main effect here would send z outside its knot range at
    # a real rotation angle, which is not what this guard test is about. Coordinates 0 and 5 are
    # otherwise inactive (see INACTIVE), so a small extended-domain pair there is disjoint from
    # both PAIR and UNPAIRED; the deep rotation math lives in test_rotation.py -- this only pins
    # that SyntheticObjective's guard now lets a real PairRotation through.
    #
    # `inters` (a real Interaction on PAIR) is deliberately left out here (fix-round finding,
    # ruling R39): combining it with this rotation is a separate, rejected configuration -- see
    # test_rotation_combined_with_an_interaction_is_rejected below.
    mains, inters = pieces
    grid = make_grid(EXT_LO, EXT_HI, 256)
    factor = eigen_factor(EXT_LO, EXT_HI, 256, 0.5)
    f0 = Component.from_raw(0, 0.5, 0.25, grid, gp_draw(factor, np.random.default_rng(50)))
    f5 = Component.from_raw(5, 0.5, 0.25, grid, gp_draw(factor, np.random.default_rng(51)))
    rotation = PairRotation(30.0, ((0, 5),))

    built = SyntheticObjective(D, mains + (f0, f5), (), rotation, None, 0)
    assert built.rotation is rotation
    assert built.labels.rotation_deg == 30.0
    assert built.labels.rotation_pairs == ((0, 5),)
    assert np.isfinite(built.f_star)


def test_rotation_combined_with_an_interaction_is_rejected(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    """Ruling R39: this module's own `inters` fixture (a real Interaction on `PAIR`) combined with
    a rotation on a disjoint pair is rejected at construction, not silently assembled with an
    undefined `s_axis` for the interaction-paired coordinates -- see test_rotation.py's
    `test_rotation_and_interaction_together_is_rejected_at_construction` for the full
    label-corruption demonstration this guards against."""
    mains, inters = pieces
    grid = make_grid(EXT_LO, EXT_HI, 256)
    factor = eigen_factor(EXT_LO, EXT_HI, 256, 0.5)
    f0 = Component.from_raw(0, 0.5, 0.25, grid, gp_draw(factor, np.random.default_rng(50)))
    f5 = Component.from_raw(5, 0.5, 0.25, grid, gp_draw(factor, np.random.default_rng(51)))
    rotation = PairRotation(30.0, ((0, 5),))

    with pytest.raises(ValueError, match="rotation"):
        SyntheticObjective(D, mains + (f0, f5), inters, rotation, None, 0)


def test_a_non_pairrotation_object_is_rejected(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    mains, inters = pieces
    with pytest.raises(TypeError, match="PairRotation"):
        SyntheticObjective(D, mains, inters, object(), None, 0)


def test_overlapping_pairs_are_rejected(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    mains, inters = pieces
    inter = inters[0]
    # A second pair reusing coordinate PAIR[0]: the blocks are no longer independent, so the
    # per-block maxima no longer sum to f_star and the assembly rule (decision D1) breaks.
    overlapping = (inter, Interaction(PAIR[0], UNPAIRED[0], 0.1, inter.u_i, inter.u_j))

    with pytest.raises(ValueError, match="disjoint"):
        SyntheticObjective(D, mains, overlapping, None, None, 0)
    with pytest.raises(NotImplementedError, match="overlapping"):
        SyntheticObjective(D, mains, overlapping, None, None, 0, allow_overlap=True)

    # allow_overlap is inert when the pairs really are disjoint.
    assert SyntheticObjective(D, mains, inters, None, None, 0, allow_overlap=True).f_star > 0.0


def test_duplicate_component_coordinates_are_rejected(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    # Two main effects on one coordinate would be summed by __call__ but only one of them would
    # reach the f_star assembly, and max(f + g) != max f + max g, so f_star would stop being the
    # maximum. Rejected rather than silently mis-labelled.
    mains, inters = pieces
    with pytest.raises(ValueError, match="one component per coordinate"):
        SyntheticObjective(D, mains + (mains[0],), inters, None, None, 0)


def test_a_paired_coordinate_needs_a_main_effect(
    pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    # The block maximum is over f_i(x) + f_j(y) + h_ij(x, y): drop f_j and there is no block to
    # maximize. Without the guard this is a bare `KeyError: 11` from the assembly's dict lookup.
    mains, inters = pieces
    with pytest.raises(ValueError, match="main effect"):
        SyntheticObjective(D, (mains[0],), inters, None, None, 0)


def test_an_objective_needs_at_least_one_component() -> None:
    # Otherwise `g` is 0/0, which under -W error fails as a RuntimeWarning from inside the label
    # assembly rather than as a statement about the input.
    with pytest.raises(ValueError, match="at least one component"):
        SyntheticObjective(D, (), (), None, None, 0)


def test_precomputed_labels_are_used_verbatim(
    obj: SyntheticObjective, pieces: tuple[tuple[Component, ...], tuple[Interaction, ...]]
) -> None:
    # The save/load path (Task 8) passes stored labels and must not recompute f_star. A sentinel
    # value no assembly would produce shows the stored record is passed through untouched.
    mains, inters = pieces
    stored = replace(obj.labels, f_star=-123.5)
    rebuilt = SyntheticObjective(D, mains, inters, None, None, SEED, labels=stored)

    assert rebuilt.labels is stored
    assert rebuilt.f_star == -123.5
