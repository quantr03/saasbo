"""Tests for synthobj.families: the study's 14-variant registry and the family table it encodes.

Three kinds of assertion live here.

*Table* tests check that each `FAMILIES` entry matches the family table published in the task
brief -- not a tautology against the spec's own fields, but a comparison of the *realized*,
per-coordinate `Labels` (after `S` selection, rank-to-coordinate mapping and `Component`
construction) against the brief's independently stated (share, ell) values. A rank-shuffling or
share/ell-resolution bug would show up here even though it would not show up in a naive
`obj.spec.shares == FAMILIES[name]().shares` check, which tests nothing (the "no power" trap the
brief warns about).

*Statistical design* tests cover the two families whose defining property only shows up in
aggregate over draws: `decoupled`'s cell ordering (10/10 seeds, wide margin) and `anti_aligned`'s
reversed variance ranking (ruling R13 replaces the plan's flaky per-seed Spearman test with a
deterministic design check plus a pooled, rank-aligned realized check). Both are `slow` per the
brief's instruction, even though neither takes anywhere near the pytest.ini 10 s threshold on this
machine -- the brief asks for the marker regardless of measured cost.

*Sanity* tests spot-check a handful of properties from the brief's "Measured before re-dispatch"
table on the family layer itself (as opposed to the hand-built pieces `test_objective.py` and
`test_rotation.py` already cover): `Var_nu f` and `E_nu f` by Sobol QMC, and that `gamma` stays 0
for a rotated family (no explicit `Interaction`) while `gamma_axis` captures the genuine
axis-misalignment remainder.

No test here builds `dense_weak` in a way that conflates `Labels.active` (membership in S) with
`ACTIVE_EPS` (ruling R29): `dense_weak` clears the cutoff at D=20 and falls under it at D=100, but
`active` is all-True at both, since the family always gives every coordinate a nonzero share.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc, spearmanr

from synthobj.families import FAMILIES, STUDY_GRID, FamilySpec, build, make_family
from synthobj.kernel import v
from synthobj.objective import ACTIVE_EPS


# --- the family table (task-7 brief, "final" values) -----------------------------------------

FAMILY_TABLE: dict[str, dict] = {
    "aligned3": dict(n_active=3, shares=(1 / 3,) * 3, ells=(0.5,) * 3, gamma=0.0),
    "aligned10": dict(n_active=10, shares=(0.1,) * 10, ells=(0.5,) * 10, gamma=0.0),
    "aligned10_sin": dict(n_active=10, shares=(0.1,) * 10, ells=(0.5,) * 10, gamma=0.0),
    "decoupled": dict(
        n_active=8,
        shares=(0.21, 0.21, 0.21, 0.21, 0.04, 0.04, 0.04, 0.04),
        ells=(1.5, 1.5, 0.08, 0.08, 1.5, 1.5, 0.08, 0.08),
        gamma=0.0,
    ),
    "anti_aligned": dict(
        n_active=6,
        shares=(0.35, 0.25, 0.18, 0.12, 0.07, 0.03),
        ells=(3.0, 0.5, 0.27, 0.18, 0.11, 0.06),
        gamma=0.0,
    ),
    "interaction_g0.00": dict(n_active=5, shares=(0.20,) * 5, ells=(0.5,) * 5, gamma=0.00, n_pairs=2),
    "interaction_g0.10": dict(n_active=5, shares=(0.18,) * 5, ells=(0.5,) * 5, gamma=0.10, n_pairs=2),
    "interaction_g0.25": dict(n_active=5, shares=(0.15,) * 5, ells=(0.5,) * 5, gamma=0.25, n_pairs=2),
    "interaction_g0.50": dict(n_active=5, shares=(0.10,) * 5, ells=(0.5,) * 5, gamma=0.50, n_pairs=2),
    "interaction_g0.75": dict(n_active=5, shares=(0.05,) * 5, ells=(0.5,) * 5, gamma=0.75, n_pairs=2),
    "rotated_t0": dict(n_active=10, shares=(0.1,) * 10, ells=(0.5,) * 10, gamma=0.0, n_rot_pairs=5),
    "rotated_t15": dict(n_active=10, shares=(0.1,) * 10, ells=(0.5,) * 10, gamma=0.0, n_rot_pairs=5),
    "rotated_t45": dict(n_active=10, shares=(0.1,) * 10, ells=(0.5,) * 10, gamma=0.0, n_rot_pairs=5),
}

ANTI_ALIGNED_SHARES = (0.35, 0.25, 0.18, 0.12, 0.07, 0.03)
ANTI_ALIGNED_ELLS = (3.0, 0.5, 0.27, 0.18, 0.11, 0.06)


def test_study_grid_has_14_variants() -> None:
    assert len(STUDY_GRID) == 14
    assert len(FAMILIES) == 14
    names = [name for name, _, _ in STUDY_GRID]
    assert len(set(names)) == 14
    assert set(names) == set(FAMILIES) == set(FAMILY_TABLE) | {"dense_weak"}


@pytest.mark.parametrize("name", sorted(FAMILY_TABLE))
def test_variant_matches_family_table_at_D20(name: str) -> None:
    """Every non-dense variant builds at D=20 with the table's |S| and (share, ell) multiset."""
    expected = FAMILY_TABLE[name]
    obj = make_family(name, seed=0, D=20)
    labels = obj.labels
    S = list(labels.S)

    assert len(S) == expected["n_active"]

    real_shares, real_ells = zip(*sorted(zip(labels.s[S].tolist(), labels.ell[S].tolist())))
    want_shares, want_ells = zip(*sorted(zip(expected["shares"], expected["ells"])))
    assert np.allclose(real_shares, want_shares, atol=1e-9)
    assert np.allclose(real_ells, want_ells, atol=1e-9)

    assert sum(labels.s[S]) == pytest.approx(1.0 - expected["gamma"], abs=1e-9)
    assert labels.gamma == pytest.approx(expected["gamma"], abs=1e-9)

    if "n_pairs" in expected:
        assert len(labels.pairs) == expected["n_pairs"]
    if "n_rot_pairs" in expected:
        assert len(labels.rotation_pairs) == expected["n_rot_pairs"]
        assert labels.pairs == ()  # rotated pairs are a change of basis, not an Interaction


def test_family_flags_match_the_table_for_variants_that_deviate_from_defaults() -> None:
    """aligned10_sin is the one sinusoid family; dense_weak is the one monotone family."""
    assert make_family("aligned10_sin", seed=0, D=20).labels.generator == "sinusoid"
    assert make_family("aligned10", seed=0, D=20).labels.generator == "matern"
    assert make_family("dense_weak", seed=0, D=20).labels.monotone is True
    assert make_family("aligned10", seed=0, D=20).labels.monotone is False


def test_aligned10_sin_actually_draws_from_the_sinusoid_generator() -> None:
    """`labels.generator == "sinusoid"` alone doesn't prove the draw came from it: `aligned10` and
    `aligned10_sin` share S, shares, ells and the entire `main` stream at a given seed, so if
    `_draw_fn` silently fell back to `gp_draw` for "sinusoid" too, the two families would be
    bit-identical while every one of `aligned10_sin`'s 10 files still claimed the sinusoid label
    (fix-round finding: exactly the R27 silent-mislabel failure mode, one layer down). Comparing
    the actual drawn values is independent of, and stronger than, the label check above."""
    matern = make_family("aligned10", 0, D=20).components[0].values
    sinusoid = make_family("aligned10_sin", 0, D=20).components[0].values
    assert not np.array_equal(matern, sinusoid)


@pytest.mark.parametrize("name", sorted(FAMILY_TABLE))
def test_sparse_variant_clears_active_eps(name: str) -> None:
    """Every sparse variant's smallest active share clears ACTIVE_EPS (dense_weak is excluded:
    it is not sparse, and its share depends on D -- see the dedicated D=20/D=100 tests below,
    ruling R29)."""
    obj = make_family(name, seed=0, D=20)
    S = list(obj.labels.S)
    assert min(obj.labels.s[S]) >= ACTIVE_EPS


def test_dense_weak_at_D20_has_20_components_of_share_1_over_20() -> None:
    """R29: 1/D = 0.05 at D=20, two and a half times *above* ACTIVE_EPS -- assert only the
    build shape here, nothing about the cutoff."""
    obj = make_family("dense_weak", seed=0, D=20)
    labels = obj.labels
    assert len(labels.S) == 20
    assert labels.active.all()
    assert np.allclose(labels.s[list(labels.S)], 1 / 20)


def test_dense_weak_at_D100_share_clears_D20_but_not_active_eps() -> None:
    """R29's actual claim: at D=100 the *share* 1/D = 0.01 falls under ACTIVE_EPS, but `active`
    -- membership in S, not a comparison against the cutoff -- stays all-True regardless."""
    obj = make_family("dense_weak", seed=0, D=100)
    labels = obj.labels
    assert len(labels.S) == 100
    assert np.allclose(labels.s[list(labels.S)], 0.01)
    assert 0.01 < ACTIVE_EPS
    assert labels.active.all()


def test_dense_weak_all_components_monotone() -> None:
    """Derivative sign is constant across 10**4 points on the leading uniform-grid portion."""
    obj = make_family("dense_weak", seed=0, D=20)
    xs = np.linspace(0.0, 1.0, 10_000)
    for comp in obj.components:
        d = comp.derivative(xs)
        assert np.all(d >= 0.0) or np.all(d <= 0.0)


def test_dense_weak_sign_is_not_collapsed_to_one_direction() -> None:
    """Both signs occur among the 20 components at D=20 (measured 10 up / 10 down at seed 0).

    The monotonicity test above accepts either direction by design, so it alone cannot catch a
    collapsed `sign = 1.0` (fix-round finding): that would make every coordinate increasing, put
    `x_star` on the all-ones corner, and turn a 10-file study family into a systematically easier
    problem than intended. This checks the actual realized direction of each component, not an
    internal "sign" attribute -- independent of how `_draw_component` happens to implement it."""
    obj = make_family("dense_weak", seed=0, D=20)
    xs = np.linspace(0.0, 1.0, 10_000)
    increasing = decreasing = False
    for comp in obj.components:
        d = comp.derivative(xs)
        if np.all(d >= 0.0):
            increasing = True
        if np.all(d <= 0.0):
            decreasing = True
    assert increasing and decreasing


def test_rotated_variants_share_S_and_pairs_across_theta() -> None:
    r0 = make_family("rotated_t0", seed=3, D=20)
    r45 = make_family("rotated_t45", seed=3, D=20)
    assert r0.labels.S == r45.labels.S
    assert r0.labels.rotation_pairs == r45.labels.rotation_pairs


def test_make_family_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        make_family("nope", 0)


def test_make_family_unknown_override_raises_type_error() -> None:
    with pytest.raises(TypeError):
        make_family("aligned3", 0, bogus=1)


def test_mismatched_n_active_override_raises_instead_of_silently_truncating() -> None:
    """`decoupled`'s registered `shares`/`ells` have 8 entries; overriding `n_active` alone
    (leaving the 8-entry tuples in place) used to zip the first 4 shares against the first 4 ells
    and build a real objective with `sum(labels.s[S]) == 0.84`, no error and no label recording
    the mistake (fix-round finding: `make_family("decoupled", 0, D=20, n_active=4)`)."""
    with pytest.raises(ValueError, match="shares has 8 entries"):
        make_family("decoupled", 0, D=20, n_active=4)


def test_shares_not_summing_to_one_minus_gamma_raises() -> None:
    bad = FamilySpec(name="bad", D=20, n_active=3, shares=(0.5, 0.3, 0.1), ells=0.5, gamma=0.0)
    with pytest.raises(ValueError, match="must equal 1"):
        build(bad, 0)


# --- decoupled's cell ordering (statistical design, 10/10 seeds) -------------------------------


def test_decoupled_theoretical_slope_energy_matches_the_closed_form() -> None:
    """5/(3 ell^2 v(ell)) = 316.81 at ell=0.08 and 13.80 at ell=1.5 (verified before dispatch);
    these are what the 66.5/12.7/2.90/0.552 cell figures below are derived from."""
    assert 5.0 / (3.0 * 0.08**2 * v(0.08)) == pytest.approx(316.81, abs=0.01)
    assert 5.0 / (3.0 * 1.5**2 * v(1.5)) == pytest.approx(13.80, abs=0.01)


@pytest.mark.slow
def test_decoupled_argmax_is_strong_rough_and_argmin_is_weak_smooth_over_seeds() -> None:
    """10/10 seeds at D=100: g's argmax sits in the strong-rough cell (share 0.21, ell 0.08) and
    its argmin in weak-smooth (share 0.04, ell 1.5), with a wide margin (brief's measured cell
    slope shares ~0.39/0.09/0.02/0.005). `g[S]` is read via `list(labels.S)` (sorted by
    coordinate), never assumed to follow design rank order -- classification below reads each
    coordinate's own (s, ell) off `Labels`, not its position."""
    for seed in range(10):
        obj = build(FAMILIES["decoupled"](D=100), seed)
        labels = obj.labels
        S = list(labels.S)
        g_S = labels.g[S]
        argmax_coord = S[int(np.argmax(g_S))]
        argmin_coord = S[int(np.argmin(g_S))]
        assert labels.s[argmax_coord] == pytest.approx(0.21)
        assert labels.ell[argmax_coord] == pytest.approx(0.08)
        assert labels.s[argmin_coord] == pytest.approx(0.04)
        assert labels.ell[argmin_coord] == pytest.approx(1.5)


# --- anti_aligned's reversed variance ranking (ruling R13) --------------------------------------


def test_anti_aligned_theoretical_slope_shares_reverse_the_variance_ranking() -> None:
    """Design assertion (R13), deterministic and never flaky: the lengthscales were chosen so the
    *theoretical* slope shares exactly reverse the share ranking -- Spearman == -1.0 exactly."""
    raw = [s * 5.0 / (3.0 * ell**2 * v(ell)) for s, ell in zip(ANTI_ALIGNED_SHARES, ANTI_ALIGNED_ELLS)]
    theoretical = [r / sum(raw) for r in raw]
    assert theoretical == pytest.approx([0.0770, 0.1037, 0.1441, 0.1706, 0.2223, 0.2823], abs=1e-4)
    assert spearmanr(theoretical, ANTI_ALIGNED_SHARES).correlation == pytest.approx(-1.0)


@pytest.mark.slow
def test_anti_aligned_realized_slope_share_pooled_over_seeds_is_reversed() -> None:
    """Realized assertion (R13), pooled over the study's own seeds 0-9. A single draw's slope
    energy is high-variance (R13 struck the per-seed rule for exactly this reason), so this pools
    `g` across seeds instead -- but *by rank*, never by coordinate: `Labels.g[S]` is ordered by
    sorted coordinate index and every seed selects a different S, so averaging by position would
    average incomparable quantities (the brief's rank-alignment trap). Sorting each seed's own
    `s[S]` descending recovers the design vector exactly, which doubles as a free correctness
    check on the alignment itself."""
    rank_aligned = []
    for seed in range(10):
        obj = build(FAMILIES["anti_aligned"](D=100), seed)
        labels = obj.labels
        S = list(labels.S)
        s_S, g_S = labels.s[S], labels.g[S]
        order = np.argsort(-s_S)
        s_sorted, g_sorted = s_S[order], g_S[order]
        assert s_sorted == pytest.approx(ANTI_ALIGNED_SHARES)
        rank_aligned.append(g_sorted)

    mean_g = np.mean(rank_aligned, axis=0)
    rho = spearmanr(mean_g, ANTI_ALIGNED_SHARES).correlation
    assert rho <= -0.7


# --- variance budget and axis-vs-rotation bookkeeping on assembled families --------------------


@pytest.mark.parametrize(
    "name", ["aligned3", "decoupled", "anti_aligned", "interaction_g0.50", "rotated_t45", "dense_weak"]
)
def test_var_and_mean_under_nu_hold_for_representative_variants(name: str) -> None:
    """Var_nu f == 1 and E_nu f == 0 under the reference measure, by a 2**15-point scrambled
    Sobol QMC estimate independent of every internal quadrature the objective itself uses.
    Tolerances are looser than the brief's single measured run (which used the same Sobol
    construction and landed within 0.2% / 2.2e-4) to allow for QMC sample variance across
    environments while still catching a real scale or centering regression, which would be
    orders of magnitude larger."""
    obj = make_family(name, seed=0, D=100)
    sampler = qmc.Sobol(d=100, scramble=True, seed=0)
    X = sampler.random_base2(m=15)
    f = obj(X)
    assert np.var(f) == pytest.approx(1.0, rel=0.01)
    assert abs(np.mean(f)) < 1e-3


@pytest.mark.parametrize("name", ["aligned3", "decoupled", "anti_aligned", "interaction_g0.50", "dense_weak"])
def test_component_means_sum_to_exactly_zero_for_non_rotated_variants(name: str) -> None:
    """Each `Component.from_raw` centers to E_nu[f_i] == 0 to quadrature roundoff, so the sum
    across all of a non-rotated variant's components is exact to ~1e-12, not merely close (measured
    1e-17 to 1e-18). `rotated_t45` is deliberately excluded: rotation moves the mean entirely into
    `mu` (ruling R30), and a component's own `nu_mean()` stays ~0 regardless (measured 5.0e-17 for
    rotated_t45 too) -- summing components can never see a dropped `mu` for a rotated family, which
    is exactly why the Sobol test above is kept at its 1e-3 tolerance rather than tightened here."""
    obj = make_family(name, seed=0, D=100)
    total = sum(c.nu_mean() for c in obj.components)
    assert total == pytest.approx(0.0, abs=1e-12)


def test_gamma_and_gamma_axis_are_distinct() -> None:
    """`gamma` is 0 unless the family has an explicit `Interaction`; `gamma_axis` is the
    axis-aligned remainder a rotation leaves behind even with none. interaction_g0.50 has a real
    Interaction (gamma == gamma_axis == 0.5); rotated_t45 has none (gamma == 0) but its rotation
    still leaves a nonzero axis remainder a first-order additive model cannot see."""
    inter = make_family("interaction_g0.50", seed=0, D=100)
    rotated = make_family("rotated_t45", seed=0, D=100)
    assert inter.labels.gamma == pytest.approx(0.5)
    assert inter.labels.gamma_axis == pytest.approx(0.5)
    assert rotated.labels.gamma == pytest.approx(0.0)
    assert rotated.labels.gamma_axis == pytest.approx(0.3457, abs=1e-3)
