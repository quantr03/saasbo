"""Tests for sagp.gp's posterior readouts: Sobol indices, shares and the two active rules.

The thesis's primary identification readout is the first-order Sobol index of the posterior mean
under the reference measure, and it is the one number the four cells are compared on, so it has to
mean the same thing in each. Three of the four have it in closed form -- every centered component
integrates to *exactly* zero against the 64-node rule that defines nu -- and the fourth needs a
Saltelli estimator. What is pinned here is therefore: that each closed form is the quantity it
claims to be, checked against a brute-force tensor-grid quadrature of the very posterior mean
`FittedGP.posterior` returns; that the estimator agrees with the closed form where both apply; and
that on a synthetic objective whose first-order shares are known the index recovers them, on and
off the active set. The shares and the two active rules are checked against arithmetic, not
against themselves.

Every test that does not need a sampler builds its `FittedGP` from hand-made draws (ruling R7):
these are tests of the readout formulas, and a NUTS fit inside them would be slow and would leave
a failure ambiguous between the sampler and the code under test.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
import sagp.gp as gp
import sagp.readouts as readouts
from sagp.gp import ACTIVE_EPS, GL_NODES, GL_WEIGHTS, FittedGP, NUTSConfig

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import qmc, spearmanr

from synthobj import kernel
from synthobj.families import make_family, noise_rng

NODES = np.asarray(GL_NODES)
WEIGHTS = np.asarray(GL_WEIGHTS)


def _fitted(cell, X, y, samples, *, fixed_noise=None, active=None) -> FittedGP:
    """A `FittedGP` with the fit-status fields a readout test does not care about filled in."""
    return FittedGP(
        cell=cell,
        X_train=jnp.asarray(X),
        Y_train=jnp.asarray(y),
        samples={site: jnp.asarray(draws) for site, draws in samples.items()},
        fixed_noise=fixed_noise,
        active=active,
        status="ok",
        status_reason="",
        attempts=(),
    )


def _brute_force_sobol(fitted, D) -> tuple[np.ndarray, float]:
    """First-order indices and total variance of the posterior mean by 64^D tensor quadrature.

    The independent construction the closed forms are held to: it asks `posterior` for the
    sample-averaged mean at every point of the tensor grid the reference measure is defined on and
    integrates it directly, with no appeal to the kernel's structure. Only usable at D = 2 (4096
    points); that is exactly why the closed forms exist.
    """
    axes = np.meshgrid(*([NODES] * D), indexing="ij")
    points = np.stack([axis.ravel() for axis in axes], axis=1)
    mean = np.asarray(fitted.posterior(points)[0].mean(axis=0)).reshape([NODES.size] * D)

    weights = WEIGHTS
    for _ in range(D - 1):
        weights = weights[..., None] * WEIGHTS
    total_var = (weights * mean * mean).sum() - ((weights * mean).sum()) ** 2

    indices = []
    for i in range(D):
        marginal = mean
        for j in sorted((j for j in range(D) if j != i), reverse=True):
            marginal = np.tensordot(marginal, WEIGHTS, axes=([j], [0]))
        variance = (WEIGHTS * marginal * marginal).sum() - ((WEIGHTS * marginal).sum()) ** 2
        indices.append(variance / total_var)
    return np.array(indices), total_var


def test_exact_product_amplitude_matches_brute_force():
    # The product kernel's closed form (plan D6) reads nothing like the integral it computes: it
    # collapses E_nu[m_s m_s'] into a product over coordinates of 64-node contractions, over the
    # S(S+1)/2 sample pairs. Small enough here that the integral itself can be done by tensor
    # quadrature, so the two are the same number by construction and must agree to roundoff.
    D, n = 2, 3
    rng = np.random.default_rng(21)
    samples = {
        # Two samples, deliberately: the cross term s != s' is where the closed form differs most
        # from the sum of per-sample variances a single draw could not tell apart.
        "a_sq": np.array([[0.9, 0.2], [1.4, 0.05]]),
        "kernel_ell": np.array([[0.4, 0.8], [0.6, 0.3]]),
        "kernel_noise": np.array([0.02, 0.05]),
    }
    fitted = _fitted(
        ("product", "amplitude"), rng.uniform(0.0, 1.0, (n, D)), rng.normal(size=n), samples
    )

    sobol_hat, total_var = readouts._sobol_exact_product_amplitude(fitted)
    brute_sobol, brute_var = _brute_force_sobol(fitted, D)

    assert abs(total_var - brute_var) < 1.0e-10
    assert np.max(np.abs(sobol_hat - brute_sobol)) < 1.0e-10


@pytest.mark.parametrize("prior", ["amplitude", "lengthscale"])
def test_exact_additive_matches_brute_force(prior):
    # The additive closed form claims two things the brute force can refute: that the components
    # are nu-orthogonal (so the indices sum to one) and that each cell's weight/normalization pair
    # is the one its kernel actually uses -- a normalized kbar scaled by a_sq_i, or an
    # unnormalized k~ scaled by kernel_var.
    D, n = 2, 3
    rng = np.random.default_rng(21)
    samples = {
        "amplitude": {
            "a_sq": np.array([[0.9, 0.2], [1.4, 0.05]]),
            "kernel_ell": np.array([[0.4, 0.8], [0.6, 0.3]]),
        },
        "lengthscale": {
            "kernel_var": np.array([1.1, 0.7]),
            "kernel_inv_length_sq": np.array([[3.0, 0.4], [1.2, 2.5]]),
        },
    }[prior] | {"kernel_noise": np.array([0.02, 0.05])}
    fitted = _fitted(
        ("additive", prior), rng.uniform(0.0, 1.0, (n, D)), rng.normal(size=n), samples
    )

    sobol_hat, total_var = readouts._sobol_exact_additive(fitted)
    brute_sobol, brute_var = _brute_force_sobol(fitted, D)

    assert abs(total_var - brute_var) < 1.0e-10
    assert np.max(np.abs(sobol_hat - brute_sobol)) < 1.0e-10
    assert abs(sobol_hat.sum() - 1.0) < 1.0e-12  # a sum of one-coordinate functions has no residue


def test_qmc_matches_exact_on_product_amplitude():
    # The QMC route is the only one the ARD Matern-5/2 cell has, and that cell has no closed form
    # to check it against -- so its accuracy is pinned on a cell that does. The surface is one
    # dominant coordinate among eight near-flat ones, the regime identification cares about; 2048
    # Saltelli points then have to place the dominant index and leave the others near zero.
    D, n, S = 8, 40, 2
    rng = np.random.default_rng(31)
    X, y = rng.uniform(0.0, 1.0, (n, D)), rng.normal(size=n)
    a_sq = np.full((S, D), 0.01)
    a_sq[:, 3] = [1.2, 0.9]
    samples = {
        "a_sq": a_sq,
        "kernel_ell": rng.uniform(0.3, 1.2, (S, D)),
        "kernel_noise": np.array([0.02, 0.04]),
    }
    fitted = _fitted(("product", "amplitude"), X, y, samples)

    exact, exact_var = readouts._sobol_exact_product_amplitude(fitted)
    estimated, estimated_var = readouts._sobol_qmc(fitted, 2048, 0)

    assert np.max(np.abs(estimated - exact)) <= 0.02
    assert abs(estimated_var - exact_var) / exact_var <= 0.05
    assert np.argmax(estimated) == 3


def test_qmc_skips_coordinates_outside_active():
    # Task 7's oracle is handed X[:, S] only, so a coordinate outside `active` cannot move the
    # posterior mean and its index is exactly zero -- not "small". Skipping those columns is what
    # keeps the estimator's cost |S| + 2 rather than D + 2 evaluations at D = 100.
    D, n, S = 6, 15, 2
    active = np.asarray([1, 4])
    rng = np.random.default_rng(41)
    samples = {
        "kernel_var": np.full(S, 1.3),
        "kernel_inv_length_sq": rng.uniform(0.5, 4.0, (S, len(active))),
        "kernel_noise": np.array([0.02, 0.04]),
    }
    fitted = _fitted(
        "oracle_S", rng.uniform(0.0, 1.0, (n, D)), rng.normal(size=n), samples, active=active
    )

    sobol_hat, total_var = readouts._sobol_qmc(fitted, 256, 0)

    assert np.array_equal(np.delete(sobol_hat, active), np.zeros(D - len(active)))
    assert np.all(sobol_hat[active] > 0.0)
    assert total_var > 0.0


def test_shares_and_active_rules():
    # `share_hat` is a_sq divided by the signal fraction of the standardized variance (plan D7),
    # so with the noise fixed at 0.01 it is exactly a_sq / 0.99 -- checked as arithmetic, not
    # against a re-derivation. The two active rules then have to disagree in the right way: the
    # native one counts how often a *sample* clears the cutoff, the neutral one thresholds a
    # single index of the sample-averaged mean, so a coordinate that is large in one draw of four
    # is 0.25-active natively while its averaged component carries a quarter of the amplitude.
    S, D, n = 4, 5, 8
    rng = np.random.default_rng(11)
    a_sq = np.full((S, D), 1.0e-4)
    a_sq[:, 0] = 0.5  # above the cutoff in every sample
    a_sq[:3, 1] = 0.5  # in three of four
    a_sq[:1, 2] = 0.5  # in one of four
    samples = {"a_sq": a_sq, "kernel_ell": np.full((S, D), 0.5)}
    fitted = _fitted(
        ("additive", "amplitude"),
        rng.uniform(0.0, 1.0, (n, D)),
        rng.normal(size=n),
        samples,
        fixed_noise=0.01,
    )

    cheap = readouts.readouts(fitted, compute_sobol=False)

    assert np.array_equal(cheap["native"], a_sq)
    assert np.max(np.abs(cheap["share_hat"] - a_sq / 0.99)) < 1.0e-15
    assert np.array_equal(cheap["native_median"], np.median(a_sq, axis=0))
    assert np.array_equal(cheap["p_active"], [1.0, 0.75, 0.25, 0.0, 0.0])
    assert np.array_equal(cheap["active_native"], [True, True, False, False, False])
    # compute_sobol=False is the cheap in-loop row: the neutral readout is absent, not zero, and
    # NaN > eps leaves the rule that reads it uniformly False.
    assert np.all(np.isnan(cheap["sobol_hat"])) and np.isnan(cheap["total_var_hat"])
    assert not cheap["active_neutral"].any()

    full = readouts.readouts(fitted)

    assert abs(full["sobol_hat"].sum() - 1.0) < 1.0e-12
    assert np.array_equal(full["active_neutral"], full["sobol_hat"] > ACTIVE_EPS)
    assert full["active_neutral"][0] and full["active_native"][0]
    assert not full["active_neutral"][3:].any() and not full["active_native"][3:].any()


def test_degenerate_noise_gives_nan_shares():
    # `kernel_noise ~ LogNormal(0, 10)` is unbounded, so a draw from a fit that failed its
    # diagnostics can carry sigma^2 >= 1: no signal is left for a_sq_i to be a share of, and the
    # 1 - sigma^2 correction would hand back a *negative* share. Such a draw reads NaN and is
    # dropped from the active count -- p_active[0] is 3/3, not the 3/4 counting it as inactive.
    S, D, n = 4, 3, 8
    rng = np.random.default_rng(15)
    a_sq = np.full((S, D), 1.0e-4)
    a_sq[:, 0] = 0.5
    samples = {
        "a_sq": a_sq,
        "kernel_ell": np.full((S, D), 0.5),
        "kernel_noise": np.array([0.01, 0.01, 0.01, 1.5]),
    }
    fitted = _fitted(
        ("additive", "amplitude"), rng.uniform(0.0, 1.0, (n, D)), rng.normal(size=n), samples
    )

    out = readouts.readouts(fitted, compute_sobol=False)

    assert np.all(np.isnan(out["share_hat"][3])) and not np.any(np.isnan(out["share_hat"][:3]))
    assert np.array_equal(out["p_active"], [1.0, 0.0, 0.0])


def test_readouts_lengthscale_cells_have_no_share():
    # rho_i is not a variance share, so `share_hat` is None rather than a plausible-looking
    # number, and `p_active` falls back to the rho cutoff RHO_EPS -- the same cutoff as
    # ACTIVE_EPS by construction of ELL_EPS, which is what makes the two priors' active counts
    # comparable at all.
    S, D, n = 4, 3, 8
    rng = np.random.default_rng(12)
    rho = np.full((S, D), 0.5 * gp.RHO_EPS)
    rho[:, 0] = 2.0 * gp.RHO_EPS
    samples = {"kernel_var": np.full(S, 1.0), "kernel_inv_length_sq": rho}
    fitted = _fitted(
        ("additive", "lengthscale"),
        rng.uniform(0.0, 1.0, (n, D)),
        rng.normal(size=n),
        samples,
        fixed_noise=0.01,
    )

    out = readouts.readouts(fitted, compute_sobol=False)

    assert out["share_hat"] is None
    assert np.array_equal(out["p_active"], [1.0, 0.0, 0.0])
    assert np.array_equal(out["native"], rho)


def test_readouts_widen_the_oracle_to_full_D():
    # The oracle's draws are |S|-wide but every log row is D-wide, so the readouts pad outside
    # `active` with zeros -- a rho of zero being an infinitely long lengthscale, i.e. exactly the
    # inactive coordinate the oracle was told about.
    D, n, S = 6, 15, 1
    active = np.asarray([1, 4])
    rng = np.random.default_rng(42)
    samples = {
        "kernel_var": np.full(S, 1.3),
        "kernel_inv_length_sq": np.array([[3.0, 0.4]]),
    }
    fitted = _fitted(
        "oracle_S",
        rng.uniform(0.0, 1.0, (n, D)),
        rng.normal(size=n),
        samples,
        fixed_noise=1.0e-4,
        active=active,
    )

    out = readouts.readouts(fitted, compute_sobol=False)

    assert out["native"].shape == (S, D)
    assert np.array_equal(out["native"][0], [0.0, 3.0, 0.0, 0.0, 0.4, 0.0])
    assert np.array_equal(out["p_active"], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0])


def _readout_at_D10():
    """A hand-made additive/amplitude readout at D = 10 whose amplitudes track aligned3's shares."""
    labels = make_family("aligned3", 0, D=10).labels
    S, n = 4, 20
    rng = np.random.default_rng(13)
    a_sq = np.full((S, 10), 5.0e-4) + rng.uniform(0.0, 1.0e-4, (S, 10))
    a_sq[:, list(labels.S)] = np.array([0.30, 0.20, 0.35]) + rng.uniform(0.0, 0.02, (S, 3))
    samples = {"a_sq": a_sq, "kernel_ell": np.full((S, 10), 0.5)}
    fitted = _fitted(
        ("additive", "amplitude"),
        rng.uniform(0.0, 1.0, (n, 10)),
        rng.normal(size=n),
        samples,
        fixed_noise=0.01,
    )
    return readouts.readouts(fitted), labels


def test_manipulation_checks_keys():
    # The manipulation checks are what turn one fit's readouts into the SQ1 record's columns, so
    # what matters is that all six exist, are finite on a well-behaved readout, and pair the
    # cell's *stated* variance with the objective's realized one on the active set in the labels'
    # own order. Both readouts are scored against both labels: `sobol_hat` against `labels.g` is
    # what tells "the native readout tracks g" apart from "every readout of this fit does".
    readout, labels = _readout_at_D10()

    checks = readouts.manipulation_checks(readout, labels)

    assert set(checks) == {
        "spearman_native_vs_s",
        "spearman_native_vs_g",
        "spearman_sobol_vs_s",
        "spearman_sobol_vs_g",
        "spearman_native_vs_sobol",
        "amplitude_vs_realized",
    }
    assert all(np.isfinite(value) for key, value in checks.items() if key.startswith("spearman"))
    # `labels.s` is zero on seven of the ten coordinates, and those ties cap Spearman against it
    # below 1 however well a readout orders them; the amplitudes here rank all three active
    # coordinates above all seven inactive ones, so both readouts attain that cap exactly.
    ceiling = spearmanr(np.argsort(np.argsort(labels.s)), labels.s)[0]
    assert checks["spearman_native_vs_s"] == pytest.approx(ceiling)
    assert checks["spearman_sobol_vs_s"] == pytest.approx(ceiling)
    # `labels.g` orders the active coordinates differently from `labels.s`, and neither readout
    # can order seven coordinates that carry nothing, so these two only have to be strongly
    # positive rather than exact.
    assert checks["spearman_native_vs_g"] > 0.7
    assert checks["spearman_native_vs_sobol"] > 0.7

    pairs = checks["amplitude_vs_realized"]
    assert pairs.shape == (len(labels.S), 2)
    assert np.array_equal(pairs[:, 0], np.median(readout["share_hat"], axis=0)[list(labels.S)])
    assert np.array_equal(pairs[:, 1], labels.s[list(labels.S)])


def test_manipulation_checks_state_the_native_readout_without_shares():
    # Where a_sq_i does not exist -- the lengthscale cells and the MAP references -- the stated
    # column falls back to `native_median` rather than being dropped, so a calibration plot has
    # the same shape for every cell.
    readout, labels = _readout_at_D10()
    readout = readout | {"share_hat": None}

    checks = readouts.manipulation_checks(readout, labels)

    assert np.array_equal(
        checks["amplitude_vs_realized"][:, 0], readout["native_median"][list(labels.S)]
    )


# --- against the labels ---

# `scipy.stats.qmc.Sobol` warns whenever the point count is not a power of two, which n = 120 and
# n = 200 are not; the plan's `identify` takes n from the caller, so the warning is about the
# design the study specifies rather than about a mistake here.
_SOBOL_WARNING = "The balance properties of Sobol' points require n to be a power of 2."


def _identification_data(family: str, D: int, n: int, seed: int):
    """The SQ1 design of plan §2's `identify`: Sobol X, observed y, standardized and negated."""
    objective = make_family(family, seed, D=D)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SOBOL_WARNING)
        X = qmc.Sobol(D, scramble=True, seed=seed).random(n)
    y = objective.observe(X, noise_rng(seed, run=0))
    return objective, jnp.asarray(X), jnp.asarray(-(y - y.mean()) / y.std())


def _assert_recovers_shares(D: int, n: int, nuts: NUTSConfig) -> None:
    """Fit additive/amplitude to `aligned3` and check the neutral readout against `labels.s`."""
    objective, X, z = _identification_data("aligned3", D, n, 0)

    fitted = gp.fit(X, z, jax.random.PRNGKey(0), ("additive", "amplitude"), nuts=nuts)
    out = readouts.readouts(fitted)

    assert np.max(np.abs(out["sobol_hat"] - objective.labels.s)) <= 0.03


def test_additive_sobol_recovers_labels():
    # The whole point of the neutral readout: `aligned3` puts a third of the variance on each of
    # three coordinates and nothing on the rest, and the index has to say so -- on the active set
    # *and* off it, since a method that spreads a few percent over 7 inactive coordinates would
    # identify the wrong support. The cell's own `share_hat` medians are nowhere near a third
    # here, which is exactly why the comparison across cells is made on this readout instead.
    _assert_recovers_shares(D=10, n=120, nuts=NUTSConfig(64, 64, 4))


@pytest.mark.slow
def test_additive_sobol_recovers_labels_production_budget():
    """The same claim under the production sampler budget, at D = 30 with n = 120 (ruling R24).

    What the fast test above leaves open is whether the recovery survives `NUTSConfig()`'s own
    warm-up and thinning rather than the short one -- a sampler question, not a readout question,
    so it needs the production budget but not the production width. The study's own design point,
    D = 100 with n = 200, is a single NUTS fit of well over an hour and is measured by the Task 11
    pilot script rather than by pytest; D = 30 keeps three active coordinates among thirty at a
    length a test suite can carry.
    """
    _assert_recovers_shares(D=30, n=120, nuts=NUTSConfig())


# --- Gate 1 ---

# The check script's D = 1 study (`Research Context/research direction/Kernel normalization
# check.py`, Table 2): five data sets whose realized component variance under nu is known, read
# through the amplitude readout. Its own grid posterior over (a, ell) reported P(var > 0.09) =
# 0.01 / 0.98 / 1.00 / 0.07 / 0.05 for noise / strong-smooth / strong-rough / weak-smooth /
# weak-rough; what the thesis's cell has to reproduce is that ordering -- a normalized amplitude
# separates a strong component from a weak one from pure noise -- under its own prior and its own
# sampler rather than a grid.
_GATE1_N = 80
_GATE1_NOISE_SD = 0.1
_GATE1_CASES = (
    ("strong-smooth", 1.5, 0.25),
    ("strong-rough", 0.08, 0.25),
    ("weak-smooth", 1.5, 0.02),
    ("weak-rough", 0.08, 0.02),
)


def _gate1_cases() -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """The script's five data sets, in its draw order, built from `synthobj.kernel`'s primitives.

    One generator drives the design, the four component draws and all five noise vectors in the
    order the script draws them, so the data are the script's data and not merely data like it.
    Each component is drawn jointly on the design and the 64 quadrature nodes and rescaled so that
    its *node* values have the prescribed quadrature variance -- the variance the readout claims
    to recover.
    """
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, _GATE1_N)
    Z = np.concatenate([x, kernel.GL_NODES])

    def realized(ell: float, target_var: float) -> np.ndarray:
        covariance = kernel.normalized(Z, Z, ell) + 1.0e-8 * np.eye(Z.size)
        draw = rng.multivariate_normal(np.zeros(Z.size), covariance)
        return draw[:_GATE1_N] * np.sqrt(target_var / kernel.nu_var(draw[_GATE1_N:]))

    cases = [("pure noise", rng.normal(0.0, _GATE1_NOISE_SD, _GATE1_N))]
    for name, ell, target_var in _GATE1_CASES:
        cases.append((name, realized(ell, target_var) + rng.normal(0.0, _GATE1_NOISE_SD, _GATE1_N)))
    return x, cases


@pytest.mark.slow
def test_gate1_replication():
    # Five production fits at D = 1. `y` is passed in the script's own units rather than
    # standardized: the script's noise sd 0.1 is what `fixed_noise=0.01` states and what the 0.09
    # threshold is a variance on, and rescaling y to unit variance would divide the signal-to-
    # noise ratio out of both (see the task report). The exceedance probability is quantized to
    # sixteenths by the retained draws, so the separation is asserted on it *and* on the posterior
    # median share, which is not.
    x, cases = _gate1_cases()
    X = jnp.asarray(x[:, None])

    exceedance, median_share = {}, {}
    for i, (name, y) in enumerate(cases):
        fitted = gp.fit(
            X,
            jnp.asarray(y),
            jax.random.PRNGKey(i),
            ("additive", "amplitude"),
            fixed_noise=_GATE1_NOISE_SD**2,
        )
        share_hat = readouts.readouts(fitted, compute_sobol=False)["share_hat"]
        exceedance[name] = float(np.mean(share_hat > 0.09))
        median_share[name] = float(np.median(share_hat))

    strong = [exceedance[name] for name, _, _ in _GATE1_CASES[:2]]
    weak = [exceedance[name] for name, _, _ in _GATE1_CASES[2:]]

    assert min(strong) >= 0.9
    assert max(weak) <= 0.3
    assert exceedance["pure noise"] <= 0.05
    assert min(strong) > max(weak) >= exceedance["pure noise"]
    # The medians carry the same claim without the sixteenths: a strong component's share sits
    # above the 0.09 cutoff, a weak one below it but well clear of noise, and the four realized
    # variances (0.25, 0.25, 0.02, 0.02) are ordered correctly whatever the lengthscale.
    assert min(median_share[name] for name, _, _ in _GATE1_CASES[:2]) > 0.09
    assert max(median_share[name] for name, _, _ in _GATE1_CASES[2:]) < 0.09
    assert min(median_share[name] for name, _, _ in _GATE1_CASES[2:]) > 5.0 * median_share[
        "pure noise"
    ]
