"""sagp.readouts: what one fit says about which coordinates matter.

Three readouts per fit: the `native` one the cell's own sparsity prior lives on -- a_sq_i or
rho_i -- the noise-corrected variance share where the cell states one, and the first-order Sobol
index of the posterior mean under the reference measure, which is the only one comparable across
the four cells (exact for three of them, a QMC estimate for the fourth). Every one of them reads
a `FittedGP`'s retained draws and refits nothing, so one call serves both consumers: SQ1's
identification record, which asks for all of them, and the BO run log, which asks for the cheap
ones each iteration.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.typing import ArrayLike
from scipy.stats import qmc, spearmanr

from sagp.gp import ACTIVE_EPS, CELLS, GL_NODES, GL_WEIGHTS, RHO_EPS, FittedGP, _kbar_all


# --- readouts ---

# Points per `FittedGP.posterior` call inside the QMC estimator (plan D6). At the study's largest
# size -- 200 training points in D = 100 -- 512 x (n + Q) x D_used stays under `_CHUNK_THRESHOLD`,
# so a block is one `posterior` call and the estimator's own blocking is what bounds the memory.
_QMC_CHUNK: int = 512


def shares_from_amplitudes(a_sq: ArrayLike, noise: ArrayLike) -> Array:
    """Noise-corrected variance shares of the amplitude cells: s_i = a_sq_i / (1 - sigma^2).

    The GP sees z = -(y - ybar)/s_y, whose variance is 1 = (Var_nu f + sigma^2)/s_y^2 up to the
    design's sampling error, so a_sq_i -- coordinate i's variance under the reference measure,
    normalization being what buys that -- is its share of *signal plus noise*. Dividing by
    1 - sigma^2 divides the inflation back out and leaves a share of the signal, the quantity
    `labels.s` states (plan D7); at synthobj's Var f = 1, sigma = 0.1 the correction is 1 %.
    `noise` is the per-sample observation variance in those same standardized units, (S,),
    broadcast across the coordinates.

    The correction presupposes sigma^2 < 1, i.e. that some of the standardized variance is
    signal. `kernel_noise ~ LogNormal(0, 10)` is unbounded, so a draw from a fit that failed its
    diagnostics can violate that, and there the formula would divide by a non-positive number and
    hand back a *negative* share -- an inactive coordinate reading as active with the sign lost in
    a median. Such a draw's shares are NaN instead, and `readouts` drops it from the active count
    rather than counting it as inactive.
    """
    signal = 1.0 - jnp.reshape(jnp.asarray(noise), (-1, 1))
    return jnp.where(signal > 0.0, jnp.asarray(a_sq) / signal, jnp.nan)


def _centered_parts(fitted: FittedGP, s: int) -> tuple[Array, Array, bool]:
    """Sample `s`'s per-coordinate lengthscales, component weights and normalization flag.

    The two additive cells write the same component w_i * kbar_i(x_i, z_i) two ways: the amplitude
    cell puts the scale on a_sq_i and normalizes kbar, so a_sq_i *is* the component's variance
    under nu; the lengthscale cell puts one kernel_var in front of the *un*normalized k~_i, so
    shrinking rho_i shrinks the component through v(ell_i) instead. Reading that off the cell in
    one place is what lets `component_means` and `_sobol_exact_additive` be written once.
    """
    params = fitted._params(s)
    if fitted._param_sites()[0] == "a_sq":
        return params["kernel_ell"], params["a_sq"], True
    ell = params["kernel_inv_length_sq"] ** -0.5
    return ell, params["kernel_var"] * jnp.ones_like(ell), False


@partial(jit, static_argnums=(5,))
def _components_at(
    T: Array, X: Array, ell: Array, weights: Array, alpha: Array, normalize: bool
) -> Array:
    """One sample's additive components at the rows of T; T (G, D), X (n, D) -> (G, D).

    Entry [g, i] is m_i(T[g, i]) = w_i * sum_n kbar_i(T[g, i], X[n, i]) alpha_n, coordinate i's
    term of that sample's posterior mean sum_n k(x, X_n) alpha_n. The peak is the single
    (G, n, D) tensor `_kbar_all` hands the contraction.
    """
    return weights * jnp.einsum("gnd,n->gd", _kbar_all(T, X, ell, normalize), alpha)


def component_means(fitted: FittedGP, x_grid: ArrayLike) -> Array:
    """Additive cells only: per-component posterior means at `x_grid`, (S, D, G).

    m_{s,i}(t) = w_{s,i} * sum_n kbar_i(t, X[n, i]) alpha_{sn} with (w, kbar) the cell's own pair
    (see `_centered_parts`). Because the additive kernel is k(x, z) = sum_i w_i kbar_i(x_i, z_i),
    the posterior mean separates coordinate-wise and these sum to it exactly -- evaluating at a
    point whose coordinates all equal t reproduces `posterior(...)[0]`, which is the test that
    keeps them components of the surrogate the loop optimizes rather than of a re-fit of it. One
    grid serves every coordinate: the reference measure is U[0,1] on each.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    X = fitted._columns(fitted.X_train)
    T = jnp.tile(x_grid[:, None], (1, X.shape[1]))
    alphas = fitted.alphas()

    components = []
    for s in range(alphas.shape[0]):
        ell, weights, normalize = _centered_parts(fitted, s)
        components.append(_components_at(T, X, ell, weights, alphas[s], normalize).T)
    return jnp.stack(components)


def _quadrature_var(values: Array) -> Array:
    """Var_nu of functions tabulated at the 64 nodes: sum_q w_q f_q^2 - (sum_q w_q f_q)^2.

    Column-wise on a (Q, D) table. The reference measure's own variance -- literally the rule
    `synthobj.kernel.nu_var` applies to the objectives' components -- so an index computed with it
    and a label computed with it are the same quantity, not two approximations of one.
    """
    mean = GL_WEIGHTS @ values
    return GL_WEIGHTS @ (values * values) - mean * mean


def _sobol_exact_additive(fitted: FittedGP) -> tuple[np.ndarray, float]:
    """First-order Sobol indices of the sample-averaged posterior mean, additive cells (plan D6).

    m(x) = S^-1 sum_s sum_i m_{s,i}(x_i) is a sum of one-coordinate functions, each of which
    integrates to *exactly* zero against the 64-node rule that defines nu -- that is what
    centering the components buys -- so the variance decomposition is exact and needs no
    estimator: Var_nu(m) = sum_i Var_nu(m_i) and S_i = Var_nu(m_i) / Var_nu(m), both by the same
    quadrature. Averaging over samples before taking the variance is deliberate: the readout
    describes the one surrogate mean the acquisition function sees, not the mean of S of them.
    Returns (S_hat (D,), Var_nu(m)); the indices sum to 1. The components are accumulated one
    sample at a time, so the peak is one (Q, n, D) tensor rather than S of them.
    """
    X = fitted._columns(fitted.X_train)
    T = jnp.tile(GL_NODES[:, None], (1, X.shape[1]))
    alphas = fitted.alphas()
    S = alphas.shape[0]

    mean_components = jnp.zeros((GL_NODES.size, X.shape[1]))
    for s in range(S):
        ell, weights, normalize = _centered_parts(fitted, s)
        mean_components = mean_components + _components_at(T, X, ell, weights, alphas[s], normalize)

    variances = _quadrature_var(mean_components / S)
    total_var = float(jnp.sum(variances))
    return np.asarray(variances / total_var), total_var


@jit
def _pair_term(
    kbar_s: Array, kbar_t: Array, a_sq_s: Array, a_sq_t: Array, alpha_s: Array, alpha_t: Array
) -> Array:
    """One (s, s') term of the product/amplitude Var_nu(m); a scalar (plan D6).

    sum_{n,m} alpha_{sn} alpha_{s'm} [prod_j (1 + a_{sj}^2 a_{s'j}^2 G_j(X_{nj}, X_{mj})) - 1] with
    G_j(u, u') = sum_q w_q kbar_{sj}(t_q, u) kbar_{s'j}(t_q, u'): the product's factors are
    independent under nu, so E_nu[m_s m_s'] is a product over coordinates of
    E_{x_j}[(1 + a_{sj}^2 kbar)(1 + a_{s'j}^2 kbar)] = 1 + a_{sj}^2 a_{s'j}^2 G_j, and the "- 1"
    subtracts (E_nu m_s)(E_nu m_s') = (sum_n alpha_{sn})(sum_m alpha_{s'm}). Symmetric in (s, s'),
    which is why the caller can weight the off-diagonal pairs by 2. The (n, n, D) tensor is this
    function's peak -- 32 MB at n = 200, D = 100 -- and taking one pair at a time is what keeps
    the (S, S, D, n, n) tensor the formula reads like from ever existing.
    """
    G = jnp.einsum("q,qnd,qmd->nmd", GL_WEIGHTS, kbar_s, kbar_t)
    return alpha_s @ (jnp.prod(1.0 + a_sq_s * a_sq_t * G, axis=-1) - 1.0) @ alpha_t


def _sobol_exact_product_amplitude(fitted: FittedGP) -> tuple[np.ndarray, float]:
    """First-order Sobol indices of the sample-averaged posterior mean, product/amplitude (D6).

    m_s(x) = sum_n alpha_{sn} prod_j (1 + a_{sj}^2 kbar_{sj}(x_j, X_{nj})). Every centered
    component integrates to exactly zero against the 64-node rule, so E_{x_j}[1 + a^2 kbar] = 1
    and the marginal over all coordinates but i collapses to
    E_{x_-i} m_s(x_i) = sum_n alpha_{sn} (1 + a_{si}^2 kbar_{si}(x_i, X_{ni})), whose quadrature
    variance -- after averaging over samples, as in `_sobol_exact_additive` -- is the numerator.
    The denominator is the exact Var_nu(m) = S^-2 sum_{s,s'} `_pair_term`, summed over the
    S(S+1)/2 pairs with the off-diagonal ones doubled. Returns (S_hat (D,), Var_nu(m)); unlike the
    additive cells these do *not* sum to 1, because expanding the product leaves interaction terms.
    """
    X = fitted._columns(fitted.X_train)
    T = jnp.tile(GL_NODES[:, None], (1, X.shape[1]))
    a_sq = fitted.samples["a_sq"]
    ells = fitted.samples["kernel_ell"]
    alphas = fitted.alphas()
    S = alphas.shape[0]

    # One (Q, n, D) tensor per sample -- 10 MB each at n = 200, D = 100 -- built once because
    # every one of the S(S+1)/2 pairs below needs two of them.
    kbars = [_kbar_all(T, X, ells[s], True) for s in range(S)]

    marginal = (
        sum(
            jnp.sum(alphas[s]) + a_sq[s] * jnp.einsum("qnd,n->qd", kbars[s], alphas[s])
            for s in range(S)
        )
        / S
    )
    variances = _quadrature_var(marginal)

    total_var = float(
        sum(
            (1.0 if s == t else 2.0)
            * _pair_term(kbars[s], kbars[t], a_sq[s], a_sq[t], alphas[s], alphas[t])
            for s in range(S)
            for t in range(s, S)
        )
        / S**2
    )
    return np.asarray(variances / total_var), total_var


def _posterior_mean(fitted: FittedGP, X: np.ndarray) -> np.ndarray:
    """The sample-averaged posterior mean at the rows of X, (len(X),), in `_QMC_CHUNK` blocks."""
    return np.concatenate(
        [
            np.asarray(fitted.posterior(X[start : start + _QMC_CHUNK])[0].mean(axis=0))
            for start in range(0, X.shape[0], _QMC_CHUNK)
        ]
    )


def _sobol_qmc(fitted: FittedGP, n: int = 2048, seed: int = 0) -> tuple[np.ndarray, float]:
    """Saltelli's first-order estimator of the posterior mean's Sobol indices (plan D6).

    The ARD Matern-5/2 -- ("product", "lengthscale") and the MAP references, which share its
    kernel -- is not a product of one-coordinate factors under nu, so no closed form is available
    and the index is estimated. A and B are the two halves of one scrambled Sobol point set in 2D
    dimensions, A_B^(i) is A with column i taken from B, and
    S_i = N^-1 sum_j m(B_j) (m(A_B^(i)_j) - m(A_j)) / V, with V the variance of m over A and B
    together; `m` is the sample-averaged posterior mean, the same quantity the exact routes take.
    `test_qmc_matches_exact_on_product_amplitude` is what pins the estimator's accuracy, by running
    it on a cell whose exact indices are known. When `active` is set -- Task 7's oracle -- the
    coordinates outside it are skipped rather than evaluated: the surrogate never reads them, so
    swapping their column cannot move m and their index is exactly zero. Returns (S_hat (D,), V).
    """
    D = fitted.X_train.shape[1]
    points = qmc.Sobol(2 * D, scramble=True, seed=seed).random(n)
    A, B = points[:, :D], points[:, D:]

    m_A, m_B = _posterior_mean(fitted, A), _posterior_mean(fitted, B)
    total_var = float(np.var(np.concatenate([m_A, m_B])))

    s_hat = np.zeros(D)
    for i in range(D) if fitted.active is None else np.asarray(fitted.active):
        A_B = A.copy()
        A_B[:, i] = B[:, i]
        s_hat[i] = float(np.mean(m_B * (_posterior_mean(fitted, A_B) - m_A))) / total_var
    return s_hat, total_var


def readouts(
    fitted: FittedGP,
    *,
    eps: float = ACTIVE_EPS,
    compute_sobol: bool = True,
    sobol_n: int = 2048,
    sobol_seed: int = 0,
) -> dict[str, object]:
    """Every identification summary of one fit, from its retained draws alone -- no refitting.

    Two kinds of readout, deliberately (plan D6, D7). `native` is whatever the cell's own sparsity
    lives on -- a_sq_i or rho_i -- so it is the number that cell's own user would threshold, and
    `p_active` thresholds it in that cell's own units: ACTIVE_EPS on a share, RHO_EPS on rho, which
    by construction of ELL_EPS are the same cutoff. `share_hat` exists only where a_sq_i *is* a
    variance share (the amplitude cells) and is None elsewhere rather than faked. `sobol_hat` is
    the parameterization-neutral one -- the first-order Sobol index of the posterior mean under nu
    -- and is therefore the only readout comparable across the four cells; it is exact for the two
    additive cells and for product/amplitude, and QMC for the ARD Matern-5/2 (the three `_sobol_*`
    functions). `compute_sobol=False` leaves it and `total_var_hat` NaN for the cheap in-loop rows.

    Keys: `native` (S, D), `native_median` (D,), `share_hat` (S, D) or None, `p_active` (D,),
    `sobol_hat` (D,), `total_var_hat` float, `active_neutral` (D,) bool = sobol_hat > eps,
    `active_native` (D,) bool = p_active > 0.5. Numpy throughout: these go to npz and to the run
    log, and every per-coordinate array is widened back to D by zero-padding outside `active` so
    that a log row has one width whatever produced it. A draw whose noise leaves no signal reads
    NaN in `share_hat` and is left out of `p_active` entirely (`shares_from_amplitudes`).
    """
    D = fitted.X_train.shape[1]
    is_amplitude = isinstance(fitted.cell, tuple) and fitted.cell[1] == "amplitude"
    site = (
        CELLS[fitted.cell].native_site
        if isinstance(fitted.cell, tuple)
        else "kernel_inv_length_sq"
    )

    def widen(draws: np.ndarray) -> np.ndarray:
        """(S, D_used) -> (S, D), zero outside `active`; the identity when `active` is None."""
        if fitted.active is None:
            return draws
        widened = np.zeros((draws.shape[0], D))
        widened[:, np.asarray(fitted.active)] = draws
        return widened

    native = widen(np.asarray(fitted.samples[site]))

    if is_amplitude:
        share_hat = widen(
            np.asarray(shares_from_amplitudes(fitted.samples["a_sq"], fitted._noises()))
        )
        # A degenerate draw's shares are NaN (see `shares_from_amplitudes`), and NaN > eps is
        # False, which would count it as *inactive* rather than omit it; lifting the comparison to
        # float and putting the NaNs back makes `nanmean` average over the usable draws alone.
        counted = np.where(np.isnan(share_hat), np.nan, np.asarray(share_hat > eps, dtype=float))
        p_active = np.nanmean(counted, axis=0)
    else:
        share_hat = None
        p_active = np.mean(native > RHO_EPS, axis=0)

    sobol_hat, total_var = np.full(D, np.nan), float("nan")
    if compute_sobol:
        if isinstance(fitted.cell, tuple) and fitted.cell[0] == "additive":
            sobol_hat, total_var = _sobol_exact_additive(fitted)
        elif fitted.cell == ("product", "amplitude"):
            sobol_hat, total_var = _sobol_exact_product_amplitude(fitted)
        else:
            sobol_hat, total_var = _sobol_qmc(fitted, sobol_n, sobol_seed)

    return {
        "native": native,
        "native_median": np.median(native, axis=0),
        "share_hat": share_hat,
        "p_active": p_active,
        "sobol_hat": sobol_hat,
        "total_var_hat": total_var,
        "active_neutral": sobol_hat > eps,
        "active_native": p_active > 0.5,
    }


def manipulation_checks(readout: dict[str, object], labels: object) -> dict[str, object]:
    """Rank agreement between one fit's readouts and the objective's labels (plan D7).

    Spearman rather than Pearson throughout, over all D coordinates: what a manipulation check
    asks is whether the surrogate *orders* the coordinates the way the objective does, and in three
    of the four cells the two quantities are not even on the same scale (rho_i is not a share).
    The targets are `labels.s`, the first-order variance share `sobol_hat` estimates; `labels.g`,
    the realized slope share, which is what a lengthscale readout can be expected to track; and
    `sobol_hat` itself, against which `native_median` measures how much the cell's own
    parameterization already tells you. Both readouts are scored against both labels, `sobol_hat`
    against `labels.g` included: it is the parameterization-neutral readout, so without that pair
    "the native readout tracks g" could not be told from "every readout of this fit does".
    `amplitude_vs_realized` pairs, on the active coordinates only, the sample median of the
    readout that states a *variance* -- `share_hat` where the cell has one, `native_median`
    otherwise -- with `labels.s`, so a calibration plot of stated against realized share needs no
    further arithmetic. That median is over the usable draws only: a draw
    whose noise left no signal reads NaN (`shares_from_amplitudes`) and must not drag the median.
    """
    native_median = np.asarray(readout["native_median"])
    sobol_hat = np.asarray(readout["sobol_hat"])
    share_hat = readout["share_hat"]
    stated = native_median if share_hat is None else np.nanmedian(np.asarray(share_hat), axis=0)
    active = list(labels.S)

    return {
        "spearman_native_vs_s": float(spearmanr(native_median, labels.s)[0]),
        "spearman_native_vs_g": float(spearmanr(native_median, labels.g)[0]),
        "spearman_sobol_vs_s": float(spearmanr(sobol_hat, labels.s)[0]),
        "spearman_sobol_vs_g": float(spearmanr(sobol_hat, labels.g)[0]),
        "spearman_native_vs_sobol": float(spearmanr(native_median, sobol_hat)[0]),
        "amplitude_vs_realized": np.column_stack([stated[active], labels.s[active]]),
    }
