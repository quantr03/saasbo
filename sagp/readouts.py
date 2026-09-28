"""sagp.readouts: what one fit says about which coordinates matter.

Three readouts per fit, all read off a `FittedGP`'s retained draws with no refitting: the `native`
parameter the cell's own sparsity prior lives on (a_sq_i or rho_i), the noise-corrected variance
share where the cell states one, and the Sobol index under nu, the only readout comparable across
cells.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.typing import ArrayLike
from scipy.stats import qmc, spearmanr

from sagp.gp import ACTIVE_EPS, CELLS, GL_NODES, GL_WEIGHTS, RHO_EPS, FittedGP, kbar_all


# Points per `FittedGP.posterior` call inside the QMC estimator. The Saltelli points are
# independent -- each is its own batch entry in `FittedGP.posterior` -- so a chunk is an exact
# partition and the block size changes no number, only what one call costs. It is small because
# GPyTorch expands the training inputs over the test batch: a 512-row call recomputes the
# train-side quadrature 512 times, measured at 5.3 GB peak / 130 s for additive/amplitude at
# n = 200, D = 100, S = 16, against 1.0 GB / 9.8 s at 64.
_QMC_CHUNK: int = 64


def shares_from_amplitudes(a_sq: ArrayLike, noise: ArrayLike) -> Array:
    """Noise-corrected variance shares of the amplitude cells: s_i = a_sq_i / (1 - sigma^2).

    Normalizing the components makes a_sq_i coordinate i's variance under nu, of a target
    standardized to 1 = (Var_nu f + sigma^2)/s_y^2 and so a share of *signal plus noise*; dividing
    by 1 - sigma^2 leaves the signal's share, which `labels.s` states. `noise` is that variance per
    sample, (S,), in the same units. It is unbounded, so a draw with sigma^2 >= 1 would give a
    *negative* share; its shares are NaN instead and `readouts` leaves it out of the active count.
    """
    signal = 1.0 - jnp.reshape(jnp.asarray(noise), (-1, 1))
    return jnp.where(signal > 0.0, jnp.asarray(a_sq) / signal, jnp.nan)


def r2d2_r2(a_sq: ArrayLike) -> Array:
    """The R2-D2 prior's R2 per draw, omega / (1 + omega) with omega = sum_i a_sq_i; (S,).

    `a_sq` is (S, D), from any amplitude cell. Normalized components make omega the first-order
    variance of the standardized target, and under the R2-D2 prior it is exactly the global scale
    R2 / (1 - R2), the shares summing to one: this is the R2 its Beta is on, recomputed from the
    retained amplitudes rather than stored as a site. It is not the first-order R^2 of the data --
    a well-fitted standardized target has omega near 1, so this sits near 0.5 however much of the
    target is noise; `first_order_r2` is that R^2.
    """
    omega = jnp.sum(jnp.asarray(a_sq), axis=-1)
    return omega / (1.0 + omega)


def first_order_r2(a_sq: ArrayLike, noise: ArrayLike) -> Array:
    """First-order R^2 per draw, omega / (omega + sigma^2) with omega = sum_i a_sq_i; (S,).

    The fitted model's: `a_sq` is (S, D), from any amplitude cell, and `noise` is sigma^2 per
    draw, (S,), as `FittedGP.noises()` gives it -- the only source when the noise is fixed, since
    then there is no `noise` site. Interaction variance an additive cell cannot represent goes to
    the noise, so on the interaction sweep this falls to about 1 - gamma: the misspecification
    readout.
    """
    omega = jnp.sum(jnp.asarray(a_sq), axis=-1)
    return omega / (omega + jnp.reshape(jnp.asarray(noise), (-1,)))


def _centered_parts(fitted: FittedGP, s: int) -> tuple[Array, Array, bool]:
    """Sample `s`'s per-coordinate lengthscales, component weights and normalization flag.

    The amplitude cell scales the normalized kbar by a_sq_i, so a_sq_i *is* that component's
    variance under nu; the lengthscale cell scales the *un*normalized k~_i by its outputscale, so
    shrinking rho_i shrinks the component through v(ell_i).
    """
    params = fitted.params(s)
    if fitted.param_sites()[0] == "a_sq":
        return params["kernel_ell"], params["a_sq"], True
    ell = params["kernel_inv_length_sq"] ** -0.5
    return ell, params["outputscale"] * jnp.ones_like(ell), False


@partial(jit, static_argnums=(5,))
def _components_at(
    T: Array, X: Array, ell: Array, weights: Array, alpha: Array, normalize: bool
) -> Array:
    """One sample's additive components at the rows of T; T (G, D), X (n, D) -> (G, D).

    Entry [g, i] is w_i * sum_n kbar_i(T[g, i], X[n, i]) alpha_n; the peak is the (G, n, D) tensor
    `kbar_all` feeds the contraction.
    """
    return weights * jnp.einsum("gnd,n->gd", kbar_all(T, X, ell, normalize), alpha)


def component_means(fitted: FittedGP, x_grid: ArrayLike) -> Array:
    """Additive cells only: per-component posterior means at `x_grid`, (S, D, G).

    m_{s,i}(t) = w_{s,i} * sum_n kbar_i(t, X[n, i]) alpha_{sn}, (w, kbar) the cell's own pair
    (`_centered_parts`). The additive kernel being sum_i w_i kbar_i(x_i, z_i), the posterior mean is
    `fitted.means()[s]` plus these summed: at a point whose coordinates all equal t,
    `means()[s] + sum_i m_{s,i}(t)` reproduces `posterior(...)[0]`. One grid serves every
    coordinate.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    X = fitted.columns(fitted.X_train)
    T = jnp.tile(x_grid[:, None], (1, X.shape[1]))
    alphas = fitted.alphas()

    components = []
    for s in range(alphas.shape[0]):
        ell, weights, normalize = _centered_parts(fitted, s)
        components.append(_components_at(T, X, ell, weights, alphas[s], normalize).T)
    return jnp.stack(components)


def _quadrature_var(values: Array) -> Array:
    """Var_nu of functions tabulated at the 64 nodes: sum_q w_q f_q^2 - (sum_q w_q f_q)^2.

    Column-wise on a (Q, D) table; the same rule `synthobj.kernel.nu_var` applies to the
    objectives, so an index and a label are one quantity.
    """
    mean = GL_WEIGHTS @ values
    return GL_WEIGHTS @ (values * values) - mean * mean


def _sobol_exact_additive(fitted: FittedGP) -> tuple[np.ndarray, float]:
    """First-order Sobol indices of the sample-averaged posterior mean, additive cells.

    m(x) = S^-1 sum_s sum_i m_{s,i}(x_i) is a sum of one-coordinate functions, each integrating to
    *exactly* zero against the 64-node rule that defines nu -- what centering buys -- so no
    estimator is needed: S_i = Var_nu(m_i) / sum_j Var_nu(m_j), over the sample-averaged mean the
    acquisition sees. Returns (S_hat (D,), Var_nu(m)), summing to 1.
    """
    X = fitted.columns(fitted.X_train)
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
    """One (s, s') term of the product/amplitude Var_nu(m); a scalar.

    sum_{n,m} alpha_{sn} alpha_{s'm} [prod_j (1 + a_{sj}^2 a_{s'j}^2 G_j(X_{nj}, X_{mj})) - 1],
    G_j(u, u') = sum_q w_q kbar_{sj}(t_q, u) kbar_{s'j}(t_q, u'), exact because the factors are
    independent under nu. Symmetric in (s, s'), so the caller doubles the off-diagonal pairs; one
    pair at a time holds the peak at (n, n, D), 32 MB at n = 200, D = 100.
    """
    G = jnp.einsum("q,qnd,qmd->nmd", GL_WEIGHTS, kbar_s, kbar_t)
    return alpha_s @ (jnp.prod(1.0 + a_sq_s * a_sq_t * G, axis=-1) - 1.0) @ alpha_t


def _sobol_exact_product_amplitude(fitted: FittedGP) -> tuple[np.ndarray, float]:
    """First-order Sobol indices of the sample-averaged posterior mean, product/amplitude.

    m_s(x) = sum_n alpha_{sn} prod_j (1 + a_{sj}^2 kbar_{sj}(x_j, X_{nj})), whose marginal in x_i
    is sum_n alpha_{sn} (1 + a_{si}^2 kbar_{si}(x_i, X_{ni})) because centering makes E_{x_j}[1 +
    a^2 kbar] = 1; its sample-averaged quadrature variance is the numerator over the exact
    Var_nu(m) = S^-2 sum_{s,s'} `_pair_term`. These do *not* sum to 1.
    """
    X = fitted.columns(fitted.X_train)
    T = jnp.tile(GL_NODES[:, None], (1, X.shape[1]))
    a_sq = fitted.samples["a_sq"]
    ells = fitted.samples["kernel_ell"]
    alphas = fitted.alphas()
    S = alphas.shape[0]

    # One (Q, n, D) tensor per sample -- 10 MB each at n = 200, D = 100 -- built once because
    # every one of the S(S+1)/2 pairs below needs two of them.
    kbars = [kbar_all(T, X, ells[s], True) for s in range(S)]

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
    """Saltelli's first-order estimator of the posterior mean's Sobol indices.

    The ARD Matern-5/2 -- ("product", "lengthscale") and the MAP references, which share its kernel
    -- does not factor over coordinates under nu, so its index is estimated over one scrambled
    Sobol set in 2D dimensions, on the sample-averaged posterior mean.
    `test_qmc_matches_exact_on_product_amplitude` pins the accuracy against a cell whose exact
    indices are known; coordinates outside `active` are skipped. Returns (S_hat (D,), V), where `V`
    is Saltelli's denominator: the variance of the sample-averaged posterior mean over the A and B
    points together. That is not the quadrature `Var_nu(m)` the exact routes return.
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

    `native` is the cell's own sparsity parameter, a_sq_i or rho_i, and `p_active` thresholds it in
    that cell's own units: ACTIVE_EPS on a share, RHO_EPS on rho. `compute_sobol=False` leaves
    `sobol_hat` and `total_var_hat` NaN for the cheap in-loop rows. Keys: `native` (S, D),
    `native_median` (D,), `share_hat` (S, D) or None, `p_active` (D,), `sobol_hat` (D,),
    `total_var_hat` float, `active_neutral` = sobol_hat > eps, `active_native` = p_active > 0.5;
    numpy, zero-padded back to D outside `active` so a row has one width.
    """
    D = fitted.X_train.shape[1]
    is_amplitude = isinstance(fitted.cell, tuple) and CELLS[fitted.cell].native_site == "a_sq"
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
            np.asarray(shares_from_amplitudes(fitted.samples["a_sq"], fitted.noises()))
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
        elif is_amplitude and fitted.cell[0] == "product":
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
    """Rank agreement between one fit's readouts and the objective's labels.

    Spearman rather than Pearson, over all D coordinates: the question is whether the surrogate
    *orders* the coordinates the way the objective does, and in three of the four cells the two
    quantities are not even on the same scale (rho_i is not a share). Both readouts are scored
    against both labels: `labels.s`, the variance share `sobol_hat` estimates, and `labels.g`, the
    realized slope share a lengthscale readout can track; `spearman_native_vs_sobol` measures how
    much the cell's own parameterization already tells you, against the parameterization-neutral
    readout. `amplitude_vs_realized` pairs `labels.s` on the active coordinates with the median,
    over the usable draws, of the readout that states a *variance*.
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
