"""sagp.r2d2: the R2-D2 shrinkage prior on a cell's native coordinates, through a Gaussian copula.

The prior (Zhang et al. 2016) puts theta_i = omega * phi_i on the D native coordinates of a cell
(its amplitudes a_sq or its inverse squared lengthscales rho): a global omega = R2 / (1 - R2) with
R2 ~ Beta(a, b), and shares phi ~ Dir(k, ..., k). Both of its exact parameterizations draw the
shares from standard normals -- phi_i = lam_i / sum_j lam_j with lam_i = F_k^-1(Phi(z_i)) the
Gamma(k, 1) quantile of z_i ~ N(0, 1) -- so NUTS sees an isotropic Gaussian whatever k is and
the non-Gaussianity sits in a deterministic map with an exact Jacobian, with no `numpyro.factor`:

- "reference": `r2d2_R2` ~ Beta(a, b) and the copula shares `r2d2_z_lam`; any (a, b, k).
- "tied": theta_i = lam_i / xi with xi the Gamma(b, 1) quantile of `r2d2_z_xi` ~ N(0, 1), every
  site N(0, 1). It is the same law exactly when a = k * dim (the paper's tie, its Proposition 5),
  and no other.

`sample_log_theta` is the one entry point, so a cell never branches on the form it runs.

The map y = log F_k^-1(Phi(z)) exists twice. `log_gamma_icdf_newton` solves for it by Newton in
log space (JAX has no inverse incomplete gamma): accurate, but a while-loop around `gammainc` and
`gammaincc`, which are while-loops themselves -- 15-21 ms per value-and-gradient at D = 100 on a
laptop CPU, 11-15 times a GP likelihood gradient -- so it only builds the table and serves as the
accuracy reference. `log_gamma_icdf`, the map the models run, is loop-free: a quintic Hermite
table built once per shape from the solver, outside any trace, and cached.

Imports jax and numpyro only -- nothing from `sagp`, nothing from torch -- so the prior-only
harness can import it without the BoTorch stack.
"""
from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array, lax
from jax.scipy.special import gammainc, gammaincc, gammaln, log_ndtr, logsumexp
from jax.typing import ArrayLike

FORMS: tuple[str, ...] = ("reference", "tied")
SAMPLED_SITES: dict[str, tuple[str, ...]] = {
    "reference": ("r2d2_R2", "r2d2_z_lam"),   # Beta site on R2, copula shares
    "tied": ("r2d2_z_xi", "r2d2_z_lam"),      # every site N(0, 1)
}

# --- the reference solver ---

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
# Terms of the asymptotic series `log_ndtr` switches to below z = -20. At its default of 3 the
# series is off by 4e-9 at the switch -- a jump in log Phi there, 2e-11 relative in y -- and at 10
# by 3e-19, below double precision. Solver and map share this one log Phi.
_LOG_NDTR_TERMS = 10
# u = Phi(z) = 0.9. Below it the solver matches the lower tail P(k, x) to u, above it the upper
# tail Q(k, x) to 1 - u, so each branch works with the smaller probability and loses no digits.
_Z_SPLIT = 1.2816
# Below x = 1e-6, log P(k, x) comes from its series, the only way to evaluate it once x underflows.
_LOG_X_SERIES = math.log(1e-6)
# The floor on P and Q before their logs. Q reaches it once 1 - Phi(z) < 1e-300, past z = 37,
# which is why the table stops at z = 36.
_TINY = 1e-300
_STEP_CLIP = 2.0
_TOL = 1e-10
_MAX_ITER = 100


def _log_phi(z: ArrayLike) -> Array:
    """log Phi(z), the one used by the solver and the map alike."""
    return log_ndtr(z, series_order=_LOG_NDTR_TERMS)


def _log_p(y: Array, k: float) -> Array:
    """log P(k, e^y), the log lower regularized incomplete gamma at x = e^y.

    Below x = 1e-6 the series k y - lgamma(k + 1) - k x / (k + 1): its neglected term is below
    k x^2 / 4, which moves the solved y by less than 3e-13. Above it, log `gammainc`.
    """
    x = jnp.exp(y)
    series = k * y - gammaln(k + 1.0) - k * x / (k + 1.0)
    direct = jnp.log(jnp.maximum(gammainc(k, jnp.maximum(x, _TINY)), _TINY))
    return jnp.where(y < _LOG_X_SERIES, series, direct)


def _dy_dz(z: ArrayLike, y: ArrayLike, k: float) -> Array:
    """dy/dz = phi(z) / (x f_k(x)) at x = e^y: F_k(x) = Phi(z) differentiated implicitly.

    Formed as one exponential of log phi(z) - log(x f_k(x)), which stays finite wherever y is.
    """
    return jnp.exp(-0.5 * z * z - _LOG_SQRT_2PI - (k * y - jnp.exp(y) - gammaln(k)))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def log_gamma_icdf_newton(z: ArrayLike, k: float) -> Array:
    """y = log F_Gamma(k,1)^-1(Phi(z)) by Newton in log space: the verified reference solver.

    `icdf2.py::log_gamma_icdf_from_normal` from the scouting, hardened. It solves for y = log x,
    not x, because x itself underflows (at k = 0.01 the median is x = 4.5e-31, and Phi(-8) maps
    to y = -3502), and takes Phi(z) and 1 - Phi(z) as `log_ndtr(z)` and `log_ndtr(-z)`:

    - lower branch, z < 1.2816 (Phi(z) < 0.9): Newton on g(y) = log P(k, e^y) - log Phi(z), with
      g' = x f_k(x) / P, log P from its series below x = 1e-6 and from log `gammainc` above,
      started at the small-x asymptote y0 = (log Phi(z) + lgamma(k + 1)) / k;
    - upper branch, z >= 1.2816: Newton on g(y) = log Q(k, e^y) - log(1 - Phi(z)), with
      g' = -x f_k(x) / Q and Q from `gammaincc`, started at the large-x asymptote
      y0 = log(t + (k - 1) log t) (the log term only where t > 1), t = -log(1 - Phi(z)).

    Steps are clipped to [-2, 2] as a safeguard far from the root. Each element stops once its
    step falls below 1e-10 max(1, |y|) -- the next would be below 1e-20 by quadratic convergence,
    so the value is converged to rounding -- and is frozen there, so its value does not depend on
    the rest of the batch; one that has not stopped after 100 iterations is returned as NaN
    rather than as a silently wrong number. At the study's shapes (0.005 to 20.93) every z on
    [-38, 36], |z| <= 8 included, stops within 14 iterations at k = 0.005, within 10 at k = 0.01
    and within 8 above. There it agrees with 40-digit mpmath to 6e-13 relative on [-38, 36] --
    the worst at k = 0.005 near z = 3, where `gammaincc` forms Q as 1 - P; below 5e-14 from
    k = 0.0333 up -- and with SciPy's inverse incomplete gamma to 7e-13 relative on z in
    [-2.4, 6] wherever SciPy's x is a normal double (below e^-708 SciPy's x is subnormal and
    rounds: 1.5e-11 relative off at k = 0.005, z = -1.94). The upper branch's floor on Q at
    1e-300 makes it invalid past z = 37, where 1 - Phi(z) is smaller.

    The derivative is `custom_jvp`'s closed form dy/dz = phi(z) / (x f_k(x)), never a derivative
    of the iterations. Off the hot path: it builds the runtime table and is the tests' reference.
    `k` is a Python float, a constant of the cell.
    """
    z = jnp.asarray(z, dtype=float)
    log_u, log_1mu = _log_phi(z), _log_phi(-z)
    lower = z < _Z_SPLIT
    t = jnp.maximum(-log_1mu, 1e-6)
    y0 = jnp.where(
        lower,
        (log_u + gammaln(k + 1.0)) / k,
        jnp.log(t + (k - 1.0) * jnp.log(t) * (t > 1.0)),
    )

    def newton_step(y: Array) -> Array:
        x = jnp.exp(y)
        log_xf = k * y - x - gammaln(k)  # log(x f_k(x)), the slope of P(k, e^y) in y
        log_p = _log_p(y, k)
        log_q = jnp.log(jnp.maximum(gammaincc(k, x), _TINY))
        g = jnp.where(lower, log_p - log_u, log_q - log_1mu)
        dg = jnp.where(lower, jnp.exp(log_xf - log_p), -jnp.exp(log_xf - log_q))
        return jnp.clip(g / dg, -_STEP_CLIP, _STEP_CLIP)

    def not_done(state: tuple[int, Array, Array]) -> Array:
        i, _, done = state
        return (i < _MAX_ITER) & ~jnp.all(done)

    def iterate(state: tuple[int, Array, Array]) -> tuple[int, Array, Array]:
        i, y, done = state
        step = newton_step(y)
        y = jnp.where(done, y, y - step)
        return i + 1, y, done | (jnp.abs(step) <= _TOL * jnp.maximum(1.0, jnp.abs(y)))

    _, y, done = lax.while_loop(not_done, iterate, (0, y0, jnp.zeros(z.shape, dtype=bool)))
    return jnp.where(done, y, jnp.nan)


@log_gamma_icdf_newton.defjvp
def _log_gamma_icdf_newton_jvp(k: float, primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    (z,), (dz,) = primals, tangents
    y = log_gamma_icdf_newton(z, k)
    return y, _dy_dz(z, y, k) * dz


# --- the runtime map ---

# Knots z_j = -38 + j / 64 on [-38, 36]: 4,736 intervals. The power-of-two spacing makes every
# knot exact and the index arithmetic exact up to the subtraction z + 38.
_Z_LO, _Z_HI = -38.0, 36.0
_KNOTS_PER_UNIT = 64
_N_INTERVALS = int((_Z_HI - _Z_LO) * _KNOTS_PER_UNIT)


class _Table(NamedTuple):
    """One shape's table: r(z) = k y(z) - log Phi(z) on [-38, 36] and the two tails."""

    coef: Array  # (4736, 6): on interval j, r = sum_m coef[j, m] t^m, t = 64 (z - z_j) in [0, 1]
    r_lo: float  # r(-38), the constant of the asymptote below the table
    y_hi: float  # y(36) and ...
    dy_hi: float  # ... y'(36): the tangent line above the table


_TABLES: dict[float, _Table] = {}


def _table(k: float) -> _Table:
    """The table for shape k, built on first use and cached per shape."""
    k = float(k)
    if k not in _TABLES:
        _TABLES[k] = _build_table(k)
    return _TABLES[k]


def _build_table(k: float) -> _Table:
    """Solve at every knot and fit a quintic Hermite interpolant of r = k y - log Phi(z).

    Built under `jax.ensure_compile_time_eval`, so a first call inside `jit` or `grad` computes
    the table rather than staging the solver's while-loops into the traced program. The knot
    data are the solver's y and the closed forms y' = phi(z) / (x f_k(x)) and
    y'' = y' (-z - (k - x) y') (x = e^y), with (log Phi)' = phi / Phi and
    (log Phi)'' = -(log Phi)' (z + (log Phi)'). The residual r rather than y is tabulated because
    y reaches -1.5e5 at k = 0.005 and z = -38, while r tends to lgamma(k + 1) as z -> -inf: its
    interpolation error, divided by k, stays below 1e-12 relative in y at every study shape
    (the worst is 9e-13, at k = 0.005, z = 3.06). r's value, slope and curvature at both ends of an
    interval fix the quintic; its six coefficients in t are stored per interval.
    """
    h = 1.0 / _KNOTS_PER_UNIT
    with jax.ensure_compile_time_eval():
        z = _Z_LO + h * jnp.arange(_N_INTERVALS + 1, dtype=float)
        y = log_gamma_icdf_newton(z, k)
        if not bool(jnp.all(jnp.isfinite(y))):
            raise RuntimeError(f"log_gamma_icdf_newton did not converge on every knot at k = {k}")
        x = jnp.exp(y)
        dy = _dy_dz(z, y, k)
        d2y = dy * (-z - (k - x) * dy)
        log_phi = _log_phi(z)
        m1 = jnp.exp(-0.5 * z * z - _LOG_SQRT_2PI - log_phi)  # (log Phi)'
        m2 = -m1 * (z + m1)  # (log Phi)''
        r, dr, d2r = k * y - log_phi, k * dy - m1, k * d2y - m2
        # In t: value f, slope d = h r' and curvature s = h^2 r'' at each end of each interval.
        f0, f1 = r[:-1], r[1:]
        d0, d1 = h * dr[:-1], h * dr[1:]
        s0, s1 = h * h * d2r[:-1], h * h * d2r[1:]
        c0, c1, c2 = f0, d0, 0.5 * s0
        rest_f, rest_d, rest_s = f1 - (c0 + c1 + c2), d1 - (c1 + 2.0 * c2), s1 - 2.0 * c2
        coef = jnp.stack(
            [
                c0, c1, c2,
                10.0 * rest_f - 4.0 * rest_d + 0.5 * rest_s,
                -15.0 * rest_f + 7.0 * rest_d - rest_s,
                6.0 * rest_f - 3.0 * rest_d + 0.5 * rest_s,
            ],
            axis=-1,
        )
        return _Table(coef=coef, r_lo=float(r[0]), y_hi=float(y[-1]), dy_hi=float(dy[-1]))


def log_gamma_icdf(z: ArrayLike, k: float) -> Array:
    """The runtime map: the same y, loop-free, from a table cached per shape k.

    y = (log Phi(z) + r(z)) / k with r = k y - log Phi(z) interpolated from the shape's table
    (`_build_table`: quintic Hermite, knots every 1/64 on [-38, 36], built once per shape from
    `log_gamma_icdf_newton`): the knot index is arithmetic on z, the coefficients a gather, the
    quintic a Horner sum, so the map has no loop and costs about 20 microseconds per
    value-and-gradient at D = 100 on a laptop CPU. It agrees with the solver to 1e-12 relative
    over the table at every study shape (0.005 to 20.93). Outside the table:

    - below z = -38, the small-x asymptote y = (log Phi(z) + r(-38)) / k: r(-38) is lgamma(k + 1)
      up to k x / (k + 1), which is far below rounding there, and up to the solver's rounding,
      which is not -- the two differ by up to 3e-13 at the study's shapes (the tail test bounds it
      at 1e-12). Taking r(-38) from the table makes the map continuous there, and it agrees with
      the solver there to 1e-14 relative in y (measured 2e-16);
    - above z = 36, the tangent line at 36, y(36) + y'(36) (z - 36) with the solver's y and
      closed-form y' (past z = 37 the solver's Q underflows; the prior mass beyond 36 is about
      1e-284).

    So the map is finite and strictly increasing with a finite positive derivative for every
    finite z, and continuously differentiable everywhere. Autodiff through the interpolant is
    its gradient, consistent with the map by construction. `k` is a Python float.
    """
    table = _table(k)
    z = jnp.asarray(z, dtype=float)
    # Inside the table z itself, so the gradient flows; at either edge beyond it, a constant.
    z_in = jnp.where(z < _Z_LO, _Z_LO, jnp.where(z > _Z_HI, _Z_HI, z))
    u = (z_in - _Z_LO) * _KNOTS_PER_UNIT
    j = jnp.minimum(jnp.floor(u), _N_INTERVALS - 1)
    t = u - j
    c = table.coef[j.astype(int)]
    r = c[..., 5]
    for m in (4, 3, 2, 1, 0):  # Horner
        r = c[..., m] + t * r
    log_phi = _log_phi(z)
    return jnp.where(
        z < _Z_LO,
        (log_phi + table.r_lo) / k,
        jnp.where(z > _Z_HI, table.y_hi + table.dy_hi * (z - _Z_HI), (log_phi + r) / k),
    )


# --- the prior ---


def sample_reference(dim: int, *, a: float, b: float, k: float) -> Array:
    """log theta: r2d2_R2 ~ Beta(a, b), r2d2_z_lam ~ N(0, I); y = log_gamma_icdf(z_lam, k),
    log theta = log R2 - log1p(-R2) + y - logsumexp(y). Any (a, b, k).

    y - logsumexp(y) is log phi, the Dirichlet shares normalized in log space (the Gammas
    themselves underflow at small k). Returns (dim,).
    """
    r2 = numpyro.sample("r2d2_R2", dist.Beta(a, b))
    z_lam = numpyro.sample("r2d2_z_lam", dist.Normal(jnp.zeros(dim), 1.0))
    y = log_gamma_icdf(z_lam, k)
    return jnp.log(r2) - jnp.log1p(-r2) + y - logsumexp(y)


def sample_tied(dim: int, *, b: float, k: float) -> Array:
    """log theta = log_gamma_icdf(z_lam, k) - log_gamma_icdf(z_xi, b): the paper's a = k * dim.

    theta_i = lam_i / xi with lam_i ~ Gamma(k) and xi ~ Gamma(b), so R2 = sum lam / (sum lam + xi)
    is Beta(k * dim, b), independent of the shares (the Gamma-Dirichlet theorem). Sites
    r2d2_z_xi ~ N(0, 1), then r2d2_z_lam ~ N(0, I). Returns (dim,).
    """
    z_xi = numpyro.sample("r2d2_z_xi", dist.Normal(0.0, 1.0))
    z_lam = numpyro.sample("r2d2_z_lam", dist.Normal(jnp.zeros(dim), 1.0))
    return log_gamma_icdf(z_lam, k) - log_gamma_icdf(z_xi, b)


def sample_log_theta(dim: int, *, form: str, k: float, b: float, a: float | None = None) -> Array:
    """The one entry point: theta_i = omega * phi_i, R2 ~ Beta(a, b), phi ~ Dir(k, ..., k).

    `a` None is the paper's tie a = k * dim (valid in both forms); "tied" with any other `a` raises
    ValueError (an `a` equal to k * dim up to rounding is the tie). Creates the sampled sites of
    `SAMPLED_SITES[form]` only; the caller names the deterministic native site. Returns (dim,)
    log theta.
    """
    if form == "reference":
        return sample_reference(dim, a=k * dim if a is None else a, b=b, k=k)
    if form == "tied":
        if a is not None and not math.isclose(a, k * dim, rel_tol=1e-12):
            raise ValueError(f"the tied form fixes a = k * dim = {k * dim}; got a = {a}")
        return sample_tied(dim, b=b, k=k)
    raise ValueError(f"unknown R2-D2 form {form!r}; the forms are {FORMS}")
