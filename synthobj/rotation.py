"""Planar rotation of disjoint active-coordinate pairs: the family's only source of non-additivity.

`PairRotation.forward` maps `x` to `z = 1/2 + R_theta (x - 1/2)` on each of its coordinate pairs and
leaves every other coordinate unchanged, so `f_a(z_a) + f_b(z_b)` is additive in the rotated
coordinates but not in the axis ones -- the non-additivity is entirely a change of basis, with no
explicit interaction term (plan decision D2). For `x` in the unit square the offset `x - 1/2` has
norm at most `sqrt(2)/2`, so `z` ranges over `0.5 +/- 0.7071`; `EXT_LO`/`EXT_HI` extend that by a
0.01 margin on each side (ruling R7), and rotated components are drawn on
`make_grid(EXT_LO, EXT_HI, n_ext)` rather than on `[0, 1]` so `z` never leaves their knot range and
is never clipped or contracted -- keeping the components' nominal lengthscales. Contracting the
rotated square back into the cube was rejected: it multiplies effective lengthscales by
`|cos theta| + |sin theta|` and would confound rotation with smoothness.

Rotating moves variance around: `Var_nu f_rot` is in general not 1, since `kernel.normalized` and
`Component.from_raw` both normalize under `U[0,1]`, and the centered kernel's marginal variance is
larger outside that reference interval. `SyntheticObjective` renormalizes by a `scale` computed from
`rotated_block_stats`'s per-pair quadrature (blocks are independent under the product measure, so
their variances add). The same quadrature also splits each pair's variance into axis-aligned
first-order Sobol shares (`Labels.s_axis`) and a remainder (`Labels.gamma_axis`) that a first-order
additive model cannot see -- unlike `Labels.gamma`, which stays 0 for this family because it has no
explicit `Interaction`.

Rotating also moves the *mean*: `f_a`/`f_b` are centered under their own coordinate's `U[0,1]`
marginal, but `z_a`, `z_b` are linear combinations of two independent uniforms, so `E_nu[phi]` is
generically nonzero away from a multiple of 90 degrees (ruling R30). `rotated_block_mean` computes
that quantity by the same quadrature `rotated_block_stats` uses internally for its own mean, and
`SyntheticObjective` subtracts the sum of it over every rotated pair (once, alongside `scale`) to
restore `E_nu[f] = 0`.

`rotated_block_argmax` is `interaction.block_argmax`'s analogue for a rotated pair (ruling R8): a
rotated component's knots live on the extended domain and are not the set `phi` is ever evaluated
on, so the search is a uniform 1024x1024 grid on `[0,1]^2` in x-space, with the best `top_k` cells
refined by L-BFGS-B on the analytic gradient through the chain rule.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from synthobj.component import Component
from synthobj.kernel import GL_NODES, GL_WEIGHTS

# For x in [0,1]^2, |x - (0.5,0.5)| <= sqrt(2)/2 = 0.7071..., so a rotated coordinate ranges over
# 0.5 +/- sqrt(2)/2; the extra 0.01 margin on each side (ruling R7) keeps z strictly inside the knot
# range of a component built on this box, even at a floating-point corner of [0,1]^2.
EXT_LO = 0.5 - np.sqrt(2) / 2 - 0.01
EXT_HI = 1.0 - EXT_LO


def _rotated_pair(xa: np.ndarray, xb: np.ndarray, c: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    """`z_a, z_b = 1/2 + R_theta (x_a - 1/2, x_b - 1/2)`, given `c = cos theta`, `s = sin theta`."""
    da, db = xa - 0.5, xb - 0.5
    return 0.5 + c * da - s * db, 0.5 + s * da + c * db


@dataclass(frozen=True, eq=False)
class PairRotation:
    """A Givens rotation by `theta_deg` about the cube centre, applied to each pair in `pairs`.

    `pairs` must be disjoint coordinate pairs; `SyntheticObjective` checks that, not this class.
    Every coordinate not mentioned in `pairs` is left unchanged by `forward`. Declared `eq=False`
    for the same reason as `Component` and `Interaction` (ruling R4): equality is identity.
    """

    theta_deg: float
    pairs: tuple[tuple[int, int], ...]

    def R(self) -> np.ndarray:
        """The 2x2 rotation matrix `[[cos, -sin], [sin, cos]]` at `theta_deg`."""
        theta = np.deg2rad(self.theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def forward(self, X: np.ndarray) -> np.ndarray:
        """`z = 1/2 + R_theta (x - 1/2)` on each pair; other coordinates unchanged; a copy of `X`.

        The coordinate axis is `X`'s last axis, so this accepts a single point of shape `(D,)` or a
        batch of shape `(n, D)` alike.
        """
        Z = np.array(X, dtype=float)
        theta = np.deg2rad(self.theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        for i, j in self.pairs:
            Z[..., i], Z[..., j] = _rotated_pair(X[..., i], X[..., j], c, s)
        return Z


def rotated_block_mean(f_a: Component, f_b: Component, rot: PairRotation) -> float:
    """`E_nu[phi]` for `phi(x_a, x_b) = f_a(z_a) + f_b(z_b)`, by 64x64 Gauss-Legendre quadrature.

    `f_a`/`f_b` are centered so `E_{u~U[0,1]}[f_a(u)] = 0` under their OWN coordinate's marginal
    (`Component.from_raw`), but `z_a = 1/2 + cos(theta)(x_a - 1/2) - sin(theta)(x_b - 1/2)` is a
    linear combination of two independent uniforms, whose marginal is not `U[0,1]` -- so integrating
    a nonlinear draw against it is generically nonzero away from a multiple of 90 degrees (ruling
    R30). `SyntheticObjective` subtracts the sum of this over its rotated pairs once, alongside
    `scale`, to restore `E_nu[f] = 0`.
    """
    theta = np.deg2rad(rot.theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    xa, xb = np.meshgrid(GL_NODES, GL_NODES, indexing="ij")
    za, zb = _rotated_pair(xa, xb, c, s)
    phi = f_a(za) + f_b(zb)
    w = GL_WEIGHTS[:, None] * GL_WEIGHTS[None, :]
    return float((w * phi).sum())


def rotated_block_stats(f_a: Component, f_b: Component, rot: PairRotation) -> tuple[float, float, float]:
    """64x64 Gauss-Legendre quadrature of `phi(x_a, x_b) = f_a(z_a) + f_b(z_b)` under `U[0,1]^2`.

    Returns `(Var phi, Var_{x_a} E_{x_b} phi, Var_{x_b} E_{x_a} phi)`: the block's total variance and
    its two axis-aligned first-order Sobol variances. `phi` is additive in `(z_a, z_b)` but the
    rotation mixes `(x_a, x_b)`, so in general `Var phi >= Var_{x_a}E_{x_b}phi + Var_{x_b}E_{x_a}phi`
    (the ANOVA remainder is a variance and cannot be negative), with equality only when
    `rot.theta_deg` is a multiple of 90 degrees -- and even then the two first-order terms reduce to
    `f_a.nu_var()` and `f_b.nu_var()` **in order** only at an even multiple (0, 180, ...); at an odd
    multiple (90, 270, ...) the rotation exchanges the two coordinates (`z_a` depends only on `x_b`,
    `z_b` only on `x_a`), so the pair comes out swapped: `Var_{x_a}E_{x_b}phi = f_b.nu_var()` and
    `Var_{x_b}E_{x_a}phi = f_a.nu_var()`.

    `mean_phi` is computed via `rotated_block_mean` rather than re-derived here, so it is the exact
    same quantity `SyntheticObjective` subtracts for `mu` (ruling R30) -- not a second, merely
    numerically-close copy of the same quadrature.
    """
    theta = np.deg2rad(rot.theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    xa, xb = np.meshgrid(GL_NODES, GL_NODES, indexing="ij")
    za, zb = _rotated_pair(xa, xb, c, s)
    phi = f_a(za) + f_b(zb)

    w = GL_WEIGHTS[:, None] * GL_WEIGHTS[None, :]
    mean_phi = rotated_block_mean(f_a, f_b, rot)
    var_phi = float((w * phi**2).sum()) - mean_phi**2

    e_given_a = phi @ GL_WEIGHTS  # E_{x_b}[phi | x_a], one value per x_a node
    e_given_b = GL_WEIGHTS @ phi  # E_{x_a}[phi | x_b], one value per x_b node
    var_a = float((GL_WEIGHTS * e_given_a**2).sum()) - mean_phi**2
    var_b = float((GL_WEIGHTS * e_given_b**2).sum()) - mean_phi**2
    return var_phi, var_a, var_b


def rotated_block_value_and_grad(
    f_a: Component, f_b: Component, rot: PairRotation, xy: np.ndarray
) -> tuple[float, np.ndarray]:
    """`phi(x_a, x_b) = f_a(z_a) + f_b(z_b)` and its gradient at the single point `xy`.

    `xy` is a 2-vector. The gradient follows from the chain rule through the rotation:
    `d phi/d x_a = f_a'(z_a) cos theta + f_b'(z_b) sin theta` and
    `d phi/d x_b = -f_a'(z_a) sin theta + f_b'(z_b) cos theta`. `z_a`, `z_b` are clamped into
    `f_a`'s/`f_b`'s knot range before evaluation (the task-6 brief's "integration hazard"): the 0.01
    margin in `EXT_LO`/`EXT_HI` keeps every quadrature and search-grid point strictly inside, but a
    floating-point corner of `[0,1]^2` can overshoot it by a handful of ulps, and the spline does not
    extrapolate.
    """
    theta = np.deg2rad(rot.theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    xa, xb = float(xy[0]), float(xy[1])
    za, zb = _rotated_pair(xa, xb, c, s)
    za = float(np.clip(za, f_a.grid[0], f_a.grid[-1]))
    zb = float(np.clip(zb, f_b.grid[0], f_b.grid[-1]))

    value = float(f_a(za)) + float(f_b(zb))
    dfa, dfb = float(f_a.derivative(za)), float(f_b.derivative(zb))
    grad = np.array([dfa * c + dfb * s, -dfa * s + dfb * c])
    return value, grad


def rotated_block_argmax(
    f_a: Component, f_b: Component, rot: PairRotation, top_k: int = 20
) -> tuple[np.ndarray, float]:
    """Maximum of `phi(x_a, x_b) = f_a(z_a) + f_b(z_b)` on `[0,1]^2` in x-space: `(x*, phi*)`.

    As `interaction.block_argmax`, but the search grid is a uniform 1024x1024 grid on `[0,1]^2` in
    x-space rather than the components' knots (ruling R8): a rotated component's knots live on the
    extended domain `[EXT_LO, EXT_HI]`, which is not the set `phi` is ever evaluated on. The `top_k`
    best cells are each refined by L-BFGS-B with the analytic gradient of
    `rotated_block_value_and_grad`, bounds `[(0,1), (0,1)]`, `ftol=1e-15` and `gtol=1e-12`. `z` is
    clamped into each component's knot range before every evaluation, coarse grid included.
    """
    theta = np.deg2rad(rot.theta_deg)
    c, s = np.cos(theta), np.sin(theta)

    def negated(p: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = rotated_block_value_and_grad(f_a, f_b, rot, p)
        return -value, -grad

    n = 1024
    xs = np.linspace(0.0, 1.0, n)
    xa, xb = np.meshgrid(xs, xs, indexing="ij")
    za, zb = _rotated_pair(xa, xb, c, s)
    za = np.clip(za, f_a.grid[0], f_a.grid[-1])
    zb = np.clip(zb, f_b.grid[0], f_b.grid[-1])
    phi = f_a(za) + f_b(zb)
    flat = phi.ravel()

    best_cell = int(flat.argmax())
    p, q = np.unravel_index(best_cell, phi.shape)
    best_xy, best_value = np.array([xs[p], xs[q]]), float(flat[best_cell])

    for index in np.argpartition(flat, -top_k)[-top_k:]:
        p, q = np.unravel_index(int(index), phi.shape)
        result = minimize(
            negated,
            np.array([xs[p], xs[q]]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        if -float(result.fun) > best_value:
            best_xy, best_value = result.x, -float(result.fun)

    return best_xy, best_value
