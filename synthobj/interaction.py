"""Pairwise product interactions, and the exact maximum of one paired block.

`Interaction` is the study's only source of non-additivity: `h_ij(x_i, x_j) = c u_i(x_i)
u_j(x_j)`, built from two `Component`s drawn with `share = 1`, so each factor has quadrature
mean 0 and quadrature variance 1 under the uniform reference measure U[0,1]. Under that product
measure the identities the thesis's variance budget rests on hold to machine precision, not
merely to quadrature accuracy (measured below 4e-16 on every one of them):

- `E_nu[h_ij | x_i] = c u_i(x_i) E_nu[u_j] = 0` for every `x_i`, because the centering of `u_j`
  was solved at exactly the 64 Gauss-Legendre nodes the quadrature sums over;
- `E_nu[h_ij f_i] = c E_nu[u_i f_i] E_nu[u_j] = 0`, again from the `u_j` factor alone --
  `E_nu[u_i f_i]` is in general *not* zero;
- `Var_nu h_ij = c^2 Var u_i Var u_j = c^2`, so the prescribed interaction share is `s_ij = c^2`
  and `c = sqrt(s_ij)`.

`block_argmax` is the second half of the plan's exact-`f_star` claim (decision D1). Interaction
pairs are disjoint, so a block's maximum is independent of every other block and `f_star` is the
sum of per-block maxima plus the unpaired 1-D maxima from `Component.argmax`. The two-dimensional
maximum has no closed form, so it is found by scanning the components' knot grid and refining the
best cells with L-BFGS-B on the analytic gradient. Measured over four independent blocks at
ell = 0.5 (main shares 0.1875, c = 0.5) on the production 1088 x 1088 knot grid: refinement moves
the value by 0 to 7.1e-6, and the refined maximum beat the best of 10^7 uniform random probes in
every trial, by 6.3e-7 to 6.1e-3. A block costs 0.02 to 0.04 s.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from synthobj.component import Component


@dataclass(frozen=True, eq=False)
class Interaction:
    """`h_ij(x_i, x_j) = c u_i(x_i) u_j(x_j)` on the coordinate pair `(i, j)`.

    `u_i` and `u_j` must be built with `share = 1`; `c = sqrt(s_ij)` then makes `Var_nu h_ij`
    exactly the prescribed interaction share `s_ij`. `i` and `j` are the coordinates in the
    objective's input space, carried as labels -- neither is used to evaluate `h`.

    Declared `eq=False` for the same reason as `Component` (controller ruling R4): frozen
    dataclasses holding arrays or `Component`s on this branch compare by identity.
    """

    i: int
    j: int
    c: float
    u_i: Component
    u_j: Component

    def __call__(self, xi: float | np.ndarray, xj: float | np.ndarray) -> np.ndarray:
        """`c u_i(xi) u_j(xj)`, broadcasting `xi` against `xj` as numpy does.

        Elementwise for two arrays of the same shape, and the full outer product for `xi[:, None]`
        against `xj[None, :]`. NaN wherever either factor is outside its knot range, since
        `Component` does not extrapolate.
        """
        return self.c * self.u_i(xi) * self.u_j(xj)

    def nu_var(self) -> float:
        """`Var_nu h_ij = c^2 Var u_i Var u_j` under U[0,1]; equals `s_ij` to roundoff."""
        return float(self.c**2 * self.u_i.nu_var() * self.u_j.nu_var())


def block_value_and_grad(
    f_i: Component, f_j: Component, inter: Interaction, xy: np.ndarray
) -> tuple[float, np.ndarray]:
    """`phi(x, y) = f_i(x) + f_j(y) + c u_i(x) u_j(y)` and its gradient at the single point `xy`.

    `xy` is a 2-vector. The returned gradient has shape (2,) and is analytic, read from the
    splines' first derivatives: `(f_i'(x) + c u_i'(x) u_j(y), f_j'(y) + c u_i(x) u_j'(y))`.
    """
    x, y = float(xy[0]), float(xy[1])
    ui, uj = float(inter.u_i(x)), float(inter.u_j(y))
    value = float(f_i(x)) + float(f_j(y)) + inter.c * ui * uj
    grad = np.array(
        [
            float(f_i.derivative(x)) + inter.c * float(inter.u_i.derivative(x)) * uj,
            float(f_j.derivative(y)) + inter.c * ui * float(inter.u_j.derivative(y)),
        ]
    )
    return value, grad


def _unit_knots(comp: Component) -> np.ndarray:
    """`comp`'s knots inside [0,1], with 0.0 and 1.0 added: one axis of the search grid.

    The objective is only ever evaluated on the cube, so a component drawn on the wider box the
    rotated family needs (plan decision D2) must not be searched outside it. Adding the two
    endpoints keeps the corners of the cube in the scan when the wider grid has no knot there;
    for a component drawn on [0,1] they are already knots and `np.unique` drops the duplicates.
    """
    inside = comp.grid[(comp.grid >= 0.0) & (comp.grid <= 1.0)]
    return np.unique(np.concatenate([[0.0, 1.0], inside]))


def block_argmax(
    f_i: Component, f_j: Component, inter: Interaction, top_k: int = 20
) -> tuple[np.ndarray, float]:
    """Maximum of `phi(x, y) = f_i(x) + f_j(y) + h_ij(x, y)` on [0,1]^2: `((x*, y*), phi*)`.

    `phi` is evaluated on the outer product of the two components' knots restricted to the cube
    (1088 x 1088 for the production grid), and the `top_k` best cells are each refined by L-BFGS-B
    with the analytic gradient of `block_value_and_grad`, bounds [(0,1), (0,1)], `ftol=1e-15` and
    `gtol=1e-12`. The best of the knot maximum and the refinements is returned, so the result can
    never be worse than the scan; when the scan already sits on the maximum, refinement gains
    exactly nothing and the knot point is returned unchanged. The knot maximum is within about
    1e-5 of the true maximum and refinement takes it below 1e-10 -- accurate but not closed form,
    unlike the exact 1-D `Component.argmax`.

    `u_i` and `u_j` are assumed to cover [0,1], which holds for every component the plan builds.
    """

    def negated(p: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = block_value_and_grad(f_i, f_j, inter, p)
        return -value, -grad

    xs, ys = _unit_knots(f_i), _unit_knots(f_j)
    phi = f_i(xs)[:, None] + f_j(ys)[None, :] + inter(xs[:, None], ys[None, :])
    flat = phi.ravel()

    best_cell = int(flat.argmax())
    p, q = np.unravel_index(best_cell, phi.shape)
    best_xy, best_value = np.array([xs[p], ys[q]]), float(flat[best_cell])

    for index in np.argpartition(flat, -top_k)[-top_k:]:
        p, q = np.unravel_index(int(index), phi.shape)
        result = minimize(
            negated,
            np.array([xs[p], ys[q]]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        if -float(result.fun) > best_value:
            best_xy, best_value = result.x, -float(result.fun)

    return best_xy, best_value
