"""One coordinate's contribution to a synthetic objective: a spline with an exact variance share.

A `Component` wraps a raw 1-D draw (`synthobj.draws`) into the form the rest of the package
uses: centered to zero mean and rescaled to an exact variance share under the uniform reference
measure U[0,1], stored as values on a fixed knot vector, and evaluated by a not-a-knot cubic
spline with `extrapolate=False` (plan decision D3).

The share is exact, not approximate, because the thesis reads relevance off variance shares.
`from_raw` computes the mean and variance by 64-node Gauss-Legendre quadrature on the trailing
node values of the draw, and the same 64 nodes are then knots of the spline, so `nu_var()` sees
exactly the numbers the rescale was solved from and returns `share` to floating-point roundoff
rather than to interpolation error.

`nu_mean`, `nu_var` and `slope_energy` always integrate over [0,1], since they read `GL_NODES`.
That stays true for a component drawn on the wider box the rotated family needs (plan decision
D2): the extended knots only extend where the spline can be evaluated, not what it is
normalized against.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from synthobj import kernel

# Knots closer together than this are the same point (plan decision D3): a uniform grid point
# can land on a Gauss-Legendre node, and CubicSpline requires strictly increasing knots.
KNOT_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class Component:
    """A 1-D component f(x) stored as `values` on the knot vector `grid`.

    `grid` is the merged, sorted, deduplicated union of the draw grid with the 64 GL nodes
    (1088 entries for a 1024-point draw grid), and `values` are the final centered, rescaled
    and signed values there, so `(grid, values)` alone rebuilds the component -- constructing
    `Component(coord, ell, share, grid, values)` directly is how a saved component is loaded,
    with no random draw involved. `share` is `Var_nu f` under U[0,1] and `ell` is the
    lengthscale the draw came from, carried as a label; neither is used to evaluate f.

    Declared `eq=False` (controller ruling R4): the generated `__eq__` of a frozen dataclass
    compares the fields as a tuple, which raises on the two ndarray fields. Equality is
    identity.
    """

    coord: int
    ell: float
    share: float
    grid: np.ndarray
    values: np.ndarray
    spline: CubicSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spline",
            CubicSpline(self.grid, self.values, bc_type="not-a-knot", extrapolate=False),
        )

    @classmethod
    def from_raw(
        cls,
        coord: int,
        ell: float,
        share: float,
        grid: np.ndarray,
        raw_joint_values: np.ndarray,
        sign: float = 1.0,
    ) -> "Component":
        """Center, rescale to `Var_nu = share` exactly, apply `sign`, merge knots, build the spline.

        `grid` is the uniform draw grid alone and `raw_joint_values` are the raw, uncentered
        draw values at `kernel.joint_points(grid)`, so its trailing 64 entries are the values
        at the GL nodes. The quadrature mean and variance are taken from those 64 entries, and
        every value -- grid and node alike -- is mapped by `x -> sign * sqrt(share / var) *
        (x - mean)`. Since `sqrt(share / var)` is positive, that map is affine with positive
        scale at `sign=+1`, preserving the raw draw's shape and monotonicity, and `sign=-1`
        mirrors the result about zero.

        The stored knots are `grid` unioned with the GL nodes, sorted and deduplicated at
        `KNOT_TOL`; on a tie the node's value is the one kept, so the quadrature in `nu_var`
        stays exact.
        """
        node_values = raw_joint_values[-kernel.GL_NODES.size :]
        scale = sign * np.sqrt(share / kernel.nu_var(node_values))
        joint_values = scale * (raw_joint_values - kernel.nu_mean(node_values))

        knots = kernel.joint_points(grid)
        # Stable sort puts a grid point before a node of the same value; keeping the last of
        # each near-duplicate run therefore keeps the node.
        order = np.argsort(knots, kind="stable")
        knots, joint_values = knots[order], joint_values[order]
        keep = np.ones(knots.size, dtype=bool)
        keep[:-1] = np.diff(knots) >= KNOT_TOL
        return cls(coord, ell, share, knots[keep], joint_values[keep])

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        """f(x); NaN outside the knot range, since the spline does not extrapolate."""
        return self.spline(x)

    def derivative(self, x: float | np.ndarray) -> np.ndarray:
        """f'(x); NaN outside the knot range, since the spline does not extrapolate."""
        return self.spline(x, 1)

    def nu_mean(self) -> float:
        """E_nu[f] under U[0,1] by 64-node quadrature; 0 to roundoff after `from_raw`."""
        return kernel.nu_mean(self.spline(kernel.GL_NODES))

    def nu_var(self) -> float:
        """Var_nu[f] under U[0,1] by 64-node quadrature; equals `share` to roundoff after `from_raw`."""
        return kernel.nu_var(self.spline(kernel.GL_NODES))

    def slope_energy(self) -> float:
        """E_nu[(f')^2] = sum_q w_q f'(t_q)^2, the realized mean squared slope under U[0,1]."""
        return float((kernel.GL_WEIGHTS * self.derivative(kernel.GL_NODES) ** 2).sum())

    def argmax(self, lo: float = 0.0, hi: float = 1.0) -> tuple[float, float]:
        """Exact `(x*, f*)`, the maximizer and maximum of f on [lo, hi] (plan decision D1).

        The spline's derivative is piecewise quadratic, so `roots()` returns every critical
        point of f, and `extrapolate=False` keeps it from reporting roots of the polynomial
        pieces continued outside the knot range. The maximum over those roots that lie in
        [lo, hi], together with the two endpoints, is the global maximum to floating point --
        there is no grid and no iterative search.
        """
        roots = self.spline.derivative().roots()
        candidates = np.concatenate([roots[(roots >= lo) & (roots <= hi)], [lo, hi]])
        values = self.spline(candidates)
        best = int(np.argmax(values))
        return float(candidates[best]), float(values[best])
