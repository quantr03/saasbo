"""The assembled objective and the `Labels` record it is scored against.

A `SyntheticObjective` is a sum of one-dimensional `Component`s on distinct coordinates plus
product `Interaction`s on disjoint coordinate pairs. Every term is centered under the uniform
reference measure U[0,1] and an interaction is orthogonal to both of its coordinates' main effects
(see `synthobj.interaction`), so under nu = U[0,1]^D the terms are uncorrelated and their variances
add: `Var_nu f = sum_i s_i + sum_pairs s_ij`, which the families set to 1.

Which quantities are exact and which are quadrature:

- `Labels.s` and `Labels.s_pairs` are the *prescribed* shares -- `Component.share` and `c**2` --
  and are exact by construction, not measured.
- `component_variances()` is the *realized* variance of each block, read back by 64-node
  Gauss-Legendre quadrature. It agrees with the prescribed shares to roundoff because
  `Component.from_raw` solved the rescale at exactly those nodes, and the two disagree only if a
  block was built wrongly -- which is what makes the sum-to-one check a real test.
- `Labels.g` is the *realized* slope share, `E_nu[(f_i')^2]` per main effect normalized to sum to
  1 over S. It is a quadrature of the spline's derivative, accurate to about 0.1 % (plan decision
  D3). Interaction factors do not contribute: `g` is a main-effect statistic.
- `f_star` is exact per unpaired component (`Component.argmax` finds every critical point of a
  piecewise-quadratic spline derivative in closed form) and accurate to about 1e-10 per paired
  block (`block_argmax` scans the knot grid and refines with L-BFGS-B). Pairs are disjoint, so
  each block is an independent function of its own two coordinates and the maximum separates
  exactly into a sum of per-block maxima (plan decision D1).

`ACTIVE_EPS` lives here rather than in `families.py` (controller ruling R14) because
`Labels.active_eps` is built here and `families.py` does not exist yet; `families.py` imports it
from this module. It is recorded for downstream posterior scoring and does **not** define S:
`Labels.active` is membership in S, and the dense-weak family's shares all sit below it by design.

`FamilySpec` is imported only under `typing.TYPE_CHECKING` (ruling R15). `families.py` imports this
module, so a runtime import here would be a cycle; `from __future__ import annotations` keeps the
annotation readable without one. Only `name`, `noise_sd`, `generator` and `monotone` are read off
a spec, all through `getattr` with defaults, so `spec=None` is legal and yields a "custom" family
with the study's noise sd of 0.1.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from synthobj.component import Component
from synthobj.interaction import Interaction, block_argmax

if TYPE_CHECKING:
    from synthobj.families import FamilySpec

# Global active cutoff recorded on every Labels for downstream posterior scoring (ruling R14).
ACTIVE_EPS = 0.02

# Points within this distance of a face of the cube are accepted and clipped onto it; further out
# is a caller error. Components do not extrapolate, so without the clip a point at 1 + 1e-12 would
# evaluate to NaN instead of to f(1).
BOX_TOL = 1e-9


@dataclass(frozen=True, eq=False)
class Labels:
    """The ground truth a posterior is scored against: which coordinates matter, and how much.

    `S` is the sorted tuple of coordinates carrying a main effect and `active` marks membership in
    it -- not a comparison against `active_eps`, which is carried only so downstream scoring code
    reads one number rather than re-deriving it. `s` is the first-order variance share of the
    pre-rotation main effect (0 off S), `ell` its lengthscale (NaN off S), and `g` its realized
    slope share, normalized over S. `pairs`/`s_pairs`/`gamma` describe the product interactions,
    and `s_axis`/`gamma_axis`/`scale`/`rotation_deg`/`rotation_pairs` describe the objective in
    axis coordinates; without a rotation they are `s`, `gamma`, 1.0, 0.0 and `()`.

    Declared `eq=False` (controller ruling R4): the generated `__eq__` of a frozen dataclass
    compares the fields as a tuple, which raises on the five ndarray fields. Equality is identity;
    the save/load round-trip test compares field by field.
    """

    D: int
    S: tuple[int, ...]
    active: np.ndarray
    s: np.ndarray
    ell: np.ndarray
    g: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    s_pairs: tuple[float, ...]
    gamma: float
    f_star: float
    x_star: np.ndarray
    rotation_deg: float
    rotation_pairs: tuple[tuple[int, int], ...]
    s_axis: np.ndarray
    gamma_axis: float
    scale: float
    generator: str
    monotone: bool
    family: str
    seed: int
    noise_sd: float
    active_eps: float


class SyntheticObjective:
    """`f(x) = sum_i f_i(x_i) + sum_{(i,j)} c_ij u_i(x_i) u_j(x_j)` on [0,1]^D, with its labels.

    Maximization convention: `f_star` is the maximum, and the design brief's regret is
    `f_star - max_t f(x_t)`.

    `components` must hold at most one entry per coordinate and `interactions` must sit on
    disjoint coordinate pairs -- both are what make the `f_star` assembly a sum of independent
    per-block maxima. Overlapping pairs raise `ValueError`, or `NotImplementedError` under
    `allow_overlap=True`: no family in this project uses them, so the joint-grid fallback the plan
    sketches is not built.

    `rotation` must be `None` in this version; the rotated family (plan decision D2) extends this
    class. `spec` may be `None`, in which case the labels record `family="custom"` and the study's
    default noise sd of 0.1. `labels` is for the save/load path: when it is given it is stored
    verbatim and `f_star` is taken from it, so loading never re-runs the block maximizations.
    """

    def __init__(
        self,
        D: int,
        components: Sequence[Component],
        interactions: Sequence[Interaction],
        rotation: None,
        spec: FamilySpec | None,
        seed: int,
        labels: Labels | None = None,
        allow_overlap: bool = False,
    ) -> None:
        if rotation is not None:
            raise NotImplementedError(
                "rotation is not implemented in this version of SyntheticObjective; pass None"
            )

        self.D = int(D)
        self.components: tuple[Component, ...] = tuple(components)
        self.interactions: tuple[Interaction, ...] = tuple(interactions)
        self.rotation = rotation
        self.spec = spec
        self.seed = int(seed)
        self.noise_sd = float(getattr(spec, "noise_sd", 0.1))

        coords = [comp.coord for comp in self.components]
        if len(set(coords)) != len(coords):
            # __call__ would sum both, but only one would reach the f_star assembly, and
            # max(f + g) is not max f + max g -- f_star would silently stop being the maximum.
            raise ValueError(f"one component per coordinate is required; got coordinates {coords}")

        paired = [coord for inter in self.interactions for coord in (inter.i, inter.j)]
        if len(set(paired)) != len(paired):
            if allow_overlap:
                raise NotImplementedError(
                    "overlapping interaction pairs are not implemented; no family uses them"
                )
            raise ValueError(f"interaction pairs must be disjoint; got coordinates {paired}")

        self.labels = self._compute_labels() if labels is None else labels
        self.f_star = self.labels.f_star

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Noise-free `f` at `X` of shape `(n, D)` -> `(n,)`, or `(D,)` -> a scalar (0-d) array.

        Raises `ValueError` if `X` is not `D` coordinates wide, holds a non-finite entry, or lies
        more than `BOX_TOL` outside [0,1]; points inside that tolerance are clipped onto the face.
        """
        points, single = self._as_points(X)
        values = np.zeros(points.shape[0])
        for comp in self.components:
            values += comp(points[:, comp.coord])
        for inter in self.interactions:
            values += inter(points[:, inter.i], points[:, inter.j])
        return values.reshape(()) if single else values

    def observe(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`f(X)` plus independent `N(0, noise_sd^2)` noise, one draw per row.

        The noise comes entirely from the caller's `rng`: the objective holds no random state, so
        paired BO runs can reproduce an observation stream exactly by re-seeding their generator.
        """
        values = self(X)
        # np.asarray keeps the return shape identical to __call__'s: numpy demotes the sum of two
        # 0-d arrays to a numpy scalar, which would break the single-point case's array contract.
        return np.asarray(values + self.noise_sd * rng.standard_normal(values.shape))

    def component_variances(self) -> np.ndarray:
        """Realized `Var_nu` of each block by 64-node quadrature: main effects, then interactions.

        For blocks built by `Component.from_raw` this equals the prescribed shares to roundoff,
        and it sums to `Var_nu f` because the blocks are uncorrelated under the product measure.
        It is the measured counterpart of `Labels.s` and `Labels.s_pairs`, which are prescribed:
        the two diverge exactly when a block was not built to the share it claims.
        """
        return np.array(
            [comp.nu_var() for comp in self.components]
            + [inter.nu_var() for inter in self.interactions]
        )

    def _as_points(self, X: np.ndarray) -> tuple[np.ndarray, bool]:
        """Validate `X`, return it as a `(n, D)` array clipped into the cube, and whether it was 1-D."""
        points = np.asarray(X, dtype=float)
        single = points.ndim == 1
        if single:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != self.D:
            raise ValueError(f"X must have {self.D} coordinates; got shape {np.shape(X)}")
        # Before the box test, since every comparison against NaN is False.
        if not np.all(np.isfinite(points)):
            raise ValueError("X has non-finite entries")
        if points.min() < -BOX_TOL or points.max() > 1.0 + BOX_TOL:
            raise ValueError(f"X lies outside the unit cube by more than {BOX_TOL}")
        return np.clip(points, 0.0, 1.0), single

    def _compute_labels(self) -> Labels:
        """Assemble `Labels`, including `f_star` and its maximizer (plan decision D1).

        Pairs are disjoint, so each paired block is an independent function of its own two
        coordinates and the maximum separates: `f_star` is the sum of the per-block maxima from
        `block_argmax` (accurate to about 1e-10) and the per-component maxima from
        `Component.argmax` (exact to floating point). `x_star` records the achieving coordinates,
        with every coordinate carrying no main effect left at 0.5 -- `f` ignores those entirely.
        """
        D = self.D
        s = np.zeros(D)
        ell = np.full(D, np.nan)
        slope = np.zeros(D)
        for comp in self.components:
            s[comp.coord] = comp.share
            ell[comp.coord] = comp.ell
            slope[comp.coord] = comp.slope_energy()

        by_coord = {comp.coord: comp for comp in self.components}
        x_star = np.full(D, 0.5)
        f_star = 0.0
        paired: set[int] = set()
        for inter in self.interactions:
            (x_i, x_j), phi = block_argmax(by_coord[inter.i], by_coord[inter.j], inter)
            x_star[inter.i], x_star[inter.j] = float(x_i), float(x_j)
            f_star += phi
            paired.update((inter.i, inter.j))
        for comp in self.components:
            if comp.coord in paired:
                continue
            x, value = comp.argmax()
            x_star[comp.coord] = x
            f_star += value

        s_pairs = tuple(float(inter.c**2) for inter in self.interactions)
        gamma = float(sum(s_pairs))
        return Labels(
            D=D,
            S=tuple(sorted(by_coord)),
            active=s > 0.0,
            s=s,
            ell=ell,
            g=slope / slope.sum(),
            pairs=tuple((inter.i, inter.j) for inter in self.interactions),
            s_pairs=s_pairs,
            gamma=gamma,
            f_star=float(f_star),
            x_star=x_star,
            # No rotation: the axis-aligned shares are the pre-rotation ones and nothing is
            # renormalized. Plan decision D2 generalizes these five fields.
            rotation_deg=0.0,
            rotation_pairs=(),
            s_axis=s.copy(),
            gamma_axis=gamma,
            scale=1.0,
            generator=str(getattr(self.spec, "generator", "matern")),
            monotone=bool(getattr(self.spec, "monotone", False)),
            family=str(getattr(self.spec, "name", "custom")),
            seed=self.seed,
            noise_sd=self.noise_sd,
            active_eps=ACTIVE_EPS,
        )
