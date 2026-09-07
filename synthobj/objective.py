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
a spec, through `_spec_field`: `spec=None` is legal and yields a "custom" family with the study's
noise sd of 0.1, but a spec that is *present* must carry all four, and a missing one raises
`AttributeError` at construction rather than silently taking the default (ruling R27). A default
firing on a real spec whose fields were renamed would put the wrong noise level on every regret
curve and the wrong generator on two study variants, and nothing downstream would notice.
"""
from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from synthobj.component import Component
from synthobj.interaction import Interaction, block_argmax
from synthobj.kernel import GL_NODES
from synthobj.rotation import PairRotation, rotated_block_argmax, rotated_block_mean, rotated_block_stats

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

    `components` must be non-empty and hold exactly one entry per coordinate, and `interactions`
    must sit on disjoint coordinate pairs inside that set -- together these are what make the
    `f_star` assembly a sum of independent per-block maxima. Each is checked at construction and
    raises `ValueError`; overlapping pairs raise `NotImplementedError` instead under
    `allow_overlap=True`, since no family in this project uses them and the joint-grid fallback
    the plan sketches is not built.

    `rotation` is `None` or a `PairRotation` (plan decision D2). When it is given, `__call__` applies
    `rotation.forward` to `X` before evaluating every component and interaction, then subtracts `mu`
    and multiplies by `scale` -- `scale` renormalizes `Var_nu f` back to 1, and `mu` (ruling R30,
    `Sum_p rotated_block_mean` over rotated pairs) recenters `E_nu f` back to 0, since rotation moves
    both: a component is centered under its own coordinate's `U[0,1]` marginal, but a rotated
    coordinate's marginal is not `U[0,1]`. `f_star` for a rotated pair comes from
    `rotated_block_argmax` (searched in x-space, ruling R8) rather than `block_argmax`, and its
    axis-aligned first-order shares (`Labels.s_axis`) and remainder (`Labels.gamma_axis`) come from
    `rotated_block_stats` -- neither needs a `mu` correction, since both are computed as
    `E[X**2] - E[X]**2` and are already shift-invariant. `spec` may be `None`, in which case the
    labels record `family="custom"` and the study's
    default noise sd of 0.1; a spec that is present must carry `name`, `noise_sd`, `generator` and
    `monotone`, and a missing one raises `AttributeError` (ruling R27). `labels` is for the
    save/load path: when it is given it is stored verbatim and `f_star` is taken from it, so
    loading never re-runs the block maximizations.
    """

    def __init__(
        self,
        D: int,
        components: Sequence[Component],
        interactions: Sequence[Interaction],
        rotation: PairRotation | None,
        spec: FamilySpec | None,
        seed: int,
        labels: Labels | None = None,
        allow_overlap: bool = False,
    ) -> None:
        if rotation is not None and not isinstance(rotation, PairRotation):
            raise TypeError(f"rotation must be a PairRotation or None; got {type(rotation).__name__}")

        self.D = int(D)
        self.components: tuple[Component, ...] = tuple(components)
        self.interactions: tuple[Interaction, ...] = tuple(interactions)
        self.rotation = rotation
        self.spec = spec
        self.seed = int(seed)
        self.noise_sd = float(self._spec_field("noise_sd", 0.1))

        if not self.components:
            raise ValueError("at least one component is required; an empty objective has no S")

        coords = [comp.coord for comp in self.components]
        if len(set(coords)) != len(coords):
            # __call__ would sum both, but only one would reach the f_star assembly, and
            # max(f + g) is not max f + max g -- f_star would silently stop being the maximum.
            raise ValueError(f"one component per coordinate is required; got coordinates {coords}")

        rotation_pairs = rotation.pairs if rotation is not None else ()
        # A rotated pair's block maximum (rotated_block_argmax) needs its two coordinates disjoint
        # from every other block for the same reason an interaction pair does: f_star separates into
        # a sum of independent per-block maxima only when the blocks partition the coordinates.
        paired = [coord for inter in self.interactions for coord in (inter.i, inter.j)]
        paired += [coord for pair in rotation_pairs for coord in pair]
        if len(set(paired)) != len(paired):
            if allow_overlap:
                raise NotImplementedError(
                    "overlapping interaction pairs are not implemented; no family uses them"
                )
            raise ValueError(f"interaction pairs must be disjoint; got coordinates {paired}")

        # The block maximum is over f_i(x) + f_j(y) + h_ij(x, y), so a paired coordinate without a
        # main effect has no block to maximize; every family puts its pairs inside S.
        missing = sorted(set(paired) - set(coords))
        if missing:
            raise ValueError(f"every paired coordinate needs a main effect; {missing} have none")

        self.scale = self._compute_scale()
        self.mu = self._compute_mu()
        self.labels = self._compute_labels() if labels is None else labels
        self.f_star = self.labels.f_star

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Noise-free `f` at `X` of shape `(n, D)` -> `(n,)`, or `(D,)` -> a scalar (0-d) array.

        Raises `ValueError` if `X` is not `D` coordinates wide, holds a non-finite entry, or lies
        more than `BOX_TOL` outside [0,1]; points inside that tolerance are clipped onto the face.
        A zero-row batch is legal and returns an empty `(0,)` result. When `rotation` is set, every
        component and interaction is evaluated at `rotation.forward(X)` rather than at `X` itself,
        and the sum is recentered by `self.mu` and scaled by `self.scale` (ruling R30) -- `mu = 0.0`
        and `scale = 1.0` when there is no rotation, so this reduces to the un-rotated sum exactly.
        """
        points, single = self._as_points(X)
        Z = self.rotation.forward(points) if self.rotation is not None else points
        values = np.zeros(points.shape[0])
        for comp in self.components:
            values += comp(Z[:, comp.coord])
        for inter in self.interactions:
            values += inter(Z[:, inter.i], Z[:, inter.j])
        values = (values - self.mu) * self.scale
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
        """Pre-rotation `Var_nu` of each block by 64-node quadrature: main effects, then interactions.

        For blocks built by `Component.from_raw` this equals the prescribed shares to roundoff -- it
        is the measured counterpart of `Labels.s` and `Labels.s_pairs`, which are prescribed, and the
        two diverge exactly when a block was not built to the share it claims. Without a rotation
        this also sums to `Var_nu f`, since the blocks are uncorrelated under the product measure.

        Ruling R31: **under a rotation it does not.** Each component's own `nu_var()` is read off
        its own coordinate and ignores both the rotation (which mixes a pair's two coordinates) and
        `scale`, so the sum here is neither the realized `Var_nu f` (which `scale` forces to exactly
        1) nor a meaningful per-coordinate axis variance -- that is `Labels.s_axis`, from
        `rotated_block_stats`. The values are still meaningful ground truth: they are exactly
        `Labels.s` (and `Labels.s_pairs`) for every block, rotated or not.
        """
        return np.array(
            [comp.nu_var() for comp in self.components]
            + [inter.nu_var() for inter in self.interactions]
        )

    def save(self, stem: str | Path) -> tuple[Path, Path]:
        """Write this objective to `<stem>.npz` and `<stem>.json`; returns `(npz_path, json_path)`.

        Same seed -> bit-identical files (Global Constraint, ruling R5): the `.npz` is a
        hand-rolled zip (every member's timestamp fixed at 1980-01-01, `ZIP_STORED`, members in
        sorted-name order) rather than `np.savez`'s, whose `zipfile` stamps each member with
        `time.localtime()` and would make two saves of the same arrays differ in bytes whenever
        they straddle a one-second boundary. The `.json` is `json.dumps(..., sort_keys=True,
        indent=1)`, so key order does not depend on dict insertion order.

        `Labels.ell` is NaN off `S` by design (ruling R6): Python's `json` module writes a bare
        `NaN` token there (not standard JSON, but round-tripped exactly by Python's own
        `json.loads`), so a reader needs a NaN-tolerant parser.

        `SyntheticObjective.mu` (ruling R30 follow-on) is stored in the JSON next to the labels'
        `scale`, since `load` must read the rotation offset back rather than re-derive it even
        though re-deriving it from the reloaded components would be numerically cheap.

        Requires every component and every interaction factor to sit on one shared knot grid --
        true of every family `synthobj.families` builds (a family either draws every active
        coordinate on [0,1], or pairs all of them into a rotation and draws all of them on the
        rotated family's extended grid) but not guaranteed by this class's own guards, since a
        hand-built objective can mix domains. Raises `ValueError` naming the offending coordinate
        rather than silently writing one component's grid under another's values.
        """
        stem = Path(stem)
        npz_path, json_path = stem.with_suffix(".npz"), stem.with_suffix(".json")

        factors = [comp for inter in self.interactions for comp in (inter.u_i, inter.u_j)]
        grid = _shared_grid(list(self.components) + factors)

        active = np.array([comp.coord for comp in self.components], dtype=np.int64)
        main_values = np.array([comp.values for comp in self.components], dtype=np.float64)
        s = np.array([comp.share for comp in self.components], dtype=np.float64)
        ell = np.array([comp.ell for comp in self.components], dtype=np.float64)

        n_pairs = len(self.interactions)
        if n_pairs:
            inter_values = np.array(
                [
                    values
                    for inter in self.interactions
                    for values in (inter.u_i.values, inter.u_j.values)
                ],
                dtype=np.float64,
            )
            pairs = np.array([[inter.i, inter.j] for inter in self.interactions], dtype=np.int64)
            c_pairs = np.array([inter.c for inter in self.interactions], dtype=np.float64)
        else:
            inter_values = np.zeros((0, grid.size))
            pairs = np.zeros((0, 2), dtype=np.int64)
            c_pairs = np.zeros((0,))

        arrays = {
            "grid": grid,
            "main_values": main_values,
            "inter_values": inter_values,
            "active": active,
            "pairs": pairs,
            "c_pairs": c_pairs,
            "s": s,
            "ell": ell,
            "x_star": self.labels.x_star,
        }
        _write_npz_deterministic(npz_path, arrays)

        spec_dict = (
            None if self.spec is None else {f.name: getattr(self.spec, f.name) for f in fields(self.spec)}
        )
        payload = {
            "version": 1,
            "spec": spec_dict,
            "labels": _labels_to_json(self.labels),
            "rotation": (
                None
                if self.rotation is None
                else {"theta_deg": self.rotation.theta_deg, "pairs": [list(p) for p in self.rotation.pairs]}
            ),
            "mu": self.mu,
            "grid_lo": float(grid[0]),
            "grid_hi": float(grid[-1]),
            "grid_n": int(grid.size - GL_NODES.size),
        }
        json_path.write_text(json.dumps(payload, sort_keys=True, indent=1))
        return npz_path, json_path

    @classmethod
    def load(cls, stem: str | Path) -> "SyntheticObjective":
        """Rebuild the `SyntheticObjective` that `save` wrote to `<stem>.npz`/`<stem>.json`.

        Never draws and never calls `synthobj.kernel.eigen_factor`. `Component`s and
        `Interaction`s are reconstructed with the plain `Component` constructor -- not
        `from_raw` -- directly from the stored knots and values, since those are already final
        and pushing them back through the centre-and-rescale path would be a no-op only in exact
        arithmetic, not in floating point. `Labels` is passed through to `__init__` verbatim, so
        `f_star` is never recomputed either.

        `__init__` recomputes `scale` and `mu` as a side effect of construction (it has no
        parameter for either), but that recomputation is immediately overwritten here by the
        stored `mu` (ruling R30 follow-on): the object this method returns carries the saved
        float, not a fresh quadrature, regardless of what `__init__` did internally on the way.
        """
        stem = Path(stem)
        npz_path, json_path = stem.with_suffix(".npz"), stem.with_suffix(".json")

        payload = json.loads(json_path.read_text())
        if payload["version"] != 1:
            raise ValueError(f"unsupported save format version {payload['version']!r}")

        spec = _spec_from_json(payload["spec"])
        labels = _labels_from_json(payload["labels"])
        rotation = _rotation_from_json(payload["rotation"])

        with np.load(npz_path) as data:
            grid = data["grid"]
            main_values = data["main_values"]
            inter_values = data["inter_values"]
            active = data["active"]
            pairs = data["pairs"]
            c_pairs = data["c_pairs"]
            s = data["s"]
            ell = data["ell"]

        components = tuple(
            Component(int(active[k]), float(ell[k]), float(s[k]), grid, main_values[k])
            for k in range(active.size)
        )
        # The interaction factors' own `ell` is not part of the file format (it is unused by any
        # Component method -- see component.py -- and not surfaced by Labels), so it is not
        # knowable here; NaN says so honestly rather than guessing a value that might not match
        # what save() started from. `share` is not a guess: every Interaction factor is built
        # with share = 1 by convention (interaction.py), so that value is genuinely known.
        interactions = tuple(
            Interaction(
                int(pairs[p, 0]),
                int(pairs[p, 1]),
                float(c_pairs[p]),
                Component(int(pairs[p, 0]), float("nan"), 1.0, grid, inter_values[2 * p]),
                Component(int(pairs[p, 1]), float("nan"), 1.0, grid, inter_values[2 * p + 1]),
            )
            for p in range(pairs.shape[0])
        )

        obj = cls(labels.D, components, interactions, rotation, spec, labels.seed, labels=labels)
        obj.mu = payload["mu"]
        return obj

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
        # np.any rather than min()/max(): both are vacuously False on a zero-row batch, where the
        # reductions would instead raise "zero-size array to reduction operation minimum". An
        # empty batch is a legal request and evaluates to an empty result.
        if np.any(points < -BOX_TOL) or np.any(points > 1.0 + BOX_TOL):
            raise ValueError(f"X lies outside the unit cube by more than {BOX_TOL}")
        return np.clip(points, 0.0, 1.0), single

    def _spec_field(self, name: str, default: Any) -> Any:
        """`self.spec`'s `name`, or `default` when there is no spec at all (ruling R27).

        The default is *not* a fallback for a spec that lacks the field: a present spec must carry
        every field this module reads, and a missing one raises `AttributeError` here rather than
        recording a plausible-looking wrong label.
        """
        return default if self.spec is None else getattr(self.spec, name)

    def _compute_scale(self) -> float:
        """`1 / sqrt(total realized Var_nu f)`, or `1.0` when there is no rotation.

        Blocks are independent under the product measure, so variances add: a rotated pair's raw
        block variance is `rotated_block_stats`'s `Var phi` (quadrature, since rotation moves the
        block outside the domain `Component.from_raw` normalized on), while an unpaired component's
        `share` and an interaction's `c**2` are exact by construction. Without a rotation the total
        is not computed at all, matching the pre-rotation objective's implicit `scale = 1`.
        """
        if self.rotation is None:
            return 1.0
        by_coord = {comp.coord: comp for comp in self.components}
        rotated_coords = {coord for pair in self.rotation.pairs for coord in pair}
        total = sum(
            rotated_block_stats(by_coord[i], by_coord[j], self.rotation)[0]
            for i, j in self.rotation.pairs
        )
        total += sum(comp.share for comp in self.components if comp.coord not in rotated_coords)
        total += sum(inter.c**2 for inter in self.interactions)
        return float(1.0 / np.sqrt(total))

    def _compute_mu(self) -> float:
        """`Sum_p E_nu[phi_p]` over rotated pairs, or `0.0` with no rotation (ruling R30).

        Unpaired components are already centered by `Component.from_raw` (`E_nu = 0`), and a product
        interaction is exactly zero-mean on its own two coordinates (`synthobj.interaction`'s
        orthogonality identities) -- only a rotated pair's mixing of two independent uniforms can
        shift the mean away from 0 (`rotated_block_mean`). Pairs are disjoint, so the total is a
        plain sum: `E[X + Y] = E[X] + E[Y]` regardless of independence. `__call__` and `_compute_labels`
        both subtract this once, in the same raw (pre-`scale`) units it is computed in.
        """
        if self.rotation is None:
            return 0.0
        by_coord = {comp.coord: comp for comp in self.components}
        return float(
            sum(rotated_block_mean(by_coord[i], by_coord[j], self.rotation) for i, j in self.rotation.pairs)
        )

    def _compute_labels(self) -> Labels:
        """Assemble `Labels`, including `f_star` and its maximizer (plan decision D1).

        Pairs are disjoint, so each paired block is an independent function of its own two
        coordinates and the maximum separates: `f_star` is `(sum - self.mu) * self.scale`, where
        `sum` is the raw total of the per-block maxima (`block_argmax` for an interaction,
        `rotated_block_argmax` for a rotated pair, both accurate to about 1e-10) and the
        per-component maxima from `Component.argmax` (exact to floating point) -- matching
        `__call__`, which recenters and scales the whole sum once rather than each block (ruling
        R30: `rotated_block_argmax`'s own `phi` is never recentered, since L-BFGS-B's relative
        `ftol` would otherwise see a shifted stopping denominator and the maximizer's last bits
        could move). `x_star` records the achieving coordinates, with every coordinate carrying no
        main effect left at 0.5 -- `f` ignores those entirely. A rotated pair's `x_star` comes
        directly from `rotated_block_argmax`'s x-space search, so it already lies in the cube.
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
        s_axis = np.zeros(D)
        scale2 = self.scale**2

        rotation_pairs = self.rotation.pairs if self.rotation is not None else ()
        for i, j in rotation_pairs:
            f_a, f_b = by_coord[i], by_coord[j]
            (x_a, x_b), phi = rotated_block_argmax(f_a, f_b, self.rotation)
            x_star[i], x_star[j] = float(x_a), float(x_b)
            f_star += phi
            paired.update((i, j))
            _, var_a, var_b = rotated_block_stats(f_a, f_b, self.rotation)
            s_axis[i] = var_a * scale2
            s_axis[j] = var_b * scale2

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
            if self.rotation is not None:
                s_axis[comp.coord] = comp.share * scale2

        f_star = (f_star - self.mu) * self.scale

        s_pairs = tuple(float(inter.c**2) for inter in self.interactions)
        gamma = float(sum(s_pairs))

        if self.rotation is None:
            # No rotation: the axis-aligned shares are the pre-rotation ones and nothing is
            # renormalized.
            rotation_deg = 0.0
            rotation_pairs_label: tuple[tuple[int, int], ...] = ()
            s_axis = s.copy()
            gamma_axis = gamma
        else:
            rotation_deg = float(self.rotation.theta_deg)
            rotation_pairs_label = rotation_pairs
            gamma_axis = 1.0 - float(s_axis.sum())

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
            rotation_deg=rotation_deg,
            rotation_pairs=rotation_pairs_label,
            s_axis=s_axis,
            gamma_axis=gamma_axis,
            scale=self.scale,
            generator=str(self._spec_field("generator", "matern")),
            monotone=bool(self._spec_field("monotone", False)),
            family=str(self._spec_field("name", "custom")),
            seed=self.seed,
            noise_sd=self.noise_sd,
            active_eps=ACTIVE_EPS,
        )


# --- save/load helpers (Task 8) ----------------------------------------------------------------

def _labels_to_json(labels: Labels) -> dict[str, Any]:
    """`Labels` as a JSON-safe dict: ndarray fields become lists; tuple fields are left as tuples,
    since `json.dumps` already writes any tuple as a JSON array.

    `ell`'s NaN entries survive as Python `float('nan')`, which `json.dumps`'s default
    `allow_nan=True` writes as a bare `NaN` token (ruling R6) -- not standard JSON, but
    round-tripped exactly by Python's own `json.loads`.
    """
    out: dict[str, Any] = {}
    for f in fields(Labels):
        value = getattr(labels, f.name)
        out[f.name] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


def _labels_from_json(data: dict[str, Any]) -> Labels:
    """Reverse of `_labels_to_json`: ndarray fields rebuilt; tuple fields restored from the lists
    JSON turned them into (ruling R9), including the inner `[i, j]` pairs of `pairs` and
    `rotation_pairs`, which must become `(i, j)` or the reconstructed record compares unequal to
    an in-memory one on those two fields.
    """
    kwargs = dict(data)
    kwargs["active"] = np.array(kwargs["active"], dtype=bool)
    for name in ("s", "ell", "g", "x_star", "s_axis"):
        kwargs[name] = np.array(kwargs[name], dtype=float)
    kwargs["S"] = tuple(kwargs["S"])
    kwargs["s_pairs"] = tuple(kwargs["s_pairs"])
    kwargs["pairs"] = tuple(tuple(pair) for pair in kwargs["pairs"])
    kwargs["rotation_pairs"] = tuple(tuple(pair) for pair in kwargs["rotation_pairs"])
    return Labels(**kwargs)


def _spec_from_json(data: dict[str, Any] | None) -> "FamilySpec | None":
    """Reverse of `save`'s inline `{field: value}` dump of `self.spec`: restores `shares` and
    `ells` from the lists JSON turned them into (ruling R9).

    Imports `FamilySpec` locally rather than at module scope: `families.py` imports this module
    (ruling R15), so a module-level `from synthobj.families import FamilySpec` here would be a
    cycle. `save` needs no such import at all -- `dataclasses.fields` works on `self.spec` by
    duck typing -- so only this direction pays for it, and only when `load` actually runs.
    """
    if data is None:
        return None
    from synthobj.families import FamilySpec

    kwargs = dict(data)
    if kwargs["shares"] is not None:
        kwargs["shares"] = tuple(kwargs["shares"])
    if isinstance(kwargs["ells"], list):
        kwargs["ells"] = tuple(kwargs["ells"])
    return FamilySpec(**kwargs)


def _rotation_from_json(data: dict[str, Any] | None) -> PairRotation | None:
    """Reverse of `save`'s `{"theta_deg": ..., "pairs": [[i, j], ...]}`; `None` stays `None`."""
    if data is None:
        return None
    return PairRotation(float(data["theta_deg"]), tuple(tuple(pair) for pair in data["pairs"]))


def _shared_grid(blocks: list[Component]) -> np.ndarray:
    """The one knot grid every component and interaction factor must share to be savable.

    True of every family `synthobj.families` builds (a family either draws every active
    coordinate on [0,1], or pairs all of them into a rotation and draws all of them on the
    rotated extended grid), but not guaranteed by `SyntheticObjective`'s own guards -- a
    hand-built objective can mix domains. Raises `ValueError` naming the offending coordinate
    rather than silently writing one component's grid under another's values.
    """
    grid = blocks[0].grid
    for comp in blocks[1:]:
        if comp.grid.shape != grid.shape or not np.array_equal(comp.grid, grid):
            raise ValueError(
                f"save requires every component to share one knot grid; coordinate {comp.coord} "
                f"does not match coordinate {blocks[0].coord}"
            )
    return grid


def _write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write `arrays` as a `.npz` with bit-identical bytes for bit-identical arrays (ruling R5).

    `np.savez` builds its zip through `zipfile`, which stamps each member with
    `time.localtime()`, so two saves of the same arrays differ in bytes whenever they straddle a
    one-second boundary. Writing the zip directly -- a fixed 1980-01-01 timestamp on every
    member, `ZIP_STORED` (no compression, whose parameters could otherwise vary by zlib version),
    and members written in sorted-name order -- removes every source of non-determinism that
    `np.save` itself (deterministic for a given array) does not already control. The result loads
    with `np.load` exactly like a normal `.npz`.
    """
    with zipfile.ZipFile(path, mode="w") as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, buf.getvalue())
