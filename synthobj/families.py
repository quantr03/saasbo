"""The family registry: a name plus a seed determines one fully-specified `SyntheticObjective`.

`FamilySpec` is the frozen, hashable description of one variant's shape; `build` turns a spec and
a seed into an assembled objective. `FAMILIES` maps each of the study's 14 variant names to a
factory that produces its `FamilySpec` (with `D` and any other field overridable), `make_family`
is the one-call convenience (`FAMILIES[name](D=D, **overrides)` then `build`), and `STUDY_GRID`
lists the 14 variants the thesis's whole study runs on: `len(STUDY_GRID) == 14`, and combined with
10 seeds each gives the 140-file study grid.

Determinism is `SeedSequence(seed).spawn(3)` (plan decision D6): `select` picks which coordinates
are active (independent of every other choice, and *nested* across `n_active` -- the same seed's
first 3 picks are `aligned3`'s S and its first 10 are `aligned10`'s), `main` gives one independent
child stream per active rank (so changing one coordinate's lengthscale or share never perturbs any
other component's draw), and `inter` gives pair selection plus, when a family builds real
`Interaction`s, the two unit-variance draws each pair needs. Noise is deliberately not part of the
objective (`SyntheticObjective.observe` takes the caller's own generator); `noise_rng` exists so a
paired BO run can reproduce its observation noise independently of the objective's own draws.

`ACTIVE_EPS` is imported from `synthobj.objective`, not redefined here (ruling R14): this module
has no runtime use for the constant itself (whether a variant's shares clear it is a property
tests check against the family table, not something `build` enforces), so it is left to the
`synthobj.objective` import in `tests/test_families.py` and `tests/test_seeding.py`.

Rotated components live on the extended grid `make_grid(EXT_LO, EXT_HI, ROTATED_GRID_N)`, not on
[0,1] (ruling R7): a rotated coordinate's argument to its component ranges over `0.5 +/- sqrt(2)/2`,
which is outside the unit interval a plain `Component` is built on. `ROTATED_GRID_N = 1469` is
fixed independently of `FamilySpec.grid_n` (which only ever applies to a component that is *not*
part of a rotated pair) so that the joint grid (1469 draw points + 64 GL nodes) is 1533 long, as
ruling R7 fixes it.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.random import Generator, SeedSequence
from typing import NamedTuple

from synthobj.component import Component
from synthobj.draws import draw_until_monotone, gp_draw, sinusoid_draw
from synthobj.interaction import Interaction
from synthobj.kernel import eigen_factor, joint_points, make_grid
from synthobj.objective import SyntheticObjective
from synthobj.rotation import EXT_HI, EXT_LO, PairRotation

# Fixed independently of FamilySpec.grid_n (ruling R7): 1469 + 64 GL nodes = 1533.
ROTATED_GRID_N = 1469

# 0x4E4F ("NO", for "noise") separates the noise stream's entropy from a seed used elsewhere for
# the same integer; `run` lets one seed reproduce several independent noise sequences (e.g. one
# per paired BO repetition).
_NOISE_SALT = 0x4E4F


@dataclass(frozen=True)
class FamilySpec:
    """One variant's shape: how many coordinates are active, their shares and lengthscales, and
    whether they carry an interaction, a rotation, a non-default generator or the monotone
    rejection wrapper.

    `n_active=None` means dense (every one of `D` coordinates is active, as `dense_weak` needs).
    `shares=None` means an equal split of `1 - gamma` over the active ranks; a given tuple must
    already sum to `1 - gamma` and is used verbatim, ordered by rank (`decoupled` and
    `anti_aligned` both need a specific share paired with a specific lengthscale, not just the
    right multiset). `ells` broadcasts a scalar to every active rank or is given per rank the same
    way. `n_pairs` disjoint rank pairs are drawn from the active ranks regardless of what they
    become: with `theta_deg is None` they are `Interaction`s carrying `gamma` split equally
    between them; with `theta_deg` set they are the pairs a `PairRotation` rotates instead, and
    `gamma` plays no part (`shares` alone already sums to 1).

    Keeps the default `eq=True` (unlike `Labels` and `Component`, ruling R4): every field is a
    plain hashable value (`str`, `int`, `float`, `bool`, or a `tuple` of them), so value equality
    is well defined and a save/load round trip can compare specs directly.
    """

    name: str
    D: int = 100
    n_active: int | None = None
    shares: tuple[float, ...] | None = None
    ells: float | tuple[float, ...] = 0.5
    n_pairs: int = 0
    gamma: float = 0.0
    theta_deg: float | None = None
    generator: str = "matern"
    monotone: bool = False
    noise_sd: float = 0.1
    grid_n: int = 1024


class Streams(NamedTuple):
    """The three independent `SeedSequence`s one seed spawns (plan decision D6)."""

    select: SeedSequence
    main: SeedSequence
    inter: SeedSequence


def streams(seed: int) -> Streams:
    """`SeedSequence(seed).spawn(3)` as `(select, main, inter)`."""
    select, main, inter = SeedSequence(seed).spawn(3)
    return Streams(select, main, inter)


def noise_rng(seed: int, run: int = 0) -> Generator:
    """An observation-noise generator independent of the objective's own draws.

    Salted with `_NOISE_SALT` so it never collides with `streams(seed)`'s children, and `run`
    lets one seed reproduce several independent noise sequences (e.g. one per paired BO repeat)
    without disturbing the objective itself, which holds no random state of its own.
    """
    return np.random.default_rng(SeedSequence([seed, _NOISE_SALT, run]))


def select_active(D: int, n_active: int, ss: SeedSequence) -> np.ndarray:
    """The `n_active` active coordinates, in the rank order shares and lengthscales are assigned in.

    `default_rng(ss)` is a pure function of `ss` -- calling it again reproduces the same draws --
    so `permutation(D)` is the same full ordering regardless of `n_active`, and truncating it to
    different lengths at the same `(D, seed)` nests: `aligned3`'s three coordinates are `aligned10`'s
    first three.
    """
    return np.random.default_rng(ss).permutation(D)[:n_active]


def select_pairs(n_active: int, n_pairs: int, ss: SeedSequence) -> tuple[tuple[int, int], ...]:
    """`n_pairs` disjoint rank pairs, cut from a permutation of `range(n_active)`.

    Ranks, not coordinates: a caller maps `(a, b)` through `select_active`'s output to get the
    actual coordinate pair. Raises `ValueError` if there are not enough ranks to pair up.
    """
    if 2 * n_pairs > n_active:
        raise ValueError(
            f"cannot draw {n_pairs} disjoint pairs from {n_active} active coordinates"
        )
    perm = np.random.default_rng(ss).permutation(n_active)
    return tuple((int(perm[2 * k]), int(perm[2 * k + 1])) for k in range(n_pairs))


def _resolve_shares(spec: FamilySpec, n_active: int) -> tuple[float, ...]:
    """`spec.shares` verbatim, or an equal `(1 - gamma) / n_active` split.

    A given tuple must have exactly `n_active` entries summing to `1 - gamma`: `decoupled` and
    `anti_aligned` both need a specific share paired with a specific lengthscale, not just a
    multiset that happens to be long enough. Without this check, a mismatched `n_active` override
    (e.g. `make_family("decoupled", 0, n_active=4)`, which leaves the 8-entry `shares` from the
    registry untouched) would silently zip the first `n_active` shares against the first `n_active`
    ells and build a real objective whose `Var_nu f` is not 1, with no label recording the mistake
    (fix-round finding). The too-long case was previously silent; a too-short one already raised
    `IndexError` inside `build`'s per-rank loop, just without an informative message.
    """
    if spec.shares is None:
        return tuple([(1.0 - spec.gamma) / n_active] * n_active)
    if len(spec.shares) != n_active:
        raise ValueError(f"shares has {len(spec.shares)} entries; expected n_active={n_active}")
    total = sum(spec.shares) + spec.gamma
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"shares (sum={sum(spec.shares)}) plus gamma ({spec.gamma}) must equal 1; got {total}"
        )
    return spec.shares


def _resolve_ells(spec: FamilySpec, n_active: int) -> tuple[float, ...]:
    """`spec.ells` broadcast to `n_active` ranks if it is a scalar, else taken verbatim.

    A given tuple must have exactly `n_active` entries, one per active rank -- the same
    mismatched-override hazard `_resolve_shares` guards against.
    """
    if isinstance(spec.ells, (int, float)):
        return tuple([float(spec.ells)] * n_active)
    ells = tuple(spec.ells)
    if len(ells) != n_active:
        raise ValueError(f"ells has {len(ells)} entries; expected n_active={n_active}")
    return ells


def _draw_fn(generator: str, ell: float, lo: float, hi: float, grid_n: int) -> Callable[[Generator], np.ndarray]:
    """A zero-argument-but-`rng` draw function returning raw joint values at this domain and `ell`."""
    if generator == "matern":
        F = eigen_factor(lo, hi, grid_n, ell)
        return lambda rng: gp_draw(F, rng)
    if generator == "sinusoid":
        Z = joint_points(make_grid(lo, hi, grid_n))
        return lambda rng: sinusoid_draw(Z, ell, rng)
    raise ValueError(f"unknown generator {generator!r}; expected 'matern' or 'sinusoid'")


def _draw_component(
    coord: int,
    ell: float,
    share: float,
    lo: float,
    hi: float,
    grid_n: int,
    generator: str,
    monotone: bool,
    rng: Generator,
) -> Component:
    """One component: draw raw values with `rng`, optionally reject until monotone, rescale."""
    draw = _draw_fn(generator, ell, lo, hi, grid_n)
    if monotone:
        raw = draw_until_monotone(draw, grid_n, rng)
        # The random monotone direction (dense_weak's design): drawn from the same rank stream,
        # after the rejection loop, so it is still fully determined by that rank's own seed.
        sign = float(rng.choice([-1.0, 1.0]))
    else:
        raw = draw(rng)
        sign = 1.0
    return Component.from_raw(coord, ell, share, make_grid(lo, hi, grid_n), raw, sign=sign)


def build(spec: FamilySpec, seed: int) -> SyntheticObjective:
    """Turn `spec` and `seed` into a fully-specified, deterministic `SyntheticObjective`.

    See the module docstring for the seeding scheme. Pairs (interaction or rotation) are always
    drawn the same way regardless of what they become, so a gamma sweep at one seed keeps S, the
    pairs and every component's raw shape identical (only the shares move).
    """
    D = spec.D
    n_active = D if spec.n_active is None else spec.n_active
    ss = streams(seed)

    S = select_active(D, n_active, ss.select)
    shares = _resolve_shares(spec, n_active)
    ells = _resolve_ells(spec, n_active)

    rotation_pairs: tuple[tuple[int, int], ...] = ()
    rank_pairs: tuple[tuple[int, int], ...] = ()
    inter_draw_ss: SeedSequence | None = None
    if spec.n_pairs > 0:
        select_ss, inter_draw_ss = ss.inter.spawn(2)
        rank_pairs = select_pairs(n_active, spec.n_pairs, select_ss)
        coord_pairs = tuple((int(S[a]), int(S[b])) for a, b in rank_pairs)
        if spec.theta_deg is not None:
            rotation_pairs = coord_pairs

    rotated_coords = {coord for pair in rotation_pairs for coord in pair}

    main_children = ss.main.spawn(n_active)
    components: list[Component] = []
    for rank in range(n_active):
        coord = int(S[rank])
        on_extended = coord in rotated_coords
        lo, hi = (EXT_LO, EXT_HI) if on_extended else (0.0, 1.0)
        grid_n = ROTATED_GRID_N if on_extended else spec.grid_n
        rng = np.random.default_rng(main_children[rank])
        components.append(
            _draw_component(
                coord, ells[rank], shares[rank], lo, hi, grid_n, spec.generator, spec.monotone, rng
            )
        )

    interactions: list[Interaction] = []
    if rank_pairs and spec.theta_deg is None:
        assert inter_draw_ss is not None
        pair_children = inter_draw_ss.spawn(len(rank_pairs))
        s_ij = spec.gamma / spec.n_pairs
        c = float(np.sqrt(s_ij))
        for k, (ra, rb) in enumerate(rank_pairs):
            ca, cb = int(S[ra]), int(S[rb])
            ui_ss, uj_ss = pair_children[k].spawn(2)
            u_i = _draw_component(
                ca, ells[ra], 1.0, 0.0, 1.0, spec.grid_n, spec.generator, False,
                np.random.default_rng(ui_ss),
            )
            u_j = _draw_component(
                cb, ells[rb], 1.0, 0.0, 1.0, spec.grid_n, spec.generator, False,
                np.random.default_rng(uj_ss),
            )
            interactions.append(Interaction(ca, cb, c, u_i, u_j))

    rotation = PairRotation(spec.theta_deg, rotation_pairs) if spec.theta_deg is not None else None

    return SyntheticObjective(D, components, interactions, rotation, spec, seed)


# --- the study's 14 variants -----------------------------------------------------------------

FAMILIES: dict[str, Callable[..., FamilySpec]] = {
    "aligned3": partial(FamilySpec, name="aligned3", n_active=3, ells=0.5),
    "aligned10": partial(FamilySpec, name="aligned10", n_active=10, ells=0.5),
    "aligned10_sin": partial(
        FamilySpec, name="aligned10_sin", n_active=10, ells=0.5, generator="sinusoid"
    ),
    "decoupled": partial(
        FamilySpec,
        name="decoupled",
        n_active=8,
        shares=(0.21, 0.21, 0.21, 0.21, 0.04, 0.04, 0.04, 0.04),
        ells=(1.5, 1.5, 0.08, 0.08, 1.5, 1.5, 0.08, 0.08),
    ),
    "anti_aligned": partial(
        FamilySpec,
        name="anti_aligned",
        n_active=6,
        shares=(0.35, 0.25, 0.18, 0.12, 0.07, 0.03),
        ells=(3.0, 0.5, 0.27, 0.18, 0.11, 0.06),
    ),
    "interaction_g0.00": partial(
        FamilySpec, name="interaction_g0.00", n_active=5, ells=0.5, n_pairs=2, gamma=0.00
    ),
    "interaction_g0.10": partial(
        FamilySpec, name="interaction_g0.10", n_active=5, ells=0.5, n_pairs=2, gamma=0.10
    ),
    "interaction_g0.25": partial(
        FamilySpec, name="interaction_g0.25", n_active=5, ells=0.5, n_pairs=2, gamma=0.25
    ),
    "interaction_g0.50": partial(
        FamilySpec, name="interaction_g0.50", n_active=5, ells=0.5, n_pairs=2, gamma=0.50
    ),
    "interaction_g0.75": partial(
        FamilySpec, name="interaction_g0.75", n_active=5, ells=0.5, n_pairs=2, gamma=0.75
    ),
    "rotated_t0": partial(
        FamilySpec, name="rotated_t0", n_active=10, ells=0.5, n_pairs=5, theta_deg=0.0
    ),
    "rotated_t15": partial(
        FamilySpec, name="rotated_t15", n_active=10, ells=0.5, n_pairs=5, theta_deg=15.0
    ),
    "rotated_t45": partial(
        FamilySpec, name="rotated_t45", n_active=10, ells=0.5, n_pairs=5, theta_deg=45.0
    ),
    "dense_weak": partial(FamilySpec, name="dense_weak", n_active=None, ells=3.0, monotone=True),
}

# (variant, family, overrides): every FAMILIES entry already carries its full configuration, so
# overrides is always {} here; `family` groups variants for downstream analysis (e.g. plotting
# all three rotation angles together) without re-deriving it from the variant name.
STUDY_GRID: list[tuple[str, str, dict]] = [
    ("aligned3", "aligned", {}),
    ("aligned10", "aligned", {}),
    ("aligned10_sin", "aligned", {}),
    ("decoupled", "decoupled", {}),
    ("anti_aligned", "anti_aligned", {}),
    ("interaction_g0.00", "interaction", {}),
    ("interaction_g0.10", "interaction", {}),
    ("interaction_g0.25", "interaction", {}),
    ("interaction_g0.50", "interaction", {}),
    ("interaction_g0.75", "interaction", {}),
    ("rotated_t0", "rotated", {}),
    ("rotated_t15", "rotated", {}),
    ("rotated_t45", "rotated", {}),
    ("dense_weak", "dense_weak", {}),
]


def make_family(name: str, seed: int, D: int = 100, **overrides) -> SyntheticObjective:
    """Build the named study variant at `seed` (and `D`, default 100): `KeyError` if `name` is
    not in `FAMILIES`, `TypeError` if `overrides` names a field `FamilySpec` does not have.
    """
    spec = FAMILIES[name](D=D, **overrides)
    return build(spec, seed)
