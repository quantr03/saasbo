"""Tests for synthobj.families' seeding scheme (plan decision D6).

`SeedSequence(seed).spawn(3)` gives three independent streams -- `select`, `main`, `inter` -- and
the separability this buys is the point of the whole scheme: changing one part of a `FamilySpec`
must never perturb a component or a coordinate selection it has no business touching. Each test
below isolates one such independence claim rather than re-deriving it from `build`'s own bookkeeping:

- a gamma sweep at one seed must move only the share scaling, never S, the pairs, or a component's
  underlying shape (checked via `values / sqrt(share)`, an independent renormalization the
  implementation is not simply asked to echo back);
- changing one coordinate's lengthscale must leave every other rank's raw draw bit-identical, not
  merely close;
- `select_active` truncating one permutation is what makes `aligned3`'s S a *prefix* of
  `aligned10`'s at the same seed, checked both at the primitive level and through the actual
  `FAMILIES` entries;
- different seeds give different S, and `noise_rng` separates its `run` argument from the
  objective's own draws.

A last test pins the task's core deliverable directly: same seed, same D, same variant -> a
bit-identical objective (`S`, `f_star`, and `f` at a shared batch of points).
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from synthobj.families import FamilySpec, build, make_family, noise_rng, select_active, streams


@pytest.fixture(scope="module", autouse=True)
def _restore_families_import_state():
    """See test_families.py's fixture of the same name (ruling R15).

    This module sorts after test_objective.py, so its own import cannot break that test's
    `"synthobj.families" not in sys.modules` check under the default alphabetical run order --
    test_families.py's identical fixture already restores that state before test_objective.py
    runs. This is included anyway so the property holds regardless of run order (e.g. a partial
    invocation that runs this file without test_families.py).
    """
    yield
    sys.modules.pop("synthobj.families", None)


def test_gamma_sweep_keeps_S_pairs_and_component_shapes_identical() -> None:
    """Same seed, gamma in {0, 0.5}: identical S and pairs; every component's values divided by
    sqrt(its own share) agree to 1e-12 -- the only thing gamma is allowed to move is that scale."""
    seed = 42
    base = dict(name="t", D=20, n_active=5, ells=0.5, n_pairs=2)
    spec0 = FamilySpec(gamma=0.0, **base)
    spec5 = FamilySpec(gamma=0.5, **base)
    obj0, obj5 = build(spec0, seed), build(spec5, seed)

    assert obj0.labels.S == obj5.labels.S
    assert obj0.labels.pairs == obj5.labels.pairs

    for c0, c5 in zip(obj0.components, obj5.components):
        assert c0.coord == c5.coord
        v0 = c0.values / np.sqrt(c0.share)
        v5 = c5.values / np.sqrt(c5.share)
        assert np.allclose(v0, v5, atol=1e-12, rtol=0.0)


def test_changing_one_coordinates_ell_leaves_the_others_bit_identical() -> None:
    seed = 42
    ells_a = (0.5, 0.5, 0.5, 0.5, 0.5)
    ells_b = (0.5, 0.5, 0.9, 0.5, 0.5)  # only rank 2 differs
    spec_a = FamilySpec(name="t", D=20, n_active=5, ells=ells_a)
    spec_b = FamilySpec(name="t", D=20, n_active=5, ells=ells_b)
    obj_a, obj_b = build(spec_a, seed), build(spec_b, seed)

    for rank, (ca, cb) in enumerate(zip(obj_a.components, obj_b.components)):
        assert ca.coord == cb.coord
        if rank == 2:
            assert not np.array_equal(ca.values, cb.values)
            continue
        assert np.array_equal(ca.grid, cb.grid)
        assert np.array_equal(ca.values, cb.values)


def test_n_active_nests_across_ranks() -> None:
    """select_active truncates one permutation, so a smaller n_active is a prefix of a larger one
    at the same seed -- checked both at the primitive level and through aligned3/aligned10."""
    seed = 7
    ss = streams(seed)
    s3 = select_active(100, 3, ss.select)
    s10 = select_active(100, 10, ss.select)
    assert np.array_equal(s3, s10[:3])

    o3 = make_family("aligned3", seed=seed, D=100)
    o10 = make_family("aligned10", seed=seed, D=100)
    assert set(o3.labels.S) <= set(o10.labels.S)


def test_different_seeds_give_different_S() -> None:
    o0 = make_family("aligned10", seed=0, D=20)
    o1 = make_family("aligned10", seed=1, D=20)
    assert o0.labels.S != o1.labels.S


def test_noise_rng_streams_are_independent_of_run() -> None:
    r0, r1 = noise_rng(0, 0), noise_rng(0, 1)
    assert r0.standard_normal() != r1.standard_normal()


def test_same_seed_gives_a_bit_identical_objective() -> None:
    """Determinism is the task's core deliverable: same seed, same D, same variant -> the same S,
    f_star and f(X)."""
    seed = 3
    obj_a = make_family("aligned10", seed=seed, D=20)
    obj_b = make_family("aligned10", seed=seed, D=20)
    X = np.random.default_rng(0).uniform(size=(5, 20))

    assert obj_a.labels.S == obj_b.labels.S
    assert obj_a.f_star == obj_b.f_star
    assert np.array_equal(obj_a(X), obj_b(X))
