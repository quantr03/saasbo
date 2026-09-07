"""Tests for SyntheticObjective.save/load: persisting an objective without ever re-drawing.

The Global Constraint this task exists to serve is "same seed -> bit-identical saved files", so
that the study's 140-file grid can be generated once and reloaded reproducibly. Three kinds of
assertion live here, and per the branch's own caution about assertions with no power against the
mutant they appear to guard, the load-bearing ones are the ones that cross a real boundary:

- `test_repeated_save_is_byte_identical` compares the raw bytes of two files independently
  produced by two independently-built objects at the same seed -- not one file compared to
  itself, which `np.savez` (ruling R5) would fail unpredictably depending on wall-clock timing.
- `test_round_trip_agrees_with_the_original_to_1e_15` compares `f` at 10**4 points between the
  object that wrote the files and the object `load` reconstructs from them -- an independent
  evaluation, not a readback of stored state compared to itself. Dropping or corrupting `mu`
  (ruling R30 follow-on) or `main_values`/`inter_values` would shift this by orders of magnitude
  more than the tolerance; a manual check (see task-8-report.md) confirms a corrupted `mu` alone
  moves it to ~5e-3.
- `test_load_does_not_call_eigen_factor` monkeypatches every name that function is bound under
  (`synthobj.kernel.eigen_factor` and `synthobj.families.eigen_factor`, which imports it by name
  at module scope) to raise, then loads anyway -- the strongest available evidence that `load`
  never re-draws.

The remaining tests (`Labels` field-by-field, `spec ==`, `version`, different-seed `S`) are
weaker, stored-state-round-trips-to-itself checks; they are kept because the brief asks for them
and because `FamilySpec.__eq__` (kept `eq=True`, ruling R9) does have real power against a missed
tuple conversion when the round-tripped spec actually holds a tuple -- which is why
`test_family_spec_shares_and_ells_round_trip_as_tuples` deliberately uses `decoupled`, the family
whose `shares` and `ells` are real per-rank tuples rather than `None`/a scalar. `interaction_g0.25`
and `rotated_t45` (used for the 1e-15 test) both have `shares=None` and a scalar `ells`, so their
own `spec ==` check has no power over that specific ruling.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import synthobj.families as families
import synthobj.kernel as kernel
from synthobj.families import make_family
from synthobj.objective import Labels, SyntheticObjective

D = 20
N_POINTS = 10_000


def _labels_equal(a: Labels, b: Labels) -> bool:
    """Field-by-field comparison (`Labels` is `eq=False`, ruling R4): ndarray fields via
    `array_equal` (`equal_nan=True` for the float ones -- `ell` is NaN off `S` by design, ruling
    R6 -- and plain for `active`, which is bool and rejects `equal_nan`), everything else by `==`.
    """
    for f in fields(Labels):
        av, bv = getattr(a, f.name), getattr(b, f.name)
        if isinstance(av, np.ndarray):
            if av.dtype.kind == "f":
                if not np.array_equal(av, bv, equal_nan=True):
                    return False
            elif not np.array_equal(av, bv):
                return False
        elif av != bv:
            return False
    return True


def test_repeated_save_is_byte_identical(tmp_path: Path) -> None:
    """Same seed, two independently-built objectives, saved to two different stems -> identical
    bytes on both files (ruling R5). `np.savez` cannot pass this: its zip member timestamps
    (`time.localtime()`) would make two saves of the same arrays differ in bytes whenever they
    straddle a one-second boundary, so this test would pass or fail by timing rather than by
    correctness.
    """
    obj_a = make_family("interaction_g0.25", seed=5, D=D)
    obj_b = make_family("interaction_g0.25", seed=5, D=D)

    npz_a, json_a = obj_a.save(tmp_path / "a")
    npz_b, json_b = obj_b.save(tmp_path / "b")

    assert npz_a.read_bytes() == npz_b.read_bytes()
    assert json_a.read_bytes() == json_b.read_bytes()


@pytest.mark.parametrize("variant", ["interaction_g0.25", "rotated_t45"])
def test_round_trip_agrees_with_the_original_to_1e_15(tmp_path: Path, variant: str) -> None:
    """`load` reconstructs an object whose `f` agrees with the one that saved it to 1e-15 at
    10**4 random points -- the assertion that would fail if `main_values`/`inter_values` were
    corrupted, if an interaction's factors or `c` were dropped, or if the rotation offset `mu`
    (ruling R30 follow-on) were lost, since the whole function would then shift by a constant.
    `rotated_t45` exercises the `mu != 0` path; `interaction_g0.25` exercises real `Interaction`s.
    """
    obj = make_family(variant, seed=7, D=D)
    obj.save(tmp_path / variant)
    loaded = SyntheticObjective.load(tmp_path / variant)

    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(N_POINTS, D))
    assert np.max(np.abs(obj(X) - loaded(X))) < 1e-15
    assert loaded.f_star == obj.f_star
    assert loaded.mu == obj.mu

    assert _labels_equal(obj.labels, loaded.labels)
    assert loaded.spec == obj.spec


def test_load_reads_mu_from_the_file_rather_than_rederiving_it(tmp_path: Path) -> None:
    """Ruling R30 follow-on's actual claim, made to have power. `SyntheticObjective.__init__`
    recomputes `mu` from the (bit-identical) reloaded components as a side effect of
    construction, and that recompute happens to land on the same float the stored one does --
    so `assert loaded.mu == obj.mu` above passes whether or not `load` reads the file at all, and
    measurably does: an earlier version of `load` that omitted the post-construction override
    passed every other test in this module, including that one and the 1e-15 `f` agreement, while
    silently never touching `payload["mu"]`.

    Corrupting the saved JSON's `mu` to a value `_compute_mu` would never produce and reloading
    is the only way to tell the two implementations apart: a `load` that truly reads the file
    carries the corrupted value onto the returned object (and `f` shifts by exactly the
    difference, scaled), while one that re-derives silently discards the corruption and returns
    the original, correct number.
    """
    obj = make_family("rotated_t45", seed=7, D=D)
    npz_path, json_path = obj.save(tmp_path / "obj")
    true_mu = obj.mu

    payload = json.loads(json_path.read_text())
    corrupted_mu = true_mu + 1.0
    payload["mu"] = corrupted_mu
    json_path.write_text(json.dumps(payload, sort_keys=True, indent=1))

    loaded = SyntheticObjective.load(tmp_path / "obj")
    assert loaded.mu == corrupted_mu

    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(N_POINTS, D))
    # __call__ computes (raw - mu) * scale, so obj(X) - loaded(X) == scale * (corrupted - true).
    shift = (obj(X) - loaded(X)) / obj.scale
    assert np.allclose(shift, corrupted_mu - true_mu, atol=1e-12)


def test_family_spec_shares_and_ells_round_trip_as_tuples(tmp_path: Path) -> None:
    """`decoupled` is the family whose `FamilySpec.shares` and `.ells` are real per-rank tuples
    (every interaction/rotated variant's `shares` is `None` and `ells` a scalar, so their own
    `spec ==` check cannot see this). JSON has no tuple type (ruling R9): a `load` that left
    either field as a list would fail this `==`, since `FamilySpec` keeps `eq=True` and
    `(1, 2) != [1, 2]` in Python -- that is the point of the test.
    """
    obj = make_family("decoupled", seed=2, D=D)
    obj.save(tmp_path / "decoupled")
    loaded = SyntheticObjective.load(tmp_path / "decoupled")

    assert loaded.spec == obj.spec
    assert isinstance(loaded.spec.shares, tuple)
    assert isinstance(loaded.spec.ells, tuple)


def test_load_does_not_call_eigen_factor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`load` must never re-draw. `eigen_factor` is defined in `synthobj.kernel` but imported by
    name into `synthobj.families` (`_draw_fn`'s only call site), so patching `kernel.eigen_factor`
    alone would not raise for a regression that reloaded by calling `families.build` -- both
    bindings are patched here so the test has power against that failure mode, not just against
    the one the brief names literally.
    """
    obj = make_family("rotated_t45", seed=1, D=D)
    obj.save(tmp_path / "obj")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load must not call eigen_factor")

    monkeypatch.setattr(kernel, "eigen_factor", _boom)
    monkeypatch.setattr(families, "eigen_factor", _boom)

    loaded = SyntheticObjective.load(tmp_path / "obj")
    assert loaded.f_star == obj.f_star


def test_version_is_present(tmp_path: Path) -> None:
    obj = make_family("aligned3", seed=0, D=D)
    _, json_path = obj.save(tmp_path / "obj")
    payload = json.loads(json_path.read_text())
    assert payload["version"] == 1


def test_different_seeds_give_different_S(tmp_path: Path) -> None:
    obj_a = make_family("aligned10", seed=0, D=D)
    obj_b = make_family("aligned10", seed=1, D=D)
    obj_a.save(tmp_path / "a")
    obj_b.save(tmp_path / "b")

    loaded_a = SyntheticObjective.load(tmp_path / "a")
    loaded_b = SyntheticObjective.load(tmp_path / "b")
    assert loaded_a.labels.S != loaded_b.labels.S
