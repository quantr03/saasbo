"""Tests for SyntheticObjective.save/load: persisting an objective without ever re-drawing.

The Global Constraint this task exists to serve is "same seed -> bit-identical saved files", so
that the study's 140-file grid can be generated once and reloaded reproducibly. Three kinds of
assertion live here, and per the branch's own caution about assertions with no power against the
mutant they appear to guard, the load-bearing ones are the ones that cross a real boundary:

- `test_repeated_save_is_byte_identical` compares the raw bytes of two files independently
  produced by two independently-built objects at the same seed -- not one file compared to
  itself. The hand-rolled `.npz` writer (ruling R5) exists because `np.savez` writes each member
  through `zipfile`'s streaming, unknown-size code path (`zipf.open(name, "w",
  force_zip64=True)`), which lays out a member differently -- and slightly larger, measured 20
  bytes per member -- than a `writestr` call with a pre-built, known-size payload; not, as an
  earlier draft of this file claimed, because `np.savez` stamps a wall-clock timestamp (on numpy
  2.4.6, measured: it does not -- both paths default to a fixed 1980-01-01 `ZipInfo.date_time`).
  This test's own correctness does not depend on wall-clock timing either way, by construction.
- `test_round_trip_agrees_with_the_original_to_1e_15` compares `f` at 10**4 points between the
  object that wrote the files and the object `load` reconstructs from them -- an independent
  evaluation, not a readback of stored state compared to itself. Dropping or corrupting `mu`
  (ruling R30 follow-on) or `main_values`/`inter_values` would shift this by orders of magnitude
  more than the tolerance; a manual check (see task-8-report.md) confirms a corrupted `mu` alone
  moves it to ~5e-3.
- `test_load_does_not_redraw_for_any_generator` monkeypatches every generator name each of the
  three study generators is bound under (`eigen_factor` on both `synthobj.kernel` and
  `synthobj.families`, plus `sinusoid_draw`, `draw_until_monotone` and `gp_draw` on
  `synthobj.families`, all imported by name at module scope) to raise, then loads anyway for the
  matern (`rotated_t45`), sinusoid (`aligned10_sin`) and monotone-rejection (`dense_weak`)
  variants -- the strongest available evidence that `load` never re-draws, for every generator
  the study uses, not just the matern one the brief names literally.

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
import zipfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import synthobj.families as families
import synthobj.kernel as kernel
import synthobj.objective as objective_mod
from synthobj.component import Component
from synthobj.draws import gp_draw
from synthobj.families import make_family
from synthobj.kernel import eigen_factor, make_grid
from synthobj.objective import Labels, SyntheticObjective
from synthobj.rotation import EXT_HI, EXT_LO

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
    bytes on both files (ruling R5). Fixed member timestamps, `ZIP_STORED`, and sorted-name write
    order make this independent of wall-clock timing and of dict/array insertion order by
    construction -- see the module docstring for why the hand-rolled writer exists instead of
    `np.savez` (a real, measured reason, but a different one than an earlier draft claimed).
    """
    obj_a = make_family("interaction_g0.25", seed=5, D=D)
    obj_b = make_family("interaction_g0.25", seed=5, D=D)

    npz_a, json_a = obj_a.save(tmp_path / "a")
    npz_b, json_b = obj_b.save(tmp_path / "b")

    assert npz_a.read_bytes() == npz_b.read_bytes()
    assert json_a.read_bytes() == json_b.read_bytes()


def test_save_path_survives_a_dotted_stem(tmp_path: Path) -> None:
    """`Path.with_suffix` parses a decimal point in the stem as an existing suffix and replaces
    it: `Path("interaction_g0.25").with_suffix(".npz")` is `interaction_g0.npz`, not
    `interaction_g0.25.npz`. Every study variant name with a `gamma` in it
    (`interaction_g0.00/0.10/0.25/0.50/0.75`) has exactly this shape, so `save`/`load` must build
    paths by plain string concatenation, never `with_suffix`, or five of the study's variants
    collapse onto one file with nothing raising to say so.
    """
    obj = make_family("interaction_g0.25", seed=0, D=D)
    npz_path, json_path = obj.save(tmp_path / "interaction_g0.25")
    assert npz_path == tmp_path / "interaction_g0.25.npz"
    assert json_path == tmp_path / "interaction_g0.25.json"


def test_two_gamma_variants_in_one_directory_do_not_collide(tmp_path: Path) -> None:
    """The concrete failure `test_save_path_survives_a_dotted_stem` guards in the abstract:
    `interaction_g0.25` and `interaction_g0.50`, saved to their own names in the same directory,
    must produce four distinct files and reload as two distinct objectives -- not silently merge
    onto one `interaction_g0.npz`/`.json` pair, which is what happened before this fix round.
    """
    obj_25 = make_family("interaction_g0.25", seed=0, D=D)
    obj_50 = make_family("interaction_g0.50", seed=0, D=D)
    obj_25.save(tmp_path / "interaction_g0.25")
    obj_50.save(tmp_path / "interaction_g0.50")

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "interaction_g0.25.json",
        "interaction_g0.25.npz",
        "interaction_g0.50.json",
        "interaction_g0.50.npz",
    ]

    loaded_25 = SyntheticObjective.load(tmp_path / "interaction_g0.25")
    loaded_50 = SyntheticObjective.load(tmp_path / "interaction_g0.50")
    assert loaded_25.labels.family == "interaction_g0.25"
    assert loaded_50.labels.family == "interaction_g0.50"
    assert loaded_25.f_star != loaded_50.f_star


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


def test_load_passes_stored_labels_through_without_recomputing(tmp_path: Path) -> None:
    """The headline requirement's own no-power gap, found in review: nothing anywhere proved
    `load` actually passes `labels=labels` into `__init__` rather than falling through to
    `_compute_labels`. On this branch's own reloaded, bit-identical components, a full
    recomputation of `f_star`/`x_star`/`g`/`s`/`ell`/`s_axis`/`gamma_axis` lands back on the same
    bit-identical numbers `_labels_equal` and `loaded.f_star == obj.f_star` already check
    (confirmed: deleting `labels=labels` from `load` left the full suite at 244/244) -- exactly
    the `mu` lesson one level up, and this test is the fix, corrupting a value no recomputation
    could ever reproduce and confirming `load` carries it through regardless.
    """
    obj = make_family("rotated_t45", seed=7, D=D)
    npz_path, json_path = obj.save(tmp_path / "obj")

    payload = json.loads(json_path.read_text())
    payload["labels"]["f_star"] = -123.5
    payload["labels"]["g"] = [0.0] * obj.labels.D
    json_path.write_text(json.dumps(payload, sort_keys=True, indent=1))

    loaded = SyntheticObjective.load(tmp_path / "obj")
    assert loaded.f_star == -123.5
    assert loaded.labels.f_star == -123.5
    assert np.array_equal(loaded.labels.g, np.zeros(obj.labels.D))


def test_load_does_not_call_the_block_maximizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The eigen_factor lesson applied to `f_star`: `block_argmax` and `rotated_block_argmax` are
    the two functions `_compute_labels` would call to re-derive `f_star`/`x_star` if `labels=`
    were ever dropped, and both are imported by name into `synthobj.objective`
    (`from synthobj.interaction import ... block_argmax`, `from synthobj.rotation import ...
    rotated_block_argmax`), so they must be patched on `objective_mod`, not on
    `synthobj.interaction`/`synthobj.rotation` where they're defined -- patching the defining
    module would not touch the already-bound names `objective.py` actually calls.
    """
    obj = make_family("rotated_t45", seed=1, D=D)
    obj.save(tmp_path / "obj")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load must not call the block maximizers")

    monkeypatch.setattr(objective_mod, "block_argmax", _boom)
    monkeypatch.setattr(objective_mod, "rotated_block_argmax", _boom)

    loaded = SyntheticObjective.load(tmp_path / "obj")
    assert loaded.f_star == obj.f_star


def test_load_restores_scale_from_the_file_rather_than_rederiving_it(tmp_path: Path) -> None:
    """Ruling R34. `__init__` recomputes `scale` from the reloaded components as a side effect of
    construction, and on unmodified data that recompute is bit-exact -- but the stored
    `labels.f_star` was computed against the *original* `_compute_scale()`'s output, so `load`
    must read `scale` back from the file rather than trust a future `_compute_scale` to keep
    agreeing with it. Corrupting the stored `scale` to 2x and confirming `f` scales by exactly 2x
    is the only way to tell "reads the file" from "recomputes and happens to match": a `load` that
    dropped the restore would report the recomputed (uncorrupted) scale instead.
    """
    obj = make_family("rotated_t45", seed=3, D=D)
    npz_path, json_path = obj.save(tmp_path / "obj")

    payload = json.loads(json_path.read_text())
    payload["labels"]["scale"] *= 2.0
    json_path.write_text(json.dumps(payload, sort_keys=True, indent=1))

    loaded = SyntheticObjective.load(tmp_path / "obj")
    assert loaded.scale == pytest.approx(obj.scale * 2.0)

    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(N_POINTS, D))
    ratio = loaded(X) / obj(X)
    assert np.median(ratio) == pytest.approx(2.0, abs=1e-9)


def test_load_restores_noise_sd_from_the_file_rather_than_rederiving_it(tmp_path: Path) -> None:
    """Ruling R34's other half. `__init__` sets `noise_sd` from the reconstructed `spec` (or the
    0.1 default), which agrees with the stored `labels.noise_sd` today only because both paths
    compute it the same way from the same spec -- `observe` reads `self.noise_sd` directly, so a
    future change to either the spec-reading logic or the 0.1 default must not silently move a
    reloaded objective's noise onto a different number than the file was actually built with.
    """
    obj = make_family("aligned10", seed=0, D=D)
    npz_path, json_path = obj.save(tmp_path / "obj")

    payload = json.loads(json_path.read_text())
    payload["labels"]["noise_sd"] = 0.7
    json_path.write_text(json.dumps(payload, sort_keys=True, indent=1))

    loaded = SyntheticObjective.load(tmp_path / "obj")
    assert loaded.noise_sd == 0.7


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


@pytest.mark.parametrize("variant", ["rotated_t45", "aligned10_sin", "dense_weak"])
def test_load_does_not_redraw_for_any_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """`load` must never re-draw, for every generator the study uses, not just the one the brief
    names literally. `_draw_fn` calls `eigen_factor` only for `generator == "matern"`
    (`rotated_t45`); the sinusoid branch (`aligned10_sin`) goes through `joint_points`/`make_grid`/
    `sinusoid_draw` and never touches `eigen_factor` at all, so a regression that reloaded a
    sinusoid variant by calling `families.build` would raise nothing against a test that only
    patched `eigen_factor` -- confirmed empirically before writing this test. `dense_weak` adds
    `draw_until_monotone` (matern's rejection wrapper) to the same check. Every name checked here
    is imported by name into `synthobj.families` at module scope, so each is patched on
    `families`, not on `synthobj.kernel`/`synthobj.draws` where it's defined -- patching the
    defining module alone would not touch the already-bound name `families.py` actually calls
    (confirmed for `eigen_factor` specifically: patching only `kernel.eigen_factor` and then
    calling `families.make_family` still succeeds).
    """
    obj = make_family(variant, seed=1, D=D)
    obj.save(tmp_path / "obj")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load must not draw")

    monkeypatch.setattr(kernel, "eigen_factor", _boom)
    monkeypatch.setattr(families, "eigen_factor", _boom)
    monkeypatch.setattr(families, "sinusoid_draw", _boom)
    monkeypatch.setattr(families, "draw_until_monotone", _boom)
    monkeypatch.setattr(families, "gp_draw", _boom)

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
    # M3: NOTE ON POWER. The added assertions below do not distinguish "load reads Labels.S"
    # from "load re-derives S from the npz `active` array's coordinate set" -- a correct
    # re-derivation produces the identical sorted tuple on healthy data, so both readings pass
    # here regardless. What they do add over the seed comparison above is narrower but real:
    # power against wholesale S corruption (e.g. a dropped or duplicated coordinate) or a type
    # bug in the round trip (e.g. S surviving as a list or the wrong dtype), either of which
    # would break `==` against the original even though it says nothing about "read" vs.
    # "re-derived".
    assert loaded_a.labels.S == obj_a.labels.S
    assert loaded_b.labels.S == obj_b.labels.S


def test_npz_members_and_json_keys_are_written_in_sorted_order(tmp_path: Path) -> None:
    """Both determinism safeguards `_write_npz_deterministic` and `save` rely on, pinned directly
    rather than only inferred from the byte-identity test passing: `zipfile` member order follows
    insertion order, so `_write_npz_deterministic` must insert in `sorted(arrays)` order for the
    namelist to come back sorted; `json.dumps(..., sort_keys=True)` must actually be passed for
    `json.loads` (which preserves the text's own key order) to hand back a sorted dict. Confirmed
    this has power: reversing the npz write order and dropping `sort_keys=True` together leaves
    every other test in this module passing.
    """
    obj = make_family("interaction_g0.25", seed=0, D=D)
    npz_path, json_path = obj.save(tmp_path / "obj")

    with zipfile.ZipFile(npz_path) as zf:
        names = zf.namelist()
    assert names == sorted(names)

    payload = json.loads(json_path.read_text())
    assert list(payload) == sorted(payload)


def test_save_rejects_components_that_do_not_share_one_knot_grid(tmp_path: Path) -> None:
    """The file format's one shared `grid` array (true of every family `synthobj.families`
    builds) is enforced by `_shared_grid`, not merely assumed: a hand-built objective mixing a
    plain `[0,1]` component with an extended-domain one (as `test_objective.py`'s own
    `test_rotation_is_accepted` does) must raise rather than have `save` silently write one
    component's grid under another's values.
    """
    f_unit = Component.from_raw(
        2, 0.5, 0.5, make_grid(0.0, 1.0, 256),
        gp_draw(eigen_factor(0.0, 1.0, 256, 0.5), np.random.default_rng(1)),
    )
    f_ext = Component.from_raw(
        5,
        0.5,
        0.5,
        make_grid(EXT_LO, EXT_HI, 256),
        gp_draw(eigen_factor(EXT_LO, EXT_HI, 256, 0.5), np.random.default_rng(2)),
    )
    obj = SyntheticObjective(D, (f_unit, f_ext), (), None, None, 0)

    with pytest.raises(ValueError, match="share one knot grid"):
        obj.save(tmp_path / "obj")
