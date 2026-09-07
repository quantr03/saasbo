"""Tests for synthobj.generate: the study-grid CLI.

Per the branch's own caution about assertions with no power against the mutant they appear to
guard, a manifest is a description of files the CLI *just wrote*, so `assert entry["f_star"] ==
obj.f_star` against the very object the CLI built in-process mostly proves a float survived a
JSON round trip. The load-bearing checks here instead cross a real boundary:

- `test_writes_stems_matching_disk_hashes_and_independent_rebuild` hashes the files *on disk* with
  a fresh `hashlib.sha256` call (not `synthobj.generate._sha256`) and compares to the manifest's
  claim; separately rebuilds each (variant, seed) from scratch with `make_family` -- an
  independent evaluator, not a readback of stored state -- and compares `f_star`/`gamma`/`S` to
  the manifest entry; and separately again, `SyntheticObjective.load`s the stem straight off disk
  (the brief's own literal spec) and checks `f_star` a third, independent way. `family` is checked
  against a hand-written literal mapping, not `entry["family"] in (...)`, which a hardcoded wrong
  constant could satisfy for every entry.
- `test_rerun_without_overwrite_skips_and_does_not_rebuild` monkeypatches `generate.make_family`
  to raise, then reruns without `--overwrite`: if skip logic ever degraded into "rebuild every
  time", this fails loudly instead of merely reading back files that happen to already exist.
- `test_overwrite_actually_rebuilds_not_just_reruns` is that monkeypatch's mirror image: it spies
  on `make_family` under `--overwrite` and asserts it *was* called.
  `test_overwrite_produces_byte_identical_files` below it is explicitly kept as a *weak* test --
  "byte-identical across two runs" is exactly what "the second run did nothing" also produces, and
  a fix-round review confirmed deleting `and not args.overwrite` from the skip condition (making
  `--overwrite` a complete no-op) left it, and the rest of the original suite, green.
- `test_two_gamma_variants_do_not_collide_on_disk` is the concrete regression the brief's path
  hazard section warns about (Task 8's `Path.with_suffix` bug collapsed every `interaction_g0.*`
  variant onto one file): two gamma variants at the same seed must produce four distinct files
  with distinct hashes, not silently merge.
- `test_stem_path_builds_expected_literal_path`,
  `test_npz_and_json_paths_are_literal_and_do_not_collide_across_gamma_variants`, and
  `test_npz_and_json_paths_use_fstring_concatenation_not_with_suffix` compare the path helpers'
  output against hand-written literal `Path(...)` strings, not against a second call to the same
  helpers.
- `test_manifest_scans_directory_across_differently_scoped_runs` covers ruling R35/R36: the
  manifest describes `--out` itself (built by scanning it after generation), not just the
  (variant, seed) pairs the most recent invocation asked for.
- Ruling R37 (fix round 2, correcting fix round 1's own over-broad rule): the exit code answers
  "is the manifest incomplete", not "was anything ignored". Five tests each pin one class rather
  than one test asserting a bundled non-zero, because round 1's version of this scan would have
  exited non-zero on `.DS_Store` alone -- close to guaranteed on this OneDrive-synced,
  Finder-browsed checkout -- which the reviewer measured and ruled a real defect, not a hypothetical
  one: `test_scan_ignores_dotfile_silently_and_exits_zero` (exit 0, *no* warning at all),
  `test_scan_warns_but_exits_zero_on_ordinary_non_dotfile_stray` (exit 0, but warned -- the pair
  that matters most, since `notes.txt` and `.DS_Store` must land on opposite sides of "warned" but
  the same side of "exit code"), `test_scan_exits_nonzero_on_half_written_pair`,
  `test_scan_exits_nonzero_on_corrupt_pair`, and
  `test_scan_exits_nonzero_on_objective_shaped_pair_under_unknown_variant_dir` (all three: exit 1,
  warned, and excluded from `entries`, since each represents an objective that exists, or partly
  exists, on disk but that the manifest cannot describe).
- `test_dry_run_prints_full_grid_via_subprocess_and_writes_nothing` checks row *content*
  (140 distinct lines, one specific literal row present), not just a row *count* -- a `print("x")`
  substituted for the real row would still print 140 lines.
- The `wrote N stem(s), skipped M stem(s)` stdout counters (the brief's own "report how many were
  skipped") are asserted directly in `test_writes_stems_matching_disk_hashes_and_independent_rebuild`
  and `test_rerun_without_overwrite_skips_and_does_not_rebuild`, not merely implied by file counts.

One test exercises the module entry point through `subprocess` (`--dry-run` on the default grid,
which builds nothing and is fast regardless of process startup cost); everything else goes through
`main(argv)` directly, which is far cheaper per the brief's own guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from synthobj import generate
from synthobj.families import make_family
from synthobj.objective import SyntheticObjective

PYTHON = "/opt/anaconda3/envs/saasbo/bin/python"
D = 20


# --- the main write path -------------------------------------------------------------------


def test_writes_stems_matching_disk_hashes_and_independent_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The brief's own literal command: 2 variants x 2 seeds -> 4 stems + a manifest, and every
    manifest claim checked against something other than the CLI's own in-memory state.
    """
    code = generate.main(
        [
            "--out", str(tmp_path),
            "--D", str(D),
            "--seeds", "0-1",
            "--families", "aligned3,interaction_g0.25",
        ]
    )
    assert code == 0
    assert "wrote 4 stem(s), skipped 0 stem(s)" in capsys.readouterr().out

    expected_files = [
        tmp_path / "aligned3" / "seed00_D20.npz",
        tmp_path / "aligned3" / "seed00_D20.json",
        tmp_path / "aligned3" / "seed01_D20.npz",
        tmp_path / "aligned3" / "seed01_D20.json",
        tmp_path / "interaction_g0.25" / "seed00_D20.npz",
        tmp_path / "interaction_g0.25" / "seed00_D20.json",
        tmp_path / "interaction_g0.25" / "seed01_D20.npz",
        tmp_path / "interaction_g0.25" / "seed01_D20.json",
    ]
    for path in expected_files:
        assert path.is_file(), f"missing {path}"

    # Every file that actually exists on disk is exactly the 4 stems' worth plus the manifest --
    # not more (a stray write), not fewer (a silent collision).
    all_files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(all_files) == 9

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["Ds"] == [D]
    assert manifest["seeds"] == [0, 1]
    assert len(manifest["entries"]) == 4

    # Independently written, not derived from generate._FAMILY_OF: a hardcoded wrong constant for
    # every entry would pass `entry["family"] in ("aligned", "interaction")` but not this.
    expected_family = {"aligned3": "aligned", "interaction_g0.25": "interaction"}

    for entry in manifest["entries"]:
        npz_path = tmp_path / entry["npz"]
        json_path = tmp_path / entry["json"]

        # Boundary 1: hash the bytes actually on disk, independently of generate._sha256.
        assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == entry["sha256_npz"]
        assert hashlib.sha256(json_path.read_bytes()).hexdigest() == entry["sha256_json"]

        # Boundary 2: rebuild from scratch (a fresh object, not the CLI's own) and compare.
        rebuilt = make_family(entry["variant"], seed=entry["seed"], D=entry["D"])
        assert rebuilt.f_star == entry["f_star"]
        assert rebuilt.labels.gamma == entry["gamma"]
        assert len(rebuilt.labels.S) == entry["n_active"]

        # Boundary 3: the brief's own literal spec item -- load the stem straight off disk (a
        # literal stem, not built via generate._stem_path) and compare f_star a third way.
        stem = tmp_path / entry["variant"] / f"seed{entry['seed']:02d}_D{entry['D']}"
        assert SyntheticObjective.load(stem).f_star == entry["f_star"]

        assert entry["family"] == expected_family[entry["variant"]]
        assert entry["overrides"] == {}


def test_two_gamma_variants_do_not_collide_on_disk(tmp_path: Path) -> None:
    """The concrete failure the path hazard section warns about: `interaction_g0.25` and
    `interaction_g0.50` at the same seed must produce four distinct files with distinct content,
    not collapse onto `interaction_g0.{npz,json}` the way `Path.with_suffix` would.
    """
    code = generate.main(
        [
            "--out", str(tmp_path),
            "--D", str(D),
            "--seeds", "0",
            "--families", "interaction_g0.25,interaction_g0.50",
        ]
    )
    assert code == 0

    rel_paths = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    assert rel_paths == [
        "interaction_g0.25/seed00_D20.json",
        "interaction_g0.25/seed00_D20.npz",
        "interaction_g0.50/seed00_D20.json",
        "interaction_g0.50/seed00_D20.npz",
    ]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    by_variant = {e["variant"]: e for e in manifest["entries"]}
    assert by_variant["interaction_g0.25"]["sha256_npz"] != by_variant["interaction_g0.50"]["sha256_npz"]
    assert by_variant["interaction_g0.25"]["f_star"] != by_variant["interaction_g0.50"]["f_star"]
    assert by_variant["interaction_g0.25"]["gamma"] == pytest.approx(0.25)
    assert by_variant["interaction_g0.50"]["gamma"] == pytest.approx(0.50)


# --- dry-run, via subprocess (the one module-entry-point exercise) --------------------------


def test_dry_run_prints_full_grid_via_subprocess_and_writes_nothing(tmp_path: Path) -> None:
    """`--dry-run` on the default grid (all 14 variants x seeds 0-9) prints 140 rows and creates
    no files -- driven through `python -m synthobj.generate` to prove the module entry point
    itself works, not just `main(argv)`. Cheap regardless of grid size, since dry-run never builds
    an objective.

    Checks content, not just count: `len(set(lines)) == 140` fails a `print("x")`-style mutant
    that would still satisfy `len(lines) == 140`, and the literal spot-check pins the actual
    (variant, seed, D) format of a specific row.
    """
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [PYTHON, "-m", "synthobj.generate", "--out", str(tmp_path), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 140
    assert len(set(lines)) == 140
    assert "interaction_g0.25 seed=03 D=100" in lines
    assert list(tmp_path.iterdir()) == []


# --- rerun / skip / overwrite ----------------------------------------------------------------


def test_rerun_without_overwrite_skips_and_does_not_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    argv = [
        "--out", str(tmp_path),
        "--D", str(D),
        "--seeds", "0-1",
        "--families", "aligned3,interaction_g0.25",
    ]
    assert generate.main(argv) == 0
    capsys.readouterr()  # discard the first run's output

    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "manifest.json"]
    assert len(written) == 8
    before = {p: p.read_bytes() for p in written}

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a skipped stem must not be rebuilt")

    monkeypatch.setattr(generate, "make_family", _boom)

    assert generate.main(argv) == 0  # no --overwrite; must not touch _boom at all
    assert "wrote 0 stem(s), skipped 4 stem(s)" in capsys.readouterr().out

    after = {p: p.read_bytes() for p in written}
    assert after == before


def test_overwrite_actually_rebuilds_not_just_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`test_overwrite_produces_byte_identical_files` below is deliberately weak: a second run
    that did nothing at all also produces byte-identical files. A fix-round review confirmed this
    directly -- deleting `and not args.overwrite` from the skip condition, making `--overwrite` a
    complete no-op, left that test (and the rest of the original 15) green. This test instead
    spies on `make_family`, mirroring `test_rerun_without_overwrite_skips_and_does_not_rebuild`'s
    monkeypatch in the opposite direction, and asserts it *was* called under `--overwrite`.
    """
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0

    calls: list[tuple[object, ...]] = []
    real_make_family = generate.make_family

    def _spy(*args: object, **kwargs: object) -> SyntheticObjective:
        calls.append((args, kwargs))
        return real_make_family(*args, **kwargs)

    monkeypatch.setattr(generate, "make_family", _spy)

    assert generate.main(argv + ["--overwrite"]) == 0
    assert len(calls) == 1


def test_overwrite_produces_byte_identical_files(tmp_path: Path) -> None:
    """Weak by itself (see module docstring and `test_overwrite_actually_rebuilds_not_just_reruns`
    above) -- kept because the brief asks for it, and it does confirm ruling R5's byte-identity
    survives a real CLI round trip when combined with the stronger test proving a rebuild actually
    happened.
    """
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0

    npz_path = tmp_path / "aligned3" / "seed00_D20.npz"
    json_path = tmp_path / "aligned3" / "seed00_D20.json"
    before_npz, before_json = npz_path.read_bytes(), json_path.read_bytes()

    assert generate.main(argv + ["--overwrite"]) == 0
    assert npz_path.read_bytes() == before_npz
    assert json_path.read_bytes() == before_json


def test_rerun_manifest_created_changes_but_entries_stay_the_same(tmp_path: Path) -> None:
    """Per the brief: `created` is the one manifest field a rerun changes; every entry (hashes,
    f_star, gamma, ...) is unaffected by a skip-only rerun.
    """
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0
    manifest_1 = json.loads((tmp_path / "manifest.json").read_text())

    assert generate.main(argv) == 0
    manifest_2 = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest_1["created"] != manifest_2["created"]
    assert manifest_1["entries"] == manifest_2["entries"]


# --- ruling R35/R36: the manifest scans <out>, not just this run's requests -----------------


def test_manifest_scans_directory_across_differently_scoped_runs(tmp_path: Path) -> None:
    """Ruling R35: the manifest must describe everything under `--out`, not just what the most
    recent invocation was asked to produce. Two runs with different D/seeds/families into the
    same `--out` leave three `.npz` files on disk; a run-scoped manifest would report only the
    second run's one entry and silently drop the first run's two -- the defect a fix-round review
    measured directly and ruled on.
    """
    assert generate.main(
        ["--out", str(tmp_path), "--D", "20", "--seeds", "0-1", "--families", "aligned3"]
    ) == 0
    assert generate.main(
        ["--out", str(tmp_path), "--D", "100", "--seeds", "5", "--families", "decoupled"]
    ) == 0

    npz_on_disk = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.npz"))
    assert npz_on_disk == [
        "aligned3/seed00_D20.npz",
        "aligned3/seed01_D20.npz",
        "decoupled/seed05_D100.npz",
    ]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest_npz = sorted(e["npz"] for e in manifest["entries"])
    assert manifest_npz == npz_on_disk
    assert len(manifest["entries"]) == 3

    # Ruling R36: top-level fields summarize the whole directory, not the last invocation.
    assert manifest["Ds"] == [20, 100]
    assert manifest["seeds"] == [0, 1, 5]


def test_scan_ignores_dotfile_silently_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Ruling R37, corrected from round 1: a directory holding only a valid grid plus a
    `.DS_Store` must exit 0 -- a dotfile is not objective-shaped, so it cannot make the manifest
    incomplete -- and must print *no* warning at all mentioning it. A warning printed on every run
    (`.DS_Store` is close to guaranteed on this OneDrive-synced, Finder-browsed checkout) would be
    its own kind of noise; round 1 got this specific case wrong (measured: exit 1) before this fix.
    """
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0
    capsys.readouterr()

    (tmp_path / ".DS_Store").write_bytes(b"")

    code = generate.main(argv)  # same request; aligned3/seed00 already exists -> skip-eligible
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_scan_warns_but_exits_zero_on_ordinary_non_dotfile_stray(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The distinction the review's correction turns on: `notes.txt` and `.DS_Store` both exit 0
    (neither is objective-shaped, so neither makes the manifest incomplete), but only `notes.txt`
    is warned about -- it isn't invisible, it just isn't fatal.
    """
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0
    capsys.readouterr()

    (tmp_path / "notes.txt").write_text("junk")

    code = generate.main(argv)
    assert code == 0
    captured = capsys.readouterr()
    assert "notes.txt" in captured.err


def test_scan_exits_nonzero_on_half_written_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert generate.main(
        ["--out", str(tmp_path), "--D", str(D), "--seeds", "0-1", "--families", "aligned3"]
    ) == 0
    capsys.readouterr()

    # Half-written pair: a Ctrl-C between save's sequential .npz-then-.json writes leaves exactly
    # this. Seed 0 is not part of the second run's request below, so the scan finds it undisturbed.
    (tmp_path / "aligned3" / "seed00_D20.json").unlink()

    code = generate.main(
        ["--out", str(tmp_path), "--D", str(D), "--seeds", "1", "--families", "aligned3"]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "seed00_D20" in captured.err

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seeds_in_manifest = {e["seed"] for e in manifest["entries"] if e["variant"] == "aligned3"}
    assert seeds_in_manifest == {1}  # seed 0's half-pair excluded, seed 1 present


def test_scan_exits_nonzero_on_corrupt_pair(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert generate.main(
        ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    ) == 0
    capsys.readouterr()

    # Corrupt, not part of the second run's request below, so the scan finds it undisturbed.
    (tmp_path / "aligned3" / "seed00_D20.npz").write_bytes(b"not a zip file")

    code = generate.main(
        ["--out", str(tmp_path), "--D", str(D), "--seeds", "1", "--families", "decoupled"]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "seed00_D20" in captured.err

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    variants_in_manifest = {e["variant"] for e in manifest["entries"]}
    assert variants_in_manifest == {"decoupled"}  # the corrupt aligned3 pair is excluded


def test_scan_exits_nonzero_on_objective_shaped_pair_under_unknown_variant_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A `seed<N>_D<N>` pair under a directory absent from `FAMILIES` is objective-shaped -- an
    objective exists on disk -- but genuinely cannot be described (no `family`/`overrides` to
    look up for an unrecognized name), so it is undescribed like a half-pair, not merely a stray
    directory.
    """
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0
    capsys.readouterr()

    unknown_dir = tmp_path / "not_a_real_variant"
    unknown_dir.mkdir()
    (unknown_dir / "seed00_D20.npz").write_bytes(b"")
    (unknown_dir / "seed00_D20.json").write_text("{}")

    code = generate.main(argv)  # same request; aligned3/seed00 already exists -> skip-eligible
    assert code == 1
    captured = capsys.readouterr()
    assert "not_a_real_variant" in captured.err

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    variants_in_manifest = {e["variant"] for e in manifest["entries"]}
    assert "not_a_real_variant" not in variants_in_manifest


# --- bad arguments -----------------------------------------------------------------------------


def test_unknown_variant_exits_2_and_names_it(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = generate.main(
        ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "not_a_real_variant"]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert "not_a_real_variant" in captured.err
    assert list(tmp_path.iterdir()) == []


def test_missing_required_out_exits_2() -> None:
    assert generate.main(["--D", str(D)]) == 2


def test_help_exits_0_and_documents_hash_portability_caveat(capsys: pytest.CaptureFixture) -> None:
    code = generate.main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "platform" in captured.out.lower()
    assert "windows" in captured.out.lower()


def test_manifest_documents_hash_portability_caveat(tmp_path: Path) -> None:
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "platform" in manifest["hash_note"].lower()
    assert "windows" in manifest["hash_note"].lower()
    assert "scope_note" in manifest


# --- path-building helpers, checked against literals, not against each other -----------------


def test_stem_path_builds_expected_literal_path(tmp_path: Path) -> None:
    stem = generate._stem_path(tmp_path, "interaction_g0.25", seed=3, D=20)
    assert stem == tmp_path / "interaction_g0.25" / "seed03_D20"


def test_npz_and_json_paths_use_fstring_concatenation_not_with_suffix() -> None:
    """Pins `_npz_and_json` in isolation: fed a stem whose *final* path segment itself carries a
    dot (never produced by `_stem_path` today, since the dotted variant name always lands one
    level up in the directory component -- but this helper should not rely on that to stay safe),
    `Path.with_suffix` would silently truncate at that dot and return `out/interaction_g0.npz`
    instead of `out/interaction_g0.25.npz`. `test_npz_and_json_paths_are_literal_and_do_not_collide_across_gamma_variants`
    below, using the directory-shaped stems the CLI actually builds, cannot tell f-string
    concatenation apart from `with_suffix` -- both give the right answer when the final segment has
    no dot -- so this test exists specifically to close that gap.
    """
    npz, js = generate._npz_and_json(Path("out/interaction_g0.25"))
    assert npz == Path("out/interaction_g0.25.npz")
    assert js == Path("out/interaction_g0.25.json")


def test_npz_and_json_paths_are_literal_and_do_not_collide_across_gamma_variants() -> None:
    npz_25, json_25 = generate._npz_and_json(Path("out/interaction_g0.25/seed00_D100"))
    npz_50, json_50 = generate._npz_and_json(Path("out/interaction_g0.50/seed00_D100"))

    assert npz_25 == Path("out/interaction_g0.25/seed00_D100.npz")
    assert json_25 == Path("out/interaction_g0.25/seed00_D100.json")
    assert npz_50 == Path("out/interaction_g0.50/seed00_D100.npz")
    assert json_50 == Path("out/interaction_g0.50/seed00_D100.json")
    assert len({npz_25, json_25, npz_50, json_50}) == 4


# --- --seeds parsing -----------------------------------------------------------------------


def test_seed_list_parses_range_and_comma_list() -> None:
    assert generate._seed_list("0-9") == list(range(10))
    assert generate._seed_list("2,5,7") == [2, 5, 7]
    assert generate._seed_list("3") == [3]


def test_seed_list_rejects_malformed_spec() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        generate._seed_list("abc")
    with pytest.raises(argparse.ArgumentTypeError):
        generate._seed_list("9-3")
