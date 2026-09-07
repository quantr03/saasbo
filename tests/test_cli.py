"""Tests for synthobj.generate: the study-grid CLI.

Per the branch's own caution about assertions with no power against the mutant they appear to
guard, a manifest is a description of files the CLI *just wrote*, so `assert entry["f_star"] ==
obj.f_star` against the very object the CLI built in-process mostly proves a float survived a
JSON round trip. The load-bearing checks here instead cross a real boundary:

- `test_writes_stems_matching_disk_hashes_and_independent_rebuild` hashes the files *on disk* with
  a fresh `hashlib.sha256` call (not `synthobj.generate._sha256`) and compares to the manifest's
  claim, and separately rebuilds each (variant, seed) from scratch with `make_family` -- an
  independent evaluator, not a readback of stored state -- and compares `f_star`/`gamma`/`S` to
  the manifest entry.
- `test_rerun_without_overwrite_skips_and_does_not_rebuild` monkeypatches `generate.make_family`
  to raise, then reruns without `--overwrite`: if skip logic ever degraded into "rebuild every
  time", this fails loudly instead of merely reading back files that happen to already exist.
- `test_two_gamma_variants_do_not_collide_on_disk` is the concrete regression the brief's path
  hazard section warns about (Task 8's `Path.with_suffix` bug collapsed every `interaction_g0.*`
  variant onto one file): two gamma variants at the same seed must produce four distinct files
  with distinct hashes, not silently merge.
- `test_stem_path_builds_expected_literal_path` and
  `test_npz_and_json_paths_are_literal_and_do_not_collide_across_gamma_variants` compare
  `generate._stem_path`/`_npz_and_json`'s output against hand-written literal `Path(...)` strings,
  not against a second call to the same helpers -- so a regression in the helper itself (e.g.
  reintroducing `Path.with_suffix`, or flattening `<out>/<variant>/seed...` into
  `<out>/<variant>_seed...`, which *would* put the dotted variant name back into the final path
  segment) has something independent to disagree with.

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

PYTHON = "/opt/anaconda3/envs/saasbo/bin/python"
D = 20


# --- the main write path -------------------------------------------------------------------


def test_writes_stems_matching_disk_hashes_and_independent_rebuild(tmp_path: Path) -> None:
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
    assert manifest["D"] == D
    assert manifest["seeds"] == [0, 1]
    assert len(manifest["entries"]) == 4

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
        assert entry["family"] in ("aligned", "interaction")
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
    assert list(tmp_path.iterdir()) == []


# --- rerun / skip / overwrite ----------------------------------------------------------------


def test_rerun_without_overwrite_skips_and_does_not_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "--out", str(tmp_path),
        "--D", str(D),
        "--seeds", "0-1",
        "--families", "aligned3,interaction_g0.25",
    ]
    assert generate.main(argv) == 0

    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "manifest.json"]
    assert len(written) == 8
    before = {p: p.read_bytes() for p in written}

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a skipped stem must not be rebuilt")

    monkeypatch.setattr(generate, "make_family", _boom)

    assert generate.main(argv) == 0  # no --overwrite; must not touch _boom at all

    after = {p: p.read_bytes() for p in written}
    assert after == before


def test_overwrite_produces_byte_identical_files(tmp_path: Path) -> None:
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


def test_manifest_documents_hash_portability_caveat(tmp_path: Path) -> None:
    argv = ["--out", str(tmp_path), "--D", str(D), "--seeds", "0", "--families", "aligned3"]
    assert generate.main(argv) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "platform" in manifest["hash_note"].lower()


# --- path-building helpers, checked against literals, not against each other -----------------


def test_stem_path_builds_expected_literal_path(tmp_path: Path) -> None:
    stem = generate._stem_path(tmp_path, "interaction_g0.25", seed=3, D=20)
    assert stem == tmp_path / "interaction_g0.25" / "seed03_D20"


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
