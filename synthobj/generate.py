"""The study-grid generator CLI: materializes `STUDY_GRID` x seeds to `<out>/<variant>/seed{seed:02d}_D{D}.{npz,json}`
plus a `manifest.json` describing every entry.

`main(argv)` is the whole implementation; `if __name__ == "__main__"` just forwards `sys.argv` to
it and exits with its return code, so a test can call `main([...])` directly (fast, in-process,
shares one `synthobj.kernel.eigen_factor` cache across every variant/seed) or drive the same code
through `python -m synthobj.generate` via `subprocess` (slower, but the only way to prove the
module entry point itself works). Both `--D`, `--seeds` and `--families` errors and `--overwrite`
skip logic live entirely in `main`; there is no separate orchestration layer, since one process
building at most 140 objectives has no reason for one.

Path hazard (Task 8's own, ruling carried forward): a study variant name can carry a decimal point
(`interaction_g0.25`), and `Path(name).with_suffix(".npz")` silently rewrites that into
`interaction_g0.npz`, colliding every gamma variant onto one file. The dotted name here only ever
lands in a *directory* component (`<out>/<variant>/...`), never in the file stem itself
(`seed00_D100` has no dots), so the collision cannot literally recur -- but `_stem_path` and
`_npz_and_json` below are still written by plain `/` and f-string concatenation only, matching
`SyntheticObjective.save`/`.load` exactly, rather than trusting that invariant to hold forever.

Ruling R35 -- the manifest describes `<out>`, not this invocation: `main` writes/skips exactly the
`(variant, seed)` pairs this run was asked to produce, but `_scan_directory` then rebuilds the
*whole* `entries` list by scanning `<out>` itself, after generation finishes. Two runs into the
same `--out` with different `--D`/`--seeds`/`--families` leave files from both; a manifest built
only from "what this run touched" would describe a run, not a directory, and silently drop
everything from an earlier invocation. Measured cost: `SyntheticObjective.load` is ~5 ms cold and
~0.6 ms warm at D=100, so scanning a full 140-file grid costs about the same as building a single
objective -- negligible next to actually building 140. The scan tolerates, warns about, and counts
(without crashing) four kinds of debris a script-driven directory can accumulate: a stray file
directly under `<out>` (`.DS_Store` is close to guaranteed on this checkout), a subdirectory whose
name is not a `STUDY_GRID` variant, a file inside a variant directory that isn't a well-formed
`seed<N>_D<N>.{npz,json}` member, and a "half-pair" stem with only one of the two extensions
present -- exactly the state a `Ctrl-C` between `save`'s sequential `.npz`-then-`.json` writes
leaves. A *complete* pair that exists but fails to `SyntheticObjective.load` (corrupt zip, foreign
JSON, unsupported version) is caught the same way. None of this is wrapped around the generation
loop's own `load` call on a stem this run explicitly asked to skip-and-reuse: that failure must
reach the user, with `--overwrite` as the stated repair, not vanish into a manifest that still
claims success. `main` returns 1 (not 2, which is reserved for a bad argument) when the scan found
anything it had to exclude.

Ruling R36 -- the top-level fields describe the directory too: `Ds` (plural, deliberately not the
brief's scalar `"D"`) and `seeds` are the sorted union of every entry actually on disk, since a
directory populated by two differently-scoped runs can hold more than one `D` or seed set and a
scalar cannot honestly describe that. Per-entry `D`/`seed` are authoritative; see `SCOPE_NOTE`.
Entries are sorted by `(STUDY_GRID index, D, seed)` rather than left in scan order (`Path.iterdir`
order is unspecified and, for the directory listing this scan does, alphabetical by variant name,
not table order) -- so the manifest orders itself only once, on write, and two manifests describing
the same directory contents compare equal regardless of what order the scan happened to visit
things in.

Hash portability caveat (Task 8's other ruling, ruling R35's review sharpened it further):
`sha256_npz`/`sha256_json` in the manifest are reproducible only on the platform that generated
them, for two independent reasons -- see `HASH_NOTE`, stated in `--help`'s epilog and in the
manifest's own `hash_note` field, not merely implied to be portable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthobj.families import STUDY_GRID, make_family
from synthobj.objective import SyntheticObjective

_VALID_VARIANTS: list[str] = [variant for variant, _, _ in STUDY_GRID]
_FAMILY_OF: dict[str, str] = {variant: family for variant, family, _ in STUDY_GRID}
_OVERRIDES_OF: dict[str, dict] = {variant: overrides for variant, _, overrides in STUDY_GRID}
_VARIANT_INDEX: dict[str, int] = {variant: i for i, variant in enumerate(_VALID_VARIANTS)}

# Deliberately permissive on digit count (not anchored to the writer's own `seed{seed:02d}`
# zero-padding): this pattern is used to *recognize* a stem while scanning a possibly-messy
# directory, not to construct one, so it accepts anything `_stem_path` could have produced and
# more, rather than rejecting a hand-placed file just because it isn't zero-padded.
_STEM_RE = re.compile(r"seed(\d+)_D(\d+)")

HASH_NOTE = (
    "sha256_npz/sha256_json are reproducible across regenerations on the platform that produced "
    "them, but are NOT a cross-platform checksum, for two independent reasons: "
    "zipfile.ZipInfo.create_system is 3 on POSIX and 0 on win32 and is not normalized by the "
    ".npz writer, and Path.write_text's default text-mode newline translation writes \\r\\n line "
    "endings on Windows where POSIX writes \\n, changing sha256_json too. A grid generated on "
    "one OS and hash-checked on another will mismatch on every entry with no corruption present."
)

SCOPE_NOTE = (
    "'Ds' and 'seeds' are the sorted union of every stem currently found under --out (ruling "
    "R35/R36) -- the manifest describes the output directory, not just what this invocation was "
    "asked to produce -- so per-entry 'D'/'seed' are the authoritative values, not these "
    "top-level summaries. 'created' is when this manifest.json was written, not when every file "
    "it describes was generated: a partial rerun re-stamps 'created' while leaving older, "
    "untouched files exactly as they were."
)


def _seed_list(spec: str) -> list[int]:
    """Parse `--seeds`: an inclusive `a-b` range, or a comma-separated list of ints -- the two
    forms `--seeds` accepts are mutually exclusive, not combinable into `a-b,c`.
    """
    try:
        if "-" in spec:
            lo_s, hi_s = spec.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"empty range {spec!r} (hi < lo)")
            return list(range(lo, hi + 1))
        return [int(s) for s in spec.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --seeds value {spec!r}: {exc}") from exc


def _resolve_variants(spec: str) -> list[str]:
    """`"all"` -> every `STUDY_GRID` variant in table order; otherwise a comma-separated list of
    variant names (not family names), each checked against `STUDY_GRID`. Raises `ValueError`
    naming every unknown entry and listing the valid variants, for `main` to report and turn into
    exit code 2.
    """
    if spec == "all":
        return list(_VALID_VARIANTS)
    requested = [name.strip() for name in spec.split(",")]
    unknown = [name for name in requested if name not in _FAMILY_OF]
    if unknown:
        raise ValueError(
            f"unknown variant(s): {', '.join(unknown)}; valid variants are: "
            f"{', '.join(_VALID_VARIANTS)}"
        )
    return requested


def _stem_path(out: Path, variant: str, seed: int, D: int) -> Path:
    """`<out>/<variant>/seed{seed:02d}_D{D}`, the extensionless stem `SyntheticObjective.save`/
    `.load` take. Built with `/` and an f-string only -- never `Path.with_suffix`, never a
    `Path.stem`/`.suffix` round trip -- per the path hazard in the module docstring.
    """
    return out / variant / f"seed{seed:02d}_D{D}"


def _npz_and_json(stem: Path) -> tuple[Path, Path]:
    """`(<stem>.npz, <stem>.json)` by f-string concatenation, matching `SyntheticObjective.save`'s
    own path construction exactly, so a stem's on-disk names are predictable before `save` runs.
    """
    return Path(f"{stem}.npz"), Path(f"{stem}.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_entry(
    variant: str, seed: int, D: int, out: Path, stem: Path, obj: SyntheticObjective
) -> dict[str, Any]:
    """One `entries[]` row for the stem at `stem`, hashing the files as they actually sit on disk
    (not re-serializing `obj`) so a corrupted or truncated write is reflected in the manifest's
    own claim about it. `variant`/`seed`/`D` are passed in rather than re-derived from `obj` or
    from `_stem_path`, since `_scan_directory` (ruling R35) already knows all three, and `stem`
    itself, from the directory listing it just walked.
    """
    npz_path, json_path = _npz_and_json(stem)
    return {
        "variant": variant,
        "family": _FAMILY_OF[variant],
        "overrides": dict(_OVERRIDES_OF[variant]),
        "seed": seed,
        "D": D,
        "npz": npz_path.relative_to(out).as_posix(),
        "json": json_path.relative_to(out).as_posix(),
        "sha256_npz": _sha256(npz_path),
        "sha256_json": _sha256(json_path),
        "f_star": obj.f_star,
        "gamma": obj.labels.gamma,
        "n_active": len(obj.labels.S),
    }


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, int, int]:
    """`(STUDY_GRID index, D, seed)` -- ruling R36's canonical order, so two manifests describing
    the same directory contents compare equal regardless of `Path.iterdir`'s unspecified order.
    """
    return (_VARIANT_INDEX[entry["variant"]], entry["D"], entry["seed"])


def _scan_directory(out: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Build manifest entries by scanning `<out>` itself (ruling R35), not from the `(variant,
    seed)` pairs this invocation was asked to produce -- so a stem left over from an earlier,
    differently-scoped run is described too, and every stem (freshly written or merely left in
    place) is read back from disk rather than assumed unchanged.

    Returns `(entries, issues, load_failures)`.

    `issues` counts, and warns to stderr about, every item under `<out>` this scan could not make
    sense of -- none of these raise; the item is simply excluded from `entries` and counted:

    - a stray file directly under `<out>` (other than `manifest.json` itself);
    - a directory whose name is not a known `STUDY_GRID` variant (this is also what would
      otherwise `KeyError` inside `_manifest_entry`'s `_FAMILY_OF[variant]` lookup);
    - a non-file entry, or a file not matching `seed<N>_D<N>.{npz,json}`, inside an otherwise
      valid variant directory;
    - a "half-pair" stem with only one of `.npz`/`.json` present -- the exact state a `Ctrl-C`
      between `SyntheticObjective.save`'s sequential `.npz`-then-`.json` writes leaves, so this is
      not a hypothetical to guard against.

    `load_failures` counts, and warns about, a stem whose `.npz`/`.json` pair is complete but that
    `SyntheticObjective.load` could not parse. Caught with a bare `except Exception`: measured
    failure modes span `FileNotFoundError`, `zipfile.BadZipFile`, two distinct `ValueError`s
    (numpy's pickle refusal; an unsupported `version`), and `KeyError` (foreign JSON missing
    `version`) -- a narrower `except` tuple would not cover all of them, and none of them should
    abort the whole scan. This is deliberately different from the generation loop's own `load`
    call in `main`, on a stem this run explicitly asked to skip-and-reuse: that failure is left
    unwrapped, since a corrupt file the user asked to reuse must be seen by the user, with
    `--overwrite` as the repair, not silently dropped from a manifest that still claims success.
    """
    entries: list[dict[str, Any]] = []
    issues = 0
    load_failures = 0

    for child in sorted(out.iterdir(), key=lambda p: p.name):
        if child.name == "manifest.json":
            continue
        if not child.is_dir():
            print(f"warning: ignoring stray file directly under --out: {child}", file=sys.stderr)
            issues += 1
            continue

        variant = child.name
        if variant not in _FAMILY_OF:
            print(f"warning: ignoring directory that is not a known variant: {child}", file=sys.stderr)
            issues += 1
            continue

        seen: dict[str, set[str]] = {}
        for f in sorted(child.iterdir(), key=lambda p: p.name):
            if not f.is_file():
                print(f"warning: ignoring unexpected non-file entry: {f}", file=sys.stderr)
                issues += 1
                continue
            if f.name.endswith(".npz"):
                base, ext = f.name[: -len(".npz")], "npz"
            elif f.name.endswith(".json"):
                base, ext = f.name[: -len(".json")], "json"
            else:
                print(f"warning: ignoring unrecognized file: {f}", file=sys.stderr)
                issues += 1
                continue
            if not _STEM_RE.fullmatch(base):
                print(f"warning: ignoring unrecognized file: {f}", file=sys.stderr)
                issues += 1
                continue
            seen.setdefault(base, set()).add(ext)

        for base in sorted(seen):
            exts = seen[base]
            stem = child / base
            if exts != {"npz", "json"}:
                print(
                    f"warning: ignoring half-written stem (found {sorted(exts)}, not both): {stem}",
                    file=sys.stderr,
                )
                issues += 1
                continue
            match = _STEM_RE.fullmatch(base)
            assert match is not None  # guaranteed: `base` only reaches here via the check above
            seed, D = int(match.group(1)), int(match.group(2))
            try:
                obj = SyntheticObjective.load(stem)
            except Exception as exc:
                print(f"warning: failed to load {stem}: {exc!r}", file=sys.stderr)
                load_failures += 1
                continue
            entries.append(_manifest_entry(variant, seed, D, out, stem, obj))

    return entries, issues, load_failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synthobj.generate",
        description=(
            "Materialize the synthobj study grid to <out>/<variant>/seed{seed:02d}_D{D}.{npz,json} "
            "plus <out>/manifest.json."
        ),
        epilog=f"{HASH_NOTE}\n\n{SCOPE_NOTE}",
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--D", type=int, default=100, help="ambient dimension (default 100)")
    parser.add_argument(
        "--seeds",
        type=_seed_list,
        default=_seed_list("0-9"),
        help="inclusive 'a-b' range or comma list of seeds (default 0-9)",
    )
    parser.add_argument(
        "--families",
        default="all",
        help="'all' (default) or a comma list of STUDY_GRID variant names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the (variant, seed) rows that would be generated; write nothing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate stems that already exist (default: skip them)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (or `sys.argv` when `None`) and run the generator.

    Returns 0 on success, 2 on a bad argument, 1 if generation succeeded but the post-generation
    scan of `<out>` (ruling R35) found anything it had to exclude from the manifest (debris or a
    stem that failed to load) -- distinct from 2, which is reserved for an argument `main` never
    got far enough to act on. An argparse-level error (e.g. a malformed `--seeds`) reaches exit
    code 2 via `SystemExit`, caught here and turned into a plain return so `main` never raises out
    from under a caller that just wants an int; an unknown `--families` entry is checked
    afterward and reported the same way, since which variants are valid depends on
    `synthobj.families`, not on anything argparse's own `type=`/`choices` machinery can express
    with a useful message.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)

    try:
        variants = _resolve_variants(args.families)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out: Path = args.out
    D: int = args.D
    seeds: list[int] = args.seeds

    if args.dry_run:
        for variant in variants:
            for seed in seeds:
                print(f"{variant} seed={seed:02d} D={D}")
        return 0

    written = 0
    skipped = 0
    for variant in variants:
        for seed in seeds:
            stem = _stem_path(out, variant, seed, D)
            npz_path, json_path = _npz_and_json(stem)
            if npz_path.exists() and json_path.exists() and not args.overwrite:
                # An explicitly requested stem that already exists must still load cleanly, or
                # the failure must reach the user rather than be swallowed -- `--overwrite` is the
                # stated repair. Deliberately unwrapped, unlike `_scan_directory`'s `load` calls
                # below; see that function's docstring for why the two are handled differently.
                SyntheticObjective.load(stem)
                skipped += 1
            else:
                stem.parent.mkdir(parents=True, exist_ok=True)
                obj = make_family(variant, seed=seed, D=D)
                obj.save(stem)
                written += 1

    # The manifest describes <out> itself, not just this run's requests (ruling R35): scan after
    # generation, so a regenerated (--overwrite) file is hashed post-write and a stem left by an
    # earlier, differently-scoped invocation is included too.
    out.mkdir(parents=True, exist_ok=True)
    entries, issues, load_failures = _scan_directory(out)
    entries.sort(key=_entry_sort_key)

    manifest: dict[str, Any] = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "Ds": sorted({e["D"] for e in entries}),
        "seeds": sorted({e["seed"] for e in entries}),
        "hash_note": HASH_NOTE,
        "scope_note": SCOPE_NOTE,
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))

    print(f"wrote {written} stem(s), skipped {skipped} stem(s)")
    if issues:
        print(f"warning: {issues} item(s) under --out were ignored (see warnings above)", file=sys.stderr)
    if load_failures:
        print(
            f"error: {load_failures} stem(s) under --out failed to load (see warnings above); "
            "rerun with --overwrite to repair",
            file=sys.stderr,
        )
    return 1 if (issues or load_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
