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

Hash portability caveat (Task 8's other ruling): `sha256_npz`/`sha256_json` in the manifest are
reproducible only on the platform that generated them. `zipfile.ZipInfo.create_system` is 3 on
POSIX and 0 on win32, and neither `SyntheticObjective`'s hand-rolled `.npz` writer nor `np.savez`
normalizes it, so a grid generated on macOS and hash-checked on Windows mismatches on every entry
with no corruption present. This is stated in `--help`'s epilog and in the manifest's own
`hash_note` field -- not merely implied to be portable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthobj.families import STUDY_GRID, make_family
from synthobj.objective import SyntheticObjective

_VALID_VARIANTS: list[str] = [variant for variant, _, _ in STUDY_GRID]
_FAMILY_OF: dict[str, str] = {variant: family for variant, family, _ in STUDY_GRID}
_OVERRIDES_OF: dict[str, dict] = {variant: overrides for variant, _, overrides in STUDY_GRID}

HASH_NOTE = (
    "sha256_npz/sha256_json are reproducible across regenerations on the platform that produced "
    "them (measured zipfile.ZipInfo.create_system == 3 on POSIX, 0 on win32; not normalized by "
    "the .npz writer), but are NOT a cross-platform checksum -- a grid generated on one OS and "
    "hash-checked on another will mismatch on every entry with no corruption present."
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
    variant: str, seed: int, D: int, out: Path, obj: SyntheticObjective
) -> dict[str, Any]:
    """One `entries[]` row, hashing the files as they actually sit on disk (not re-serializing
    `obj`) so a corrupted or truncated write is reflected in the manifest's own claim about it.
    """
    stem = _stem_path(out, variant, seed, D)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synthobj.generate",
        description=(
            "Materialize the synthobj study grid to <out>/<variant>/seed{seed:02d}_D{D}.{npz,json} "
            "plus <out>/manifest.json."
        ),
        epilog=HASH_NOTE,
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
    """Parse `argv` (or `sys.argv` when `None`) and run the generator. Returns 0 on success, 2 on
    a bad argument -- an argparse-level error (e.g. a malformed `--seeds`) reaches that exit code
    via `SystemExit`, caught here and turned into a plain return so `main` never raises out from
    under a caller that just wants an int; an unknown `--families` entry is checked afterward and
    reported the same way, since which variants are valid depends on `synthobj.families`, not on
    anything argparse's own `type=`/`choices` machinery can express with a useful message.
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

    entries: list[dict[str, Any]] = []
    written = 0
    skipped = 0
    for variant in variants:
        for seed in seeds:
            stem = _stem_path(out, variant, seed, D)
            npz_path, json_path = _npz_and_json(stem)
            if npz_path.exists() and json_path.exists() and not args.overwrite:
                # Skip: reload rather than rebuild, so a skip never re-draws and never pays the
                # 0.02-0.60 s cold build cost it exists to avoid.
                obj = SyntheticObjective.load(stem)
                skipped += 1
            else:
                stem.parent.mkdir(parents=True, exist_ok=True)
                obj = make_family(variant, seed=seed, D=D)
                obj.save(stem)
                written += 1
            entries.append(_manifest_entry(variant, seed, D, out, obj))

    manifest: dict[str, Any] = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "D": D,
        "seeds": seeds,
        "hash_note": HASH_NOTE,
        "entries": entries,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))

    print(f"wrote {written} stem(s), skipped {skipped} stem(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
