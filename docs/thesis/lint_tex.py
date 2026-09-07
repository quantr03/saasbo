#!/usr/bin/env python3
"""Structural and notation-drift lint for the synthobj thesis appendix.

Run from docs/thesis/:

    python lint_tex.py

Scans every ``.tex`` file in this directory and in ``tables/`` and checks:

  1. Balanced ``{}`` and matching ``\\begin{X}``/``\\end{X}`` in every file.
  2. Every ``\\input{...}`` target resolves to an existing file (``\\SOroot``
     resolves to ``.``) in every file.
  3. Every ``\\ref{...}``/``\\eqref{...}`` has a matching ``\\label{...}``
     defined somewhere across the whole file set.
  4. Every ``\\label`` outside the wrapper (standalone_main.tex) is prefixed
     ``app:so:``.
  5. No ``\\documentclass``, ``\\begin{document}``, ``\\usepackage`` or
     ``\\cite`` in a fragment or a table.
  6. No raw ``\\mathrm{Var}``, ``\\mathrm{E}``, ``\\mathbb{E}``, ``\\sigma``
     or ``\\lambda`` in a fragment -- the notation-drift guard; these must
     go through the macros in synthobj-notation.sty instead. This also
     catches the subscripted/decorated forms (``\\sigma_i^2``,
     ``\\lambda_{\\max}``): matching stops only at a following letter, not
     at ``_`` or a digit.
  7. Every ``\\SO...`` macro a fragment uses is defined in
     synthobj-notation.sty (``\\newcommand``/``\\renewcommand``/
     ``\\DeclareMathOperator``) or as ``\\SOnum...`` in tables/numbers.tex.
     Deliberately scoped to the SO namespace only (fix-round 2, ruling
     R14): standard LaTeX/amsmath commands (``\\frac``, ``\\toprule``,
     ``\\le``, ...) and this file's own non-SO notation macros (``\\Enu``,
     ``\\fstar``, ...) are NOT checked here, because no finite allowlist of
     "legitimate standard LaTeX" can avoid false positives once fragments
     have real content -- fix-round 1 tried exactly that (KNOWN_STANDARD_
     MACROS) and it would have broken on the first booktabs table or the
     first derivation. Any undefined command, SO-prefixed or not, is still
     caught: by the compile. ``make pdf`` runs latexmk with
     ``-halt-on-error``, which fails on "Undefined control sequence".

"Fragment" means appendix-synthobj.tex and A1-construction.tex through
A4-design-decisions.tex. "Table" means a file under tables/. "The wrapper"
means standalone_main.tex, which is exempt from checks 4-7 since it is
allowed (and needs) \\documentclass, \\usepackage, etc.

Exits 1 and prints one "path:line: message" line per failure; exits 0 and
prints a one-line summary if everything passes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER_NAME = "standalone_main.tex"

# check_macros_defined only tests names starting with SO (fix-round 2,
# ruling R14): that is the one namespace this project fully controls, so it
# is the only one a closed allowlist can check without false positives.
# Everything else -- standard LaTeX/amsmath commands and this file's own
# non-SO notation macros alike -- is left to the compile (make pdf,
# latexmk -halt-on-error), which fails on any genuinely undefined command.
SO_MACRO_RE = re.compile(r"\\(SO[A-Za-z]*)\b")
DEFINE_RE = re.compile(
    r"\\(?:newcommand|renewcommand|DeclareMathOperator)\*?\{?\\([A-Za-z]+)\}?"
)
BEGIN_RE = re.compile(r"\\begin\{([^}]*)\}")
END_RE = re.compile(r"\\end\{([^}]*)\}")
INPUT_RE = re.compile(r"\\input\{([^}]*)\}")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(eqref|ref)\{([^}]*)\}")
BANNED_HOST_RE = re.compile(r"\\(documentclass|begin\{document\}|usepackage|cite)\b")
# No trailing \b on \sigma/\lambda: Python's \b treats "_" as a word
# character, so it would let "\sigma_i^2" and "\lambda_{\max}" -- the
# standard way either symbol is decorated -- through uncaught. A negative
# lookahead for a following letter still blocks matching as a prefix of an
# unrelated longer command name, while correctly catching "_", a digit, "^"
# or "{" immediately after.
RAW_NOTATION_RE = re.compile(
    r"\\mathrm\{Var\}|\\mathrm\{E\}|\\mathbb\{E\}|\\sigma(?![A-Za-z])|\\lambda(?![A-Za-z])"
)

failures: list[str] = []


def fail(path: Path, lineno: int, message: str) -> None:
    failures.append(f"{path.relative_to(HERE)}:{lineno}: {message}")


def strip_comments(line: str) -> str:
    """Drop a trailing %-comment, treating \\% as a literal percent."""
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            out.append(line[i : i + 2])
            i += 2
            continue
        if ch == "%":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def category(path: Path) -> str:
    if path.name == WRAPPER_NAME:
        return "wrapper"
    if path.parent.name == "tables":
        return "table"
    return "fragment"


def collect_tex_files() -> list[Path]:
    files = sorted(HERE.glob("*.tex"))
    tables_dir = HERE / "tables"
    if tables_dir.is_dir():
        files += sorted(tables_dir.glob("*.tex"))
    return files


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def check_braces_and_envs(path: Path, lines: list[str]) -> None:
    depth = 0
    stack: list[tuple[str, int]] = []
    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        depth += line.count("{") - line.count("}")
        for m in BEGIN_RE.finditer(line):
            stack.append((m.group(1), lineno))
        for m in END_RE.finditer(line):
            name = m.group(1)
            if not stack:
                fail(path, lineno, f"\\end{{{name}}} has no matching \\begin")
                continue
            open_name, open_lineno = stack.pop()
            if open_name != name:
                fail(
                    path,
                    lineno,
                    f"\\end{{{name}}} does not match \\begin{{{open_name}}} "
                    f"opened at line {open_lineno}",
                )
    if depth != 0:
        fail(path, len(lines) or 1, f"unbalanced braces (net {depth:+d})")
    for name, lineno in stack:
        fail(path, lineno, f"\\begin{{{name}}} is never closed")


def check_inputs(path: Path, lines: list[str]) -> None:
    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        for m in INPUT_RE.finditer(line):
            target = m.group(1)
            resolved = target.replace(r"\SOroot", ".")
            if not resolved.endswith(".tex"):
                resolved += ".tex"
            candidate = (HERE / resolved).resolve()
            if not candidate.is_file():
                fail(
                    path,
                    lineno,
                    f"\\input target does not exist: {target} "
                    f"(resolved {candidate})",
                )


def check_banned_host_commands(path: Path, lines: list[str]) -> None:
    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        m = BANNED_HOST_RE.search(line)
        if m:
            fail(path, lineno, f"host-only command {m.group(0)} in a fragment/table")


def check_raw_notation(path: Path, lines: list[str]) -> None:
    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        m = RAW_NOTATION_RE.search(line)
        if m:
            fail(
                path,
                lineno,
                f"raw notation {m.group(0)!r} in a fragment; use a macro from "
                f"synthobj-notation.sty instead",
            )


def check_label_prefix(path: Path, lines: list[str]) -> None:
    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        for m in LABEL_RE.finditer(line):
            name = m.group(1)
            if not name.startswith("app:so:"):
                fail(path, lineno, f"\\label{{{name}}} is not prefixed app:so:")


def collect_macro_defs(*paths: Path) -> set[str]:
    defined: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for raw in read_lines(path):
            line = strip_comments(raw)
            for m in DEFINE_RE.finditer(line):
                defined.add(m.group(1))
    return defined


def check_macros_defined(path: Path, lines: list[str], defined: set[str]) -> None:
    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        for m in SO_MACRO_RE.finditer(line):
            name = m.group(1)
            if name not in defined:
                fail(
                    path,
                    lineno,
                    f"\\{name} is used but not defined in synthobj-notation.sty "
                    f"or tables/numbers.tex",
                )


def main() -> int:
    files = collect_tex_files()
    file_lines = {path: read_lines(path) for path in files}

    for path, lines in file_lines.items():
        check_braces_and_envs(path, lines)
        check_inputs(path, lines)

    labels: dict[str, tuple[Path, int]] = {}
    refs: list[tuple[str, Path, int]] = []
    for path, lines in file_lines.items():
        for lineno, raw in enumerate(lines, start=1):
            line = strip_comments(raw)
            for m in LABEL_RE.finditer(line):
                labels.setdefault(m.group(1), (path, lineno))
            for m in REF_RE.finditer(line):
                refs.append((m.group(2), path, lineno))
    for name, path, lineno in refs:
        if name not in labels:
            fail(path, lineno, f"\\ref/\\eqref to undefined label {{{name}}}")

    for path, lines in file_lines.items():
        if category(path) == "wrapper":
            continue
        check_label_prefix(path, lines)

    for path, lines in file_lines.items():
        if category(path) in ("fragment", "table"):
            check_banned_host_commands(path, lines)

    for path, lines in file_lines.items():
        if category(path) == "fragment":
            check_raw_notation(path, lines)

    defined_macros = collect_macro_defs(
        HERE / "synthobj-notation.sty", HERE / "tables" / "numbers.tex"
    )
    for path, lines in file_lines.items():
        if category(path) == "fragment":
            check_macros_defined(path, lines, defined_macros)

    if failures:
        for line in failures:
            print(line)
        print(f"lint_tex: {len(failures)} failure(s) across {len(files)} file(s)")
        return 1

    print(f"lint_tex: ok ({len(files)} file(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
