"""Layering tests: the import rules that keep `sagp` a library and `experiments` its study.

The split is only worth having if the dependency edges run one way and only through public names,
which is a property of the source, not of any run -- so it is checked with `ast` over every file
rather than by importing anything. Each test also feeds its checker a deliberately violating
source string, so a checker that quietly stopped looking would fail here instead of passing
vacuously over a tree that happens to be clean.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SAGP_FILES = sorted((_ROOT / "sagp").glob("*.py"))
_EXPERIMENT_FILES = sorted((_ROOT / "experiments").glob("*.py"))
assert _SAGP_FILES and _EXPERIMENT_FILES, "no modules to check: the globs found nothing"

# `gp.py` and `diagnostics.py` are the bottom of the library: `bo.py`, `references.py` and the
# readouts module are layered above them and must never be imported back down.
_CORE_FILES = [_ROOT / "sagp" / "gp.py", _ROOT / "sagp" / "diagnostics.py"]
_ABOVE_CORE = ("sagp.bo", "sagp.references", "sagp.readouts")


def _imported_modules(source: str) -> set[str]:
    """Every module a file imports; `from a.b import c` counts as both `a.b` and `a.b.c`."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules |= {f"{node.module}.{alias.name}" for alias in node.names}
    return modules


def _experiments_imports(source: str) -> list[str]:
    """Every `experiments` module a file imports, in any form."""
    return sorted(m for m in _imported_modules(source) if m.split(".")[0] == "experiments")


def _private_sagp_uses(source: str) -> list[str]:
    """Every underscore-prefixed `sagp` name a file imports or reaches through a `sagp` module."""
    tree = ast.parse(source)
    bound: set[str] = set()  # local names for a `sagp` module or one of its attributes
    private: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "sagp":
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0 and node.module.split(".")[0] == "sagp":
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
                    if alias.name.startswith("_"):
                        private.append(f"{node.module}.{alias.name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in bound:
                private.append(f"{root.id}.{node.attr}")
    return sorted(private)


@pytest.mark.parametrize("path", _SAGP_FILES, ids=lambda p: p.name)
def test_the_library_never_imports_the_study(path: Path):
    """Rule (a): no module under `sagp/` imports `experiments`, in any form."""
    assert _experiments_imports(path.read_text()) == []
    violating = "import experiments\nfrom experiments.identify import identify\n"
    assert _experiments_imports(violating) == [
        "experiments", "experiments.identify", "experiments.identify.identify",
    ]


@pytest.mark.parametrize("path", _EXPERIMENT_FILES, ids=lambda p: p.name)
def test_the_study_uses_only_public_library_names(path: Path):
    """Rule (b): every `sagp` name `experiments/` imports or reaches for is public."""
    assert _private_sagp_uses(path.read_text()) == []
    assert _private_sagp_uses("from sagp.gp import _chunk_size") == ["sagp.gp._chunk_size"]
    assert _private_sagp_uses("import sagp.gp as gp\ngp._chunk_size(x)\n") == ["gp._chunk_size"]


@pytest.mark.parametrize("path", _CORE_FILES, ids=lambda p: p.name)
def test_the_core_never_imports_the_modules_above_it(path: Path):
    """Rule (c): the core modules import none of `sagp.bo`, `sagp.references`, `sagp.readouts`."""
    assert _imported_modules(path.read_text()).isdisjoint(_ABOVE_CORE)
    assert not _imported_modules("from sagp import bo").isdisjoint(_ABOVE_CORE)
    assert not _imported_modules("import sagp.readouts").isdisjoint(_ABOVE_CORE)
