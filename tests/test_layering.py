"""Layering tests: the import rules that keep `sagp` a library, `experiments` its study, and
`synthobj` a torch-free description of the objectives.

Rules (a)-(c) run one way through the `sagp`/`experiments` split; rule (d) is the edge that keeps
`synthobj` off the torch stack, so `import synthobj` works on a machine with neither torch nor
botorch installed and only `synthobj/botorch_adapter.py` -- the module nobody imports at package
import -- may name them. Rule (e) keeps a cell's parameterization where the cell declares it, in
its `native_site`: nothing under `sagp/` or `experiments/` compares a cell's prior against a prior
string, so a new cell can share its twin's parameterization under a prior string of its own. All
five are properties of the source, not of any run, so they are checked with `ast` over every file
rather than by importing anything. Each test also feeds its checker a deliberately violating source
string, so a checker that quietly stopped looking would fail here instead of passing vacuously
over a tree that happens to be clean.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SAGP_FILES = sorted((_ROOT / "sagp").glob("*.py"))
_EXPERIMENT_FILES = sorted((_ROOT / "experiments").glob("*.py"))
# Every `synthobj` module but the adapter, which is the one allowed to reach the torch stack.
_SYNTHOBJ_FILES = [
    path for path in sorted((_ROOT / "synthobj").glob("*.py")) if path.name != "botorch_adapter.py"
]
assert _SAGP_FILES and _EXPERIMENT_FILES and _SYNTHOBJ_FILES, (
    "no modules to check: the globs found nothing"
)

# The four distributions rule (d) forbids, by the first dotted component of the imported name.
_TORCH_STACK: frozenset[str] = frozenset({"torch", "botorch", "gpytorch", "linear_operator"})

# D4 written out as a table: what each module under `sagp/` may import from the package. The
# bottom of the library is `r2d2.py`, `diagnostics.py` and `kernels_torch.py`, which import
# nothing from it; `gp.py` stands on those three alone, and the three modules above the core
# reach `gp` and never each other. A module missing from this table is a decision that has
# not been made, so the test says so rather than passing.
_ALLOWED_SAGP_IMPORTS: dict[str, set[str]] = {
    "gp": {"sagp.diagnostics", "sagp.kernels_torch", "sagp.r2d2"},
    "diagnostics": set(),
    "kernels_torch": set(),  # torch only, so `gp` can import it without a cycle
    "r2d2": set(),  # jax and numpyro only: the prior-only harness imports it without BoTorch
    "references": {"sagp.gp"},
    "readouts": {"sagp.gp"},
    "bo": {"sagp.gp"},
    "__init__": {"sagp.gp", "sagp.bo"},  # the lazy imports inside `__getattr__` count
}

# The prior strings rule (e) forbids comparing against: the ones a dispatch written for the four
# half-Cauchy cells would name, and the R2-D2 cells' own, so no dispatch on those starts either.
_PRIOR_STRINGS = frozenset({"amplitude", "lengthscale", "amplitude_r2d2", "lengthscale_r2d2"})
# A literal holding one of these beside a prior string is a cell key, whatever it is compared to.
_STRUCTURE_STRINGS = frozenset({"additive", "product"})
# The comparisons a dispatch is written with.
_DISPATCH_OPS = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)


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


def _sagp_module_imports(source: str) -> set[str]:
    """Every `sagp` submodule a file imports, as `sagp.<module>`; the bare package is not one."""
    modules: set[str] = set()
    for name in _imported_modules(source):
        parts = name.split(".")
        if parts[0] == "sagp" and len(parts) > 1:
            modules.add(f"sagp.{parts[1]}")
    return modules


def _experiments_imports(source: str) -> list[str]:
    """Every `experiments` module a file imports, in any form."""
    return sorted(m for m in _imported_modules(source) if m.split(".")[0] == "experiments")


def _torch_stack_imports(source: str) -> list[str]:
    """Every torch-stack module a file imports, in any form."""
    return sorted(m for m in _imported_modules(source) if m.split(".")[0] in _TORCH_STACK)


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


def _literal_strings(node: ast.AST) -> set[str]:
    """Every string in a constant or a Tuple/List/Set literal, at any depth of nesting."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return set().union(*(_literal_strings(element) for element in node.elts))
    return set()


def _holds_a_prior(node: ast.AST) -> bool:
    """A cell-key subscript `X[1]` or a `.prior` attribute: where a cell's prior string lives."""
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == 1
    return isinstance(node, ast.Attribute) and node.attr == "prior"


def _prior_string_comparisons(source: str) -> list[int]:
    """Lines comparing a cell's prior against a prior string: a dispatch on the string.

    A comparison (==, !=, in, not in) against a literal holding a prior string at any depth counts
    when the other side is `X[1]` or `.prior`, or when the literal also holds a structure string
    (it is then a cell key). A site-name filter such as `site in ("mean", "lengthscale")` is
    neither: "lengthscale" is a site name too.
    """
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = (node.left, *node.comparators)
        pairs = [
            (literal, other)
            for op, left, right in zip(node.ops, operands, operands[1:])
            if isinstance(op, _DISPATCH_OPS)
            for literal, other in ((left, right), (right, left))
        ]
        for literal, other in pairs:
            strings = _literal_strings(literal)
            if strings & _PRIOR_STRINGS and (strings & _STRUCTURE_STRINGS or _holds_a_prior(other)):
                lines.append(node.lineno)
                break
    return lines


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


@pytest.mark.parametrize("path", _SAGP_FILES, ids=lambda p: p.name)
def test_each_library_module_imports_only_what_d4_allows(path: Path):
    """Rule (c): every `sagp` import a module makes is in that module's row of the D4 table."""
    assert path.stem in _ALLOWED_SAGP_IMPORTS, f"{path.name} has no row in the D4 table"
    assert _sagp_module_imports(path.read_text()) <= _ALLOWED_SAGP_IMPORTS[path.stem]
    # The table forbids sideways edges as well as downward ones: `bo` may reach `gp` and nothing
    # else, so a proposer taken from `references` is a violation even though it is not the core.
    sideways = _sagp_module_imports("from sagp.references import propose_sobol")
    assert sideways - _ALLOWED_SAGP_IMPORTS["bo"] == {"sagp.references"}
    # The alias forms the checker has to see through, both of which the core must not contain.
    assert _sagp_module_imports("from sagp import bo") == {"sagp.bo"}
    assert _sagp_module_imports("import sagp.readouts as r") == {"sagp.readouts"}


@pytest.mark.parametrize("path", _SYNTHOBJ_FILES, ids=lambda p: p.name)
def test_the_objectives_never_import_the_torch_stack(path: Path):
    """Rule (d): no module under `synthobj/` but `botorch_adapter.py` names torch or its stack."""
    assert _torch_stack_imports(path.read_text()) == []
    violating = (
        "import torch\n"
        "from botorch.models import SingleTaskGP\n"
        "import gpytorch.settings\n"
        "from linear_operator.utils.errors import NotPSDError\n"
    )
    assert _torch_stack_imports(violating) == [
        "botorch.models", "botorch.models.SingleTaskGP", "gpytorch.settings",
        "linear_operator.utils.errors", "linear_operator.utils.errors.NotPSDError", "torch",
    ]


@pytest.mark.parametrize("path", _SAGP_FILES + _EXPERIMENT_FILES, ids=lambda p: p.name)
def test_nothing_dispatches_on_a_prior_string(path: Path):
    """Rule (e): a cell's parameterization is its declared `native_site`, never its prior string."""
    assert _prior_string_comparisons(path.read_text()) == []
    violating = (
        'a = cell[1] == "amplitude"\n'
        'b = key == ("product", "lengthscale")\n'
        'c = key in (("product", "amplitude"),)\n'
        'd = cell[1] in {"lengthscale"}\n'
        'e = site in ("mean", "lengthscale")\n'  # a site-name filter, not a dispatch
    )
    assert _prior_string_comparisons(violating) == [1, 2, 3, 4]
