"""synthobj: synthetic ground-truth objective functions for Bayesian optimization.

The family registry's public API (`synthobj.families`) is re-exported from the package root, but
lazily, via `__getattr__` (PEP 562) rather than an eager import: `families.py` imports
`objective.py` at runtime (ruling R15), and an eager `from synthobj.families import ...` here
would import `families` into `sys.modules` the instant *any* submodule of this package is
imported -- e.g. `from synthobj.kernel import v` in a test file that has nothing to do with
families -- since importing a submodule always runs the parent package's `__init__.py` first.
Accessing `synthobj.build`, `synthobj.FamilySpec`, etc. still works; it only defers the import of
`families` to the first such access.
"""
from typing import Any

_FAMILIES_API = (
    "FAMILIES",
    "STUDY_GRID",
    "FamilySpec",
    "Streams",
    "build",
    "make_family",
    "noise_rng",
    "select_active",
    "select_pairs",
    "streams",
)

__all__ = list(_FAMILIES_API)


def __getattr__(name: str) -> Any:
    if name in _FAMILIES_API:
        from synthobj import families

        return getattr(families, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
