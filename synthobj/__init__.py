"""synthobj: synthetic ground-truth objective functions for Bayesian optimization.

The family registry's public API (`synthobj.families`) and the assembled-objective API
(`synthobj.objective`) are re-exported from the package root, but lazily, via `__getattr__`
(PEP 562) rather than an eager import: `families.py` imports `objective.py` at runtime (ruling
R15), and an eager `from synthobj.families import ...` here would import `families` into
`sys.modules` the instant *any* submodule of this package is imported -- e.g. `from synthobj.kernel
import v` in a test file that has nothing to do with families -- since importing a submodule
always runs the parent package's `__init__.py` first. Accessing `synthobj.build`,
`synthobj.FamilySpec`, `synthobj.SyntheticObjective`, etc. still works; it only defers the import
of `families`/`objective` to the first such access, and `import synthobj` alone (no attribute
access) imports neither -- so it stays torch-free, since neither module imports torch.

`synthobj.load` is `SyntheticObjective.load` itself (a bound classmethod, not a copy), matching the
plan's promised `synthobj.load(stem)` call form without a wrapper function to keep in sync with it.
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

_OBJECTIVE_API = (
    "SyntheticObjective",
    "Labels",
    "load",
)

__all__ = list(_FAMILIES_API) + list(_OBJECTIVE_API)


def __getattr__(name: str) -> Any:
    if name in _FAMILIES_API:
        from synthobj import families

        return getattr(families, name)
    if name in _OBJECTIVE_API:
        from synthobj import objective

        if name == "load":
            return objective.SyntheticObjective.load
        return getattr(objective, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
