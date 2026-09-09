"""Environment tests for the sagp package.

Task 0 makes no behavioral claims about sagp beyond two things: the pins in
requirements-sagp.txt match what's actually installed, jax and torch alike (a), and importing
sagp enables float64 process-wide on both jax and torch (b).

`import sagp` comes first, before anything else in this module creates a JAX array, so its
numpyro.enable_x64()/jax.config.update side effect is in force before this module's own
`jnp.ones(1)` call -- regardless of what else pytest has already imported earlier in the session.
"""
from pathlib import Path

import sagp

import botorch
import gpytorch
import jax
import jax.numpy as jnp
import jaxlib
import linear_operator
import numpy as np
import numpyro
import scipy
import torch


def _pinned_versions(repo_root: Path) -> dict[str, str]:
    """Parse the name==version pins out of requirements-sagp.txt."""
    pins = {}
    for line in (repo_root / "requirements-sagp.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, version = line.partition("==")
        pins[name] = version
    return pins


def test_pinned_versions_match_installed_stack(repo_root: Path) -> None:
    pins = _pinned_versions(repo_root)
    assert jax.__version__ == pins["jax"]
    assert jaxlib.__version__ == pins["jaxlib"]
    assert numpyro.__version__ == pins["numpyro"]
    assert scipy.__version__ == pins["scipy"]
    assert np.__version__ == pins["numpy"]
    assert torch.__version__ == pins["torch"]
    assert gpytorch.__version__ == pins["gpytorch"]
    assert linear_operator.__version__ == pins["linear_operator"]
    assert botorch.__version__ == pins["botorch"]


def test_import_sagp_sets_float64_in_torch_and_jax() -> None:
    assert torch.get_default_dtype() is torch.float64
    assert torch.ones(1).dtype is torch.float64
    assert jnp.ones(1).dtype == jnp.float64
