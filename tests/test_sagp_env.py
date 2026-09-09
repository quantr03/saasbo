"""Environment tests for the sagp package.

Task 0 makes no behavioral claims about sagp beyond four things: the pins in
requirements-sagp.txt match what's actually installed, jax and torch alike (a), the vendored
saasgp/saasbo/util still import and saasgp.matern_kernel still computes a sane kernel now that
saasgp.py's one jnp.clip call is patched for JAX 0.10 (b), the vendored SAASGP still fits and
predicts end to end on the pinned stack (c), and importing sagp enables float64 process-wide on
both jax and torch (d). saasbo and util are imported only to prove they still do; nothing else
here calls them.

`import sagp` comes first, before anything else in this module creates a JAX array, so its
numpyro.enable_x64()/jax.config.update side effect is in force before test (c) creates one --
regardless of what else pytest has already imported earlier in the session.
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

import saasbo
import saasgp
import util


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


def test_import_sagp_sets_torch_float64_default() -> None:
    assert torch.get_default_dtype() is torch.float64
    assert torch.ones(1).dtype is torch.float64


def test_vendored_reference_imports_and_matern_kernel_diagonal_is_sane() -> None:
    X = jnp.array(np.random.default_rng(0).uniform(size=(6, 3)))
    K = saasgp.matern_kernel(X, X, 1.0, jnp.ones(3), 0.1, True)

    assert K.shape == (6, 6)
    assert np.all(np.isfinite(K))
    # On the diagonal dsq == 0 exactly, which jnp.clip(dsq, 1e-12) floors before the sqrt; that
    # floor propagates through poly * exp(-exponent) as a fixed ~-2.5e-12 correction (measured),
    # so the diagonal matches var + noise + jitter only up to that floor, not to full float64
    # precision. atol=1e-11 covers the floor with margin while still catching a wrong noise, var
    # or jitter constant, any of which would be off by >= 1e-6.
    np.testing.assert_allclose(np.diag(K), 1.0 + 0.1 + 1.0e-6, atol=1.0e-11, rtol=0.0)


def test_import_sagp_enables_x64_and_saasgp_fit_runs_on_pinned_stack() -> None:
    assert jnp.ones(1).dtype == jnp.float64

    rng = np.random.default_rng(1)
    X_train = rng.uniform(size=(10, 3))
    Y_train = rng.standard_normal(10)
    # thinning=1 (not the brief's thinning=4): num_samples // thinning must retain a multiple of
    # 8, because saasgp.py hardcodes chunk_size=8 in compute_choleskys/posterior, and util.py's
    # get_chunks calls np.arange on the remainder without ever importing numpy -- a pre-existing
    # bug in the vendored, byte-identical-required util.py, unrelated to this task's jnp.clip
    # patch. thinning=4 retains 8 // 4 = 2 samples and hits that NameError; the reference's own
    # defaults (num_samples=256, thinning=16 -> 16 retained) never do, which is presumably why
    # this has gone unnoticed. thinning=1 retains all 8 samples, sidestepping the bug without
    # touching util.py or changing num_warmup/num_samples/kernel.
    gp = saasgp.SAASGP(num_warmup=10, num_samples=8, thinning=1, verbose=False, kernel="matern").fit(
        X_train, Y_train
    )

    X_test = rng.uniform(size=(5, 3))
    mean, var = gp.posterior(X_test)
    assert mean.shape == (8, 5)
    assert var.shape == (8, 5)
