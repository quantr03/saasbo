"""sagp: the thesis's four sparse-GP cells on the SAASBO reference code path.

`sagp.gp` implements the four model cells (additive/product structure x amplitude/lengthscale
prior; see the plan), NUTS inference and prediction, following the vendored
`saasgp.py`/`saasbo.py` reference exactly except where the plan calls for a change.
`sagp.readouts` holds the posterior readouts and is reached as the submodule it is, like
`sagp.gp` itself: callers write `from sagp.readouts import readouts`. `sagp.bo` runs the
Bayesian-optimization loop against any cell through `gp.py`'s fit/FittedGP/posterior interface
alone: `bo.py` must not know how a cell is parameterized, and `gp.py` must not know about
acquisition or budgets, so importing one must not pull in the other. `fit` and `FittedGP` are
re-exported here from `sagp.gp` and `run_bo` from `sagp.bo`, lazily via `__getattr__` (PEP 562)
rather than eagerly -- matching `synthobj/__init__.py` -- so `import sagp` imports no submodule.

`numpyro.set_platform`/`set_host_device_count`/`enable_x64` run here, at package import, before
any of this package's code creates a JAX array: every later task depends on float64 and the cpu
platform being set before the first array exists, not after.
"""
from typing import Any

import jax
import numpyro

numpyro.set_platform("cpu")
numpyro.set_host_device_count(1)
numpyro.enable_x64()
# Belt-and-suspenders: this is what numpyro.enable_x64() does internally, but a test process may
# have imported jax -- and so fixed its dtype defaults -- before this package is ever imported.
jax.config.update("jax_enable_x64", True)

_GP_API = ("fit", "FittedGP")
_BO_API = ("run_bo",)

__all__ = list(_GP_API) + list(_BO_API)


def __getattr__(name: str) -> Any:
    if name in _GP_API:
        from sagp import gp

        return getattr(gp, name)
    if name in _BO_API:
        from sagp import bo

        return getattr(bo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
