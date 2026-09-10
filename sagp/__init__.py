"""sagp: the thesis's four sparse-GP cells on BoTorch's fully Bayesian path.

`sagp.gp` implements the four model cells (additive/product structure x amplitude/lengthscale
prior; see the plan) as BoTorch `PyroModel`s, fitted by BoTorch's own NUTS scheme and predicted
through the batched GPyTorch model it loads the retained draws into.
`sagp.diagnostics` is the convergence verdict `sagp.gp.fit` calls once per attempt. `sagp.bo`
runs the Bayesian-optimization loop against any cell through `FittedGP`/`posterior` alone, never
learning how a cell is parameterized; `sagp.references` and `sagp.readouts` sit beside it, the
study's two MAP references/Sobol proposer and its posterior readouts respectively. `sagp.gp` and
`sagp.diagnostics` are the bottom of this stack and import none of the other three. `sagp.readouts`
is reached as the submodule it is: callers write `from sagp.readouts import readouts`. `fit`,
`FittedGP` and `run_bo` are re-exported here from `sagp.gp` and `sagp.bo`, lazily via
`__getattr__` (PEP 562) rather than eagerly -- matching `synthobj/__init__.py` -- so `import sagp`
imports no submodule.

`numpyro.set_platform`/`set_host_device_count`/`enable_x64` run here, at package import, before
any of this package's code creates a JAX array: every later task depends on float64 and the
platform -- a visible GPU first, the CPU otherwise -- being set before the first array exists, not
after. Torch's default dtype and thread count are set here for the same reason numpyro's are.
"""
import os
from typing import Any

import jax
import numpyro
import torch
from jax._src.hardware_utils import has_visible_nvidia_gpu

# A GPU first, the CPU otherwise. `cuda` is requested only where JAX's own test finds an NVIDIA
# device, since requesting it makes the CUDA plugin call cuInit, which on a CPU node fails and logs
# a traceback on every start. Where one is visible the list is explicit, so a GPU that will not
# start fails loudly instead of the job quietly running on its host's CPU. JAX_PLATFORMS=cpu still
# forces the CPU.
numpyro.set_platform(
    os.environ.get("JAX_PLATFORMS") or ("cuda,cpu" if has_visible_nvidia_gpu() else "cpu")
)
# JAX would otherwise claim 75 % of the GPU when its backend starts, leaving too little for torch,
# which runs the acquisition on the same device (`sagp.gp.fit`); it is read at that start, so here.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
numpyro.set_host_device_count(1)
numpyro.enable_x64()
# Belt-and-suspenders: this is what numpyro.enable_x64() does internally, but a test process may
# have imported jax -- and so fixed its dtype defaults -- before this package is ever imported.
jax.config.update("jax_enable_x64", True)
# Every torch factory call in sagp must produce float64 to match the JAX side, and an array task
# on SLURM must not oversubscribe its one core.
torch.set_default_dtype(torch.float64)
if "OMP_NUM_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

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
